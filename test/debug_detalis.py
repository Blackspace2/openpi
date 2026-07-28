"""
debug_details.py — openpi / π0.5 手动 Debug 脚本（VSCode 断点式，非 pdb）

目的
----
加载少量样本，在 VSCode 里用 Python Debugger（debugpy）打断点，逐行探究：
  1) 模型结构        —— 参数树、各子模块参数量、dtype
  2) 训练前向 / loss —— compute_loss 的张量流（图像/状态/动作/时间如何编码、走 PaliGemma）
  3) 推理采样        —— sample_actions 的去噪循环、KV-cache、动作解码
  4) 端到端管线      —— policy.infer 的数据变换（归一化/tokenize）+ 真实权重推理

关键技巧（务必理解）
--------------------
openpi 的训练用 jax.jit(train_step)、推理用 nnx_utils.module_jit(sample_actions) 包了一层 JIT。
JIT 之后函数体是“被追踪”的：断点命中的是抽象 tracer（没有具体数值），且 lax.while_loop 只追踪一次。
本脚本默认在 `jax.disable_jit()` 上下文里跑 —— 此时 jit 变 no-op，且 lax.while_loop / lax.cond
会退化成纯 Python 循环逐迭代执行，于是：
    ✅ 断点命中的是 **具体 jax.Array**（能看 .shape，用 np.asarray(x) 看数值）
    ✅ 能单步进入 sample_actions 去噪 `step()` 的**每一次迭代**
代价是慢（几十秒~几分钟，样本量小无所谓）。想跑快 / 复现真实执行图时加 `--jit`。

用法
----
命令行：
    conda activate openpi        # 或直接用绝对路径解释器 /root/miniforge3/envs/openpi/bin/python
    cd <repo>/project/openpi
    python test/debug_detalis.py --mode train           # 只跑训练前向（默认）
    python test/debug_detalis.py --mode all --jit        # 全部 + 开启 jit
    python test/debug_detalis.py --mode infer --random-init  # 随机权重, 最快
VSCode：
    打开本文件 → 在下面标了「👉 断点」的行、或直接在 src/openpi/models/pi0.py 里打断点
    → 按 F5 选 "openpi: debug_details"（见 .vscode/launch.json，已设 justMyCode=false）
    → 单步 / Step Into 观察张量。变量面板里对 jax.Array 用 `np.asarray(x)` 看数值。

推荐断点位置（src/openpi/models/pi0.py 内）
    Pi0.compute_loss   (~L189)  单次前向，全 eager，最适合看张量流
    Pi0.embed_prefix   (~L106)  图像经 SigLIP、文本经 Gemma embedding
    Pi0.embed_suffix   (~L140)  state / 带噪动作 / 时间的编码
    Pi0.sample_actions (~L217)  去噪循环（默认 eager 才能逐迭代进 step()）
"""

import argparse
import gc
import os
import sys

# ── 必须在 import jax 之前设置环境变量 ────────────────────────────────────────
# 设备选择：默认 GPU。用 `--device cpu` 或 env DEBUG_DEVICE=cpu 切到 CPU。
#
# 背景（为什么曾经默认 CPU，现在改回 GPU）：
#   本脚本默认关 jit（eager）以便断点命中具体张量、逐迭代进去噪循环。但 eager 下
#   XLA 不做算子融合/缓冲复用。cfg.model.create 随机初始化用 flax 默认
#   param_dtype=float32（3.35B 参数 ≈13.4GB），前向的中间激活会把 24G 显存撑爆。
#   —— 只是学习/看结构和张量流，不是真训练，精度无所谓，所以 load_model 里已把
#   随机初始化参数转成 bf16（≈6.7GB），前向（structure/train 的 forward/infer）
#   实测能在空闲的 24G 卡上跑通。反向传播（value_and_grad）eager 下没有梯度检查点/
#   重计算优化，需要同时保留全网络中间激活，比前向贵得多，即使 bf16 仍可能 OOM ——
#   debug_train_forward 里已把这一步包了 try/except，OOM 只警告不影响已跑通的前向。
#   想要反向也稳定跑通，或想更快/更贴近真实执行图，用 `--device cpu` 或 `--jit`。
_DEVICE = os.environ.get("DEBUG_DEVICE", "gpu").lower()
if "--device" in sys.argv:
    _DEVICE = sys.argv[sys.argv.index("--device") + 1].lower()
if _DEVICE == "cpu":
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
else:
    # 锁到指定 GPU（默认 0）；共享机器上关掉显存预分配，避免一上来吃满。
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("DEBUG_GPU", "0"))
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from jax import tree_util as jtu
import numpy as np

import openpi.models.model as _model
import openpi.shared.download as _download
import openpi.training.config as _config

# ── 全局配置 ─────────────────────────────────────────────────────────────────
CONFIG_NAME = "pi05_libero"
CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"  # 已缓存于 ~/.cache/openpi，不会重下
BATCH_SIZE = 2  # 少量样本；调大可观察批维度如何传播


# ── 工具函数 ─────────────────────────────────────────────────────────────────
def _key_str(k) -> str:
    """把 jax pytree path 的键转成可读字符串。"""
    for attr in ("name", "key", "idx"):
        if hasattr(k, attr):
            return str(getattr(k, attr))
    return str(k)


def _path_str(path) -> str:
    return "/".join(_key_str(k) for k in path)


def load_model(cfg, *, from_checkpoint: bool):
    """构建 π0.5 模型。

    from_checkpoint=True : 加载真实微调权重（bf16），数值真实，适合看推理效果。
    from_checkpoint=False: 随机初始化（cfg.model.create），最快，只想看结构/张量形状时用。
    """
    if from_checkpoint:
        ckpt_dir = _download.maybe_download(CHECKPOINT)  # 返回本地缓存路径
        print(f"[load] restoring params from {ckpt_dir}/params ...")
        params = _model.restore_params(ckpt_dir / "params", dtype=jnp.bfloat16)
        model = cfg.model.load(params)
    else:
        # 注意：cfg.model.create 用 flax 默认 param_dtype=float32 初始化参数
        # （Pi0Config.dtype="bfloat16" 只管激活计算 dtype，不管参数存储 dtype），
        # 3.35B 参数 -> 约 13.4GB。eager 模式下 GPU 显存吃紧时手动转 bf16 省一半。
        print("[load] random init (cfg.model.create) ...")
        model = cfg.model.create(jax.random.key(0))
        # 只是用来学习/看结构和张量流，不是真训练，精度无所谓 —— 转 bf16 省一半参数显存。
        print("[load] casting random-init params to bfloat16 ...")
        state = nnx.state(model, nnx.Param)
        state = jtu.tree_map(lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x, state)
        nnx.update(model, state)
    return model


# ── 1) 模型结构 ──────────────────────────────────────────────────────────────
def debug_model_structure(model) -> None:
    """打印参数树 / 各子模块参数量 / dtype。断点后在变量面板展开 model 逐层探索。"""
    print("\n================= 模型结构 =================")
    # 👉 断点：在这里展开 model.PaliGemma / model.action_in_proj / model.state_proj ...
    #    model 是 nnx.Module，可点进 model.PaliGemma.llm、model.action_out_proj 逐层看。
    state = nnx.state(model, nnx.Param)
    leaves = jtu.tree_leaves_with_path(state)

    total = 0
    groups: dict[str, list[int]] = {}
    sample_rows = []
    for path, arr in leaves:
        if not hasattr(arr, "shape"):
            continue
        n = int(np.prod(arr.shape))
        total += n
        top = _key_str(path[0]) if path else "?"
        g = groups.setdefault(top, [0, 0])
        g[0] += 1
        g[1] += n
        if len(sample_rows) < 25:
            sample_rows.append((_path_str(path), tuple(arr.shape), str(arr.dtype)))

    print(f"总参数量: {total:,} ({total / 1e9:.3f} B)")
    print("\n-- 顶层子模块参数量 --")
    for name, (cnt, num) in sorted(groups.items(), key=lambda kv: -kv[1][1]):
        print(f"  {name:<24} tensors={cnt:<5} params={num:,}")
    print("\n-- 前 25 个参数张量 (path / shape / dtype) --")
    for p, s, d in sample_rows:
        print(f"  {p:<60} {str(s):<22} {d}")


# ── 2) 训练前向 / loss（纯 eager，最佳张量流断点目标）───────────────────────────
def debug_train_forward(cfg, model) -> None:
    """构造 fake batch → compute_loss（flow-matching 回归 loss）。全程 eager，断点命中具体张量。"""
    print("\n================= 训练前向 / loss =================")
    obs = cfg.model.fake_obs(batch_size=BATCH_SIZE)  # 随机观测（图像/状态/tokenized prompt）
    actions = cfg.model.fake_act(batch_size=BATCH_SIZE)  # 随机动作 (B, action_horizon=10, action_dim=32)
    rng = jax.random.key(0)
    model.train()

    print("obs.state:", obs.state.shape, "| actions:", actions.shape)
    # 👉 断点：Step Into 进入 src/openpi/models/pi0.py::Pi0.compute_loss
    #    里面依次进 embed_prefix（图像→SigLIP、文本→Gemma emb）、embed_suffix（state/动作/时间）、
    #    PaliGemma.llm（transformer 前向），观察每个中间张量的 shape 与数值。
    per_chunk_loss = model.compute_loss(rng, obs, actions, train=True)
    print("per-chunk loss:", per_chunk_loss.shape, "| mean =", float(jnp.mean(per_chunk_loss)))

    # 反向：value_and_grad（注意 grad 会“追踪”函数，断点在此步命中的是 tracer，不是具体值）
    # 显存注意：eager 模式下反向传播没有 XLA 的重计算/激活检查点优化，需要同时保留
    # 整个网络所有层的中间激活，比前向贵得多。即使 bf16 参数，24G 卡仍可能 OOM ——
    # 这不是参数精度问题，是 eager 反向传播本身的固有开销。OOM 时下面会捕获并跳过，
    # 不影响你已经看到的前向张量流（这才是这个 debug 脚本的主要目的）。
    diff_state = nnx.DiffState(0, cfg.trainable_filter)

    def loss_fn(m):
        return jnp.mean(m.compute_loss(rng, obs, actions, train=True))

    try:
        loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model)
        gnorm = jtu.tree_reduce(lambda a, x: a + jnp.sum(x.astype(jnp.float32) ** 2), grads, 0.0) ** 0.5
        # 👉 断点：看 loss 标量与梯度全局范数；展开 grads 看各参数梯度
        print("loss =", float(loss), "| grad global-norm =", float(gnorm))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] backward (value_and_grad) 失败/OOM，跳过: {type(e).__name__}: {e}")
        print("[warn] 前向张量流已经看完；反向想跑通可以试 --jit 或 --device cpu。")


# ── 3) 推理采样（去噪循环）─────────────────────────────────────────────────────
def debug_inference_eager(cfg, model) -> None:
    """直接调 model.sample_actions（绕开 policy 的 module_jit）。默认 eager 下能逐迭代进去噪 step()。

    实测：sample_actions 先填 KV-cache 再逐步 decode，eager/无融合下峰值显存逼近 24G 卡
    上限（bf16 参数、mem_fraction 提到 0.97 都不够，缺口稳定在同一个 128MB 请求处，
    说明是真实显存不够而不是 cap 设小了）。GPU 上 OOM 会被捕获跳过；想让推理稳定跑通
    并逐步进 step()，用 `--device cpu`（主机内存够，慢但对单样本调试无所谓）。
    """
    print("\n================= 推理采样 sample_actions =================")
    obs = cfg.model.fake_obs(batch_size=BATCH_SIZE)
    rng = jax.random.key(0)
    model.eval()

    # 👉 断点：Step Into 进入 pi0.py::Pi0.sample_actions
    #    先看 embed_prefix + 一次 PaliGemma.llm 填 KV-cache（eager 具体值）；
    #    再进 jax.lax.while_loop 的 step()：disable_jit 下每个去噪步都会停，可看 x_t 逐步收敛。
    try:
        actions = model.sample_actions(rng, obs, num_steps=10)
        print("sampled actions:", actions.shape, "| dtype", actions.dtype)
        print("first action vector:", np.asarray(actions[0, 0]))
    except Exception as e:  # noqa: BLE001
        print(f"[warn] sample_actions 失败/OOM，跳过: {type(e).__name__}: {e}")
        print("[warn] eager+GPU 下 sample_actions 需要接近满卡显存；想稳定单步调试推理用 --device cpu。")


# ── 4) 端到端管线（真实变换 + 真实权重）────────────────────────────────────────
def debug_inference_pipeline(cfg) -> None:
    """create_trained_policy + 一个 LIBERO 样例 → policy.infer。用于调数据变换管线与真实输出。"""
    print("\n================= 端到端 policy.infer =================")
    import openpi.policies.libero_policy as libero_policy
    import openpi.policies.policy_config as policy_config

    policy = policy_config.create_trained_policy(cfg, CHECKPOINT)
    example = libero_policy.make_libero_example()  # 随机 LIBERO 观测（双路图像 + state + prompt）
    print("example keys:", list(example.keys()))
    # 👉 断点：Step Into policy.infer → 依次经 LiberoInputs 变换、Normalize(归一化)、tokenize prompt，
    #    再到 sample_actions（eager 下可继续深入），最后 Unnormalize 还原动作。
    # 实测：即使前面已释放主 model，这里内部走的仍是同一条 sample_actions（图像走 SigLIP
    # nn.scan），在满载 24G 卡上跟 debug_inference_eager 一样卡在几 MB 的缺口上，偶发 OOM。
    try:
        result = policy.infer(example)
        print("actions:", np.asarray(result["actions"]).shape, "| infer_ms =", result["policy_timing"]["infer_ms"])
    except Exception as e:  # noqa: BLE001
        print(f"[warn] policy.infer 失败/OOM，跳过: {type(e).__name__}: {e}")
        print("[warn] 同 sample_actions 的显存缺口；想稳定跑通用 --device cpu。")


# ── 入口 ─────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="openpi π0.5 手动 debug 脚本")
    parser.add_argument(
        "--mode",
        default="train",
        choices=["structure", "train", "infer", "pipeline", "all"],
        help="调试哪一部分（默认 train：最快且最适合看张量流）",
    )
    parser.add_argument("--jit", action="store_true", help="开启 JIT（更快/更贴近真实执行，但断点看到的是 tracer）")
    parser.add_argument("--random-init", action="store_true", help="随机初始化模型而非加载 12G checkpoint（更快）")
    parser.add_argument(
        "--device",
        default="gpu",
        choices=["cpu", "gpu"],
        help="gpu(默认，随机初始化已转 bf16，前向能跑；反向 OOM 会被捕获跳过) 或 cpu(反向也想稳定跑通时用)",
    )
    args = parser.parse_args()

    print(f"JAX devices: {jax.devices()}  (DEBUG_DEVICE={_DEVICE})")
    print(f"mode={args.mode}  jit={'on' if args.jit else 'OFF(eager, 可断点)'}  random_init={args.random_init}")

    cfg = _config.get_config(CONFIG_NAME)

    # pipeline 模式内部会用 create_trained_policy 自行加载**第二份**权重；
    # 其余模式在这里加载/构建模型。--mode all 时两份权重会同时驻留显存，
    # 叠加 eager 模式本就吃紧的激活显存很容易 OOM —— 所以 pipeline 开始前
    # 把第一份模型显式释放掉（见下方 holder["model"] = None）。
    need_model = args.mode in ("structure", "train", "infer", "all")
    holder = {"model": load_model(cfg, from_checkpoint=not args.random_init) if need_model else None}

    def run() -> None:
        if args.mode in ("structure", "all"):
            debug_model_structure(holder["model"])
        if args.mode in ("train", "all"):
            debug_train_forward(cfg, holder["model"])
        if args.mode in ("infer", "all"):
            debug_inference_eager(cfg, holder["model"])
        if args.mode in ("pipeline", "all"):
            if holder["model"] is not None:
                holder["model"] = None
                gc.collect()
            debug_inference_pipeline(cfg)

    if args.jit:
        run()
    else:
        # 默认：关闭 jit，使 while_loop / module_jit 退化为 Python 执行，断点命中具体张量。
        with jax.disable_jit():
            run()

    print("\n[done]")


if __name__ == "__main__":
    main()

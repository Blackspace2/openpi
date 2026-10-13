import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05  # 判断 pi0 or pi0.5
        self.tt_rtc_max_delay = config.tt_rtc_max_delay  # Training-Time RTC 开关，0=关闭；见 docs/apply_tt-rtc.md
        self.tt_rtc_delay_distribution = config.tt_rtc_delay_distribution  # delay 采样分布："exp" / "uniform"
        # 两个 Transformer expert
        paligemma_config = _gemma.get_config(config.paligemma_variant)  # Gemma-2B，负责 prefix
        action_expert_config = _gemma.get_config(config.action_expert_variant)  # 动作头 action-expert 300M，负责 suffix
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                # 两个 expert 会在 attention 中交互
                # action expert 的 动作 token 可以关注 PaliGemma 的图像和语言 token
                configs=[paligemma_config, action_expert_config], 
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])  # action expert 使用 adaRMSNorm
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",  # patch_size=14
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        # 动作输入投影 [B,H,action_dim] -> [B,H,width]
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)  
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        # 动作输出投影，把 action expert 输出投影为其在速度场中的速度
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(  # 模型条件信息，图像 tokens、任务/state tokens
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:  # b,s,emb -> batch_size, seq_len, emb_dim
        input_mask = []  # 图像（patch）有效性掩码
        ar_mask = []
        """
          ar_mask: autoregressive mask 自回归注意力掩码

        这里是控制 Attention Block 的边界：

        - False: 与前一个 token 属于同一块，可以块内双向注意。
        - True: 从当前 token 开始新的块，不能被前面的块看到。
            [imgs, prompt,state | action]
            [True, False,False | True False]
        """
        tokens = []  # prefix token
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)  # [B, 256, 2048],图像大小是224，patchsize=14,(224/14)^2=16^2=256

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )   # [B, 256]
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs) ，tokenized_prompt 是整数 token id, [B, 200]
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")  # 浮点 embedding, [B, 200, 2048]
            tokens.append(tokenized_inputs)  # [... imgs tokens ...  ; ... prompt tokens ...]
            input_mask.append(obs.tokenized_prompt_mask)  # obs.tokenized_prompt_mask [B, 200]
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)  # [B, 3*256+200, 2048]
        input_mask = jnp.concatenate(input_mask, axis=1)  # [B, 3*256+200]
        ar_mask = jnp.array(ar_mask)  # [3*256+200]
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self,
        obs: _model.Observation,
        noisy_actions: _model.Actions,
        timestep: at.Float[at.Array, " b"] | at.Float[at.Array, "b ah"],
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | at.Float[at.Array, "b ah emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token, state:[B,32]
            state_token = self.state_proj(obs.state)[:, None, :]  # [B, 1, 1024]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)  # [B, 10, 32] -> [B, 10, 1024]
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        # timestep is either shared across the whole action chunk ([B], the default) or, when
        # Training-Time RTC (tt_rtc_max_delay > 0, see docs/apply_tt-rtc.md) is active, one value per
        # action-chunk position ([B, ah]).
        per_token_time = timestep.ndim == 2
        time_flat = timestep.reshape(-1) if per_token_time else timestep
        time_emb = posemb_sincos(time_flat, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if per_token_time:
            time_emb = time_emb.reshape(*timestep.shape, -1)  # [b, ah, emb]
        if self.pi05:
            # time MLP (for adaRMS); Linear/swish act on the last axis, so this is unchanged whether
            # time_emb is [b, emb] (default) or [b, ah, emb] (Training-Time RTC per-token conditioning).
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)  # action chunk 共用一个 t
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)  # [B, 10, 2048]
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)  # [B, 10, 1024]
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)  # [B, 10, 1024]
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)  # 🚩 pi0: [state_token, action_time_tokens]  pi0.5: [action_tokens]
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))  
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))  # action 开始新的 attention block
        tokens = jnp.concatenate(tokens, axis=1)  # pi0:[B, 11, 1024] pi0.5:[B, 10, 1024]
        input_mask = jnp.concatenate(input_mask, axis=1)  # pi0:[B, 11] pi0.5:[B, 10]
        ar_mask = jnp.array(ar_mask)  # pi0:[11,] pi0.5 [10,]
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)  # obs 的图像预处理

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)  # 采样噪声 [B, 10, 32]
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        """
        time ~ beta(1.5, 1)
            range:[0.001, 1.000]，不采样极端干净的动作 
            shape:[B]
        均匀采样会让所有噪声程度获得同样的采样概率
        但是 π0 作者认为，高噪声区域更难：
            低噪声：
                x_t 已经很像正确动作，预测比较容易
            高噪声：
                x_t 几乎没有动作信息
        模型必须主要依靠图像、语言、机器人状态推断动作
        所以训练时让高噪声区域出现得更多
        """

        time_expanded = time[..., None, None]  # [B, 1, 1]
        x_t = time_expanded * noise + (1 - time_expanded) * actions  # 构造带噪动作 t=0, x_t=action 真实动作；t=1, x_t=noise 纯噪声
        u_t = noise - actions   # 速度场  [B, 10, 32], 真值 -> 噪声

        rtc_prefix_mask = None
        if self.tt_rtc_max_delay > 0:
            # Training-Time RTC (docs/TT-RTC.md, docs/apply_tt-rtc.md): simulate an inference delay by
            # pinning a random-length action-chunk prefix (positions already "committed" by a previous,
            # now-superseded inference call) to its ground-truth clean value, and only train the
            # flow-matching loss on the remaining postfix positions. openpi's time convention is
            # t=0 -> clean action, t=1 -> noise (opposite of the TT-RTC paper's own convention), so
            # "clean" below means time=0.0.
            # Sample delay ~ Exp-decay over {0, ..., tt_rtc_max_delay}, biased toward delay=0 (matches the
            # official reference `third_party/real-time-chunking-kinetix/src/model.py:280-282`, and real
            # deployment latency: most inference calls finish with little/no delay, occasional slow calls
            # produce a long tail up to tt_rtc_max_delay). Was plain uniform (jax.random.randint) before --
            # switched to match the reference now that it's confirmed which end of the range it favors.
            time_rng, delay_rng = jax.random.split(time_rng)
            n_delay_values = self.tt_rtc_max_delay + 1
            if self.tt_rtc_delay_distribution == "uniform":
                # Every delay in {0, ..., tt_rtc_max_delay} equally likely (TT-RTC paper, real-robot setup).
                delay = jax.random.randint(delay_rng, batch_shape, 0, n_delay_values)  # [*b]
            else:
                delay_weights = jnp.exp(jnp.arange(n_delay_values, dtype=jnp.float32)[::-1])
                delay_weights = delay_weights / jnp.sum(delay_weights)
                delay = jax.random.choice(delay_rng, n_delay_values, batch_shape, p=delay_weights)  # [*b]
            rtc_prefix_mask = jnp.arange(self.action_horizon) < delay[..., None]  # [*b, ah]
            time = jnp.where(rtc_prefix_mask, 0.0, time[..., None])  # [*b, ah]
            time_expanded = time[..., None]  # [*b, ah, 1], overrides the [*b,1,1] version above
            x_t = time_expanded * noise + (1 - time_expanded) * actions
            u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)  # [B, 3*256+200=968, 2048] [B, 968] [968,]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)  # pi0.5:[B, 10, 1024] [B, 10] [10] [B, 1024] (或 tt_rtc 时 time:[B,ah], adarms_cond:[B,ah,1024])
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)  # pi0.5 [B, 978]
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)  # pi0.5 [978,]
        attn_mask = make_attn_mask(input_mask, ar_mask)  # [B, 978, 978]
        positions = jnp.cumsum(input_mask, axis=1) - 1  # toekn 在序列中的位置，padding 会导致的空洞。利用 input_mask 跳过无效的 padding，让所有有效 Token 获得连续的位置编号，从而保证 RoPE 的位置计算正确
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )  # pi0.5 suffix_out [B, 10, 1024]
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])  # [B, 10, 32]

        per_token_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)  # [B ,10]
        if rtc_prefix_mask is not None:
            # zero out the (already-committed, non-predicted) prefix positions and rescale so that a
            # later plain jnp.mean(...) over the full `ah` axis (done by the training loop, not here)
            # still averages correctly over just the postfix positions -- matching the reference
            # implementation's `sum(loss*mask)/sum(mask)` normalization.
            postfix_mask = jnp.logical_not(rtc_prefix_mask)
            n_postfix = jnp.sum(postfix_mask, axis=-1, keepdims=True).clip(min=1)
            per_token_loss = jnp.where(postfix_mask, per_token_loss, 0.0) * (self.action_horizon / n_postfix)
        return per_token_loss

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,  # 默认是 10，从积分噪声到动作的迭代求解步数
        noise: at.Float[at.Array, "b ah ad"] | None = None,  # noise 就是 x_1 初始噪声 [B, action_horinzon, action_dim]
        action_prefix: at.Float[at.Array, "b ah ad"] | None = None,
        prefix_delay: at.Int[at.Array, " b"] | int | None = None,
    ) -> _model.Actions:
        """Training-Time RTC 的推理侧用法(需要模型是用 tt_rtc_max_delay>0 训练出来的，见
        docs/apply_tt-rtc.md)：传入上一次推理已产出、当前已提交执行的动作块 `action_prefix`
        (shape 与 noise 相同 [B, ah, ad])，以及本次预计的延迟步数 `prefix_delay`(标量或 [B,] 整数，
        表示前多少个动作位置保持 `action_prefix` 的值不变)。函数只会对 `prefix_delay:` 之后的位置去
        噪，`action_prefix` 对应位置的动作原样返回、全程不加噪也不积分——是 inference-time RTC
        (pseudoinverse guidance)的零额外计算开销替代方案。`action_prefix`/`prefix_delay` 都为 None
        时(默认)行为与之前完全一致。
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps  # 积分方向：采样过程从 t=1 (纯噪声) 减小到 t=0 (动作)，这里 t 是 Flow Matching 的噪声时间
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        rtc_prefix_mask = None
        if action_prefix is not None:
            if prefix_delay is None:
                raise ValueError("prefix_delay must be given when action_prefix is given")
            prefix_delay = jnp.broadcast_to(jnp.asarray(prefix_delay), (batch_size,))
            rtc_prefix_mask = jnp.arange(self.action_horizon)[None, :] < prefix_delay[:, None]  # [b, ah]
            # prefix positions are already-committed clean actions, not noise, right from t=1 onward
            noise = jnp.where(rtc_prefix_mask[..., None], action_prefix, noise)

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            if rtc_prefix_mask is not None:
                x_t = jnp.where(rtc_prefix_mask[..., None], action_prefix, x_t)
                time_input = jnp.where(
                    rtc_prefix_mask, 0.0, jnp.broadcast_to(time, (batch_size, self.action_horizon))
                )  # openpi convention: 0.0 = clean
            else:
                time_input = jnp.broadcast_to(time, batch_size)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, time_input
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            x_next = x_t + dt * v_t
            if rtc_prefix_mask is not None:
                x_next = jnp.where(rtc_prefix_mask[..., None], action_prefix, x_next)  # keep prefix pinned exactly
            return x_next, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))  # 最终的 action 结果
        return x_0

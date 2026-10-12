#!/usr/bin/env python3
"""Launcher that quiets tqdm_loggable before handing off to scripts/train.py.

`scripts/train.py:16` does `import tqdm_loggable.auto as tqdm`, and that package
emits one "Progress on: ..." log line every `log_message_rate` (default 10 s --
tqdm_logging.py:33). At ~25 s/step over 70 h that is ~25k lines of noise.

`log_message_rate` is a plain class attribute, so we raise it here *before* the
training script builds its progress bar. Nothing in the repo is modified.

Override the interval with TQDM_LOG_INTERVAL_SEC (default 600 = every 10 min).

Usage -- identical to invoking scripts/train.py, just swap the entrypoint:

    PYTHONUNBUFFERED=1 TQDM_LOG_INTERVAL_SEC=600 \
    uv run python tools/train_quiet.py pi05_mobile_aloha --exp-name=... [...]
"""

from __future__ import annotations

import os
import pathlib
import runpy
import sys
from datetime import timedelta

interval = int(os.environ.get("TQDM_LOG_INTERVAL_SEC", "600"))

from tqdm_loggable.tqdm_logging import tqdm_logging  # noqa: E402

tqdm_logging.log_message_rate = timedelta(seconds=interval)
print(f"[train_quiet] tqdm_loggable log_message_rate -> {interval}s", file=sys.stderr, flush=True)

repo_root = pathlib.Path(__file__).resolve().parent.parent
train_py = repo_root / "scripts" / "train.py"

# Hand argv over as if train.py had been invoked directly.
sys.argv = [str(train_py), *sys.argv[1:]]
runpy.run_path(str(train_py), run_name="__main__")

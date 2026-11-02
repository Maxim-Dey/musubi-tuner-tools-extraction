"""CLI SIGINT handling without models, training or CUDA."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("handler", ["ignored", "default", "custom", "worker"])
@pytest.mark.parametrize("entrypoint", ["wrapper", "module"])
def test_cli_interrupt_handler_in_subprocess(handler, entrypoint):
    root = Path(__file__).resolve().parents[1]
    script = r'''
import os
import runpy
import signal
import sys
import threading

mode, entrypoint, root = sys.argv[1:]
custom_calls = []
def custom(signum, frame):
    custom_calls.append(signum)
original = {"ignored": signal.SIG_IGN, "default": signal.default_int_handler,
            "custom": custom, "worker": signal.SIG_IGN}[mode]
signal.signal(signal.SIGINT, original)
from musubi_tuner import krea2_train_network as trainer
assert signal.getsignal(signal.SIGINT) == original  # Import alone has no effect.
cleaned = []
def train(self, args):
    expected = signal.default_int_handler if mode == "ignored" else original
    assert signal.getsignal(signal.SIGINT) == expected
    try:
        signal.raise_signal(signal.SIGINT)
    finally:
        cleaned.append(True)
trainer.NetworkTrainer.train = train
sys.argv = ["krea2_train_network"]
interrupted = []
def invoke():
    try:
        if entrypoint == "wrapper":
            runpy.run_path(root + "/krea2_train_network.py", run_name="__main__")
        else:
            runpy.run_module("musubi_tuner.krea2_train_network", run_name="__main__")
    except KeyboardInterrupt:
        interrupted.append(True)
if mode == "worker":
    thread = threading.Thread(target=invoke)
    thread.start()
    thread.join()
else:
    invoke()
assert cleaned == [True]
assert interrupted == ([True] if mode in ("ignored", "default") else [])
assert custom_calls == ([signal.SIGINT] if mode == "custom" else [])
assert signal.getsignal(signal.SIGINT) == (signal.default_int_handler if mode == "ignored" else original)
print("INTERRUPT_OK", flush=True)
# Avoid native-library shutdown handlers reacting a second time to the Windows
# console event; all Python interruption/cleanup assertions have completed.
os._exit(130 if interrupted else 0)
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, "-c", script, handler, entrypoint, str(root)],
        cwd=root, env=environment, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == (130 if handler in ("ignored", "default") else 0), result.stdout + result.stderr
    assert "INTERRUPT_OK" in result.stdout

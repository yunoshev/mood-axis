"""Unified backend for running pipeline scripts locally or remotely.

Shared across all scripts (01-07). Provides:
- add_backend_args(parser) — adds --backend / --gpu to argparse
- detect_backend() — auto-detect best local backend
- load_generate_fn(model_config, backend) — returns (generate_fn, cleanup_fn)
- run_on_vast(script, model, ...) — re-invokes script on Vast.ai with --backend pytorch

Usage in any script:
    from src.backend import add_backend_args, detect_backend, load_generate_fn, run_on_vast

    parser = argparse.ArgumentParser(...)
    add_backend_args(parser)
    args = parser.parse_args()

    if args.backend == "vast":
        run_on_vast(__file__, args.model, args.gpu, ...)
        return

    backend = resolve_backend(args.backend)
    generate_fn, cleanup = load_generate_fn(model_config, backend)
    # ... use generate_fn ...
    cleanup()
"""

import gc
import sys
import time
import os
import subprocess
import json
import logging
import tempfile
from pathlib import Path
from typing import Callable, Tuple, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
#  Argparse
# ---------------------------------------------------------------------------

BACKENDS = ["pytorch", "mlx", "modal", "vast", "auto"]


def add_backend_args(parser):
    """Add --backend and --gpu arguments to any script's argparse."""
    parser.add_argument(
        "--backend", choices=BACKENDS, default="auto",
        help="Compute backend: pytorch, mlx (local), modal, vast (remote), auto (detect)")
    parser.add_argument(
        "--gpu", default="RTX_3090",
        help="GPU type for remote backends (default: RTX_3090)")


# ---------------------------------------------------------------------------
#  Backend detection
# ---------------------------------------------------------------------------

def detect_backend() -> str:
    """Auto-detect best available local backend."""
    try:
        import mlx.core  # noqa: F401
        return "mlx"
    except ImportError:
        return "pytorch"


def resolve_backend(backend: str) -> str:
    """Resolve 'auto' to a concrete backend name."""
    if backend == "auto":
        return detect_backend()
    return backend


# ---------------------------------------------------------------------------
#  Local backends: pytorch / mlx
# ---------------------------------------------------------------------------

def load_generate_fn(model_config, backend: str, max_new_tokens: int = None,
                     log_fn: Callable = None):
    """Load model and return (generate_fn, cleanup_fn).

    generate_fn(messages, max_new_tokens=N) -> GenerationResult
    cleanup_fn() -> None (frees GPU memory)

    Args:
        model_config: ModelConfig from config/models.py
        backend: "pytorch" or "mlx"
        max_new_tokens: default max_new_tokens for generate_fn (overridable per-call)
        log_fn: optional logging function (default: logger.info)
    """
    _log = log_fn or logger.info
    template_kwargs = model_config.chat_template_kwargs

    if max_new_tokens is None:
        from config.settings import MAX_NEW_TOKENS
        max_new_tokens = MAX_NEW_TOKENS

    # Base models have no chat template / EOS — they generate to max_tokens.
    # Stop at natural boundaries (next "Q:" or blank line) and cap tokens.
    # Same pattern as scripts/01_calibrate.py.
    if model_config.is_base_model:
        base_stop_strings = ["\nQ:", "\nQuestion:", "\n\n"]
        max_new_tokens = min(max_new_tokens, 1024)
    else:
        base_stop_strings = None

    if backend == "mlx":
        from src.model.mlx_inference import (
            load_mlx_model,
            generate_with_hidden_states as mlx_gen,
        )
        _log(f"Loading model: {model_config.model_id} (MLX"
             f"{', ' + model_config.quantize if model_config.quantize else ''})")
        ctx = load_mlx_model(model_config.model_id,
                             quantize=model_config.quantize,
                             mlx_model_id=model_config.mlx_model_id)

        def generate_fn(messages, max_new_tokens=max_new_tokens):
            return mlx_gen(ctx, messages, max_new_tokens=max_new_tokens,
                           chat_template_kwargs=template_kwargs, save_extra=False,
                           stop_strings=base_stop_strings)

        def cleanup_fn():
            del ctx["model"]
            gc.collect()

        return generate_fn, cleanup_fn

    elif backend == "pytorch":
        from src.model.loader import load_model
        from src.model.inference import generate_with_hidden_states
        import torch

        _log(f"Loading model: {model_config.model_id} (PyTorch)")
        model, tokenizer = load_model(model_config.model_id)

        def generate_fn(messages, max_new_tokens=max_new_tokens):
            return generate_with_hidden_states(
                model, tokenizer, messages, max_new_tokens=max_new_tokens,
                save_extra=False, chat_template_kwargs=template_kwargs,
                stop_strings=base_stop_strings)

        def cleanup_fn():
            nonlocal model, tokenizer
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return generate_fn, cleanup_fn

    else:
        raise ValueError(f"load_generate_fn only supports pytorch/mlx, got: {backend}")


# ---------------------------------------------------------------------------
#  Vast.ai backend
# ---------------------------------------------------------------------------

VAST_SSH_KEY = Path.home() / ".ssh" / "vastai_key"
VAST_DOCKER_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel"
VAST_DISK_GB = 50
VAST_DEPS = [
    "transformers==4.57.3",
    "accelerate>=0.26",
    "numpy",
    "scipy",
    "sentencepiece",
    "protobuf",
    "filelock>=3.13",
    "huggingface_hub>=0.23",
]
VAST_SYNC_INTERVAL = 10  # seconds

# Timeouts for slow machine detection
VAST_SSH_TIMEOUT = 120       # seconds to wait for SSH
VAST_SETUP_TIMEOUT = 300     # seconds for deps install (5 min)
VAST_DOWNLOAD_TIMEOUT = 600  # seconds for model download start (10 min)
VAST_MAX_PROVISION_RETRIES = 3  # rent new machine if current is slow


class SlowMachineError(Exception):
    """Raised when a Vast instance is too slow (download, setup, etc)."""
    pass


def run_on_vast(script_path: str, model: str, gpu: str = "RTX_3090",
                extra_args: str = "", sync_paths: list = None,
                final_paths: list = None, upload_extra: list = None,
                instance_id: str = None, keep: bool = False):
    """Run a pipeline script on Vast.ai with incremental result sync.

    Re-invokes the same script on a rented GPU instance with --backend pytorch.
    Syncs partial results every 10s. Downloads final results when done.

    Auto-retries on slow machines: if SSH, deps install, or model download
    takes too long, destroys instance and rents a new one.

    Args:
        script_path: __file__ of the calling script
        model: model key (e.g. "qwen_7b")
        gpu: GPU type (e.g. "RTX_3090", "RTX_4090")
        extra_args: extra CLI args to pass (e.g. "--force --scenarios 30")
        sync_paths: glob patterns to sync during run (e.g. ["data/drift/*_partial.json"])
        final_paths: glob patterns to download at end (e.g. ["data/drift/*.json"])
        upload_extra: extra file paths to include in code upload (e.g. partial checkpoints)
        instance_id: reuse existing instance (skip provisioning)
        keep: don't destroy instance after run
    """
    sync_paths = sync_paths or []
    final_paths = final_paths or []

    script_rel = os.path.relpath(script_path, PROJECT_ROOT)
    remote_cmd = (f"python3 -u {script_rel} --model {model} "
                  f"--backend pytorch {extra_args}").strip()

    print(f"=== Vast.ai: {script_rel} ===", flush=True)
    print(f"Model: {model}", flush=True)
    print(f"GPU:   {gpu}", flush=True)
    print(f"Cmd:   {remote_cmd}", flush=True)
    print("", flush=True)

    skip_offers = set()

    for provision_attempt in range(VAST_MAX_PROVISION_RETRIES):
        rented_here = False
        instance = instance_id or os.environ.get("INSTANCE")
        ssh = scp_base = host = None

        def _destroy_instance():
            if not keep and rented_here and instance:
                print(f"Destroying instance {instance}...", flush=True)
                subprocess.run(["vastai", "destroy", "instance", instance],
                               capture_output=True)

        try:
            # --- Provision ---
            if not instance:
                instance = _vast_rent(gpu, skip_offers=skip_offers)
                rented_here = True

            ssh_cmd, scp_cmd = _vast_ssh_info(instance)
            ssh = ssh_cmd

            # --- Wait for SSH (with timeout) ---
            _vast_wait_ssh(ssh, timeout=VAST_SSH_TIMEOUT)

            # --- Setup with timeout ---
            _vast_setup_with_timeout(ssh, scp_cmd, model,
                                     extra_tar_paths=upload_extra,
                                     timeout=VAST_SETUP_TIMEOUT)

            # --- Verify GPU ---
            gpu_check = _ssh_run(ssh,
                'python3 -c "import torch; '
                "assert torch.cuda.is_available(), 'CUDA not available'; "
                "print(f'GPU: {torch.cuda.get_device_name(0)}, "
                "VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')\"")
            if "CUDA not available" in gpu_check or "Error" in gpu_check:
                raise SlowMachineError(f"CUDA not working: {gpu_check[:100]}")

            # --- Run + sync loop ---
            print(f"\n=== Running: {remote_cmd} ===", flush=True)
            print(f"Started at {time.strftime('%H:%M:%S')}", flush=True)

            remote_log = "/workspace/run.log"
            hf_token = _get_hf_token()
            hf_env = f"HF_TOKEN={hf_token} " if hf_token else ""
            launch_cmd = (
                f"cd /workspace && "
                f"setsid bash -c '{hf_env}HF_HUB_DISABLE_PROGRESS_BARS=1 TRANSFORMERS_VERBOSITY=error "
                f"{remote_cmd}' > {remote_log} 2>&1 </dev/null &\n"
                f"echo $! > /workspace/run.pid\n"
                f"sleep 1 && cat /workspace/run.pid"
            )
            pid_out = _ssh_run(ssh, launch_cmd)
            if pid_out:
                print(f"  Remote PID: {pid_out.strip().splitlines()[-1]}", flush=True)

            host = ssh[-1].split("@")[1]
            port = ssh[ssh.index("-p") + 1]
            scp_base = ["scp", "-q", "-o", "StrictHostKeyChecking=no",
                        "-i", str(VAST_SSH_KEY), "-P", port]

            # --- Monitor with model download timeout ---
            local_log = f"/tmp/vast_{model}_{Path(script_rel).stem}.log"
            _vast_sync_loop(ssh, scp_base, host, remote_log, local_log,
                            sync_paths, download_timeout=VAST_DOWNLOAD_TIMEOUT)

            # --- Download final results ---
            print("\n=== Downloading final results ===", flush=True)
            for pattern in final_paths:
                _scp_download(ssh, scp_base, host, pattern)

            _destroy_instance()
            return  # Success!

        except SlowMachineError as e:
            print(f"\n!!! Slow machine detected: {e}", flush=True)
            # Remember this machine to skip all its offers
            if rented_here and instance:
                try:
                    r = subprocess.run(
                        ["vastai", "show", "instance", instance, "--raw"],
                        capture_output=True, text=True)
                    info = json.loads(r.stdout)
                    machine_id = str(info.get("machine_id", ""))
                    if machine_id:
                        skip_offers.add(machine_id)
                        print(f"  Blacklisted machine {machine_id}", flush=True)
                except Exception:
                    pass
            _destroy_instance()
            if provision_attempt < VAST_MAX_PROVISION_RETRIES - 1:
                print(f"Retrying with different machine "
                      f"({provision_attempt + 2}/{VAST_MAX_PROVISION_RETRIES})...\n",
                      flush=True)
                instance_id = None  # Force new provisioning
                continue
            else:
                print(f"ERROR: Failed after {VAST_MAX_PROVISION_RETRIES} attempts",
                      flush=True)
                sys.exit(1)

        except KeyboardInterrupt:
            print(f"\n\n=== Ctrl+C ===", flush=True)
            if ssh and scp_base and host:
                print("Downloading partial results...", flush=True)
                for pattern in list(set(sync_paths + final_paths)):
                    _scp_download(ssh, scp_base, host, pattern)
            _destroy_instance()
            sys.exit(0)


def _vast_rent(gpu: str, max_retries: int = 3, skip_offers: set = None) -> str:
    """Search and rent GPU instance with retry on start failure."""
    skip_offers = skip_offers or set()

    for attempt in range(max_retries):
        print(f"Searching for {gpu}..." + (f" (attempt {attempt+1})" if attempt else ""),
              flush=True)
        result = subprocess.run(
            ["vastai", "search", "offers",
             f"gpu_name={gpu} num_gpus=1 rentable=true "
             f"disk_space>={VAST_DISK_GB} inet_down>=2000 reliability>0.95",
             "-o", "dph", "--raw"],
            capture_output=True, text=True)
        offers = json.loads(result.stdout)
        # Skip previously failed offers/machines
        offers = [o for o in offers
                  if str(o["id"]) not in skip_offers
                  and str(o.get("machine_id", "")) not in skip_offers]
        if not offers:
            print(f"ERROR: No {gpu} offers found", flush=True)
            sys.exit(1)

        offer_id = str(offers[0]["id"])
        price = offers[0].get("dph_total", 0)
        print(f"Renting offer {offer_id} (${price:.3f}/hr)...", flush=True)

        result = subprocess.run(
            ["vastai", "create", "instance", offer_id,
             "--image", VAST_DOCKER_IMAGE,
             "--disk", str(VAST_DISK_GB),
             "--ssh", "--direct", "--raw"],
            capture_output=True, text=True)
        data = json.loads(result.stdout)
        instance = str(data["new_contract"])
        print(f"Instance: {instance}", flush=True)

        # Wait for running (7 min timeout)
        print("Waiting for start", end="", flush=True)
        for _ in range(84):  # 84 × 5s = 7 min
            r = subprocess.run(
                ["vastai", "show", "instance", instance, "--raw"],
                capture_output=True, text=True)
            status = json.loads(r.stdout).get("actual_status", "")
            if status == "running":
                print(" running!", flush=True)
                return instance
            print(".", end="", flush=True)
            time.sleep(5)

        # Failed to start — destroy and retry with different offer
        print(f"\nOffer {offer_id} didn't start, trying next...", flush=True)
        subprocess.run(["vastai", "destroy", "instance", instance], capture_output=True)
        skip_offers.add(offer_id)

    print(f"ERROR: Failed to start after {max_retries} attempts", flush=True)
    sys.exit(1)


def _vast_ssh_info(instance: str) -> Tuple[list, list]:
    """Get SSH and SCP command prefixes for an instance."""
    r = subprocess.run(
        ["vastai", "show", "instance", instance, "--raw"],
        capture_output=True, text=True)
    info = json.loads(r.stdout)
    host = info.get("ssh_host", "")
    port = str(info.get("ssh_port", ""))

    if not host or not port:
        print("ERROR: Can't get SSH info", flush=True)
        sys.exit(1)

    ssh_opts = ["-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
                "-i", str(VAST_SSH_KEY)]
    ssh = ["ssh"] + ssh_opts + ["-p", port, f"root@{host}"]
    scp = ["scp", "-q"] + ssh_opts + ["-P", port]

    print(f"SSH: {host}:{port}", flush=True)
    return ssh, scp


def _vast_wait_ssh(ssh: list, timeout: int = 120):
    """Wait for SSH to become available on the instance."""
    print("Waiting for SSH", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = subprocess.run(
                ssh + ["echo", "OK"],
                capture_output=True, text=True, timeout=15)
            if r.returncode == 0 and "OK" in r.stdout:
                print(" ready!", flush=True)
                return
        except subprocess.TimeoutExpired:
            pass
        print(".", end="", flush=True)
        time.sleep(5)
    print(flush=True)
    raise SlowMachineError(f"SSH not available after {timeout}s")


def _ssh_run(ssh: list, cmd: str, quiet: bool = False) -> str:
    """Run command via SSH, print output, return stdout."""
    r = subprocess.run(ssh + [cmd], capture_output=True, text=True)
    if not quiet:
        if r.stdout.strip():
            print(r.stdout.strip(), flush=True)
        if r.returncode != 0 and r.stderr.strip():
            print(f"  stderr: {r.stderr.strip()}", flush=True)
    return r.stdout.strip()


def _get_hf_token() -> Optional[str]:
    """Read local HuggingFace token (for passing to remote instances)."""
    # 1. Environment variable
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    # 2. huggingface-cli cache (standard location)
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        token = token_path.read_text().strip()
        if token:
            return token
    return None


def _vast_setup_with_timeout(ssh: list, scp: list, model: str,
                              extra_tar_paths: list = None,
                              timeout: int = 300):
    """Install deps with timeout, configure HF token, upload code.

    Raises SlowMachineError if deps install takes longer than timeout.
    """
    print("\n=== Installing dependencies ===", flush=True)
    deps_str = " ".join(f"'{d}'" for d in VAST_DEPS)
    try:
        r = subprocess.run(
            ssh + [f"pip install -q {deps_str} 2>&1 | tail -3"],
            capture_output=True, text=True, timeout=timeout)
        if r.stdout.strip():
            print(r.stdout.strip(), flush=True)
        if r.returncode != 0:
            raise SlowMachineError(f"Deps install failed (exit {r.returncode})")
    except subprocess.TimeoutExpired:
        raise SlowMachineError(f"Deps install timed out after {timeout}s")

    _vast_setup_rest(ssh, scp, model, extra_tar_paths)


def _vast_setup_rest(ssh: list, scp: list, model: str, extra_tar_paths: list = None):
    """Configure HF token and upload code (split from setup for timeout handling)."""

    # Pass HF token for fast authenticated downloads
    hf_token = _get_hf_token()
    if hf_token:
        print("Configuring HF token...", flush=True)
        _ssh_run(ssh, f"mkdir -p /root/.cache/huggingface && "
                 f"echo '{hf_token}' > /root/.cache/huggingface/token", quiet=True)
        verify = _ssh_run(ssh, "cat /root/.cache/huggingface/token | head -c 12", quiet=True)
        print(f"  Token: {verify}...", flush=True)
    else:
        print("WARNING: No HF token found — model downloads will be throttled!", flush=True)

    print("\n=== Uploading code ===", flush=True)
    tarball = tempfile.mktemp(suffix=".tar.gz")
    axes_file = f"data/axes/{model}_axes.npz"

    # Build tar with project files. Global fw is infrastructure, not vendored source.
    tar_cmd = ["tar", "czf", tarball,
               "-C", str(PROJECT_ROOT), "src/", "config/", "scripts/", "data/calibration/"]

    # Axes and extra files are relative to PROJECT_ROOT
    axes_path = PROJECT_ROOT / axes_file
    if axes_path.exists():
        tar_cmd.extend(["-C", str(PROJECT_ROOT), axes_file])

    if extra_tar_paths:
        for p in extra_tar_paths:
            if (PROJECT_ROOT / p).exists():
                tar_cmd.extend(["-C", str(PROJECT_ROOT), p])

    subprocess.run(tar_cmd, capture_output=True)
    subprocess.run(scp + [tarball, "root@" + ssh[-1].split("@")[1] + ":/workspace/code.tar.gz"],
                   capture_output=True)
    _ssh_run(ssh, "cd /workspace && tar xzf code.tar.gz && rm code.tar.gz")
    os.unlink(tarball)
    print("Code uploaded OK", flush=True)


def _scp_download(ssh: list, scp_base: list, host: str, pattern: str):
    """Download file(s) matching pattern from remote."""
    local_dir = str(PROJECT_ROOT / os.path.dirname(pattern))
    os.makedirs(local_dir, exist_ok=True)

    # First list matching files on remote with full paths
    # Use -d to prevent ls from listing directory contents without paths
    # Use find for directories to get full paths
    remote_pattern = f"/workspace/{pattern}"
    files_str = _ssh_run(ssh,
        f"find {remote_pattern} -maxdepth 1 -type f 2>/dev/null || "
        f"ls -d {remote_pattern} 2>/dev/null || true")
    if not files_str.strip():
        print(f"  Skip: {pattern} (no files)", flush=True)
        return

    for remote_path in files_str.strip().split("\n"):
        remote_path = remote_path.strip()
        if not remote_path or not remote_path.startswith("/"):
            continue  # skip SSH banner lines, empty lines
        r = subprocess.run(
            scp_base + [f"root@{host}:{remote_path}", local_dir + "/"],
            capture_output=True, text=True)
        fname = os.path.basename(remote_path)
        if r.returncode == 0:
            print(f"  OK: {fname}", flush=True)
        else:
            print(f"  FAIL: {fname} — {r.stderr.strip()}", flush=True)


def _vast_sync_loop(ssh: list, scp_base: list, host: str, remote_log: str,
                    local_log: str, sync_paths: list,
                    download_timeout: int = 600):
    """Tail remote log + sync partial results until remote process exits.

    Raises SlowMachineError if the script doesn't produce real output
    (beyond model loading) within download_timeout seconds.
    This catches slow model downloads that would waste hours.
    """
    prev_lines = 0
    start_time = time.time()
    saw_real_progress = False  # True once we see actual computation output

    while True:
        # Tail new log lines first (so we always see output, even if process died)
        total_s = _ssh_run(ssh, f"wc -l < {remote_log} 2>/dev/null || echo 0", quiet=True)
        # Parse: last pure-integer line (Vast SSH may prepend welcome banner)
        total = 0
        for _line in total_s.strip().split('\n'):
            _line = _line.strip()
            if _line.isdigit():
                total = int(_line)
        new_text = ""
        if total > prev_lines:
            skip = prev_lines + 1
            new_text = _ssh_run(ssh, f"sed -n '{skip},{total}p' {remote_log}")
            if new_text:
                with open(local_log, "a") as f:
                    f.write(new_text + "\n")
            prev_lines = total

        # Detect real progress: script started actual computation
        # (not just "Loading model..." or "Loading weights...")
        if new_text and not saw_real_progress:
            for line in new_text.split("\n"):
                # Real progress indicators from our scripts
                if any(marker in line for marker in [
                    "/50 (", "/30 (", "/20 (", "/15 (",  # sample progress
                    "d'=", "acc=",                        # calibration result
                    "alpha=", "slope=",                    # steering result
                    "Axis ", "=== Axis",                   # axis start
                    "Saved to ", "Saved partial",          # output saved
                ]):
                    saw_real_progress = True
                    elapsed = time.time() - start_time
                    print(f"  [Model loaded in {elapsed:.0f}s]", flush=True)
                    break

        # Check download timeout
        if not saw_real_progress and (time.time() - start_time) > download_timeout:
            # Check for .incomplete files as confirmation of slow download
            incomplete = _ssh_run(ssh,
                "ls -la /root/.cache/huggingface/hub/models--*/blobs/*.incomplete 2>/dev/null "
                "| wc -l", quiet=True)
            n_incomplete = 0
            for _line in incomplete.strip().split('\n'):
                _line = _line.strip()
                if _line.isdigit():
                    n_incomplete = int(_line)
            raise SlowMachineError(
                f"No real progress after {download_timeout}s "
                f"(likely slow model download, {n_incomplete} .incomplete files)")

        # Check if remote process is alive
        alive = _ssh_run(ssh, "ps -p $(cat /workspace/run.pid 2>/dev/null) "
                              "> /dev/null 2>&1 && echo YES || echo NO", quiet=True)

        if "NO" in alive:
            # Process died — check if it was an error (e.g. OOM, import error)
            if not saw_real_progress:
                # Process died without producing real output — likely a crash
                last_lines = _ssh_run(ssh, f"tail -5 {remote_log}", quiet=True)
                print(f"\nRemote process died without progress at "
                      f"{time.strftime('%H:%M:%S')}", flush=True)
                if last_lines:
                    print(f"  Last output:\n{last_lines}", flush=True)
                raise SlowMachineError(
                    f"Process died before producing results")
            print(f"\nRemote process finished at {time.strftime('%H:%M:%S')}", flush=True)
            break

        # Sync partial results
        for pattern in sync_paths:
            local_dir = str(PROJECT_ROOT / os.path.dirname(pattern))
            os.makedirs(local_dir, exist_ok=True)
            subprocess.run(
                scp_base + [f"root@{host}:/workspace/{pattern}", local_dir + "/"],
                capture_output=True)

        time.sleep(VAST_SYNC_INTERVAL)

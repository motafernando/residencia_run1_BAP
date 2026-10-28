"""Wait for the Mistral grid, then download, check and run Llama P0 once."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
GRID = ROOT / "results" / "grid-20261026t211027z" / "status.json"
CACHE = ROOT.parent / "reproducao-bias-after-prompting" / "smoke-winobias" / "cache"
REPO = "meta-llama/Meta-Llama-3-8B-Instruct"
REVISION = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
VARIANTS = {"refined", "evaluation_task", "evaluation_gender_bias"}


def grid_gate(state):
    if state.get("status") in {"starting", "running"}:
        return "wait"
    if state.get("status") != "completed_needs_review":
        return "blocked"
    full = {row.get("variant"): row for row in state.get("completed", []) if row.get("stage") == "full"}
    return "ready" if set(full) == VARIANTS and all(
        item.get("records_written") == 9504 and item.get("execution_errors") == 0
        for item in full.values()) else "blocked"


def required_files(files):
    names = {"config.json", "generation_config.json", "model.safetensors.index.json",
             "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    names.update(f"model-{i:05d}-of-00004.safetensors" for i in range(1, 5))
    if not names.issubset(files):
        raise ValueError("Official inventory is missing required files: " + ", ".join(sorted(names - set(files))))
    return sorted(names)


def find_resumable_run(parent):
    runs = sorted(p for p in Path(parent).glob('*-inference') if (p / 'run.json').is_file())
    if len(runs) > 1:
        raise ValueError('More than one initialized run found; review required')
    return runs[0] if runs else None


def valid_stage_result(data, stage, variant, require_full_gpu=False):
    expected = 72 if stage == 'pilot' else 9504
    return (data.get('status') == 'inference_finished_needs_manual_review'
            and data.get('records_written') == expected and data.get('execution_errors') == 0
            and data.get('prompt_variant') == variant
            and (not require_full_gpu or data.get('model_device_map') == {'': '0'}))


def image_reference(state, started_inference):
    return state['image_id'] if started_inference else state.get('image_tag', 'bias-winobias-v4:0.1')


def manifest_for(session, stage, state):
    variant = state.get('prompt_variant', 'original')
    return session / 'configs' / f'{stage}-{variant}.json'


def reusable_gpu_pilot(pilot_config, full_config, run_data, image_id):
    ignored = {'sample', 'output_root', 'purpose'}
    relevant = lambda config: {k: v for k, v in config.items() if k not in ignored}
    return (relevant(pilot_config) == relevant(full_config)
            and run_data.get('status') == 'inference_finished_needs_manual_review'
            and run_data.get('sample_size') == 24 and run_data.get('records_written') == 72
            and run_data.get('execution_errors') == 0 and run_data.get('model_device_map') == {'': '0'}
            and run_data.get('environment', {}).get('image_id') == image_id)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open('w', encoding='utf-8') as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def file_digests(path):
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1(b"blob " + str(size).encode() + b"\0")
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            sha256.update(block)
            git_sha1.update(block)
    return size, sha256.hexdigest(), git_sha1.hexdigest()


def run_command(command, log_path):
    with log_path.open("ab") as log:
        log.write(f"\n--- invocation {utc_now()} ---\n".encode())
        log.flush()
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if result.returncode:
        raise RuntimeError(f"Command failed with exit code {result.returncode}; see {log_path.name}")


def powershell(manifest, image_id, *switches):
    return ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", str(ROOT / "run-docker.ps1"), "-Manifest", str(manifest.relative_to(ROOT)),
            "-ImageTag", image_id, *switches, "-CacheDirectory", str(CACHE)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id")
    parser.add_argument("--resume-existing", action="store_true")
    args = parser.parse_args()
    if not args.session_id.startswith("llama-") or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.session_id):
        raise ValueError("Invalid session identifier")
    session = ROOT / "results" / args.session_id
    if not args.resume_existing:
        session.mkdir(parents=True, exist_ok=False)
        (session / "configs").mkdir()
    elif not session.is_dir():
        raise FileNotFoundError('Session to resume does not exist')
    import msvcrt
    supervisor_lock = (session / '.supervisor.lock').open('a+b')
    supervisor_lock.seek(0)
    if supervisor_lock.read(1) == b'':
        supervisor_lock.write(b'1')
        supervisor_lock.flush()
    supervisor_lock.seek(0)
    try:
        msvcrt.locking(supervisor_lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError as exc:
        supervisor_lock.close()
        raise RuntimeError('Another supervisor is already active') from exc
    state_path = session / "status.json"
    state = json.loads(state_path.read_text(encoding='utf8')) if args.resume_existing else {"session_id": args.session_id, "status": "waiting_for_mistral_grid", "created_utc": utc_now(),
             "grid_status_file": str(GRID.relative_to(ROOT)), "repo_id": REPO, "revision": REVISION,
             "prompt_variant": "original", "seed": 2026, "dtype": "bfloat16", "quantization": None,
             "completed": [], "downloaded_files": []}
    write_json(state_path, state)
    monitor = None
    try:
        if state['status'] == 'completed_needs_review':
            return
        old_status = state['status']
        if args.resume_existing:
            state.setdefault('supervisor_restarts', []).append({'started_utc': utc_now(), 'previous_status': old_status,
                                                                 'previous_error': state.get('error')})
            state.pop('error', None)
            state.pop('error_type', None)
        os.environ['WINOBIAS_QUEUE_ID'] = args.session_id
        while subprocess.run(['docker', 'info'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            state.update(status='waiting_for_docker', stage_started_utc=utc_now())
            write_json(state_path, state)
            time.sleep(30)
        started_inference = any(find_resumable_run(session / stage) for stage in ('pilot', 'full'))
        image_id = subprocess.check_output(["docker", "image", "inspect", "--format", "{{.Id}}",
                                            image_reference(state, started_inference)], text=True).strip()
        if not image_id.startswith("sha256:"):
            raise ValueError("Could not identify the Docker image")
        if args.resume_existing and started_inference and state.get('image_id') != image_id:
            raise ValueError('Docker image differs from the previous execution')
        state["image_id"] = image_id
        state['supervisor_pid'] = os.getpid()
        state['python_path'] = sys.executable
        if not args.resume_existing:
            state["input_sha256"] = {}
            for name in ("preflight.py", "data/prepared/all_records.json", "data/prepared/smoke24.json", "data/prepared/inventory.json"):
                state["input_sha256"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            state["frozen_manifests"] = {}
            for stage in ("pilot", "full"):
                source = ROOT / "configs" / ("llama-grid-pilot_original.json" if stage == "pilot" else "llama-grid-original.json")
                config = json.loads(source.read_text(encoding="utf-8"))
                config["output_root"] = f"results/{args.session_id}/{stage}"
                manifest = session / "configs" / f"{stage}-original.json"
                write_json(manifest, config)
                state["frozen_manifests"][stage] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        state['status'] = 'waiting_for_mistral_grid'
        write_json(state_path, state)
        while True:
            grid = json.loads(GRID.read_text(encoding="utf-8"))
            gate = grid_gate(grid)
            if gate == "ready":
                break
            if gate == "blocked":
                raise RuntimeError("Mistral grid did not finish successfully; Llama was not started")
            time.sleep(60)

        state.update(status="checking_checkpoint", stage_started_utc=utc_now())
        write_json(state_path, state)
        inventory = json.loads((ROOT / "checkpoint-history-llama.json").read_text(encoding="utf-8"))
        snapshot = inventory["snapshots"]["candidate_2025"]
        if inventory["repo_id"] != REPO or snapshot["revision"] != REVISION:
            raise ValueError("Reviewed official inventory does not match pinned checkpoint")
        files = snapshot["files"]
        selected = required_files(files)
        CACHE.mkdir(parents=True, exist_ok=True)
        required_space = sum(files[name]["size_bytes"] for name in selected) + 2 * 1024**3
        if shutil.disk_usage(CACHE).free < required_space:
            raise RuntimeError("Insufficient free disk space for the checkpoint and download headroom")
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        from huggingface_hub import HfApi, hf_hub_download
        info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True, timeout=30)
        if info.sha != REVISION:
            raise ValueError("Official repository resolved to a different revision")
        upstream = {item.rfilename: item for item in info.siblings}
        state['downloaded_files'] = []
        for name in selected:
            recorded = files[name]
            item = upstream.get(name)
            if item is None or item.size != recorded["size_bytes"]:
                raise ValueError("Official metadata differs from the reviewed inventory: " + name)
            if recorded["lfs_sha256"]:
                if not item.lfs or item.lfs.sha256 != recorded["lfs_sha256"]:
                    raise ValueError("Official LFS hash differs from the reviewed inventory: " + name)
            elif item.blob_id != recorded["git_blob_id"]:
                raise ValueError("Official Git blob differs from the reviewed inventory: " + name)
        for name in selected:
            state.update(status="downloading", current_file=name, stage_started_utc=utc_now())
            write_json(state_path, state)
            local = Path(hf_hub_download(REPO, filename=name, revision=REVISION,
                                         cache_dir=str(CACHE / "huggingface" / "hub")))
            size, sha256, git_sha1 = file_digests(local)
            expected = files[name]
            if size != expected["size_bytes"] or (expected["lfs_sha256"] and sha256 != expected["lfs_sha256"]) or (not expected["lfs_sha256"] and git_sha1 != expected["git_blob_id"]):
                raise ValueError("Downloaded file failed verification: " + name)
            state["downloaded_files"].append({"file": name, "bytes": size, "sha256": sha256})
            write_json(state_path, state)
        state.pop("current_file", None)

        for name, expected in state["input_sha256"].items():
            if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError("An input changed while waiting: " + name)
        for stage, expected in state["frozen_manifests"].items():
            if hashlib.sha256(manifest_for(session, stage, state).read_bytes()).hexdigest() != expected:
                raise RuntimeError("A frozen manifest changed while waiting: " + stage)

        while subprocess.check_output(['docker', 'ps', '-q', '--filter', f'label=winobias.queue={args.session_id}'], text=True).strip():
            state.update(status='waiting_for_existing_container', stage_started_utc=utc_now())
            write_json(state_path, state)
            time.sleep(60)
        while True:
            used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
            state["gpu_used_before_launch_mib"] = int(used.strip().splitlines()[0])
            if state["gpu_used_before_launch_mib"] <= 5000:
                break
            state.update(status="waiting_for_gpu", stage_started_utc=utc_now())
            write_json(state_path, state)
            time.sleep(60)

        monitor_file = f"gpu-monitor-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
        monitor = subprocess.Popen([
            "nvidia-smi", "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits", "-l", "1", "-f", str(session / monitor_file)
        ], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        state["gpu_monitor_pid"] = monitor.pid
        state["gpu_monitor_file"] = monitor_file
        full_manifest = manifest_for(session, 'full', state)
        state.update(status="preflight", stage_started_utc=utc_now(), log="preflight.log")
        write_json(state_path, state)
        run_command(powershell(full_manifest, image_id, "-Preflight"), session / "preflight.log")

        for stage in ("pilot", "full"):
            manifest = manifest_for(session, stage, state)
            config = json.loads(manifest.read_text(encoding="utf-8"))
            if stage == 'pilot' and state.get('reuse_pilot'):
                previous = (ROOT / state['reuse_pilot']).resolve()
                if not previous.is_relative_to((ROOT / 'results').resolve()):
                    raise ValueError('Pilot is outside results')
                previous_data = json.loads((previous / 'run.json').read_text(encoding='utf8'))
                previous_config = json.loads((previous / 'config.json').read_text(encoding='utf8'))
                full_config = json.loads(manifest_for(session, 'full', state).read_text(encoding='utf8'))
                if not reusable_gpu_pilot(previous_config, full_config, previous_data, image_id):
                    raise ValueError('Prior GPU pilot is not compatible with this full run')
                state['completed'] = [item for item in state['completed'] if item.get('stage') != 'pilot']
                state['completed'].append({'stage': 'pilot', 'run_dir': str(previous.relative_to(ROOT)),
                                           'records_written': 72, 'execution_errors': 0, 'reused': True,
                                           'finished_utc': utc_now()})
                write_json(state_path, state)
                continue
            state.update(status="running", stage=stage, stage_started_utc=utc_now(),
                         result_parent=config["output_root"], manifest=str(manifest.relative_to(ROOT)),
                         manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(), log=f"{stage}-{state.get('prompt_variant', 'original')}.log")
            write_json(state_path, state)
            expected_rows = 72 if stage == "pilot" else 9504
            for attempt in range(1, 4):
                run_dir = find_resumable_run(ROOT / config['output_root'])
                run_data = json.loads((run_dir / 'run.json').read_text(encoding='utf8')) if run_dir else {}
                if valid_stage_result(run_data, stage, state['prompt_variant'], state.get('device_map_requested') == {'': 0}):
                    break
                switches = ['-Inference'] + (['-ResumePath', str(run_dir)] if run_dir else [])
                state['stage_attempt'] = attempt
                write_json(state_path, state)
                try:
                    run_command(powershell(manifest, image_id, *switches), session / state['log'])
                except RuntimeError:
                    if attempt == 3:
                        raise
                    time.sleep(30)
            run_dir = find_resumable_run(ROOT / config['output_root'])
            if run_dir is None:
                raise RuntimeError(f'No initialized {stage} run')
            run_data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            if not valid_stage_result(run_data, stage, state['prompt_variant'], state.get('device_map_requested') == {'': 0}):
                raise RuntimeError(f"Inconsistent Llama {stage} run manifest")
            state['completed'] = [item for item in state['completed'] if item.get('stage') != stage]
            state["completed"].append({"stage": stage, "run_dir": str(run_dir.relative_to(ROOT)),
                                       "records_written": expected_rows, "execution_errors": 0,
                                       "finished_utc": utc_now()})
            write_json(state_path, state)
        state.update(status="completed_needs_review", finished_utc=utc_now())
    except Exception as error:
        state.update(status="failed", finished_utc=utc_now(), error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if monitor is not None:
            monitor.terminate()
            try:
                monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill()
                monitor.wait()
        write_json(state_path, state)
        supervisor_lock.close()


if __name__ == "__main__":
    main()

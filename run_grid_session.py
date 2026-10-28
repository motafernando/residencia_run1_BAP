"""Run three WinoBias prompt variants sequentially, with pilots and GPU logging."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
VARIANTS = ("refined", "evaluation_task", "evaluation_gender_bias")
CACHE = ROOT.parent / "reproducao-bias-after-prompting" / "smoke-winobias" / "cache"


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session_id")
    args = parser.parse_args()
    if not args.session_id.startswith("grid-") or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in args.session_id):
        raise ValueError("Invalid session identifier")
    session = ROOT / "results" / args.session_id
    session.mkdir(parents=True, exist_ok=False)
    (session / "configs").mkdir()
    state_path = session / "status.json"
    state = {"session_id": args.session_id, "status": "starting", "created_utc": utc_now(),
             "seed": 2026, "dtype": "bfloat16", "quantization": None,
             "variants": list(VARIANTS), "completed": [], "gpu_monitor_file": "gpu-monitor.csv"}
    write_json(state_path, state)

    monitor = None
    try:
        if not CACHE.is_dir():
            raise FileNotFoundError("Mistral cache not found")
        monitor = subprocess.Popen([
            "nvidia-smi", "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits", "-l", "1", "-f", str(session / "gpu-monitor.csv")
        ], creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        state["gpu_monitor_pid"] = monitor.pid
        write_json(state_path, state)
        for stage in ("pilot", "full"):
            for variant in VARIANTS:
                base = ROOT / "configs" / f"mistral-grid-{variant}.json"
                config = json.loads(base.read_text(encoding="utf-8"))
                config["sample"] = "data/prepared/smoke24.json" if stage == "pilot" else "data/prepared/all_records.json"
                config["output_root"] = f"results/{args.session_id}/{stage}/{variant}"
                config["purpose"] = f"WinoBias prompt grid {stage}: {variant}, Mistral 7B Instruct v0.3"
                manifest = session / "configs" / f"{stage}-{variant}.json"
                write_json(manifest, config)
                state.update(status="running", stage=stage, variant=variant, stage_started_utc=utc_now(),
                             result_parent=config["output_root"], manifest=str(manifest.relative_to(ROOT)),
                             manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                             log=f"{stage}-{variant}.log")
                write_json(state_path, state)
                log_path = session / f"{stage}-{variant}.log"
                command = ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                           "-File", str(ROOT / "run-docker.ps1"), "-Manifest", str(manifest.relative_to(ROOT)),
                           "-Inference", "-CacheDirectory", str(CACHE)]
                with log_path.open("wb") as log:
                    process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                if process.returncode:
                    raise RuntimeError(f"{stage} {variant} failed with exit code {process.returncode}; see {log_path.name}")
                result_parent = ROOT / config["output_root"]
                runs = sorted(result_parent.glob("*-inference"))
                if len(runs) != 1:
                    raise RuntimeError(f"Unexpected number of run folders for {stage} {variant}: {len(runs)}")
                run_dir = runs[0]
                run_data = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
                expected_rows = 72 if stage == "pilot" else 9504
                if (run_data.get("execution_errors") != 0 or run_data.get("records_written") != expected_rows
                        or run_data.get("prompt_variant") != variant):
                    raise RuntimeError(f"Inconsistent run manifest for {stage} {variant}")
                state["completed"].append({"stage": stage, "variant": variant,
                                           "run_dir": str(run_dir.relative_to(ROOT)),
                                           "records_written": run_data["records_written"],
                                           "execution_errors": run_data["execution_errors"],
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


if __name__ == "__main__":
    main()

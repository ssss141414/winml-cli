from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path


workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
controller = workspace / "controller"
candidate = workspace / "candidate"
evidence = workspace / "hosted-evidence-fp16-refresh"
logs = evidence / "logs"
artifact_dir = evidence / "artifacts" / "fp16"
prior_download = workspace / "prior-acceptance"
logs.mkdir(parents=True)
artifact_dir.mkdir(parents=True)

CANDIDATE = "d951fccaba37275540cdf2757bfb6b3cf60e8ec8"
DEPENDENCY = "3708969b731425b0c6d4b97920d1b5e6519bb013"
LOCK_SHA = "f6672f52199fa442a7b1414ba943c4eccc365940206e430777a51a1374b29053"
ARTIFACT_SHA = "3146daac451a022a3f36740eaaa168936f718343e33a5eed5a83d9ea7260eab9"
RECIPE_SHA = "7cca30667bcf14d56ad1afdb3d273fe2284835bd28daa1826fc839ebf69cdeb8"
PRIOR_RUN_ID = "32695311485"
PRIOR_ARTIFACT = "tester-l4-d951fcca-evidence-4c68f03a"
records: list[dict[str, object]] = []


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(name: str, argv: list[str], cwd: Path, timeout: int) -> dict[str, object]:
    index = len(records) + 1
    stdout_path = logs / f"{index:03d}-{name}.stdout.txt"
    stderr_path = logs / f"{index:03d}-{name}.stderr.txt"
    started = time.time()
    timed_out = False
    try:
        completed = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        exit_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        timed_out = True
        exit_code = 124
        stdout = error.stdout or ""
        stderr = error.stderr or ""
    duration = round(time.time() - started, 3)
    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    record: dict[str, object] = {
        "index": index,
        "name": name,
        "argv": argv,
        "command": subprocess.list2cmdline(argv),
        "cwd": str(cwd),
        "exit_code": exit_code,
        "duration_seconds": duration,
        "timed_out": timed_out,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    metadata_path = logs / f"{index:03d}-{name}.metadata.json"
    record["metadata"] = str(metadata_path)
    write_json(metadata_path, record)
    records.append(record)
    print(f"{name}: exit={exit_code} duration={duration}s timeout={timed_out}", flush=True)
    return record


def combined(record: dict[str, object]) -> str:
    return Path(str(record["stdout"])).read_text(encoding="utf-8") + "\n" + Path(
        str(record["stderr"])
    ).read_text(encoding="utf-8")


def fail(message: str, exit_code: int = 1) -> None:
    write_json(evidence / "command-index.v2.json", records)
    write_json(evidence / "terminal-state.v1.json", {
        "phase": "failed",
        "success": False,
        "outer_exit_code": exit_code,
        "timed_out": any(bool(record["timed_out"]) for record in records),
        "error": message,
        "candidate_sha": CANDIDATE,
        "dependency_sha": DEPENDENCY,
        "artifact_sha256": ARTIFACT_SHA,
    })
    raise SystemExit(message)


git = shutil.which("git")
uv = shutil.which("uv")
gh = shutil.which("gh")
if not git or not uv or not gh:
    fail("git, uv, and gh are required")

head = run("candidate-head", [git, "rev-parse", "HEAD"], candidate, 60)
parent = run("candidate-parent", [git, "rev-parse", "HEAD^"], candidate, 60)
status = run("candidate-status", [git, "status", "--porcelain=v1"], candidate, 60)
if combined(head).strip() != CANDIDATE or combined(parent).strip() != DEPENDENCY or combined(status).strip():
    fail("immutable candidate identity or cleanliness mismatch")

lock = candidate / "uv.lock"
recipe = candidate / "examples/recipes/cross-encoder_ms-marco-MiniLM-L4-v2/cpu/cpu/reranking_fp16_config.json"
if sha256(lock) != LOCK_SHA or sha256(recipe) != RECIPE_SHA:
    fail("candidate lock or recipe hash mismatch")

sync = run("uv-sync-locked", [uv, "sync", "--locked", "--all-extras", "--all-groups"], candidate, 1800)
if sync["exit_code"] != 0:
    fail("hosted candidate-local exact-lock hydration failed")
python = candidate / ".venv" / "Scripts" / "python.exe"

probe_code = (
    "import importlib,json,pathlib,sys; names=['winml','onnx','onnxruntime','transformers']; "
    "print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':"
    "{n:{'version':getattr(importlib.import_module(n),'__version__',None),'root':"
    "str(pathlib.Path(importlib.import_module(n).__file__).resolve())} for n in names}},sort_keys=True))"
)
probe = run("environment-import-roots", [str(python), "-c", probe_code], candidate, 300)
if probe["exit_code"] != 0:
    fail("candidate-local import probe failed")

download = run(
    "download-prior-sealed-artifact",
    [gh, "run", "download", PRIOR_RUN_ID, "--repo", "ssss141414/winml-cli", "--name", PRIOR_ARTIFACT, "--dir", str(prior_download)],
    controller,
    1800,
)
if download["exit_code"] != 0:
    fail("prior sealed hosted artifact download failed")
models = [path for path in prior_download.rglob("model.onnx") if path.parent.name == "fp16" and path.parent.parent.name == "artifacts"]
configs = [path for path in prior_download.rglob("winml_build_config.json") if path.parent.name == "fp16" and path.parent.parent.name == "artifacts"]
if len(models) != 1 or len(configs) != 1 or sha256(models[0]) != ARTIFACT_SHA:
    fail("downloaded prior final fp16 artifact identity mismatch")
target_model = artifact_dir / "model.onnx"
target_config = artifact_dir / "winml_build_config.json"
shutil.copy2(models[0], target_model)
shutil.copy2(configs[0], target_config)

semantic_code = r'''
import collections, hashlib, json, sys
from pathlib import Path
import onnx
from onnx import TensorProto
model_path, build_config_path, recipe_path = map(Path, sys.argv[1:])
model = onnx.load(model_path)
counts = collections.Counter(TensorProto.DataType.Name(item.data_type) for item in model.graph.initializer)
fp16_payload = sum(len(item.raw_data) for item in model.graph.initializer if item.data_type == TensorProto.FLOAT16)
fp32_payload = sum(len(item.raw_data) for item in model.graph.initializer if item.data_type == TensorProto.FLOAT)
build_config = json.loads(build_config_path.read_text(encoding="utf-8"))
recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
cross_file = {
    "task_equal": build_config["loader"]["task"] == recipe["loader"]["task"] == "reranking",
    "model_class_equal": build_config["loader"]["model_class"] == recipe["loader"]["model_class"] == "AutoModelForSequenceClassification",
    "model_type_equal": build_config["loader"]["model_type"] == recipe["loader"]["model_type"] == "bert",
    "quant_mode_equal": build_config["quant"]["mode"] == recipe["quant"]["mode"] == "fp16",
    "keep_io_types_equal": build_config["quant"]["fp16_keep_io_types"] == recipe["quant"]["fp16_keep_io_types"] is True,
    "input_tensors_equal": build_config["export"]["input_tensors"] == recipe["export"]["input_tensors"],
}
result = {
    "success": counts["FLOAT16"] == 76 and counts["FLOAT"] == 0 and fp16_payload > 0 and fp32_payload == 0 and all(cross_file.values()),
    "model_bytes": model_path.stat().st_size,
    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    "initializer_types": dict(sorted(counts.items())),
    "float16_raw_payload_bytes": fp16_payload,
    "float_raw_payload_bytes": fp32_payload,
    "ir_version": model.ir_version,
    "default_opset": next(item.version for item in model.opset_import if not item.domain),
    "recipe_sha256": hashlib.sha256(recipe_path.read_bytes()).hexdigest(),
    "build_config_sha256": hashlib.sha256(build_config_path.read_bytes()).hexdigest(),
    "cross_file": cross_file,
}
print(json.dumps(result, sort_keys=True))
'''
semantic = run("fp16-semantics", [str(python), "-c", semantic_code, str(target_model), str(target_config), str(recipe)], candidate, 300)
if semantic["exit_code"] != 0:
    fail("fresh fp16 semantic inspection failed")
semantics = json.loads(Path(str(semantic["stdout"])).read_text(encoding="utf-8"))
write_json(evidence / "fp16-semantics.v2.json", semantics)
if not semantics["success"] or semantics["model_sha256"] != ARTIFACT_SHA:
    fail("fresh fp16 semantic checks failed")

providers = run("providers", [str(python), "-c", "import onnxruntime as ort; print(ort.get_available_providers())"], candidate, 120)
precision_perf_code = r'''
import subprocess, sys
import onnx
from winml.modelkit.session.session import WinMLSession
model_path = sys.argv[1]
precision = WinMLSession._get_precision(onnx.load(model_path, load_external_data=False))
print(f"Model Precision: {precision}", flush=True)
completed = subprocess.run([sys.executable, "-m", "winml.modelkit", "perf", "-m", model_path, "--ep", "cpu", "--device", "cpu", "--warmup", "2", "--iterations", "10"], check=False)
raise SystemExit(completed.returncode)
'''
perf = run("perf-fp16", [str(python), "-c", precision_perf_code, str(target_model)], candidate, 900)
plain = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", combined(perf))
precision = re.search(r"Model Precision:\s*(\S+)", plain)
latency = re.search(r"│\s*([0-9.]+)\s*│\s*([0-9.]+)\s*│\s*([0-9.]+)\s*│\s*([0-9.]+)\s*│\s*([0-9.]+)\s*│\s*([0-9.]+)\s*│\s*([0-9.]+)\s*│", plain)
throughput = re.search(r"Throughput:\s*([0-9.]+)\s*samples/sec", plain)
memory = re.search(r"RAM:\s*([0-9.]+)\s*MB\s*->\s*model load:\s*\+?([0-9.]+)\s*MB\s*\|\s*inference:\s*\+?([0-9.]+)\s*MB\s*\|\s*total:\s*\+?([0-9.]+)\s*MB", plain, re.S)
if perf["exit_code"] != 0 or providers["exit_code"] != 0 or not precision or precision.group(1) != "fp16" or not latency or not throughput or not memory:
    fail("fresh fp16 Perf output incomplete or precision mismatch")

metrics = {
    "model_precision_line": precision.group(0),
    "mean_ms": float(latency.group(1)),
    "p50_ms": float(latency.group(2)),
    "p90_ms": float(latency.group(3)),
    "p95_ms": float(latency.group(4)),
    "p99_ms": float(latency.group(5)),
    "min_ms": float(latency.group(6)),
    "max_ms": float(latency.group(7)),
    "throughput_samples_per_second": float(throughput.group(1)),
    "ram_before_mb": float(memory.group(1)),
    "ram_model_load_delta_mb": float(memory.group(2)),
    "ram_inference_delta_mb": float(memory.group(3)),
    "ram_total_delta_mb": float(memory.group(4)),
    "duration_seconds": perf["duration_seconds"],
    "exit_code": perf["exit_code"],
    "artifact_sha256": ARTIFACT_SHA,
}
write_json(evidence / "fp16-perf-metrics.v2.json", metrics)
write_json(evidence / "environment-provenance.v2.json", {
    "candidate_sha": CANDIDATE,
    "parent_sha": DEPENDENCY,
    "checkout_root": str(candidate),
    "environment_root": str(candidate / ".venv"),
    "interpreter_path": str(python),
    "command_runner_path": uv,
    "lock_path": str(lock),
    "lock_sha256": sha256(lock),
    "sync_command": sync["command"],
    "sync_exit_code": sync["exit_code"],
    "versions_record": probe["stdout"],
})
write_json(evidence / "command-index.v2.json", records)
write_json(evidence / "terminal-state.v1.json", {
    "phase": "fp16-refresh-completed",
    "success": True,
    "outer_exit_code": 0,
    "timed_out": False,
    "error": None,
    "candidate_sha": CANDIDATE,
    "dependency_sha": DEPENDENCY,
    "artifact_sha256": ARTIFACT_SHA,
})
print(json.dumps({"metrics": metrics, "semantics": semantics}, sort_keys=True))
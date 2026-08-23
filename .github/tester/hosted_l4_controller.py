import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


workspace = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
controller = workspace / "controller"
candidate = workspace / "candidate"
evidence = workspace / "hosted-evidence"
logs = evidence / "logs"
artifacts = evidence / "artifacts"
harness = evidence / "harness"
for path in (logs, artifacts, harness):
    path.mkdir(parents=True, exist_ok=True)
shutil.copy2(controller / ".github/tester/hosted_l4_controller.py", harness)
shutil.copy2(controller / ".github/tester/validate_l4_artifacts.py", harness)
shutil.copy2(controller / ".github/tester/l4_parity.py", harness)

candidate_sha = "d951fccaba37275540cdf2757bfb6b3cf60e8ec8"
parent_sha = "3708969b731425b0c6d4b97920d1b5e6519bb013"
model_revision = "777b2f369bc1c2f850df8bd367ed1654bda4497b"
dataset_revision = "56a6d0140cf6356659e2a7c1413286a774468d44"
fp32_recipe = candidate / "examples/recipes/cross-encoder_ms-marco-MiniLM-L4-v2/cpu/cpu/reranking_fp32_config.json"
fp16_recipe = candidate / "examples/recipes/cross-encoder_ms-marco-MiniLM-L4-v2/cpu/cpu/reranking_fp16_config.json"
fp32_dir = artifacts / "fp32"
fp16_dir = artifacts / "fp16"
model_dir = evidence / "hf"
records: list[dict[str, object]] = []


def run(name: str, args: list[str], timeout: int = 0, env: dict[str, str] | None = None) -> dict[str, object]:
    index = len(records) + 1
    stdout_path = logs / f"{index:03d}-{name}.stdout.txt"
    stderr_path = logs / f"{index:03d}-{name}.stderr.txt"
    metadata_path = logs / f"{index:03d}-{name}.metadata.json"
    started = time.time()
    timed_out = False
    exit_code = 124
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        try:
            completed = subprocess.run(
                args,
                cwd=candidate,
                env={**os.environ, **(env or {})},
                stdout=stdout,
                stderr=stderr,
                text=True,
                timeout=timeout or None,
                check=False,
            )
            exit_code = completed.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
    ended = time.time()
    record = {
        "index": index,
        "name": name,
        "executable": args[0],
        "arguments": args[1:],
        "command": subprocess.list2cmdline(args),
        "working_directory": str(candidate),
        "started_epoch": started,
        "ended_epoch": ended,
        "duration_seconds": round(ended - started, 3),
        "timeout_seconds": timeout,
        "timed_out": timed_out,
        "exit_code": exit_code,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "metadata": str(metadata_path),
    }
    metadata_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    records.append(record)
    print(f"{name}: exit={exit_code} duration={record['duration_seconds']}s timeout={timed_out}", flush=True)
    return record


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


uv = shutil.which("uv")
git = shutil.which("git")
gh = shutil.which("gh")
assert uv and git and gh
sync = run("uv-sync-locked", [uv, "sync", "--locked", "--all-extras", "--all-groups"], 1800)
python = candidate / ".venv/Scripts/python.exe"
ruff = candidate / ".venv/Scripts/ruff.exe"
mypy = candidate / ".venv/Scripts/mypy.exe"
pytest = candidate / ".venv/Scripts/pytest.exe"
if sync["exit_code"] != 0:
    raise SystemExit("Hosted exact-lock hydration failed")

probe_code = (
    "import importlib,json,pathlib,sys; names=['winml','ruff','mypy','pytest','transformers','onnx','onnxruntime','torch','datasets']; "
    "print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':{n:{'version':getattr(importlib.import_module(n),'__version__',None),'root':str(pathlib.Path(importlib.import_module(n).__file__).resolve())} for n in names}},sort_keys=True))"
)
versions = run("environment-import-roots", [str(python), "-c", probe_code], 300)
provenance = {
    "candidate_sha": candidate_sha,
    "parent_sha": parent_sha,
    "checkout_root": str(candidate),
    "environment_root": str(candidate / ".venv"),
    "interpreter_path": str(python),
    "command_runner_path": uv,
    "tool_paths": {"ruff": str(ruff), "mypy": str(mypy), "pytest": str(pytest)},
    "lock_path": str(candidate / "uv.lock"),
    "lock_sha256": sha256(candidate / "uv.lock"),
    "sync_command": sync["command"],
    "sync_exit_code": sync["exit_code"],
    "versions_record": versions["stdout"],
}
(evidence / "environment-provenance.hosted.v1.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

download_code = (
    "from huggingface_hub import snapshot_download; "
    f"print(snapshot_download('cross-encoder/ms-marco-MiniLM-L4-v2', revision='{model_revision}', local_dir=r'{model_dir}'))"
)
download = run("model-snapshot", [str(python), "-c", download_code], 1200)

build_fp32 = run(
    "build-fp32",
    [str(python), "-m", "winml.modelkit", "build", "-c", str(fp32_recipe), "-m", str(model_dir), "-o", str(fp32_dir)],
    1800,
)
build_fp16 = run(
    "build-fp16",
    [str(python), "-m", "winml.modelkit", "build", "-c", str(fp16_recipe), "-m", str(model_dir), "-o", str(fp16_dir), "--precision", "fp16"],
    1800,
)
build_pass = download["exit_code"] == build_fp32["exit_code"] == build_fp16["exit_code"] == 0

structure = None
perf_fp32 = None
perf_fp16 = None
parity = None
eval_result = None
analysis_records: list[dict[str, object]] = []
if build_pass:
    structure = run(
        "structure-and-precision",
        [str(python), str(controller / ".github/tester/validate_l4_artifacts.py"), str(evidence / "structure.v1.json"), str(fp32_dir / "model.onnx"), str(fp16_dir / "model.onnx")],
        300,
    )
    providers = run("providers", [str(python), "-c", "import onnxruntime as ort; print(ort.get_available_providers())"], 60)
    perf_fp32 = run(
        "perf-fp32",
        [str(python), "-m", "winml.modelkit", "perf", "-m", str(fp32_dir / "model.onnx"), "--ep", "cpu", "--device", "cpu", "--warmup", "2", "--iterations", "10"],
        900,
    )
    perf_fp16 = run(
        "perf-fp16",
        [str(python), "-m", "winml.modelkit", "perf", "-m", str(fp16_dir / "model.onnx"), "--ep", "cpu", "--device", "cpu", "--warmup", "2", "--iterations", "10"],
        900,
    )
    parity = run(
        "pytorch-onnx-parity",
        [str(python), str(controller / ".github/tester/l4_parity.py"), str(model_dir), str(fp32_dir / "model.onnx"), str(fp16_dir / "model.onnx"), str(evidence / "parity.v1.json")],
        1200,
    )
    if structure["exit_code"] == perf_fp32["exit_code"] == perf_fp16["exit_code"] == parity["exit_code"] == 0:
        eval_result = run(
            "scidocs-functional-smoke",
            [
                str(python), "-m", "winml.modelkit", "eval", "-m", str(fp32_dir / "model.onnx"),
                "--model-id", "cross-encoder/ms-marco-MiniLM-L4-v2", "--task", "reranking",
                "--dataset", "mteb/scidocs-reranking", "--dataset-revision", dataset_revision,
                "--split", "test", "--streaming", "--no-shuffle", "--samples", "2",
                "--column", "query_column=query", "--column", "positive_column=positive",
                "--column", "negative_column=negative", "--column", "max_candidates=10",
                "--ep", "cpu", "--device", "cpu", "-o", str(evidence / "scidocs-reranking-eval.json"), "--overwrite",
            ],
            1800,
        )

    rules_asset = evidence / "rules-asset"
    rules_asset.mkdir(exist_ok=True)
    rules_download = run("rules-download", [gh, "release", "download", "--repo", "microsoft/winml-cli", "--pattern", "rules-v*.zip", "--dir", str(rules_asset)], 600)
    if rules_download["exit_code"] == 0:
        archives = list(rules_asset.glob("*.zip"))
        if archives:
            rules_dir = candidate / "src/winml/modelkit/analyze/rules/runtime_check_rules"
            with zipfile.ZipFile(archives[0]) as archive:
                archive.extractall(rules_dir)
            for label, directory in (("fp32", fp32_dir), ("fp16", fp16_dir)):
                analysis_records.append(
                    run(
                        f"analyze-{label}",
                        [str(python), "-m", "winml.modelkit", "analyze", "--model", str(directory / "model.onnx"), "--ep", "all", "--output", str(evidence / f"analyze-{label}.json")],
                        1200,
                    )
                )

quality = []
quality.append(run("license-headers", [uv, "run", "pre-commit", "run", "insert-license", "--all-files"], 900))
quality.append(run("ruff", [str(ruff), "check", "src/", "tests/"], 900))
quality.append(run("mypy", [str(mypy), "-p", "winml.modelkit"], 1200))
partitions = {
    "analyze": ["tests/unit/analyze"],
    "models": ["tests/unit/models", "tests/unit/loader", "tests/unit/datasets", "tests/unit/export"],
    "optim": ["tests/unit/optim"],
    "commands": ["tests/unit/commands", "tests/unit/config", "tests/unit/build", "tests/unit/compiler", "tests/unit/session", "tests/unit/eval"],
    "remaining": ["tests/unit/core", "tests/unit/onnx", "tests/unit/cache", "tests/unit/utils", "tests/unit/test_helpers", "tests/unit/sysinfo", "tests/unit/inspect", "tests/unit/optracing", "tests/unit/serve", "tests/regression", "tests/cli"],
}
for name, paths in partitions.items():
    quality.append(run(f"pytest-{name}", [str(pytest), *paths, "--tb=short", "--no-cov", "-m", "not e2e and not npu and not gpu"], 1800))

summary = {
    "candidate_sha": candidate_sha,
    "parent_sha": parent_sha,
    "model_revision": model_revision,
    "dataset_revision": dataset_revision,
    "build_pass": build_pass,
    "structure_exit": None if structure is None else structure["exit_code"],
    "perf_exits": [None if perf_fp32 is None else perf_fp32["exit_code"], None if perf_fp16 is None else perf_fp16["exit_code"]],
    "parity_exit": None if parity is None else parity["exit_code"],
    "eval_exit": None if eval_result is None else eval_result["exit_code"],
    "analysis_exits": [record["exit_code"] for record in analysis_records],
    "quality_exits": {record["name"]: record["exit_code"] for record in quality},
    "commands": records,
}
(evidence / "hosted-summary.v1.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
(evidence / "command-index.hosted.v1.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
required = [build_pass, structure is not None and structure["exit_code"] == 0, perf_fp32 is not None and perf_fp32["exit_code"] == 0, perf_fp16 is not None and perf_fp16["exit_code"] == 0, parity is not None and parity["exit_code"] == 0, eval_result is not None and eval_result["exit_code"] == 0, all(record["exit_code"] == 0 for record in quality)]
raise SystemExit(0 if all(required) else 1)
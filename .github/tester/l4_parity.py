import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


model_dir = Path(sys.argv[1])
fp32_path = Path(sys.argv[2])
fp16_path = Path(sys.argv[3])
output = Path(sys.argv[4])
query = "What is the capital of France?"
passages = [
    "Paris is the capital and most populous city of France.",
    "France is a country in Western Europe with many historic cities.",
    "The Pacific Ocean is the largest ocean on Earth.",
]

tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
model.eval()


def tokenize(passage: str) -> dict[str, torch.Tensor]:
    return tokenizer(
        query,
        passage,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    )


with torch.no_grad():
    pytorch_logits = np.array([float(model(**tokenize(passage)).logits.reshape(-1)[0]) for passage in passages])


def onnx_logits(path: Path) -> np.ndarray:
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    scores = []
    for passage in passages:
        encoded = tokenize(passage)
        feeds = {name: encoded[name].numpy().astype(np.int32) for name in ("input_ids", "attention_mask", "token_type_ids")}
        scores.append(float(session.run(["logits"], feeds)[0].reshape(-1)[0]))
    return np.array(scores)


def compare(candidate: np.ndarray, cosine_floor: float, max_abs_ceiling: float) -> dict[str, object]:
    cosine = float(np.dot(pytorch_logits, candidate) / (np.linalg.norm(pytorch_logits) * np.linalg.norm(candidate)))
    max_abs = float(np.max(np.abs(pytorch_logits - candidate)))
    pt_order = np.argsort(-pytorch_logits).tolist()
    candidate_order = np.argsort(-candidate).tolist()
    return {
        "pytorch_logits": pytorch_logits.tolist(),
        "onnx_logits": candidate.tolist(),
        "cosine": cosine,
        "max_abs": max_abs,
        "pytorch_descending_order": pt_order,
        "onnx_descending_order": candidate_order,
        "order_preserved": pt_order == candidate_order,
        "cosine_floor": cosine_floor,
        "max_abs_ceiling": max_abs_ceiling,
        "pass": cosine >= cosine_floor and max_abs <= max_abs_ceiling and pt_order == candidate_order,
    }


result = {
    "pairs": [{"query": query, "passage": passage} for passage in passages],
    "fp32": compare(onnx_logits(fp32_path), 0.99999, 0.001),
    "fp16": compare(onnx_logits(fp16_path), 0.999, 0.1),
}
result["success"] = result["fp32"]["pass"] and result["fp16"]["pass"]
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if result["success"] else 1)

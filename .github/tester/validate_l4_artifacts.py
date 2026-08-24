import json
import sys
from collections import Counter
from pathlib import Path

import onnx
from onnx import TensorProto


def tensor_spec(value_info: onnx.ValueInfoProto) -> dict[str, object]:
    tensor_type = value_info.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        dims.append(dim.dim_value if dim.HasField("dim_value") else dim.dim_param)
    return {
        "name": value_info.name,
        "dtype": TensorProto.DataType.Name(tensor_type.elem_type),
        "shape": dims,
    }


def inspect(label: str, model_path: Path) -> dict[str, object]:
    model = onnx.load(model_path, load_external_data=False)
    initializer_types = Counter(TensorProto.DataType.Name(item.data_type) for item in model.graph.initializer)
    files = [path for path in model_path.parent.iterdir() if path.is_file()]
    return {
        "label": label,
        "model_path": str(model_path.resolve()),
        "ir_version": model.ir_version,
        "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "inputs": [tensor_spec(item) for item in model.graph.input],
        "outputs": [tensor_spec(item) for item in model.graph.output],
        "initializer_types": dict(sorted(initializer_types.items())),
        "model_bytes": model_path.stat().st_size,
        "directory_bytes": sum(path.stat().st_size for path in files),
        "files": [{"name": path.name, "bytes": path.stat().st_size} for path in sorted(files)],
    }


output = Path(sys.argv[1])
fp32 = inspect("fp32", Path(sys.argv[2]))
fp16 = inspect("fp16", Path(sys.argv[3]))
expected_inputs = [
    {"name": "input_ids", "dtype": "INT32", "shape": [1, 512]},
    {"name": "attention_mask", "dtype": "INT32", "shape": [1, 512]},
    {"name": "token_type_ids", "dtype": "INT32", "shape": [1, 512]},
]
checks = {
    "fp32_inputs": fp32["inputs"] == expected_inputs,
    "fp16_inputs": fp16["inputs"] == expected_inputs,
    "fp32_output": fp32["outputs"] == [{"name": "logits", "dtype": "FLOAT", "shape": [1, 1]}],
    "fp16_output": fp16["outputs"] == [{"name": "logits", "dtype": "FLOAT", "shape": [1, 1]}],
    "fp32_float_initializers": fp32["initializer_types"].get("FLOAT", 0) > 0,
    "fp16_float16_initializers": fp16["initializer_types"].get("FLOAT16", 0) > 0,
    "fp16_smaller": fp16["model_bytes"] < fp32["model_bytes"] * 0.7,
}
result = {"fp32": fp32, "fp16": fp16, "checks": checks, "success": all(checks.values())}
output.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, sort_keys=True))
raise SystemExit(0 if result["success"] else 1)

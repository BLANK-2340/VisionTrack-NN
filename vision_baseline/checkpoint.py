"""Deterministic, validated checkpoints for :class:`TinyConvNet`."""

from __future__ import annotations

import hashlib
import hmac
import json
import struct
from pathlib import Path

import numpy as np

from .model import TinyConvNet

MAGIC = b"VTNNCHK\x00"
VERSION = 1
MAX_METADATA_BYTES = 16 * 1024
MAX_PARAMETER_COUNT = 10_000_000
MAX_FILE_BYTES = 8 * MAX_PARAMETER_COUNT + MAX_METADATA_BYTES + 64
_ORDER = ("kernel", "conv_bias", "weight", "bias")


def _shape_size(shape: tuple[int, ...]) -> int:
    size = 1
    for dimension in shape:
        if dimension > MAX_PARAMETER_COUNT // size:
            raise ValueError("checkpoint exceeds parameter limit")
        size *= dimension
    return size


def _validated_arrays(model: TinyConvNet) -> tuple[dict[str, object], list[np.ndarray]]:
    arrays = [np.asarray(model.parameters()[name]) for name in _ORDER]
    kernel, conv_bias, weight, bias = arrays
    if kernel.ndim != 3 or kernel.shape[1:] != (3, 3):
        raise ValueError("kernel must have shape [channels, 3, 3]")
    channels = kernel.shape[0]
    if channels < 1 or conv_bias.shape != (channels,):
        raise ValueError("conv_bias shape does not match kernel channels")
    if weight.ndim != 2 or weight.shape[0] != channels:
        raise ValueError("weight shape does not match kernel channels")
    classes = weight.shape[1]
    if classes < 2 or bias.shape != (classes,):
        raise ValueError("bias shape does not match classifier classes")
    count = sum(array.size for array in arrays)
    if count > MAX_PARAMETER_COUNT:
        raise ValueError("checkpoint exceeds parameter limit")
    if any(array.dtype.kind != "f" or not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("checkpoint parameters must be finite floating-point arrays")
    metadata: dict[str, object] = {
        "architecture": "conv3x3-relu-gap-linear",
        "arrays": [[name, list(array.shape)] for name, array in zip(_ORDER, arrays)],
        "dtype": "<f8",
        "version": VERSION,
    }
    return metadata, [np.ascontiguousarray(array, dtype="<f8") for array in arrays]


def dumps_model(model: TinyConvNet) -> bytes:
    """Return canonical checkpoint bytes for a model."""
    metadata, arrays = _validated_arrays(model)
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("ascii")
    if len(encoded) > MAX_METADATA_BYTES:
        raise ValueError("checkpoint metadata is too large")
    body = MAGIC + struct.pack("<I", len(encoded)) + encoded
    body += b"".join(array.tobytes(order="C") for array in arrays)
    return body + hashlib.sha256(body).digest()


def loads_model(data: bytes | bytearray | memoryview) -> TinyConvNet:
    """Load a model after validating format, resource limits, and checksum."""
    raw = bytes(data)
    minimum = len(MAGIC) + 4 + 32
    if len(raw) < minimum or len(raw) > MAX_FILE_BYTES:
        raise ValueError("invalid checkpoint size")
    if raw[: len(MAGIC)] != MAGIC:
        raise ValueError("invalid checkpoint magic")
    metadata_size = struct.unpack_from("<I", raw, len(MAGIC))[0]
    if metadata_size == 0 or metadata_size > MAX_METADATA_BYTES:
        raise ValueError("invalid checkpoint metadata size")
    metadata_start = len(MAGIC) + 4
    metadata_end = metadata_start + metadata_size
    if metadata_end + 32 > len(raw):
        raise ValueError("truncated checkpoint metadata")
    try:
        metadata = json.loads(raw[metadata_start:metadata_end].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid checkpoint metadata") from error
    if not isinstance(metadata, dict) or metadata.get("version") != VERSION:
        raise ValueError("unsupported checkpoint version")
    if metadata.get("architecture") != "conv3x3-relu-gap-linear" or metadata.get("dtype") != "<f8":
        raise ValueError("unsupported checkpoint architecture or dtype")
    entries = metadata.get("arrays")
    if not isinstance(entries, list) or len(entries) != len(_ORDER):
        raise ValueError("invalid checkpoint array table")
    shapes: list[tuple[int, ...]] = []
    for expected_name, entry in zip(_ORDER, entries):
        if (not isinstance(entry, list) or len(entry) != 2 or
                entry[0] != expected_name or not isinstance(entry[1], list) or
                not entry[1] or any(type(value) is not int or value < 1 for value in entry[1])):
            raise ValueError("invalid checkpoint array entry")
        shapes.append(tuple(entry[1]))
    kernel_shape, conv_bias_shape, weight_shape, bias_shape = shapes
    if (len(kernel_shape) != 3 or kernel_shape[1:] != (3, 3) or
            conv_bias_shape != (kernel_shape[0],) or len(weight_shape) != 2 or
            weight_shape[0] != kernel_shape[0] or weight_shape[1] < 2 or
            bias_shape != (weight_shape[1],)):
        raise ValueError("inconsistent checkpoint shapes")
    sizes = [_shape_size(shape) for shape in shapes]
    count = sum(sizes)
    if count > MAX_PARAMETER_COUNT:
        raise ValueError("checkpoint exceeds parameter limit")
    expected_size = metadata_end + count * 8 + 32
    if len(raw) != expected_size:
        raise ValueError("checkpoint payload size mismatch")
    body, digest = raw[:-32], raw[-32:]
    if not hmac.compare_digest(hashlib.sha256(body).digest(), digest):
        raise ValueError("checkpoint checksum mismatch")
    offset = metadata_end
    arrays = []
    for shape, size in zip(shapes, sizes):
        array = np.frombuffer(raw, dtype="<f8", count=size, offset=offset).reshape(shape).copy()
        if not np.all(np.isfinite(array)):
            raise ValueError("checkpoint parameters must be finite")
        arrays.append(array)
        offset += size * 8
    model = TinyConvNet(channels=kernel_shape[0], classes=weight_shape[1], seed=0)
    for name, array in zip(_ORDER, arrays):
        model.parameters()[name][...] = array
    return model


def save_model(model: TinyConvNet, path: str | Path) -> None:
    Path(path).write_bytes(dumps_model(model))


def load_model(path: str | Path) -> TinyConvNet:
    checkpoint = Path(path)
    if checkpoint.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("checkpoint exceeds file-size limit")
    return loads_model(checkpoint.read_bytes())

"""Strict P2/P5 portable-graymap parsing for dependency-light inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np

MAX_PIXELS = 50_000_000
_WHITESPACE = b" \t\r\n\v\f"


class _Reader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.position = 0

    def token(self) -> bytes:
        while True:
            while self.position < len(self.data) and self.data[self.position] in _WHITESPACE:
                self.position += 1
            if self.position < len(self.data) and self.data[self.position] == ord("#"):
                newline = self.data.find(b"\n", self.position)
                if newline < 0:
                    self.position = len(self.data)
                else:
                    self.position = newline + 1
                continue
            break
        start = self.position
        while (self.position < len(self.data) and
               self.data[self.position] not in _WHITESPACE and
               self.data[self.position] != ord("#")):
            self.position += 1
        if start == self.position:
            raise ValueError("truncated PGM header")
        return self.data[start:self.position]


def _integer(token: bytes, name: str) -> int:
    try:
        text = token.decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid PGM {name}") from error
    if not text.isdecimal():
        raise ValueError(f"invalid PGM {name}")
    return int(text)


def loads_pgm(data: bytes) -> np.ndarray:
    """Decode P2 or P5 PGM bytes to a float64 image in [0, 1]."""
    reader = _Reader(data)
    magic = reader.token()
    if magic not in (b"P2", b"P5"):
        raise ValueError("only P2 and P5 PGM images are supported")
    width = _integer(reader.token(), "width")
    height = _integer(reader.token(), "height")
    maximum = _integer(reader.token(), "maximum")
    if width < 1 or height < 1 or width * height > MAX_PIXELS:
        raise ValueError("invalid or excessive PGM dimensions")
    if maximum < 1 or maximum > 65535:
        raise ValueError("PGM maximum must be in [1, 65535]")
    count = width * height
    if magic == b"P2":
        values = np.empty(count, dtype=np.float64)
        for index in range(count):
            values[index] = _integer(reader.token(), "sample")
        try:
            reader.token()
        except ValueError:
            pass
        else:
            raise ValueError("trailing PGM samples")
    else:
        if reader.position >= len(data) or data[reader.position] not in _WHITESPACE:
            raise ValueError("missing PGM binary separator")
        if data[reader.position:reader.position + 2] == b"\r\n":
            reader.position += 2
        else:
            reader.position += 1
        bytes_per_sample = 1 if maximum < 256 else 2
        payload = data[reader.position:]
        if len(payload) != count * bytes_per_sample:
            raise ValueError("PGM binary payload size mismatch")
        dtype = np.uint8 if bytes_per_sample == 1 else np.dtype(">u2")
        values = np.frombuffer(payload, dtype=dtype).astype(np.float64)
    if np.any(values > maximum):
        raise ValueError("PGM sample exceeds declared maximum")
    return (values / maximum).reshape(height, width)


def load_pgm(path: str | Path) -> np.ndarray:
    return loads_pgm(Path(path).read_bytes())

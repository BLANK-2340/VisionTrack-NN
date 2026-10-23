"""Emit a deterministic, privacy-conscious numeric environment report."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np


_DEPENDENCY_FIELDS = {
    "name": "name",
    "found": "found",
    "version": "version",
    "detection method": "detection_method",
    "openblas configuration": "configuration",
}


def _scalar(value: Any) -> str | bool | int | float | None:
    """Return a JSON scalar, excluding paths and arbitrary object strings."""
    if isinstance(value, str):
        return value.strip()
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return None


def _structured_numpy_build(config: dict[str, Any]) -> dict[str, Any]:
    dependencies: dict[str, Any] = {}
    raw_dependencies = config.get("Build Dependencies", {})
    if isinstance(raw_dependencies, dict):
        for dependency in ("blas", "lapack"):
            raw = raw_dependencies.get(dependency)
            if not isinstance(raw, dict):
                continue
            clean: dict[str, Any] = {}
            for source, target in _DEPENDENCY_FIELDS.items():
                value = _scalar(raw.get(source))
                if value is not None:
                    clean[target] = value
            dependencies[dependency] = clean

    compilers: dict[str, Any] = {}
    raw_compilers = config.get("Compilers", {})
    if isinstance(raw_compilers, dict):
        for language in ("c", "c++", "cython"):
            raw = raw_compilers.get(language)
            if not isinstance(raw, dict):
                continue
            clean = {
                key: value
                for key in ("name", "version", "linker")
                if (value := _scalar(raw.get(key))) is not None
            }
            compilers[language] = clean

    simd: dict[str, list[str]] = {}
    raw_simd = config.get("SIMD Extensions", {})
    if isinstance(raw_simd, dict):
        for key in ("baseline", "found", "not found"):
            value = raw_simd.get(key)
            if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
                simd[key.replace(" ", "_")] = list(value)

    return {
        "source": "numpy_structured_config",
        "build_dependencies": dependencies,
        "compilers": compilers,
        "simd_extensions": simd,
    }


def _legacy_dependency(config_module: Any, candidates: tuple[str, ...]) -> dict[str, Any]:
    getter = getattr(config_module, "get_info", None)
    if not callable(getter):
        return {}
    for candidate in candidates:
        raw = getter(candidate)
        if not isinstance(raw, dict) or not raw:
            continue
        libraries = raw.get("libraries", ())
        if isinstance(libraries, (list, tuple)) and all(isinstance(item, str) for item in libraries):
            return {"configuration": candidate, "libraries": sorted(libraries)}
        return {"configuration": candidate}
    return {}


def _numpy_build() -> dict[str, Any]:
    config_module = np.__config__
    config = getattr(config_module, "CONFIG", None)
    if isinstance(config, dict):
        return _structured_numpy_build(config)
    return {
        "source": "numpy_legacy_get_info",
        "build_dependencies": {
            "blas": _legacy_dependency(
                config_module, ("blas_opt_info", "openblas_info", "blas_mkl_info")
            ),
            "lapack": _legacy_dependency(
                config_module, ("lapack_opt_info", "openblas_lapack_info", "lapack_mkl_info")
            ),
        },
        "compilers": {},
        "simd_extensions": {},
    }


def build_environment_report() -> dict[str, Any]:
    """Describe runtime and numeric-library identity without host-specific paths."""
    return {
        "schema_version": 1,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "compiler": platform.python_compiler(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "byteorder": sys.byteorder,
        },
        "numpy": {
            "version": np.__version__,
            "float64_byteorder": np.dtype(np.float64).byteorder,
            "build": _numpy_build(),
        },
    }


def canonical_json(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write canonical JSON to this file")
    args = parser.parse_args()
    rendered = canonical_json(build_environment_report())
    if args.output is None:
        sys.stdout.write(rendered)
    else:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()

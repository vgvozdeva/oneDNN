#!/usr/bin/env python3
# ==============================================================================
# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Generates Bazel substitutions for include/oneapi/dnnl/dnnl_config.h.in.

The output is a .bzl file with dictionaries suitable for
bazel_skylib//rules:expand_template.bzl.
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple


CMAKEDEFINE01_RE = re.compile(r"^#cmakedefine01\s+([A-Za-z0-9_]+)\s*$")
CMAKEDEFINE_RE = re.compile(
    r"^#cmakedefine\s+([A-Za-z0-9_]+)(?:\s+(.*?))?\s*$"
)
TOKEN_RE = re.compile(r"\$\{([A-Za-z0-9_]+)\}")
PROJECT_VERSION_RE = re.compile(
    r'^\s*set\(\s*PROJECT_VERSION\s+"([0-9]+)\.([0-9]+)\.([0-9]+)"\s*\)\s*$'
)


def cmake_truthy(value: str) -> bool:
    v = str(value).strip()
    if not v:
        return False
    vu = v.upper()
    if vu in {"0", "FALSE", "OFF", "NO", "N", "IGNORE", "NOTFOUND"}:
        return False
    if vu.endswith("-NOTFOUND"):
        return False
    return True


def substitute_tokens(text: str, variables: Dict[str, str]) -> str:
    def repl(match: re.Match) -> str:
        key = match.group(1)
        return str(variables.get(key, ""))

    return TOKEN_RE.sub(repl, text)


def render_line(template_line: str, variables: Dict[str, str]) -> Tuple[str, str]:
    line = template_line.strip()

    m01 = CMAKEDEFINE01_RE.match(line)
    if m01:
        name = m01.group(1)
        value = "1" if cmake_truthy(variables.get(name, "")) else "0"
        return template_line, f"#define {name} {value}"

    m = CMAKEDEFINE_RE.match(line)
    if not m:
        raise ValueError(f"Unsupported cmakedefine line: {template_line}")

    name = m.group(1)
    value_expr = m.group(2)

    if not cmake_truthy(variables.get(name, "")):
        return template_line, f"#undef {name}"

    if value_expr:
        expanded = substitute_tokens(value_expr, variables)
        return template_line, f"#define {name} {expanded}"

    return template_line, f"#define {name}"


def collect_cmakedefine_lines(template_path: Path) -> List[str]:
    lines = template_path.read_text(encoding="utf-8").splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#cmakedefine"):
            out.append(stripped)
    return out


def base_defaults(cpu_runtime: str) -> Dict[str, str]:
    defaults: Dict[str, str] = {
        "DNNL_CPU_RUNTIME": cpu_runtime,
        "DNNL_CPU_THREADING_RUNTIME": cpu_runtime,
        "DNNL_GPU_RUNTIME": "NONE",
        "DNNL_GPU_VENDOR": "NONE",
        "ONEDNN_BUILD_GRAPH": "ON",
    }

    # Optional feature toggles default to OFF unless explicitly enabled.
    optional_off = [
        "DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE",
        "DNNL_WITH_SYCL",
        "DNNL_SYCL_CUDA",
        "DNNL_SYCL_HIP",
        "DNNL_SYCL_GENERIC",
        "DNNL_ENABLE_STACK_CHECKER",
        "DNNL_EXPERIMENTAL",
        "DNNL_EXPERIMENTAL_UKERNEL",
        "DNNL_EXPERIMENTAL_PROFILING",
        "DNNL_EXPERIMENTAL_LOGGING",
        "DNNL_SAFE_RBP",
        "DNNL_EXPERIMENTAL_SYCL_KERNEL_COMPILER",
        "DNNL_EXPERIMENTAL_GROUPED_MEMORY",
        "DNNL_DISABLE_GPU_REF_KERNELS",
    ]
    for name in optional_off:
        defaults[name] = "OFF"

    # Workload/primitive/ISA controls aligned with CMake defaults.
    defaults.update(
        {
            "BUILD_TRAINING": "ON",
            "BUILD_INFERENCE": "OFF",
            "BUILD_PRIMITIVE_ALL": "ON",
            "BUILD_PRIMITIVE_CPU_ISA_ALL": "ON",
            "BUILD_PRIMITIVE_GPU_ISA_ALL": "ON",
            "BUILD_GEMM_KERNELS_ALL": "ON",
            "BUILD_GEMM_KERNELS_NONE": "OFF",
        }
    )

    build_01_names = [
        "BUILD_BATCH_NORMALIZATION",
        "BUILD_BINARY",
        "BUILD_CONCAT",
        "BUILD_CONVOLUTION",
        "BUILD_DECONVOLUTION",
        "BUILD_ELTWISE",
        "BUILD_GATED_MLP",
        "BUILD_GROUP_NORMALIZATION",
        "BUILD_INNER_PRODUCT",
        "BUILD_LAYER_NORMALIZATION",
        "BUILD_LRN",
        "BUILD_MATMUL",
        "BUILD_POOLING",
        "BUILD_PRELU",
        "BUILD_REDUCTION",
        "BUILD_REORDER",
        "BUILD_RESAMPLING",
        "BUILD_RNN",
        "BUILD_SDPA",
        "BUILD_SHUFFLE",
        "BUILD_SOFTMAX",
        "BUILD_SUM",
        "BUILD_SSE41",
        "BUILD_AVX2",
        "BUILD_AVX512",
        "BUILD_AMX",
        "BUILD_XELP",
        "BUILD_XEHP",
        "BUILD_XEHPG",
        "BUILD_XEHPC",
        "BUILD_XE2",
        "BUILD_XE3",
        "BUILD_XE3P",
        "BUILD_GEMM_SSE41",
        "BUILD_GEMM_AVX2",
        "BUILD_GEMM_AVX512",
    ]
    for name in build_01_names:
        defaults.setdefault(name, "OFF")

    return defaults


def parse_kv_overrides(items: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --set value '{item}'. Expected NAME=VALUE")
        name, value = item.split("=", 1)
        name = name.strip()
        value = value.strip()
        if not name:
            raise ValueError(f"Invalid --set value '{item}'. NAME is empty")
        out[name] = value
    return out


def render_bzl_dict(dict_name: str, mapping: List[Tuple[str, str]]) -> str:
    lines = [f"{dict_name} = {{"]
    for template_line, rendered_line in mapping:
        lines.append(f'    "{template_line}": "{rendered_line}",')
    lines.append("}")
    return "\n".join(lines)


def generate_mapping(
    template_lines: List[str],
    variables: Dict[str, str],
) -> List[Tuple[str, str]]:
    mapping: List[Tuple[str, str]] = []
    for line in template_lines:
        mapping.append(render_line(line, variables))
    return mapping


def parse_project_version(cmake_lists_path: Path) -> Tuple[str, str, str]:
    for line in cmake_lists_path.read_text(encoding="utf-8").splitlines():
        m = PROJECT_VERSION_RE.match(line)
        if m:
            return m.group(1), m.group(2), m.group(3)
    raise ValueError(
        f"Failed to locate PROJECT_VERSION in {cmake_lists_path.as_posix()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Bazel substitutions for dnnl_config.h.in"
    )
    parser.add_argument(
        "--template",
        default="include/oneapi/dnnl/dnnl_config.h.in",
        help="Path to dnnl_config.h.in",
    )
    parser.add_argument(
        "--output",
        default="bazel/dnnl_config_substitutions.bzl",
        help="Path to output .bzl file",
    )
    parser.add_argument(
        "--cmake-lists",
        default="CMakeLists.txt",
        help="Path to top-level CMakeLists.txt used to derive oneDNN version",
    )
    parser.add_argument(
        "--version-hash",
        default="N/A",
        help="Value for @DNNL_VERSION_HASH@ in version substitutions",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override CMake variable used by #cmakedefine, e.g. --set ONEDNN_BUILD_GRAPH=OFF",
    )
    args = parser.parse_args()

    template_path = Path(args.template)
    output_path = Path(args.output)
    cmake_lists_path = Path(args.cmake_lists)

    template_lines = collect_cmakedefine_lines(template_path)
    overrides = parse_kv_overrides(args.set)

    omp_vars = base_defaults("OMP")
    omp_vars.update(overrides)

    threadpool_vars = base_defaults("THREADPOOL")
    threadpool_vars.update(overrides)

    omp_mapping = generate_mapping(template_lines, omp_vars)
    threadpool_mapping = generate_mapping(template_lines, threadpool_vars)
    version_major, version_minor, version_patch = parse_project_version(
        cmake_lists_path
    )

    body = [
        '"""Auto-generated Bazel substitutions for dnnl_config.h.in.',
        "",
        "Regenerate with:",
        "  ./scripts/generate_bazel_dnnl_substitutions.py",
        '"""',
        "",
        render_bzl_dict("DNNL_CONFIG_SUBSTITUTIONS_OMP", omp_mapping),
        "",
        render_bzl_dict(
            "DNNL_CONFIG_SUBSTITUTIONS_THREADPOOL", threadpool_mapping
        ),
        "",
        render_bzl_dict(
            "DNNL_VERSION_SUBSTITUTIONS",
            [
                ("@DNNL_VERSION_MAJOR@", version_major),
                ("@DNNL_VERSION_MINOR@", version_minor),
                ("@DNNL_VERSION_PATCH@", version_patch),
                ("@DNNL_VERSION_HASH@", args.version_hash),
            ],
        ),
        "",
        render_bzl_dict(
            "DNNL_VERSION_HASH_SUBSTITUTIONS",
            [("@DNNL_VERSION_HASH@", args.version_hash)],
        ),
        "",
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(body), encoding="utf-8")


if __name__ == "__main__":
    main()

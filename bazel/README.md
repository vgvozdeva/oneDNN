# Bazel Configuration Helpers

This directory contains Bazel-facing configuration data generated from
oneDNN CMake sources.

## Why this exists

Downstream Bazel builds (for example TensorFlow/XLA) use
`expand_template(...)` with `include/oneapi/dnnl/dnnl_config.h.in`.
Historically this required manually mirroring `#cmakedefine` substitutions,
which is fragile when oneDNN adds/removes configuration macros.

`dnnl_config_substitutions.bzl` is generated from oneDNN templates to keep
Bazel substitutions aligned with library-side configuration sources.

## Regeneration

From repository root:

```sh
python3 scripts/generate_bazel_dnnl_substitutions.py
```

This updates [dnnl_config_substitutions.bzl](dnnl_config_substitutions.bzl) with:

- `DNNL_CONFIG_SUBSTITUTIONS_OMP`
- `DNNL_CONFIG_SUBSTITUTIONS_THREADPOOL`
- `DNNL_VERSION_SUBSTITUTIONS`
- `DNNL_VERSION_HASH_SUBSTITUTIONS`

You can override individual CMake variables while generating:

```sh
python3 scripts/generate_bazel_dnnl_substitutions.py \
  --set ONEDNN_BUILD_GRAPH=OFF \
  --set BUILD_SDPA=ON
```

## Typical downstream usage

```python
load("@mkl_dnn//bazel:dnnl_config_substitutions.bzl",
     "DNNL_CONFIG_SUBSTITUTIONS_THREADPOOL",
     "DNNL_VERSION_SUBSTITUTIONS",
     "DNNL_VERSION_HASH_SUBSTITUTIONS")

expand_template(
    name = "dnnl_config_h",
    template = "include/oneapi/dnnl/dnnl_config.h.in",
    out = "include/oneapi/dnnl/dnnl_config.h",
    substitutions = DNNL_CONFIG_SUBSTITUTIONS_THREADPOOL,
)

expand_template(
    name = "dnnl_version_h",
    template = "include/oneapi/dnnl/dnnl_version.h.in",
    out = "include/oneapi/dnnl/dnnl_version.h",
    substitutions = DNNL_VERSION_SUBSTITUTIONS,
)

expand_template(
    name = "dnnl_version_hash_h",
    template = "include/oneapi/dnnl/dnnl_version_hash.h.in",
    out = "include/oneapi/dnnl/dnnl_version_hash.h",
    substitutions = DNNL_VERSION_HASH_SUBSTITUTIONS,
)
```

## CI validation

The repository includes a GitHub workflow that validates this flow on pull
requests by:

- Regenerating substitutions with
    `scripts/generate_bazel_dnnl_substitutions.py`
- Verifying `bazel/dnnl_config_substitutions.bzl` is up to date
- Running Bazel `expand_template(...)` smoke targets for:
    - `dnnl_config.h.in` (OMP and THREADPOOL variants)
    - `dnnl_version.h.in`
    - `dnnl_version_hash.h.in`

See [Bazel Config Smoke workflow](../.github/workflows/bazel-config-smoke.yml).

For stronger compile validation without external repository dependencies, the
repository also includes
[oneDNN Bazel Build Compliance](../.github/workflows/tensorflow-integration-smoke.yml).
It generates oneDNN config/version headers via Bazel `expand_template(...)` and
builds a Bazel `cc_library` from oneDNN common sources against those generated
headers.

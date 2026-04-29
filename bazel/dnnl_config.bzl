"""Custom Starlark rule for generating dnnl_config.h with flag-driven substitutions.

The rule reads the configurable flags defined in //bazel:BUILD.bazel at analysis
time and constructs the substitution dict dynamically, so consumers can override
any option via the command line without editing any BUILD file.

Example (from the oneDNN workspace root or via @onednn):
  bazel build //:dnnl_config_h \\
    --@onednn//bazel:dnnl_cpu_runtime=OMP \\
    --@onednn//bazel:dnnl_experimental=true \\
    --@onednn//bazel:onednn_build_graph=false
"""

load("@bazel_skylib//rules/common_settings.bzl", "BuildSettingInfo")
load(
    ":dnnl_config_substitutions.bzl",
    "DNNL_CONFIG_SUBSTITUTIONS_THREADPOOL",
)

def _dnnl_config_header_impl(ctx):
    cpu_runtime = ctx.attr.dnnl_cpu_runtime[BuildSettingInfo].value
    gpu_runtime = ctx.attr.dnnl_gpu_runtime[BuildSettingInfo].value
    gpu_vendor = ctx.attr.dnnl_gpu_vendor[BuildSettingInfo].value
    experimental = ctx.attr.dnnl_experimental[BuildSettingInfo].value
    experimental_ukernel = ctx.attr.dnnl_experimental_ukernel[BuildSettingInfo].value
    experimental_profiling = ctx.attr.dnnl_experimental_profiling[BuildSettingInfo].value
    experimental_logging = ctx.attr.dnnl_experimental_logging[BuildSettingInfo].value
    build_graph = ctx.attr.onednn_build_graph[BuildSettingInfo].value
    stack_checker = ctx.attr.dnnl_enable_stack_checker[BuildSettingInfo].value
    use_rt_objects = ctx.attr.dnnl_use_rt_objects_in_primitive_cache[BuildSettingInfo].value

    # Start with the full base dict (all entries present) and override the
    # flag-controlled ones.  The base dict uses THREADPOOL defaults; any runtime
    # override is applied below.
    subs = dict(DNNL_CONFIG_SUBSTITUTIONS_THREADPOOL)

    # --- Runtime overrides ---
    subs["#cmakedefine DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_${DNNL_CPU_THREADING_RUNTIME}"] = (
        "#define DNNL_CPU_THREADING_RUNTIME DNNL_RUNTIME_" + cpu_runtime
    )
    subs["#cmakedefine DNNL_CPU_RUNTIME DNNL_RUNTIME_${DNNL_CPU_RUNTIME}"] = (
        "#define DNNL_CPU_RUNTIME DNNL_RUNTIME_" + cpu_runtime
    )
    subs["#cmakedefine DNNL_GPU_RUNTIME DNNL_RUNTIME_${DNNL_GPU_RUNTIME}"] = (
        "#define DNNL_GPU_RUNTIME DNNL_RUNTIME_" + gpu_runtime
    )
    subs["#cmakedefine DNNL_GPU_VENDOR DNNL_VENDOR_${DNNL_GPU_VENDOR}"] = (
        "#define DNNL_GPU_VENDOR DNNL_VENDOR_" + gpu_vendor
    )

    # --- Boolean feature overrides ---
    def _flag(macro, enabled):
        return "#define " + macro if enabled else "#undef " + macro

    subs["#cmakedefine DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE"] = (
        _flag("DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE", use_rt_objects)
    )
    subs["#cmakedefine DNNL_EXPERIMENTAL"] = _flag("DNNL_EXPERIMENTAL", experimental)
    subs["#cmakedefine DNNL_EXPERIMENTAL_UKERNEL"] = (
        _flag("DNNL_EXPERIMENTAL_UKERNEL", experimental_ukernel)
    )
    subs["#cmakedefine DNNL_EXPERIMENTAL_PROFILING"] = (
        _flag("DNNL_EXPERIMENTAL_PROFILING", experimental_profiling)
    )
    subs["#cmakedefine DNNL_EXPERIMENTAL_LOGGING"] = (
        _flag("DNNL_EXPERIMENTAL_LOGGING", experimental_logging)
    )
    subs["#cmakedefine ONEDNN_BUILD_GRAPH"] = _flag("ONEDNN_BUILD_GRAPH", build_graph)
    subs["#cmakedefine DNNL_ENABLE_STACK_CHECKER"] = (
        _flag("DNNL_ENABLE_STACK_CHECKER", stack_checker)
    )

    ctx.actions.expand_template(
        template = ctx.file.template,
        output = ctx.outputs.out,
        substitutions = subs,
    )

dnnl_config_header = rule(
    implementation = _dnnl_config_header_impl,
    doc = """Generates dnnl_config.h from dnnl_config.h.in.

All substitution values default to the same values produced by
scripts/generate_bazel_dnnl_substitutions.py.  Individual options can be
overridden at build time by passing the corresponding flag:

  --@onednn//bazel:dnnl_cpu_runtime=OMP
  --@onednn//bazel:dnnl_gpu_runtime=OCL
  --@onednn//bazel:dnnl_experimental=true
  --@onednn//bazel:onednn_build_graph=false
""",
    attrs = {
        "template": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "The dnnl_config.h.in template file.",
        ),
        "out": attr.output(
            mandatory = True,
            doc = "The generated dnnl_config.h output file.",
        ),
        "dnnl_cpu_runtime": attr.label(
            default = "//bazel:dnnl_cpu_runtime",
            providers = [BuildSettingInfo],
            doc = "CPU threading runtime (OMP | THREADPOOL | TBB | SEQ).",
        ),
        "dnnl_gpu_runtime": attr.label(
            default = "//bazel:dnnl_gpu_runtime",
            providers = [BuildSettingInfo],
            doc = "GPU runtime (NONE | OCL | SYCL).",
        ),
        "dnnl_gpu_vendor": attr.label(
            default = "//bazel:dnnl_gpu_vendor",
            providers = [BuildSettingInfo],
            doc = "GPU vendor (NONE | INTEL | NVIDIA | AMD).",
        ),
        "dnnl_experimental": attr.label(
            default = "//bazel:dnnl_experimental",
            providers = [BuildSettingInfo],
            doc = "Enable DNNL_EXPERIMENTAL features.",
        ),
        "dnnl_experimental_ukernel": attr.label(
            default = "//bazel:dnnl_experimental_ukernel",
            providers = [BuildSettingInfo],
            doc = "Enable DNNL_EXPERIMENTAL_UKERNEL.",
        ),
        "dnnl_experimental_profiling": attr.label(
            default = "//bazel:dnnl_experimental_profiling",
            providers = [BuildSettingInfo],
            doc = "Enable DNNL_EXPERIMENTAL_PROFILING.",
        ),
        "dnnl_experimental_logging": attr.label(
            default = "//bazel:dnnl_experimental_logging",
            providers = [BuildSettingInfo],
            doc = "Enable DNNL_EXPERIMENTAL_LOGGING.",
        ),
        "onednn_build_graph": attr.label(
            default = "//bazel:onednn_build_graph",
            providers = [BuildSettingInfo],
            doc = "Enable ONEDNN_BUILD_GRAPH (graph API).",
        ),
        "dnnl_enable_stack_checker": attr.label(
            default = "//bazel:dnnl_enable_stack_checker",
            providers = [BuildSettingInfo],
            doc = "Enable DNNL_ENABLE_STACK_CHECKER.",
        ),
        "dnnl_use_rt_objects_in_primitive_cache": attr.label(
            default = "//bazel:dnnl_use_rt_objects_in_primitive_cache",
            providers = [BuildSettingInfo],
            doc = "Enable DNNL_USE_RT_OBJECTS_IN_PRIMITIVE_CACHE.",
        ),
    },
)

load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@xla//xla/tsl:tsl.bzl", "tf_openmp_copts")
load("@xla//xla/tsl/mkl:build_defs.bzl", "if_mkl", "if_mkl_ml", "if_mkldnn_openmp")
load(":bazel/dnnl_config_substitutions.bzl",
     "DNNL_CONFIG_SUBSTITUTIONS_OMP",
     "DNNL_CONFIG_SUBSTITUTIONS_THREADPOOL",
     "DNNL_VERSION_HASH_SUBSTITUTIONS",
     "DNNL_VERSION_SUBSTITUTIONS")

exports_files(["LICENSE"])

expand_template(
    name = "dnnl_config_h",
    out = "include/oneapi/dnnl/dnnl_config.h",
    substitutions = select({
        "@xla//xla/tsl/mkl:build_with_mkldnn_openmp": DNNL_CONFIG_SUBSTITUTIONS_OMP,
        "//conditions:default": DNNL_CONFIG_SUBSTITUTIONS_THREADPOOL,
    }),
    template = "include/oneapi/dnnl/dnnl_config.h.in",
)

# Version values are derived from oneDNN CMake PROJECT_VERSION via
# scripts/generate_bazel_dnnl_substitutions.py.
expand_template(
    name = "dnnl_version_h",
    out = "include/oneapi/dnnl/dnnl_version.h",
    substitutions = DNNL_VERSION_SUBSTITUTIONS,
    template = "include/oneapi/dnnl/dnnl_version.h.in",
)

expand_template(
    name = "dnnl_version_hash_h",
    out = "include/oneapi/dnnl/dnnl_version_hash.h",
    substitutions = DNNL_VERSION_HASH_SUBSTITUTIONS,
    template = "include/oneapi/dnnl/dnnl_version_hash.h.in",
)

_COPTS_LIST = select({
    "@xla//xla/tsl:windows": [],
    "//conditions:default": ["-fexceptions"],
}) + [
    "-UUSE_MKL",
    "-UUSE_CBLAS",
    "-DDNNL_ENABLE_MAX_CPU_ISA",
    "-DDNNL_ENABLE_ITT_TASKS",
    "-DDNNL_ENABLE_GRAPH_DUMP",
] + tf_openmp_copts()

_INCLUDES_LIST = [
    "include",
    "src",
    "src/common",
    "src/common/ittnotify",
    "src/common/spdlog",
    "src/common/spdlog/details",
    "src/common/spdlog/fmt",
    "src/common/spdlog/fmt/bundled",
    "src/common/spdlog/sinks",
    "src/cpu",
    "src/cpu/gemm",
    "src/cpu/x64/xbyak",
    "src/graph",
]

_TEXTUAL_HDRS_LIST = glob([
    "include/**/*",
    "src/common/*.hpp",
    "src/common/ittnotify/**/*.h",
    "src/common/spdlog/*.h",
    "src/common/spdlog/details/*.h",
    "src/common/spdlog/fmt/**/*.h",
    "src/common/spdlog/sinks/*.h",
    "src/cpu/*.hpp",
    "src/cpu/**/*.hpp",
    "src/cpu/jit_utils/**/*.hpp",
    "src/cpu/x64/xbyak/*.h",
    "src/graph/interface/*.hpp",
    "src/graph/backend/*.hpp",
    "src/graph/backend/dnnl/*.hpp",
    "src/graph/backend/fake/*.hpp",
    "src/graph/backend/dnnl/passes/*.hpp",
    "src/graph/backend/dnnl/patterns/*.hpp",
    "src/graph/backend/dnnl/kernels/*.hpp",
    "src/graph/utils/*.hpp",
    "src/graph/utils/pm/*.hpp",
]) + [
    ":dnnl_config_h",
    ":dnnl_version_h",
    ":dnnl_version_hash_h",
]

cc_library(
    name = "onednn_autogen",
    srcs = glob(["src/cpu/x64/gemm/**/*_kern_autogen*.cpp"]),
    copts = [
        "-O1",
        "-U_FORTIFY_SOURCE",
    ] + _COPTS_LIST,
    includes = _INCLUDES_LIST,
    textual_hdrs = _TEXTUAL_HDRS_LIST,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "mkl_dnn",
    srcs = glob(
        [
            "src/common/*.cpp",
            "src/cpu/*.cpp",
            "src/cpu/**/*.cpp",
            "src/common/ittnotify/*.c",
            "src/cpu/jit_utils/**/*.cpp",
            "src/cpu/x64/**/*.cpp",
            "src/graph/interface/*.cpp",
            "src/graph/backend/*.cpp",
            "src/graph/backend/dnnl/*.cpp",
            "src/graph/backend/fake/*.cpp",
            "src/graph/backend/dnnl/passes/*.cpp",
            "src/graph/backend/dnnl/patterns/*.cpp",
            "src/graph/backend/dnnl/kernels/*.cpp",
            "src/graph/utils/*.cpp",
            "src/graph/utils/pm/*.cpp",
        ],
        exclude = [
            "src/cpu/aarch64/**",
            "src/cpu/rv64/**",
            "src/cpu/x64/gemm/**/*_kern_autogen.cpp",
            "src/cpu/sycl/**",
        ],
    ),
    copts = _COPTS_LIST,
    includes = _INCLUDES_LIST,
    linkopts = select({
        "@xla//xla/tsl:linux_aarch64": ["-lrt"],
        "@xla//xla/tsl:linux_x86_64": ["-lrt"],
        "@xla//xla/tsl:linux_ppc64le": ["-lrt"],
        "@xla//xla/tsl:linux_riscv64": ["-lrt"],
        "//conditions:default": [],
    }),
    textual_hdrs = _TEXTUAL_HDRS_LIST,
    visibility = ["//visibility:public"],
    deps = [":onednn_autogen"] + if_mkl_ml(
        ["@xla//xla/tsl/mkl:intel_binary_blob"],
        [],
    ),
)

#include <gtest/gtest.h>
#include <cstdlib>

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_config.h"

namespace dnnl {

TEST(EnvHardening, JitDumpEnvIgnoredWhenDisabled) {
#if DNNL_ENABLE_JIT_DUMP_ENV
    GTEST_SKIP() << "JIT_DUMP env hardening disabled in this build";
#endif

    // Try to enable JIT dump via environment
    setenv("ONEDNN_JIT_DUMP", "1", 1);

    int enabled = -1;
    EXPECT_EQ(dnnl_get_jit_dump(&enabled), dnnl_success);
    EXPECT_EQ(enabled, 0) << "JIT dump must ignore env var in hardened build";

    // API must still override
    EXPECT_EQ(dnnl_set_jit_dump(1), dnnl_success);
    EXPECT_EQ(dnnl_get_jit_dump(&enabled), dnnl_success);
    EXPECT_EQ(enabled, 1);

    unsetenv("ONEDNN_JIT_DUMP");
}

TEST(EnvHardening, JitDumpEnvHonoredWhenEnabled) {
#if !DNNL_ENABLE_JIT_DUMP_ENV
    GTEST_SKIP() << "JIT_DUMP env support disabled in this build";
#endif

    setenv("ONEDNN_JIT_DUMP", "1", 1);

    int enabled = -1;
    EXPECT_EQ(dnnl_get_jit_dump(&enabled), dnnl_success);
    EXPECT_EQ(enabled, 1);

    unsetenv("ONEDNN_JIT_DUMP");
}
}
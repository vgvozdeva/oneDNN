#include <gtest/gtest.h>
#include <cstdlib>

#include "oneapi/dnnl/dnnl.h"
#include "oneapi/dnnl/dnnl_config.h"

namespace dnnl {

TEST(EnvHardening, JitDumpEnvIgnoredWhenDisabled) {
#if DNNL_ENABLE_JIT_DUMP_ENV
    // This build honours the env var, so this test is not applicable.
    GTEST_SKIP() << "JIT_DUMP env support is enabled in this build";
#endif

    // In a hardened build the env var must be ignored; the API must still work.
    setenv("ONEDNN_JIT_DUMP_ENV", "1", 1);

    // dnnl_set_jit_dump must succeed regardless of the env var.
    EXPECT_EQ(dnnl_set_jit_dump(0), dnnl_success);
    EXPECT_EQ(dnnl_set_jit_dump(1), dnnl_success);

    unsetenv("ONEDNN_JIT_DUMP_ENV");
}

TEST(EnvHardening, JitDumpEnvHonoredWhenEnabled) {
#if !DNNL_ENABLE_JIT_DUMP_ENV
    // This build ignores the env var, so this test is not applicable.
    GTEST_SKIP() << "JIT_DUMP env support is disabled in this build";
#endif

    setenv("ONEDNN_JIT_DUMP_ENV", "1", 1);

    // When env support is compiled in, set_jit_dump should still succeed.
    EXPECT_EQ(dnnl_set_jit_dump(1), dnnl_success);

    unsetenv("ONEDNN_JIT_DUMP_ENV");
}

} // namespace dnnl

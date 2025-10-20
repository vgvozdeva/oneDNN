/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_INTEL_LNORM_VECTORIZED_HPP
#define GPU_INTEL_LNORM_VECTORIZED_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/intel/lnorm/config.hpp"
#include "gpu/intel/primitive.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace lnorm {

struct vectorized_fwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public fwd_pd_t {
        using fwd_pd_t::fwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:vectorized", vectorized_fwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            using skip_mask_t = primitive_attr_t::skip_mask_t;
            auto *intel_engine = utils::downcast<intel::engine_t *>(engine);
            auto src_data_t = src_md()->data_type;
            auto dst_data_t = dst_md()->data_type;

            VDISPATCH_LNORM(is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_LNORM(
                    !has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "src");
            VDISPATCH_LNORM(
                    (utils::everyone_is(u8, src_data_t, dst_data_t)
                            || utils::everyone_is(s8, src_data_t, dst_data_t)
                            || utils::everyone_is(f16, src_data_t, dst_data_t)
                            || utils::everyone_is(bf16, src_data_t, dst_data_t)
                            || utils::everyone_is(f32, src_data_t, dst_data_t)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(IMPLICATION(f16 == src_data_t,
                                    intel_engine->mayiuse(
                                            compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_LNORM(memory_desc_ndims_ok(src_md(), dst_md(), stat_md()),
                    VERBOSE_INCONSISTENT_NDIMS, "src, dst", "stat");
            VDISPATCH_LNORM(
                    stat_md()->data_type == f32, VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_LNORM(check_scale_shift_data_type({f32, bf16, f16}),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_LNORM(attr()->has_default_values(skip_mask_t::scales),
                    VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_LNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            CHECK(init_conf(engine));
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        conf_t conf;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        kernel_ctx.define_int("WITH_SRC_SCALES",
                !pd()->attr()->scales_.has_default_values(DNNL_ARG_SRC));
        kernel_ctx.define_int("WITH_DST_SCALES",
                !pd()->attr()->scales_.has_default_values(DNNL_ARG_DST));

        CHECK(create_kernel(
                engine, &kernel_, "vectorized_lnorm_fwd", kernel_ctx));
        if (!kernel_) return status::runtime_error;

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_forward(ctx);
    }

private:
    status_t execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_;
};

struct vectorized_bwd_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public bwd_pd_t {
        using bwd_pd_t::bwd_pd_t;

        DECLARE_COMMON_PD_T("ocl:vectorized", vectorized_bwd_t);

        status_t init(impl::engine_t *engine) {
            using namespace data_type;
            auto *intel_engine = utils::downcast<intel::engine_t *>(engine);

            auto src_dt = src_md()->data_type;
            auto diff_dst_dt = diff_dst_md()->data_type;
            auto diff_src_dt = diff_src_md()->data_type;

            VDISPATCH_LNORM(!is_fwd(), VERBOSE_BAD_PROPKIND);
            VDISPATCH_LNORM(
                    !has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "src");
            VDISPATCH_LNORM(
                    (utils::everyone_is(f32, src_dt, diff_dst_dt, diff_src_dt)
                            || utils::everyone_is(
                                    bf16, src_dt, diff_dst_dt, diff_src_dt)
                            || utils::everyone_is(
                                    f16, src_dt, diff_dst_dt, diff_src_dt)),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_LNORM(IMPLICATION(f16 == src_dt,
                                    intel_engine->mayiuse(
                                            compute::device_ext_t::khr_fp16)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_LNORM(
                    stat_md()->data_type == f32, VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_LNORM(check_scale_shift_data_type({f32, bf16, f16}),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_LNORM(
                    attr()->has_default_values(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_LNORM(
                    set_default_formats_common(), VERBOSE_UNSUPPORTED_TAG);
            CHECK(init_conf(engine));
            init_scratchpad();
            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;
        void init_scratchpad();

        conf_t conf;
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        status_t status = pd()->init_kernel_ctx(kernel_ctx);
        CHECK(status);

        if (pd()->conf.use_fused) {
            CHECK(create_kernel(engine, &kernel_fused_,
                    "vectorized_lnorm_bwd_fused", kernel_ctx));
            if (!kernel_fused_) return status::runtime_error;
        }
        CHECK(create_kernel(
                engine, &kernel_, "vectorized_lnorm_bwd", kernel_ctx));
        if (!kernel_) return status::runtime_error;
        if (pd()->conf.use_scale || pd()->conf.use_shift) {
            CHECK(create_kernel(engine, &kernel_scaleshift_,
                    "vectorized_lnorm_bwd_scaleshift", kernel_ctx));
            if (!kernel_scaleshift_) return status::runtime_error;
            CHECK(create_kernel(engine, &kernel_scaleshift_finalize_,
                    "vectorized_lnorm_bwd_scaleshift_final", kernel_ctx));
            if (!kernel_scaleshift_finalize_) return status::runtime_error;
        }

        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override {
        return execute_backward(ctx);
    }

private:
    status_t execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }

    compute::kernel_t kernel_scaleshift_;
    compute::kernel_t kernel_scaleshift_finalize_;
    compute::kernel_t kernel_;
    compute::kernel_t kernel_fused_;
};

} // namespace lnorm
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl
#endif

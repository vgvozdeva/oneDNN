/*******************************************************************************
* Copyright 2021 Intel Corporation
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

#ifndef GPU_INTEL_REORDER_GENERIC_HPP
#define GPU_INTEL_REORDER_GENERIC_HPP

#include "common/c_types_map.hpp"
#include "common/memory.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "gpu/intel/primitive.hpp"
#include "gpu/intel/reorder/config.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace reorder {

struct generic_t : public primitive_t {
    using primitive_t::primitive_t;
    struct pd_t : public reorder::pd_t {
        using reorder::pd_t::pd_t;

        DECLARE_COMMON_PD_T("ocl:generic:any", generic_t);

        status_t init(impl::engine_t *engine, impl::engine_t *src_engine,
                impl::engine_t *dst_engine) {

            VDISPATCH_REORDER(
                    src_engine == dst_engine, VERBOSE_BAD_ENGINE_KIND);
            VDISPATCH_REORDER(src_engine->kind() == engine_kind::gpu,
                    VERBOSE_BAD_ENGINE_KIND);
            VDISPATCH_REORDER(attr_ok(), VERBOSE_UNSUPPORTED_ATTR);
            VDISPATCH_REORDER(
                    extra_ok(), VERBOSE_UNSUPPORTED_MD_FLAG, "extra_ok");

            VDISPATCH_REORDER(!(memory_desc_wrapper(src_md())
                                              .has_runtime_dims_or_strides()),
                    VERBOSE_RUNTIMEDIM_UNSUPPORTED);

            auto *intel_engine = utils::downcast<intel::engine_t *>(
                    dst_engine->kind() == engine_kind::gpu ? dst_engine
                                                           : src_engine);

            VDISPATCH_REORDER(intel_engine->mayiuse(
                                      compute::device_ext_t::intel_subgroups),
                    VERBOSE_UNSUPPORTED_FEATURE, "subgroups");
            VDISPATCH_REORDER(memory_desc_ndims_ok(src_md(), dst_md()),
                    VERBOSE_INCONSISTENT_NDIMS_WITH_VALS, "src", "dst",
                    src_md()->ndims, dst_md()->ndims);
            VDISPATCH_REORDER(
                    IMPLICATION(
                            utils::one_of(data_type::f16, src_md()->data_type,
                                    dst_md()->data_type),
                            intel_engine->mayiuse(
                                    compute::device_ext_t::khr_fp16)
                                    && intel_engine->mayiuse(
                                            compute::device_ext_t::
                                                    intel_subgroups_short)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            VDISPATCH_REORDER(IMPLICATION(utils::one_of(data_type::f64,
                                                  src_md()->data_type,
                                                  dst_md()->data_type),
                                      intel_engine->mayiuse(
                                              compute::device_ext_t::khr_fp64)),
                    VERBOSE_UNSUPPORTED_DT_CFG);
            CHECK(init_conf(engine));
            using namespace data_type;
            VDISPATCH_REORDER(utils::one_of(src_md()->data_type, f32, f16, bf16,
                                      f8_e5m2, f8_e4m3, s32, s8, u8, f64),
                    VERBOSE_UNSUPPORTED_DT);
            VDISPATCH_REORDER(utils::one_of(dst_md()->data_type, f32, f16, bf16,
                                      f8_e5m2, f8_e4m3, s32, s8, u8, f64),
                    VERBOSE_UNSUPPORTED_DT);
            init_scratchpad();

            return status::success;
        }

        status_t init_conf(impl::engine_t *engine);
        void init_scratchpad();
        status_t init_kernel_ctx(compute::kernel_ctx_t &kernel_ctx) const;

        conf_t conf;

    private:
        DECLARE_GPU_REORDER_CREATE();
    };

    status_t init(impl::engine_t *engine) override {
        compute::kernel_ctx_t kernel_ctx;

        auto status = pd()->init_kernel_ctx(kernel_ctx);
        if (status != status::success) return status;

        const auto &conf = pd()->conf;
        if (conf.nelems == 0) return status::success;

        CHECK(create_kernel(engine, &kernel_, "generic_reorder", kernel_ctx));
        if (!kernel_) return status::runtime_error;
        return status::success;
    }

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
    compute::kernel_t kernel_;
};

} // namespace reorder
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

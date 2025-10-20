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

#include "gpu/gpu_impl_list.hpp"

#include <mutex>

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
#include "gpu/intel/conv/jit.hpp"
#include "gpu/intel/conv/ref.hpp"
#include "gpu/intel/conv/xe_wino.hpp"

#ifdef DNNL_EXPERIMENTAL
#include "common/experimental.hpp"
#include "gpu/intel/conv/jit_v2.hpp"
#endif

#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_NVIDIA
#include "gpu/nvidia/cudnn_convolution.hpp"
#endif

#if DNNL_GPU_VENDOR == DNNL_VENDOR_AMD
#include "gpu/amd/miopen_convolution.hpp"
#endif

#ifdef GENERIC_SYCL_KERNELS_ENABLED
#include "gpu/generic/sycl/ref_convolution.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {

namespace {
using namespace dnnl::impl::prop_kind;

// clang-format off
const std::map<pk_impl_key_t, std::vector<impl_list_item_t>>
        impl_list_map REG_CONV_P({
    {{forward}, {
        GPU_INSTANCE_INTEL(intel::conv::gen_fwd_t)
        GPU_INSTANCE_INTEL(intel::conv::xe_wino_fwd_t)
        GPU_INSTANCE_INTEL_REF(intel::conv::ref_fwd_t)
        GPU_INSTANCE_INTEL_EXPERIMENTAL(intel::conv::v2::gen_fwd_t)
        GPU_INSTANCE_NVIDIA(nvidia::cudnn_convolution_fwd_t)
        GPU_INSTANCE_AMD(amd::miopen_convolution_fwd_t)
        GPU_INSTANCE_GENERIC_SYCL(generic::sycl::ref_convolution_fwd_t)
        nullptr,
    }},
    {{backward_data}, REG_BWD_D_PK({
        GPU_INSTANCE_INTEL(intel::conv::gen_bwd_data_t)
        GPU_INSTANCE_INTEL_REF(intel::conv::ref_bwd_data_t)
        GPU_INSTANCE_INTEL_EXPERIMENTAL(intel::conv::v2::gen_bwd_data_t)
        GPU_INSTANCE_NVIDIA(nvidia::cudnn_convolution_bwd_data_t)
        GPU_INSTANCE_AMD(amd::miopen_convolution_bwd_data_t)
        GPU_INSTANCE_GENERIC_SYCL(generic::sycl::ref_convolution_bwd_data_t)
        nullptr,
    })},
    {{backward_weights}, REG_BWD_PK({
        GPU_INSTANCE_INTEL(intel::conv::gen_bwd_weights_t)
        GPU_INSTANCE_INTEL_REF(intel::conv::ref_bwd_weights_t)
        GPU_INSTANCE_INTEL_EXPERIMENTAL(intel::conv::v2::gen_bwd_weights_t)
        GPU_INSTANCE_NVIDIA(nvidia::cudnn_convolution_bwd_weights_t)
        GPU_INSTANCE_AMD(amd::miopen_convolution_bwd_weights_t)
        GPU_INSTANCE_GENERIC_SYCL(generic::sycl::ref_convolution_bwd_weights_t)
        nullptr,
    })},
});
// clang-format on

const std::map<pk_impl_key_t, std::vector<impl_list_item_t>> &
get_impl_list_map() {
    static std::map<pk_impl_key_t, std::vector<impl_list_item_t>> list_map;
    static std::once_flag flag;
    std::call_once(flag, [&] {
        list_map = impl_list_map;
#if (DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL) && defined(DNNL_EXPERIMENTAL)
        if (experimental::use_gpu_conv_v2()) {
            for (auto &kv : list_map) {
                auto &list = kv.second;
                int fwd_idx = impl_list_item_t::find<
                        intel::conv::v2::gen_fwd_t::pd_t>(&list[0]);
                int bwd_d_idx = impl_list_item_t::find<
                        intel::conv::v2::gen_bwd_data_t::pd_t>(&list[0]);
                int bwd_w_idx = impl_list_item_t::find<
                        intel::conv::v2::gen_bwd_weights_t::pd_t>(&list[0]);
                int idx = std::max({fwd_idx, bwd_d_idx, bwd_w_idx});
                if (idx == -1) continue;
                auto item = list[idx];
                list.erase(list.begin() + idx);
                list.insert(list.begin(), item);
            }
        }
#endif
    });
    return list_map;
}

} // namespace

const impl_list_item_t *get_convolution_impl_list(
        const convolution_desc_t *desc) {
    static const impl_list_item_t empty_list[] = {nullptr};

    const bool is_fwd = utils::one_of(
            desc->prop_kind, forward_training, forward_inference);
    prop_kind_t prop_kind = is_fwd ? forward : desc->prop_kind;

    const auto impl_list_it = get_impl_list_map().find({prop_kind});
    return impl_list_it != get_impl_list_map().cend()
            ? impl_list_it->second.data()
            : empty_list;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl

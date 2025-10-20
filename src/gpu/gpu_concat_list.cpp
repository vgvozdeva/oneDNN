/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#include "gpu/generic/ref_concat.hpp"

#if DNNL_GPU_VENDOR == DNNL_VENDOR_INTEL
#include "gpu/intel/concat/multi.hpp"
#include "gpu/intel/concat/simple.hpp"
#include "gpu/intel/concat/xe.hpp"
#endif

namespace dnnl {
namespace impl {
namespace gpu {

namespace {

// clang-format off
constexpr impl_list_item_t impl_list[] = REG_CONCAT_P({
        GPU_CONCAT_INSTANCE_INTEL(intel::concat::simple_t)
        GPU_CONCAT_INSTANCE_INTEL(intel::concat::xe_t)
        GPU_CONCAT_INSTANCE_INTEL(intel::concat::multi_t)
        GPU_CONCAT_INSTANCE_GENERIC(generic::ref_concat_t)
        nullptr,
});
// clang-format on

} // namespace

const impl_list_item_t *get_concat_impl_list() {
    return impl_list;
}

} // namespace gpu
} // namespace impl
} // namespace dnnl

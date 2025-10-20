/*******************************************************************************
* Copyright 2023 Intel Corporation
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

#include "oneapi/dnnl/dnnl_config.h"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {
namespace jit {
namespace v2 {
namespace planner {
void DNNL_API planner_main(int argc, const char **argv);
}
} // namespace v2
} // namespace jit
} // namespace conv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

int main(int argc, const char **argv) {
    dnnl::impl::gpu::intel::conv::jit::v2::planner::planner_main(argc, argv);
    return 0;
}

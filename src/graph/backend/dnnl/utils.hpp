/*******************************************************************************
* Copyright 2020 Intel Corporation
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

#ifndef GRAPH_BACKEND_DNNL_UTILS_HPP
#define GRAPH_BACKEND_DNNL_UTILS_HPP

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdlib.h>
#include <utility>
#include <vector>

#include "graph/backend/dnnl/common.hpp"

#include "graph/utils/utils.hpp"

namespace dnnl {
namespace impl {
namespace graph {
namespace dnnl_impl {
namespace utils {
template <typename F, typename T,
        typename U = decltype(std::declval<F>()(std::declval<T>()))>
std::vector<U> fmap(const std::vector<T> &vec, const F &f) {
    std::vector<U> result;
    std::transform(vec.begin(), vec.end(), std::back_inserter(result), f);
    return result;
}

inline std::pair<bool, int64_t> try_reverse_axis(
        const int64_t axis, const int32_t rank) {
    // oneDNN can not operate on the negative axis
    const auto new_axis = (axis < 0) ? rank + axis : axis;
    if (new_axis < 0 || new_axis >= static_cast<int64_t>(rank))
        return std::make_pair(false, axis);
    return std::make_pair(true, new_axis);
}

inline std::vector<int32_t> cast_to_int32(const std::vector<int64_t> &vec) {
    return fmap(vec, [](int64_t e) { return static_cast<int32_t>(e); });
}

inline bool all_zero(const std::vector<int64_t> &vec) {
    auto no_zero_pos = std::find_if(
            vec.begin(), vec.end(), [](const int64_t &e) { return e != 0; });
    return no_zero_pos == vec.end();
}

inline dim_t offset_compute(
        const dims_t &strides, const dims_t &idx, int ndims) {
    dim_t off = 0;
    for (int i = 0; i < ndims; i++) {
        off += idx[i] * strides[i];
    }
    return off;
}

} // namespace utils
} // namespace dnnl_impl
} // namespace graph
} // namespace impl
} // namespace dnnl

#endif

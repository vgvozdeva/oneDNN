/*******************************************************************************
* Copyright 2022 Intel Corporation
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

#include <stdio.h>
#include <stdlib.h>

#include "self/self.hpp"

#include "utils/compare.hpp"

namespace self {

static int check_status_change() {
    {
        res_t res {};
        res.state = EXECUTED;
        dnnl_dims_t dims {10};
        dnn_mem_t m0(1, dims, dnnl_f32, tag::abx, get_cpu_engine(),
                /* prefill = */ false);
        dnn_mem_t m1(1, dims, dnnl_f32, tag::abx, get_cpu_engine(),
                /* prefill = */ false);
        for (int i = 0; i < dims[0]; i++) {
            m0.set_elem(i, 0);
            m1.set_elem(i, 0);
        }
        compare::compare_t cmp;
        cmp.compare(m0, m1, attr_t(), &res);
        SELF_CHECK_EQ(res.state, MISTRUSTED);

        // Check that MISTRUSTED can convert into FAILED.
        for (int i = 0; i < dims[0]; i++) {
            m0.set_elem(i, i);
            m1.set_elem(i, i - 1);
        }
        cmp.compare(m0, m1, attr_t(), &res);
        SELF_CHECK_EQ(res.state, FAILED);
    }
    {
        res_t res {};
        res.state = EXECUTED;
        dnnl_dims_t dims {10};
        dnn_mem_t m0(1, dims, dnnl_f32, tag::abx, get_cpu_engine(),
                /* prefill = */ false);
        dnn_mem_t m1(1, dims, dnnl_f32, tag::abx, get_cpu_engine(),
                /* prefill = */ false);
        compare::compare_t cmp;
        for (int i = 0; i < dims[0]; i++) {
            m0.set_elem(i, i);
            m1.set_elem(i, i);
        }
        cmp.compare(m0, m1, attr_t(), &res);
        SELF_CHECK_EQ(res.state, PASSED);

        // Check that PASSED can convert into MISTRUSTED.
        for (int i = 0; i < dims[0]; i++) {
            m0.set_elem(i, 0);
            m1.set_elem(i, 0);
        }
        cmp.compare(m0, m1, attr_t(), &res);
        SELF_CHECK_EQ(res.state, MISTRUSTED);

        // Check that MISTRUSTED can't convert into PASSED back.
        for (int i = 0; i < dims[0]; i++) {
            m0.set_elem(i, i);
            m1.set_elem(i, i);
        }
        cmp.compare(m0, m1, attr_t(), &res);
        SELF_CHECK_EQ(res.state, MISTRUSTED);
    }
    {
        res_t res {};
        res.state = EXECUTED;
        dnnl_dims_t dims {10};
        dnn_mem_t m0(1, dims, dnnl_f32, tag::abx, get_cpu_engine(),
                /* prefill = */ false);
        dnn_mem_t m1(1, dims, dnnl_f32, tag::abx, get_cpu_engine(),
                /* prefill = */ false);
        for (int i = 0; i < dims[0]; i++) {
            m0.set_elem(i, i);
            m1.set_elem(i, i - 1);
        }
        compare::compare_t cmp;
        cmp.compare(m0, m1, attr_t(), &res);
        SELF_CHECK_EQ(res.state, FAILED);
    }
    {
        res_t res {};
        res.state = FAILED;
        dnnl_dims_t dims {10};
        dnn_mem_t m0(1, dims, dnnl_f32, tag::abx, get_cpu_engine(),
                /* prefill = */ false);
        dnn_mem_t m1(1, dims, dnnl_f32, tag::abx, get_cpu_engine(),
                /* prefill = */ false);
        for (int i = 0; i < dims[0]; i++) {
            m0.set_elem(i, 0);
            m1.set_elem(i, 0);
        }
        compare::compare_t cmp;
        cmp.compare(m0, m1, attr_t(), &res);
        SELF_CHECK_EQ(res.state, FAILED);
    }
    {
        res_t res {};
        res.state = FAILED;
        dnnl_dims_t dims {10};
        dnn_mem_t m0(1, dims, dnnl_f32, tag::abx, get_cpu_engine(),
                /* prefill = */ false);
        dnn_mem_t m1(1, dims, dnnl_f32, tag::abx, get_cpu_engine(),
                /* prefill = */ false);
        for (int i = 0; i < dims[0]; i++) {
            m0.set_elem(i, i);
            m1.set_elem(i, i);
        }
        compare::compare_t cmp;
        cmp.compare(m0, m1, attr_t(), &res);
        SELF_CHECK_EQ(res.state, FAILED);
    }
    {
        res_t res {};
        res.state = FAILED;
        dnnl_dims_t dims {10};
        dnn_mem_t m0(1, dims, dnnl_f32, tag::abx, get_cpu_engine(),
                /* prefill = */ false);
        dnn_mem_t m1(1, dims, dnnl_f32, tag::abx, get_cpu_engine(),
                /* prefill = */ false);
        for (int i = 0; i < dims[0]; i++) {
            m0.set_elem(i, i);
            m1.set_elem(i, i - 1);
        }
        compare::compare_t cmp;
        cmp.compare(m0, m1, attr_t(), &res);
        SELF_CHECK_EQ(res.state, FAILED);
    }
    return OK;
}

void res() {
    if (has_bench_mode_modifier(mode_modifier_t::no_ref_memory)) return;
    RUN(check_status_change());
}

} // namespace self

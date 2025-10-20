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

#ifndef GPU_INTEL_CONV_JIT_ZERO_OUT_HPP
#define GPU_INTEL_CONV_JIT_ZERO_OUT_HPP

#include "gpu/intel/jit/codegen/kernel.hpp"
#include "gpu/intel/jit/ir/kernel_desc.hpp"
#include "gpu/intel/jit/ir/kernel_info.hpp"
#include "ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace conv {
namespace jit {

using namespace intel::jit;

class zero_out_kernel_desc_t : public kernel_desc_base_t {
public:
    static const size_t bytes_per_thr;

    zero_out_kernel_desc_t() = default;
    zero_out_kernel_desc_t(int regs, int simd, bool dpas)
        : regs_(regs), simd_(simd), dpas_(dpas) {}
    std::string kernel_name() const override;
    kernel::options_t options(const impl::engine_t *engine) const override;
    bool with_dpas() const override { return dpas_; }
    compute::range_t local_range() const override;
    void init_kernel_iface(kernel::iface_t &kernel_iface) const override;
    void init_kernel_info(kernel_info_t &kernel_info,
            const kernel_params_base_t &params,
            const impl::engine_t *engine) const override;
    status_t create_kernel(compute::kernel_t &kernel, primitive_t *primitive,
            impl::engine_t *engine) const override;
    status_t create_generator(
            const intel::engine_t &engine, compute::kernel_t &kernel) const;
    serialization_stream_t serialize() const override;
    static zero_out_kernel_desc_t deserialize(const serialization_stream_t &s);

    static compute::nd_range_t nd_range(int simd, size_t size);

private:
    int regs_ = 0;
    int simd_ = 0;
    bool dpas_ = false;
};

class zero_out_kernel_params_t : public kernel_params_base_t {
public:
    zero_out_kernel_params_t() = default;
    zero_out_kernel_params_t(size_t size) : size(size) {}

    size_t size = 0;
};

// Reuse IR-to-nGEN generator as it contains useful prologue/epilogue helpers
// and emulation instructions.
template <ngen::HW hw>
class zero_out_kernel_t : public ir_to_ngen_generator_t<generator_t<hw>> {
public:
    IR_TO_NGEN_GENERATOR_FORWARD(generator_t<hw>)

    using base_type = ir_to_ngen_generator_t<generator_t<hw>>;

    zero_out_kernel_t(const kernel::options_t &options,
            const kernel_info_t &kernel_info, bool require_dpas,
            const impl::engine_t *engine)
        : zero_out_kernel_t(zero_out_kernel_desc_t(options.regs(),
                                    options.simd(), require_dpas),
                engine) {}

    zero_out_kernel_t(
            const kernel_desc_base_t &_desc, const impl::engine_t *engine)
        : base_type(get_kernel_iface(_desc), _desc.options(engine),
                debug_config_t {GENERATOR_NAME, GENERATOR_LINE}) {
        requireLocalID(3);
        requireLocalSize();
        requireGRF(options().regs());
        requireSIMD(options().simd());
        requireBarrier();

        externalName(_desc.kernel_name());
        newArgument(kernel_iface()[0].template as<var_t>().name,
                to_ngen(kernel_iface()[0].type()));
        newArgument(kernel_iface()[1].template as<var_t>().name,
                ngen::ExternalArgumentType::GlobalPtr,
                ngen::GlobalAccessType::Stateless);

        finalizeInterface();

        generate_prologue();

        ra().claim(getLocalSize(0));
        ra().claim(getLocalID(0));

        int simd_size = getSIMD();
        bool use_lsc = (hw >= ngen::HW::XeHPG);

        auto size = getArgument(kernel_iface()[0].template as<var_t>().name);
        ra().claim(size);
        auto ptr = getArgument(kernel_iface()[1].template as<var_t>().name);
        ra().claim(ptr);
        auto global_id = ra().template alloc_sub<uint32_t>();
        auto off0 = ra().template alloc_sub<uint32_t>();
        const int bytes_per_thr
                = into<int>(zero_out_kernel_desc_t::bytes_per_thr);

        if (base_type::emu_strategy_.emulate64) {
            base_type::emu_state_.temp[0] = ra().alloc();
            base_type::emu_state_.temp[1] = ra().alloc();
        }

        mul(1, global_id, r0.ud(1), getLocalSize(0).uw());
        add(1, global_id, global_id, getLocalID(0));
        shl(1, off0, global_id, math::ilog2q(bytes_per_thr / simd_size));

        int grf_size = ngen::GRF::bytes(hw);
        int bytes_per_store = 16;
        int uw_size = sizeof(uint16_t);
        int ud_size = sizeof(uint32_t);
        int uq_size = sizeof(uint64_t);

        auto zero = ra().alloc_range(bytes_per_store * ud_size / grf_size);
        auto off_vec = ra().alloc_range(bytes_per_thr * ud_size / grf_size);
        auto off_vec_q_strided
                = ra().alloc_range(bytes_per_thr * uq_size / grf_size);
        auto ptr_vec = ra().alloc_range(bytes_per_thr * uq_size / grf_size);

        for (int i = 0; i < bytes_per_store * ud_size; i += 64) {
            auto z = get_subregister(hw, ngen::DataType::ud, zero, i);
            mov(16, z, 0);
        }

        auto idx_vec = ra().alloc().uw(0);
        mov(8, idx_vec(1), ngen::Immediate::uv(0, 1, 2, 3, 4, 5, 6, 7));
        mov(8, idx_vec(2), idx_vec(1));
        for (int i = 16; i < grf_size / uw_size; i += 16) {
            mov(8, idx_vec.uw(i)(2), idx_vec(2));
        }

        reg_buf_t dst, src0, src1;
        for (int i = 0; i < bytes_per_thr; i += 8) {
            auto off_sub_vec
                    = get_subregister(hw, ngen::DataType::ud, off_vec, i)(1);
            this->eadd3(8, ngen_operand_t(reg_buf_data_t(dst, off_sub_vec)),
                    ngen_operand_t(reg_buf_data_t(
                            src1, idx_vec.uw((i % grf_size) * 2)(2))),
                    ngen_operand_t(reg_buf_data_t(src0, off0)),
                    ngen_operand_t(i));
            auto ptr_sub_vec
                    = get_subregister(hw, ngen::DataType::uq, ptr_vec, i)(1);
            auto off_sub_vec_q_strided = get_subregister(
                    hw, ngen::DataType::ud, off_vec_q_strided, i * 2)(2);
            emov(8, off_sub_vec_q_strided, off_sub_vec);
            eadd(8, ptr_sub_vec, ptr, off_sub_vec_q_strided);
        }

        for (int i = 0; i < bytes_per_thr; i += bytes_per_store) {
            auto off_sub_vec
                    = get_subregister(hw, ngen::DataType::ud, off_vec, i)(1);
            cmp(16 | lt | f0[0], off_sub_vec, size);
            auto h = get_subregister(hw, ngen::DataType::uq, ptr_vec, i);
            if (use_lsc) {
                std::unique_ptr<ngen::DataSpecLSC> lsc_spec;
                lsc_spec = utils::make_unique<ngen::DataSpecLSC>(
                        ngen::scattered(ngen::DataSizeLSC::D8U32, 1));
                store.ugm(16 | f0[0], *lsc_spec, A64, h, zero[0]);
            } else {
                store(16 | f0[0], ngen::scattered_byte(), A64, h, zero[0]);
            }
        }

        generate_epilogue();
    }

private:
    static kernel::iface_t get_kernel_iface(const kernel_desc_base_t &desc) {
        kernel::iface_t iface(desc.kernel_name());
        desc.init_kernel_iface(iface);
        return iface;
    }
};

} // namespace jit
} // namespace conv
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

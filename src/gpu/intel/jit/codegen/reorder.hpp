/*******************************************************************************
* Copyright 2022-2025 Intel Corporation
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

#ifndef GPU_INTEL_JIT_CODEGEN_REORDER_HPP
#define GPU_INTEL_JIT_CODEGEN_REORDER_HPP

#include <functional>

#include "common/utils.hpp"
#include "gpu/intel/gemm/jit/generator/pieces/copy_plan.hpp"
#include "gpu/intel/jit/codegen/operand.hpp"
#include "gpu/intel/jit/codegen/register_scope.hpp"
#include "gpu/intel/jit/ir/reorder.hpp"
#include "gpu/intel/jit/ir/tensor.hpp"
#include "gpu/intel/logging.hpp"
#include "ngen.hpp"

namespace dnnl {
namespace impl {
namespace gpu {
namespace intel {
namespace jit {

struct copy_operand_t : gemmstone::CopyOperand {
    copy_operand_t() = default;
    copy_operand_t(const CopyOperand &op) : CopyOperand(op) {}
    copy_operand_t(const reg_buf_data_t &rbd);

    copy_operand_t &advance(ngen::HW hw, dim_t elems, uint8_t stride = 1);

    std::vector<int> block_bases;
    int block_size = 0;
    int block_off = 0;
};

struct copy_plan_t : gemmstone::CopyPlan {
    using gemmstone::CopyPlan::newTemp;

    copy_plan_t(ngen_register_scope_t &scope, bool systolic_support)
        : CopyPlan(scope.hw(), systolic_support), scope_(scope) {}

    ngen::HW hw() const { return CopyPlan::hw; }

    void mov(int simd, ngen::InstructionModifier mod, const copy_operand_t &dst,
            const copy_operand_t &src);

    void mov(int simd, const copy_operand_t &dst, const copy_operand_t &src) {
        return mov(simd, {}, dst, src);
    }

    template <typename Generator>
    void execute(Generator &generator) {
        using namespace std::placeholders;
        GRFAllocator grf = std::bind(&copy_plan_t::alloc_grf, this, _1, _2);
        FlagAllocator flag = std::bind(&copy_plan_t::alloc_flag, this, _1, _2);
        CopyPlan::materializeTemps(grf, flag);
        CopyPlan::execute(generator);
    }

    void alloc_grf(int count, ngen::GRFRange &range) {
        if (count > 0)
            range = scope_.try_alloc_range(count);
        else
            scope_.safeRelease(range);
    }

    void alloc_flag(int bytes, ngen::FlagRegister &flag) {
        if (bytes > 0)
            flag = scope_.try_alloc_flag(bytes * 8);
        else
            scope_.safeRelease(flag);
    }

    int phase = 0;

protected:
    using CopyPlan::materializeTemps;

    ngen_register_scope_t &scope_;
};

template <typename GeneratorT>
void emit_reorder_1d_tile(GeneratorT *host, ngen_register_scope_t &scope,
        int width, const reg_buf_data_t &src, int src_stride,
        const reg_buf_data_t &dst, int dst_stride) {
    copy_plan_t plan(scope, host->hw_info().systolic_support());
    copy_operand_t dst_op = dst;
    copy_operand_t src_op = src;
    dst_op.stride = (uint8_t)dst_stride;
    src_op.stride = (uint8_t)src_stride;

    if (!math::is_pow2(src_stride) || !math::is_pow2(dst_stride)) {
        for (int i = 0; i < width; ++i) {
            plan.mov(1, dst_op, src_op);
            dst_op.advance(plan.hw(), dst_stride);
            src_op.advance(plan.hw(), src_stride);
        }
    } else
        plan.mov(width, dst_op, src_op);
    plan.transform();
    plan.execute(*host);
}

template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const reg_buf_data_t &dst,
        reg_buf_data_t &src) {
    int src_stride = src.hs();
    // src is broadcasted, no need to align, return.
    if (src_stride == 0) return;

    bool is_xf = ngen_is_xf(src.type()) || ngen_is_xf(dst.type());
    bool is_bf_to_f = (src.type() == ngen::DataType::bf)
            && (dst.type() == ngen::DataType::f);
    int src_type_size = ngen::getBytes(src.type());
    int dst_type_size = ngen::getBytes(dst.type());
    int src_off = src.offset();
    int dst_off = dst.offset();
    int src_byte_off = src.byte_offset();
    int dst_byte_off = dst.byte_offset();
    int esize = mod.getExecSize();
    const int grf_size = ngen::GRF::bytes(scope.hw());
    // within the current generator, HS == 0 can mean 2 things:
    //   - <0; 1, 0>, i.e. a scalar value so HS is to be treated as 1
    //   - <1; 1, 0>, which is a more compatible representation of <N; N, 1>
    int grf_src = grf_size / std::max(src.hs(), 1);
    int grf_dst = grf_size / std::max(dst.hs(), 1);

    // If src is aligned with dst, return.
    if ((is_xf || is_bf_to_f) && src_off % grf_src == dst_off % grf_dst) return;
    if (!is_xf && src_byte_off % grf_size == dst_byte_off % grf_size) return;

    int new_src_off = (is_xf ? dst_off * src_type_size / dst_type_size
                             : dst_off * dst_type_size / src_type_size);

    int src_size = std::max(src_type_size * esize * src_stride, src_type_size);
    auto new_src = scope.alloc_reg_buf_data(
            utils::div_up(src_size + new_src_off * src_type_size, grf_size));
    new_src = new_src.format(new_src_off, esize, src_stride, src.type());
    emit_reorder_1d_tile(
            host, scope, esize, src, src_stride, new_src, src_stride);
    src = std::move(new_src);
}

template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const reg_buf_data_t &dst,
        reg_buf_data_t &src0, reg_buf_data_t &src1) {
    align_src_dst_offset(host, scope, mod, dst, src0);
    align_src_dst_offset(host, scope, mod, dst, src1);
}

template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
        ngen_operand_t &src) {
    if (!src.is_reg_data()) return;
    auto rd = src.reg_buf_data();

    if (!dst.is_reg_data()) {
        // Float pipe requires src operands to align with dst, even if that's
        // the null register. In the case of the null register, we align to the
        // GRF boundary.
        reg_buf_data_t dummy(reg_buf_t(rd.hw(), ngen::GRFRange(0, 1)));
        // This call returns early if everything is already aligned nicely
        align_src_dst_offset(host, scope, mod, dummy, rd);
    } else {
        align_src_dst_offset(host, scope, mod, dst.reg_buf_data(), rd);
    }
    if (rd == src.reg_buf_data()) return;

    bool is_negated = src.is_negated();
    src = ngen_operand_t(rd, src.mod());
    if (is_negated) src = -src;
}

template <typename GeneratorT>
void align_src_dst_offset(GeneratorT *host, ngen_register_scope_t &scope,
        const ngen::InstructionModifier &mod, const ngen_operand_t &dst,
        ngen_operand_t &src0, ngen_operand_t &src1) {
    align_src_dst_offset(host, scope, mod, dst, src0);
    align_src_dst_offset(host, scope, mod, dst, src1);
}

struct reorder_operand_t {
    layout_t layout;
    copy_operand_t buffer;

    type_t type() const {
        return buffer.range == ngen::DataType::invalid ? layout.type()
                                                       : to_ir(buffer.range);
    }
    bool operator==(const reorder_operand_t &other) const {
        return layout.is_equal_normalized(other.layout)
                && buffer == other.buffer;
    }
};

// Implementation of GRF reorder between 2D dense layouts.
// Requirements for A -> B reorder:
// - A and B must have the same data type
// - Layouts must be 2D and dense
// Reorder may require several steps, in this case a temporary buffer T is
// allocated. For example: A -> T -> B or A -> B -> T -> B
class reorder_2d_impl_t {
    struct reorder_step_t;

public:
    reorder_2d_impl_t(ngen::HW hw, tile_t tile, const layout_t &src_layout,
            const layout_t &dst_layout);

    const tile_t &tile() const { return tile_; }
    const std::vector<reorder_step_t> &path() const { return path_; }

    void emit(copy_plan_t &plan, copy_operand_t &src, copy_operand_t &dst);

    static const int max_tile_blocks = 4;

private:
    // Represents 2D reorder corresponding to (a x b) tile.
    struct edge_t {
        edge_t() = default;
        edge_t(int idx, int a, int b) : idx(idx), a(a), b(b) {}

        tile_t tile() const { return tile_t(std::vector<dim_t> {a, b}); }

        std::string str() const {
            ostringstream_t oss;
            oss << "edge(idx = " << idx << ", a = " << a << ", b = " << b
                << ")";
            return oss.str();
        }

        int idx; // Identifier of the edge.
        int a = 0, b = 0; // Specify tile (a x b).
    };

    // Represents GRF layout between edges-reorders.
    struct vertex_t {
        vertex_t(ngen::HW hw, int idx, const layout_t &layout)
            : hw(hw), idx(idx), layout(layout) {}

        std::string str() const {
            ostringstream_t oss;
            oss << "vertex(idx = " << idx << ", layout = " << layout << ")";
            return oss.str();
        }

        void set_edges(const std::vector<edge_t> &edges);
        void add_neighbor(const vertex_t *v) { adj_vertices.push_back(v); }

        bool is_neighbor(const vertex_t &v) const {
            for (auto *n : adj_vertices)
                if (n == &v) return true;
            return false;
        }

        bool can_reorder(const tile_t &tile, const type_t &type) const;
        int cost(const vertex_t &v, const std::vector<edge_t> &edges,
                edge_t &min_edge, type_t &min_type) const;
        int cost(const edge_t &e, const vertex_t &v, type_t &type) const;

        ngen::HW hw;
        int idx; // Identifier of the vertex.
        layout_t layout; // Layout of the vertex.
        // Specifies a bitmask for every edge: if adj_edge_type_masks[E_idx]
        // has b-th bit set then this vertex can be reordered through E edge
        // using the data type with size 2^b bytes.
        std::vector<uint32_t> adj_edge_type_masks;
        std::vector<const vertex_t *> adj_vertices; // Adjacent vertices.
    };

    // Represents a reorder step.
    struct reorder_step_t {
        reorder_step_t() = default;
        reorder_step_t(
                const layout_t &layout, const tile_t &tile, const type_t &type)
            : layout(layout), tile(tile), type(type) {}

        layout_t layout; // Destination layout.
        tile_t tile; // Tile corresponding to one instruction.
        type_t type; // Registers should be reinterpreted to `type` for reorder.
    };

    // Extracts dimension sizes and their indices from a multidimensional
    // tensor.
    static void tile_to_2d_dims(const tile_t &tile, dim_idx_t &a_idx,
            dim_idx_t &b_idx, dim_t &a, dim_t &b);

    // Finds the optimal sequence of reorders between src and dst layouts.
    static std::vector<reorder_step_t> find_min_cost_path(ngen::HW hw,
            const layout_t &src, const layout_t &dst, dim_t tile_a,
            dim_t tile_b);

    // Returns all possible layouts for (a x b) tensor.
    static std::vector<layout_t> generate_all_layouts(
            const type_t &type, dim_t a, dim_t b) {
        std::vector<layout_t> ret;
        std::vector<layout_block_t> blocks;
        generate_all_layouts_impl(ret, blocks, type, a, b, 1);
        return ret;
    }

    static void generate_all_layouts_impl(std::vector<layout_t> &layouts,
            std::vector<layout_block_t> &blocks, const type_t &type, dim_t a,
            dim_t b, dim_t stride);

    ngen::HW hw_;
    tile_t tile_;
    layout_t src_;
    layout_t dst_;
    std::vector<reorder_step_t> path_;
};

class reorder_impl_t {
public:
    reorder_impl_t(
            ngen::HW hw, const layout_t &src_layout, const layout_t &dst_layout)
        : hw_(hw), src_layout_(src_layout), dst_layout_(dst_layout) {
        try_reinterpret_to_wider_type(src_layout_, dst_layout_);
    }

    reorder_impl_t(ngen::HW hw, const reorder_t &reorder)
        : reorder_impl_t(hw, reorder.src_layout, reorder.dst_layout) {}

    template <typename GeneratorT>
    void emit(GeneratorT *host, ngen_register_scope_t &scope,
            const reg_buf_data_t &src, const reg_buf_data_t &dst) {
        auto from_rd = [](const reg_buf_data_t &rd, int off = 0) -> op_init_t {
            return [&rd, off](int elems, ngen::DataType dt) {
                return rd.format(off, elems, 1, dt);
            };
        };

        for (const auto &tile : tiles()) {
            copy_plan_t plan(scope, host->hw_info().systolic_support());
            const auto base_phase = plan.phase;
            auto src_tile = src_layout_.sub(tile);
            auto dst_tile = dst_layout_.sub(tile);
            auto emit_tile = [&](const icoord_t &start) {
                auto src_off = src_layout_.offset<int>(start);
                auto dst_off = dst_layout_.offset<int>(start);
                auto src_op = init_operand(src_tile, from_rd(src, src_off));
                auto dst_op = init_operand(dst_tile, from_rd(dst, dst_off));
                emit(plan, src_op, dst_op);
                plan.phase = base_phase;
            };
            dst_layout_.for_each_tile(tile, emit_tile);
            plan.transform();

            try {
                plan.execute(*host);
                return;
            } catch (const ngen::out_of_registers_exception &) {
                gpu_debug() << "Reorder with tile " << tile
                            << " results in out-of-registers. Trying with a "
                               "smaller tile.";
            }
        }
        throw ngen::out_of_registers_exception();
    }

    void emit(copy_plan_t &plan, const reorder_operand_t &src,
            reorder_operand_t &dst);

private:
    using op_init_t = std::function<copy_operand_t(int, ngen::DataType)>;

    std::vector<tile_t> tiles() const;

    bool layouts_compatible(const layout_t &a, const layout_t &b) const;

    reorder_operand_t init_operand(layout_t layout, const op_init_t &init) {
        if (layout.type().is_tf32()) layout = layout.with(type_t::f32());
        auto elems = size_in_elems(layout);
        auto dt = to_ngen(layout.type());
        auto buffer = init(into<int>(elems), dt);
        buffer.stride = (uint8_t)1;
        return {std::move(layout), std::move(buffer)};
    }

    layout_t make_retyped_layout(
            const layout_t &layout, const type_t &type) const;
    layout_t make_compact_layout(const layout_t &layout, const type_t &type,
            bool is_source = false) const;

    dim_t size_in_elems(const layout_t &layout) {
        const auto &type = layout.type();
        return size_bytes(layout) * type.packing() / type.size();
    }

    type_t intermediate_data_type(const type_t &s, const type_t &d) const {
        // Force up-/down-convert of small types
        if (s.is_fp4() || d.is_fp4()) return type_t::f16();
        // int4 -> fp16 has special conversion paths
        if (s.is_x4() && (d.is_f16() || d.is_bf16())) return d;
        if (d.is_x4() && (s.is_f16() || s.is_bf16())) return s;
        if (s.is_u4() || d.is_u4()) return type_t::u16();
        if (s.is_s4() || d.is_s4()) return type_t::s16();

        if (s == d) return d; // Swizzle only
        if (s.is_fp8() || d.is_fp8()) return type_t::f16();
        return s.bitsize() > d.bitsize() ? d : s;
    }

    bool needs_saturate(const type_t &ddt, const type_t &sdt) const {
        if (!ddt.is_int() || !sdt.is_int()) return false;
        if (ddt.bitsize() >= sdt.bitsize()
                && ddt.is_signed() == sdt.is_signed())
            return false;
        return true;
    }

    void emit_1d(copy_plan_t &plan, const reorder_operand_t &dst,
            const reorder_operand_t &src);

    static std::vector<tile_t> find_2d_dense_tiles(
            const layout_t &a, const layout_t &b);

    bool try_emit_2d(copy_plan_t &plan, const reorder_operand_t &dst,
            const reorder_operand_t &src);

    static tile_t find_max_tile_with_fixed_stride(const layout_t &src,
            const layout_t &dst, int &src_stride, int &dst_stride);

    ngen::HW hw_;
    layout_t src_layout_;
    layout_t dst_layout_;
};

} // namespace jit
} // namespace intel
} // namespace gpu
} // namespace impl
} // namespace dnnl

#endif

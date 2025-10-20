/*******************************************************************************
* Copyright 2025 Intel Corporation
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

#include "gemmstone/strategy.hpp"
#include "gpu/intel/gemm/jit/generator_dsl/kernel_desc.hpp"
#include "gpu/intel/gemm/jit/include/gemmstone/config.hpp"
#include "gpu/intel/jit/dsl/dsl.hpp"
#include "gpu/intel/jit/pass/pass.hpp"
#include "gpu/intel/jit/utils/trace.hpp"
#include "gpu/intel/utils.hpp"

GEMMSTONE_NAMESPACE_START

using namespace dsl;

inline ir::type_t into_ir(Type t, int elems = 1) {
    using namespace ir;
    switch (t) {
        case Type::invalid: return type_t::undef();

        case Type::f4_e3m0: return type_t::f4_e3m0(elems);
        case Type::f4_e2m1: return type_t::f4_e2m1(elems);
        case Type::bf8: return type_t::bf8(elems);
        case Type::hf8: return type_t::hf8(elems);
        case Type::bf16: return type_t::bf16(elems);
        case Type::f16: return type_t::f16(elems);
        case Type::tf32: return type_t::tf32(elems);
        case Type::f32: return type_t::f32(elems);
        case Type::f64: return type_t::f64(elems);

        case Type::u4: return type_t::u4(elems);
        case Type::s4: return type_t::s4(elems);
        case Type::u8: return type_t::u8(elems);
        case Type::s8: return type_t::s8(elems);
        case Type::u16: return type_t::u16(elems);
        case Type::s16: return type_t::s16(elems);
        case Type::u32: return type_t::u32(elems);
        case Type::s32: return type_t::s32(elems);
        case Type::u64: return type_t::u64(elems);
        case Type::s64: return type_t::s64(elems);

        default: stub(); return type_t::undef();
    }
}

struct transform_t {
    // Sample transforms on bf16 data with pack_size 16:
    // none:           64a64b -> 64a64b
    // block:          64a64b -> 4b64a16b
    // vnni:           64a64b -> 4b32a16b2a
    // transpose_vnni: 64a64b -> 4a32b16a2b
    enum class kind_t { none, block, vnni, transpose_vnni };

    transform_t() = default;

    transform_t(kind_t t_kind, int pack_size, ngen::CacheSettingsLSC cache_hint,
            std::array<idx_t, 2> dims)
        : kind(t_kind)
        , pack_size(pack_size)
        , cache_hint(to_ir(cache_hint))
        , dims(std::move(dims)) {}

    layout_t get_layout(const ir::tile_t &sizes, ir::type_t type) const {

        auto col_var = dims[0];
        auto col = sizes[dims[0]];
        auto row_var = dims[1];
        auto row = sizes[dims[1]];
        auto t = type.size();

        auto normalized = kind;
        if (normalized == kind_t::transpose_vnni) {
            std::swap(col_var, row_var);
            std::swap(col, row);
            normalized = kind_t::vnni;
        }

        if (normalized == kind_t::vnni && t >= 4) normalized = kind_t::block;

        int col_inner = pack_size ? pack_size : grf_size();
        if (normalized == kind_t::block && col <= col_inner)
            normalized = kind_t::none;

        switch (normalized) {
            case kind_t::none:
                return layout_t(type, {{col_var, col, 1}, {row_var, row, col}});

            case kind_t::block: {
                int col_outer = (int)(col / col_inner);
                return layout_t(type,
                        {{col_var, col_inner, 1}, {row_var, row, col_inner},
                                {col_var, col_outer, row * col_inner}});
            }

            case kind_t::vnni: {
                int row_inner = 4 / t;
                int row_outer = (int)(row / row_inner);
                int col_outer = (int)(col / col_inner);
                return layout_t(type,
                        {{row_var, row_inner, 1},
                                {col_var, col_inner, row_inner},
                                {row_var, row_outer, col_inner * row_inner},
                                {col_var, col_outer,
                                        row_outer * col_inner * row_inner}});
            }

            // Impossible to hit due to normalization
            case kind_t::transpose_vnni:
            default: gpu_assert(false); return {};
        }
    }

    static ir::send_cache_hint_t to_ir(ngen::CacheSettingsLSC hint) {
        switch (hint) {
            case ngen::CacheSettingsLSC::L1C_L3C:
                return ir::send_cache_hint_t::load_once;
            case ngen::CacheSettingsLSC::Default:
                return ir::send_cache_hint_t::hw_default;
            default: gpu_assert(false); return ir::send_cache_hint_t::undef;
        }
    }

    kind_t kind = kind_t::none;
    int pack_size = 0;
    ir::send_cache_hint_t cache_hint = ir::send_cache_hint_t::undef;
    std::array<idx_t, 2> dims = {};
};

static const idx_t m_var("m");
static const idx_t n_var("n");
static const idx_t k_var("k");

struct kloop_iterator_t {

    virtual const global_tensor_t &A_prefetch() const = 0;
    virtual const global_tensor_t &A_load() const = 0;
    virtual const global_tensor_t &B_prefetch() const = 0;
    virtual const global_tensor_t &B_load() const = 0;
    virtual const global_tensor_t &C_store() const = 0;

    virtual void A_prefetch_inc(int k_block) = 0;
    virtual void A_load_inc(int k_block) = 0;

    virtual void B_prefetch_inc(int k_block) = 0;
    virtual void B_load_inc(int k_block) = 0;

    virtual void kloop_inc(int k_block) = 0;

    virtual expr_t update_C() const = 0;

    // Returns whether the given increment is in bounds
    virtual expr_t is_inbounds(int increment) const = 0;
};

transform_t get_transform(const MatrixAddressingStrategy &matrix_strategy,
        std::array<idx_t, 2> dims, bool is_prefetch = false) {
    switch (matrix_strategy.accessType) {
        case AccessType::Scattered:
            // TODO: Remove workaround unimplemented scattered->vnni support.
            if (is_prefetch)
                return transform_t(transform_t::kind_t::none, 0,
                        matrix_strategy.cachingR, dims);

            return transform_t(transform_t::kind_t::transpose_vnni,
                    matrix_strategy.tileR, matrix_strategy.cachingR, dims);

        case AccessType::ChannelScattered: stub(); return {};
        case AccessType::Block2DTranspose:
            return transform_t(transform_t::kind_t::transpose_vnni,
                    matrix_strategy.tileR, matrix_strategy.cachingR, dims);
        case AccessType::Block:
        case AccessType::PseudoBlock:
            return transform_t(transform_t::kind_t::block,
                    matrix_strategy.tileR, matrix_strategy.cachingR, dims);
        case AccessType::Block2D: {
            return transform_t(transform_t::kind_t::block,
                    matrix_strategy.tileR, matrix_strategy.cachingR, dims);
        };
        case AccessType::Block2DVNNI: {
            return transform_t(transform_t::kind_t::vnni, matrix_strategy.tileR,
                    matrix_strategy.cachingR, dims);
        }
        default: stub(); return {};
    }
};
ir::pvar_map_t<expr_t> get_strides(
        MatrixLayout layout, std::array<idx_t, 2> pvars, expr_t ld) {
    switch (layout) {
        case MatrixLayout::N: return {{pvars[0], 1}, {pvars[1], ld}};
        case MatrixLayout::T: return {{pvars[0], ld}, {pvars[1], 1}};
        default: stub(); return {};
    };
}

struct tensor_config_t {
    tensor_config_t(const global_tensor_t &g, transform_t t, int copies)
        : transform(t) {
        tile = g.tile;
        layout = t.get_layout(g.tile, g.type);
        layout = layout.with_block({k_var, copies});
    }

    ir::tile_t tile;
    layout_t layout;

    transform_t transform;
};

void apply_post_ops(const dnnl::impl::gpu::intel::gpu_post_ops_t &ops,
        const tensor_t &C, const std::vector<expr_t> &idxs,
        const std::vector<idx_t> &dims) {
    for (size_t i = 0; i < ops.len(); i++) {
        if (ops[i].is_eltwise()) {
            gpu_assert(false) << "Unimplemeted";
        } else if (ops[i].is_sum()) {
            gpu_assert(false) << "Unimplemeted";
        } else if (ops[i].is_binary()) {
            auto &e = ops[i].as_binary();
            std::string i_s = std::to_string(i);
            std::string stride_prefix = "binary" + i_s + "_stride";

            int ndims = (int)dims.size();
            ir::pvar_map_t<int> dim_to_md;
            for (int i = 0; i < ndims; i++) {
                dim_to_md[dims[i]] = i;
            };

            std::vector<expr_t> strides;
            strides.reserve(idxs.size());
            for (unsigned int j = 0; j < idxs.size(); j++) {
                if (e.src1_desc.is_inner_dim(j, ndims)) {
                    strides.emplace_back(1);
                } else if (e.src1_desc.is_broadcast(j, ndims)) {
                    strides.emplace_back(0);
                } else {
                    strides.emplace_back(
                            arg(stride_prefix + std::to_string(j), true));
                }
            }

            auto src_g = [&]() -> global_tensor_t {
                expr_t src_g_offset = simplify(arg("offset_binary" + i_s)
                        + e.src1_desc.get_offset(idxs, strides));

                ir::pvar_map_t<expr_t> g_strides;
                ir::pvar_map_t<expr_t> g_sizes;
                for (int i = 0; i < ndims; i++) {
                    g_strides[dims[i]] = strides[i];
                    g_sizes[dims[i]] = e.src1_desc.is_broadcast(i, ndims)
                            ? expr_t(1)
                            : expr_t(0); //TODO: Get actual size
                }

                return {arg("binary" + i_s),
                        dnnl::impl::gpu::intel::jit::to_ir(e.src1_desc.dt),
                        src_g_offset, coord_t(), g_sizes, g_strides, {}};
            }();

            layout_t src_layout = {src_g.type};
            for (auto &b : C.layout.blocks()) {
                if (!e.src1_desc.is_broadcast(dim_to_md[b.idx], ndims)) {
                    src_layout = src_layout.with_block({b.idx, b.size});
                } else {
                    src_layout = src_layout.with_block({b.idx, 1});
                }
            }

            tensor_t src = def("binary" + i_s + "_blk", src_layout);
            std::cout << "src_g: " << src_g.str() << "\n";
            std::cout << "src: " << src.str() << "\n";
            load(src, src_g);

            switch (e.alg) {
                case dnnl::impl::alg_kind::binary_add:
                    binary(ir::op_kind_t::_add, C, C, src);
                    break;
                default: gpu_assert(false) << "Unimplemented";
            }

        } else {
            gpu_assert(false) << "Unimplemented";
        }
    }
}

// Basic iterator with no iteration over m and n.
struct basic_iterator_t : kloop_iterator_t {
    basic_iterator_t(const global_tensor_t &A, int A_prefetch_k_blk,
            int A_load_k_blk, const global_tensor_t &B, int B_prefetch_k_blk,
            int B_load_k_blk, const global_tensor_t &C)
        : m_idx_ {C.coord[m_var]}
        , m_(C.sizes[m_var])
        , n_idx_ {C.coord[n_var]}
        , n_(C.sizes[n_var])
        , k_idx_ {A.coord[k_var]}
        , k_ {A.sizes[k_var]}
        , A_prefetch_ {A.buf, A.type, A.base_offset, A.coord, A.sizes,
                  A.strides,
                  tile_t {{m_var, C.tile[m_var]}, {k_var, A_prefetch_k_blk}}}
        , A_load_ {A.buf, A.type, A.base_offset, A.coord, A.sizes, A.strides,
                  tile_t {{m_var, C.tile[m_var]}, {k_var, A_load_k_blk}}}
        , B_prefetch_ {B.buf, B.type, B.base_offset, B.coord, B.sizes,
                  B.strides,
                  tile_t {{k_var, B_prefetch_k_blk}, {n_var, C.tile[n_var]}}}
        , B_load_ {B.buf, B.type, B.base_offset, B.coord, B.sizes, B.strides,
                  tile_t {{k_var, B_load_k_blk}, {n_var, C.tile[n_var]}}}
        , C_store_ {C}

    {
        assume(m_idx_ % C.tile[m_var] == 0);
        assume(n_idx_ % C.tile[n_var] == 0);

        assume(m_idx_ >= 0);
        assume(n_idx_ >= 0);
        assume(k_idx_ >= 0);
    }

    const global_tensor_t &A_prefetch() const override { return A_prefetch_; }
    const global_tensor_t &A_load() const override { return A_load_; }
    const global_tensor_t &B_prefetch() const override { return B_prefetch_; }
    const global_tensor_t &B_load() const override { return B_load_; }
    const global_tensor_t &C_store() const override { return C_store_; }

    void A_prefetch_inc(int k_block) override {
        A_prefetch_off += k_block;
        A_prefetch_.coord[k_var] = k_idx_ + A_prefetch_off;
    }

    void A_load_inc(int k_block) override {
        A_load_off += k_block;
        A_load_.coord[k_var] = k_idx_ + A_load_off;
    }

    void B_prefetch_inc(int k_block) override {
        B_prefetch_off += k_block;
        B_prefetch_.coord[k_var] = k_idx_ + B_prefetch_off;
    }

    void B_load_inc(int k_block) override {
        B_load_off += k_block;
        B_load_.coord[k_var] = k_idx_ + B_load_off;
    }

    void kloop_inc(int k_block) override {
        // Prefetch/load computation is relative to k_idx
        A_prefetch_inc(-k_block);
        B_prefetch_inc(-k_block);
        A_load_inc(-k_block);
        B_load_inc(-k_block);

        assign(k_idx_, k_idx_ + k_block);
    }

    expr_t update_C() const override { return false; }

    expr_t is_inbounds(int increment) const override {
        return (m_idx_ < m_) & (n_idx_ < n_) & (k_idx_ < k_ - increment);
    }

private:
    expr_t m_idx_;
    expr_t m_;
    expr_t n_idx_;
    expr_t n_;
    expr_t k_idx_;
    expr_t k_;

    int A_prefetch_off = 0;
    int A_load_off = 0;
    int B_prefetch_off = 0;
    int B_load_off = 0;

    global_tensor_t A_prefetch_;
    global_tensor_t A_load_;
    global_tensor_t B_prefetch_;
    global_tensor_t B_load_;
    global_tensor_t C_store_;
};

struct generator_dsl_t {
    generator_dsl_t(const generator_dsl_desc_t &desc)
        : problem(desc.problem), strategy(desc.strategy) {}

    kernel_t build(ir::kernel::iface_t iface, ir::ir_context_t &ctx) {
        if (strategy.kParallel || strategy.kParallelLocal) {
            gpu_warning() << "kParallel support is unimplemented";
            return {};
        }
        if (strategy.persistentLoop()) {
            gpu_warning() << "persistentLoop support is unimplemented";
            return {};
        }
        if (strategy.slmA || strategy.slmB) {
            gpu_warning() << "slm copy support is unimplemented, disabling "
                             "slm copy";
        }

        if (strategy.wgPadFactor > 1) {
            gpu_warning() << "work group padding is unimplemented";
            return {};
        }

        if (strategy.cWalkOrder != WalkOrder::HW2D) {
            gpu_warning() << "Unsupported walk order";
            return {};
        }

        if (problem.Ta != problem.Ta_ext || problem.Tb != problem.Tb_ext
                || problem.Tc != problem.Tc_ext) {
            gpu_warning() << "Type conversion support is unimplemented";
            return {};
        }

        if (problem.batch != BatchMode::None
                && problem.batch != BatchMode::Strided) {
            gpu_warning() << "Batch mode is unimplemented";
            return {};
        }

        declare_kernel(iface, ctx);

        const auto m = arg("m");
        const auto n = arg("n");
        const auto k = arg("k");

        auto m_blk = strategy.unroll[LoopM];
        auto n_blk = strategy.unroll[LoopN];
        auto k_blk = strategy.unroll[LoopK];

        std::array<idx_t, 2> A_vars = {m_var, k_var};
        std::array<idx_t, 2> B_vars = {k_var, n_var};
        std::array<idx_t, 2> C_vars = {m_var, n_var};

        auto A_prefetch_transform
                = get_transform(strategy.A_prefetch, A_vars, true);
        auto A_load_transform = get_transform(strategy.A, A_vars);

        auto B_prefetch_transform
                = get_transform(strategy.B_prefetch, B_vars, true);
        auto B_load_transform = get_transform(strategy.B, B_vars);

        ir::tile_t C_dims {{{m_var, m_blk}, {n_var, n_blk}}};
        auto C_store_transform = get_transform(strategy.C, C_vars);

        tensor_t C = def("C_blk",
                C_store_transform.get_layout(C_dims, into_ir(problem.Tc)), 0);

        idx_t subgroup_dim = C.layout[0].idx;
        int m_group_idx = strategy.loopOrder[0] == LoopM ? 0 : 1;
        auto m_idx = let("m_idx",
                (group_id(m_group_idx) * local_size(m_group_idx)
                        + local_id(m_group_idx))
                        * (subgroup_dim == m_var ? m_blk / strategy.subgroupSize
                                                 : m_blk));
        int n_group_idx = strategy.loopOrder[0] == LoopN ? 0 : 1;
        auto n_idx = let("n_idx",
                (group_id(n_group_idx) * local_size(n_group_idx)
                        + local_id(n_group_idx))
                        * (subgroup_dim == n_var ? n_blk / strategy.subgroupSize
                                                 : n_blk));
        auto k_idx = def("k_idx", k.type(), 0);

        auto offset_A = arg("offset_A");
        auto offset_B = arg("offset_B");
        auto offset_C = arg("offset_C");

        std::vector<expr_t> C_idxs = {m_idx, n_idx};
        if (problem.batch == BatchMode::Strided) {
            struct info_t {
                info_t(expr_t size, expr_t idiv_magic)
                    : size(std::move(size))
                    , idiv_magic(std::move(idiv_magic)) {}
                expr_t size;
                expr_t idiv_magic;
            };

            auto info = [&]() {
                std::vector<info_t> ret;
                ret.reserve(problem.batchDims - 1);
                for (int i = 0; i < problem.batchDims - 1; i++) {
                    std::string i_s = std::to_string(i);
                    ret.emplace_back(
                            arg("batch_size" + i_s), arg("batch_magic" + i_s));
                }
                return ret;
            }();

            auto id = let("batch_id" + std::to_string(problem.batchDims - 1),
                    group_id(2) * local_size(2) + local_id(2));
            for (int i = problem.batchDims - 1; i >= 0; i--) {
                std::string i_s = std::to_string(i);

                auto idx = let("batch_idx" + i_s, [&]() {
                    if (i == 0) return id;
                    auto id_next = let("batch_id" + std::to_string(i - 1),
                            ternary_idiv(id, info[i - 1].size,
                                    info[i - 1].idiv_magic));
                    auto ret = id - info[i - 1].size * id_next;
                    id = id_next;
                    return ret;
                }());
                C_idxs.emplace_back(idx);

                offset_A = offset_A + idx * arg("stride_A" + i_s);
                offset_B = offset_B + idx * arg("stride_B" + i_s);
                offset_C = offset_C + idx * arg("stride_C" + i_s);
            }
        }

        global_tensor_t A_base {arg("A"), into_ir(problem.Ta_ext), offset_A,
                {{m_var, m_idx}, {k_var, k_idx}}, {{m_var, m}, {k_var, k}},
                get_strides(problem.A.layout, A_vars, arg("lda")), {}};
        global_tensor_t B_base {arg("B"), into_ir(problem.Tb_ext), offset_B,
                {{k_var, k_idx}, {n_var, n_idx}}, {{k_var, k}, {n_var, n}},
                get_strides(problem.B.layout, B_vars, arg("ldb")), {}};
        global_tensor_t C_base {arg("C"), into_ir(problem.Tc_ext), offset_B,
                {{m_var, m_idx}, {n_var, n_idx}}, {{m_var, m}, {n_var, n}},
                get_strides(problem.C.layout, C_vars, arg("ldc")),
                {{m_var, m_blk}, {n_var, n_blk}}};

        basic_iterator_t kloop_it(A_base, strategy.ka_prefetch,
                strategy.ka_load, B_base, strategy.kb_prefetch,
                strategy.kb_load, C_base);

        auto store_C = [&]() {
            apply_post_ops(problem.postOps.ops, C, C_idxs, {m_var, n_var});
            store(kloop_it.C_store(), C, {}, {C_store_transform.cache_hint});
        };

        tensor_config_t A_load(
                kloop_it.A_load(), A_load_transform, strategy.A_copies);
        tensor_config_t B_load(
                kloop_it.B_load(), B_load_transform, strategy.B_copies);

        auto prefetchA = strategy.prefetchA ? dnnl::impl::utils::rnd_dn(
                                 strategy.prefetchA, strategy.ka_prefetch)
                                            : 0;
        if (prefetchA != strategy.prefetchA)
            gpu_warning() << "Unimplemented partial A tile prefetch, modifying "
                             "prefetch distance "
                          << strategy.prefetchA << " -> " << prefetchA;
        auto prefetchB = strategy.prefetchB ? dnnl::impl::utils::rnd_dn(
                                 strategy.prefetchB, strategy.kb_prefetch)
                                            : 0;
        if (prefetchB != strategy.prefetchB)
            gpu_warning() << "Unimplemented partial B tile prefetch, modifying "
                             "prefetch distance "
                          << strategy.prefetchB << " -> " << prefetchB;

        k_loop_config_t k_loop_main {k_blk, prefetchA, prefetchB, kloop_it,
                std::move(A_load), std::move(B_load), A_prefetch_transform,
                B_prefetch_transform, C};

        gpu_assert(k_loop_main.A_load_warmup() % kloop_it.A_load().tile[k_var]
                == 0);
        gpu_assert(k_loop_main.B_load_warmup() % kloop_it.B_load().tile[k_var]
                == 0);

        tensor_config_t A_load_short(kloop_it.A_load(), A_load_transform, 1);
        tensor_config_t B_load_short(kloop_it.B_load(), B_load_transform, 1);

        k_loop_config_t k_loop_short {
                (int)lcm(A_load_short.tile[k_var], B_load_short.tile[k_var]), 0,
                0, kloop_it, std::move(A_load_short), std::move(B_load_short),
                A_prefetch_transform, B_prefetch_transform, std::move(C)};
        gpu_assert(k_loop_short.k_warmup() == 0);

        if (problem.A.alignment) {
            assume(arg("lda") % (problem.A.alignment / problem.Ta_ext) == 0);
        }
        if (problem.B.alignment) {
            assume(arg("ldb") % (problem.B.alignment / problem.Tb_ext) == 0);
        }
        if (problem.C.alignment) {
            assume(arg("ldc") % (problem.C.alignment / problem.Tc_ext) == 0);
        }

        _if(kloop_it.is_inbounds(0), [&]() {
            _if(
                    k >= k_loop_main.k_warmup(),
                    [&]() { build_k_loop(k_loop_main); },
                    [&]() { build_k_loop(k_loop_short); });
            store_C();
        });

        return end_kernel();
    }

    struct k_loop_config_t {
        int k_blk;
        int A_prefetch_warmup; // Offset to A prefetch
        int B_prefetch_warmup; // Offset to B prefetch
        basic_iterator_t kloop_it;
        tensor_config_t A_load;
        tensor_config_t B_load;
        transform_t A_prefetch_transform;
        transform_t B_prefetch_transform;
        tensor_t C;

        int A_load_warmup() const {
            return A_load.layout.elems(k_var) - A_load.tile[k_var];
        }
        int B_load_warmup() const {
            return B_load.layout.elems(k_var) - B_load.tile[k_var];
        }
        int k_warmup() const {
            return std::max({A_load_warmup(), B_load_warmup(),
                    A_prefetch_warmup, B_prefetch_warmup});
        }
    };

    void build_k_loop(const k_loop_config_t &cfg) {
        auto k_blk = cfg.k_blk;
        auto kloop_it = cfg.kloop_it;
        auto &C = cfg.C;

        tensor_t A = def("A_blk", cfg.A_load.layout);
        tensor_t B = def("B_blk", cfg.B_load.layout);

        int mma_k_blk
                = std::min(cfg.A_load.tile[k_var], cfg.B_load.tile[k_var]);

        auto pipeline_idx = [&](int loop_idx, int warmup_size, int period) {
            return (loop_idx + warmup_size) % period;
        };

        int A_prefetch_blk
                = cfg.A_prefetch_warmup ? kloop_it.A_prefetch().tile[k_var] : 0;
        auto A_prefetch = [&](int k_unroll_idx) {
            if (cfg.A_prefetch_warmup == 0) return;
            int idx = pipeline_idx(
                    k_unroll_idx, cfg.A_prefetch_warmup, A_prefetch_blk);
            if (idx % A_prefetch_blk != 0) return;
            prefetch(kloop_it.A_prefetch(), {{k_var, 0}},
                    {cfg.A_prefetch_transform.cache_hint});
            kloop_it.A_prefetch_inc(A_prefetch_blk);
        };

        int A_load_blk = cfg.A_load.tile[k_var];
        auto A_load = [&](int k_unroll_idx) {
            int idx = pipeline_idx(k_unroll_idx, cfg.A_load_warmup(),
                    cfg.A_load.layout.elems(k_var));
            if (idx % A_load_blk != 0) return;
            load(A.sub(cfg.A_load.tile, {{k_var, idx}}), kloop_it.A_load(),
                    {{k_var, 0}}, {cfg.A_load.transform.cache_hint});
            kloop_it.A_load_inc(A_load_blk);
        };

        int B_prefetch_blk
                = cfg.B_prefetch_warmup ? kloop_it.B_prefetch().tile[k_var] : 0;
        auto B_prefetch = [&](int k_unroll_idx) {
            if (cfg.B_prefetch_warmup == 0) return;
            int idx = pipeline_idx(
                    k_unroll_idx, cfg.B_prefetch_warmup, B_prefetch_blk);
            if (idx % B_prefetch_blk != 0) return;
            prefetch(kloop_it.B_prefetch(), {{k_var, 0}},
                    {cfg.B_prefetch_transform.cache_hint});
            kloop_it.B_prefetch_inc(B_prefetch_blk);
        };

        int B_load_blk = cfg.B_load.tile[k_var];
        auto B_load = [&](int k_unroll_idx) {
            int idx = pipeline_idx(k_unroll_idx, cfg.B_load_warmup(),
                    cfg.B_load.layout.elems(k_var));
            if (idx % B_load_blk != 0) return;
            load(B.sub(cfg.B_load.tile, {{k_var, idx}}), kloop_it.B_load(),
                    {{k_var, 0}}, {cfg.B_load.transform.cache_hint});
            kloop_it.B_load_inc(B_load_blk);
        };

        int k_unroll_blk = [&]() {
            int ret = k_blk;
            for (auto v :
                    {A_prefetch_blk, A_load_blk, B_prefetch_blk, B_load_blk}) {
                ret = gcd(ret, v);
            }
            return ret;
        }();

        auto k_body = [&](int k_offset, bool do_A_prefetch, bool do_B_prefetch,
                              bool do_A_load, bool do_B_load, bool do_mma) {
            if (do_A_prefetch) { A_prefetch(k_offset); }

            if (do_B_prefetch) { B_prefetch(k_offset); }

            if (do_A_load) { A_load(k_offset); }

            if (do_B_load) { B_load(k_offset); }

            if (do_mma) {
                if (k_offset % mma_k_blk == 0) {
                    ir::tile_t tile = C.layout.tile();
                    tile[k_var] = mma_k_blk;
                    mma(C, A, B, tile, {{k_var, k_offset}}, strategy.systolic);
                }
            }
        };

        // Pipeline controls
        auto warmup = cfg.k_warmup();

        for (int k_unroll_idx = -warmup; k_unroll_idx < 0;
                k_unroll_idx += k_unroll_blk) {
            bool A_prefetch = k_unroll_idx + cfg.A_prefetch_warmup >= 0;
            bool B_prefetch = k_unroll_idx + cfg.B_prefetch_warmup >= 0;
            bool A_load = k_unroll_idx + cfg.A_load_warmup() >= 0;
            bool B_load = k_unroll_idx + cfg.B_load_warmup() >= 0;
            bool do_mma = false;
            k_body(k_unroll_idx, A_prefetch, B_prefetch, A_load, B_load,
                    do_mma);
        }

        _while(kloop_it.is_inbounds(warmup), [&]() {
            for (int k_unroll_idx = 0; k_unroll_idx < k_blk;
                    k_unroll_idx += k_unroll_blk) {
                k_body(k_unroll_idx, cfg.A_prefetch_warmup,
                        cfg.B_prefetch_warmup, true, true, true);
            }
            kloop_it.kloop_inc(k_blk);
        });

        auto tail_end = dnnl::impl::utils::rnd_up(warmup, k_blk);
        for (int k_unroll_idx = 0; k_unroll_idx < tail_end;
                k_unroll_idx += k_unroll_blk) {
            bool A_prefetch = k_unroll_idx + cfg.A_prefetch_warmup < tail_end;
            bool B_prefetch = k_unroll_idx + cfg.B_prefetch_warmup < tail_end;
            bool A_load = k_unroll_idx + cfg.A_load_warmup() < tail_end;
            bool B_load = k_unroll_idx + cfg.B_load_warmup() < tail_end;
            k_body(k_unroll_idx, A_prefetch, B_prefetch, A_load, B_load, true);
        }
    }

    const GEMMProblem &problem;
    const GEMMStrategy &strategy;
};

kernel_t make_kernel(
        const generator_dsl_desc_t &desc, ir::constraint_set_t cset) {
    ir::ir_context_t ctx(desc.options, cset);

    ir::trace_start();
    auto k = generator_dsl_t(desc).build(desc.kernel_iface(), ctx);
    ir::trace_pass("build generator_dsl_t", k.body, ctx);

    k.body = ir::simplify(k.body, ctx);
    k.body = ir::inject_send(k.body, ctx);

    // TODO: This should be unnecessary as it could happen at codegen
    k.body = ir::fixup_if_conditions(k.body, ctx);
    k.body = ir::eliminate_common_subexprs(
            k.body, ctx, desc.strategy.GRFs * ctx.hw().grf_size());
    return k;
}

GEMMSTONE_NAMESPACE_END

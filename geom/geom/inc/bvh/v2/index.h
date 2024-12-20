#ifndef BVH_V2_INDEX_H
#define BVH_V2_INDEX_H

#include "bvh/v2/utils.h"

#include <cassert>
#include <cstddef>

namespace bvh::v2 {

/// Packed index data structure. This index can either refer to a range of primitives for a BVH
/// leaf, or to the children of a BVH node. In either case, the index corresponds to a contiguous
/// range, which means that:
///
/// - For leaves, primitives in a BVH node should be accessed via:
///
///     size_t begin = index.first_id();
///     size_t end   = begin + index.prim_count();
///     for (size_t i = begin; i < end; ++i) {
///         size_t prim_id = bvh.prim_ids[i];
///         // ...
///     }
///
///   Note that for efficiency, reordering the original data to avoid the indirection via
///   `bvh.prim_ids` is preferable.
///
/// - For inner nodes, children should be accessed via:
///
///   auto& left_child = bvh.nodes[index.first_id()];
///   auto& right_child = bvh.nodes[index.first_id() + 1];
///
template <size_t Bits, size_t PrimCountBits>
struct Index {
    using Type = UnsignedIntType<Bits>;

    static constexpr size_t bits = Bits;
    static constexpr size_t prim_count_bits = PrimCountBits;
    static constexpr Type max_prim_count = make_bitmask<Type>(prim_count_bits);
    static constexpr Type max_first_id   = make_bitmask<Type>(bits - prim_count_bits);

    static_assert(PrimCountBits < Bits);

    Type value;

    Index() = default;
    explicit Index(Type value) : value(value) {}

    //bool operator == (const Index&) const = default;
    //bool operator != (const Index&) const = default;
    bool operator == (const Index& other) const {
        return other.value == value;
    }
    bool operator != (const Index& other) const {
        return other.value != value;
    }

    BVH_ALWAYS_INLINE Type first_id() const { return value >> prim_count_bits; }
    BVH_ALWAYS_INLINE Type prim_count() const { return value & max_prim_count; }
    BVH_ALWAYS_INLINE bool is_leaf() const { return prim_count() != 0; }
    BVH_ALWAYS_INLINE bool is_inner() const { return !is_leaf(); }

    BVH_ALWAYS_INLINE void set_first_id(size_t first_id) {
        *this = Index { first_id, prim_count() };
    }

    BVH_ALWAYS_INLINE void set_prim_count(size_t prim_count) {
        *this = Index { first_id(), prim_count };
    }

    static BVH_ALWAYS_INLINE Index make_leaf(size_t first_prim, size_t prim_count) {
        assert(prim_count != 0);
        return Index { first_prim, prim_count };
    }

    static BVH_ALWAYS_INLINE Index make_inner(size_t first_child) {
        return Index { first_child, 0 };
    }

private:
    explicit Index(size_t first_id, size_t prim_count)
        : Index(
            (static_cast<Type>(first_id) << prim_count_bits) |
            (static_cast<Type>(prim_count) & max_prim_count))
    {
        assert(first_id   <= static_cast<size_t>(max_first_id));
        assert(prim_count <= static_cast<size_t>(max_prim_count));
    }
};

} // namespace bvh::v2

#endif

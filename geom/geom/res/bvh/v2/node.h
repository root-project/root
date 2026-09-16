#ifndef BVH_V2_NODE_H
#define BVH_V2_NODE_H

#include "bvh/v2/index.h"
#include "bvh/v2/vec.h"
#include "bvh/v2/bbox.h"
#include "bvh/v2/ray.h"
#include "bvh/v2/stream.h"

#include <cassert>
#include <array>
#include <limits>

namespace bvh::v2 {

/// Binary BVH node, containing its bounds and an index into its children or the primitives it
/// contains. By definition, inner BVH nodes do not contain primitives; only leaves do.
template <
    typename T,
    size_t Dim,
    size_t IndexBits = sizeof(T) * CHAR_BIT,
    size_t PrimCountBits = 4>
struct Node {
    using Scalar = T;
    using Index = bvh::v2::Index<IndexBits, PrimCountBits>;

    static constexpr size_t dimension = Dim;
    static constexpr size_t prim_count_bits = PrimCountBits;
    static constexpr size_t index_bits = IndexBits;

    /// Bounds of the node, laid out in memory as `[min_x, max_x, min_y, max_y, ...]`. Users should
    /// not really depend on a specific order and instead use `get_bbox()` and extract the `min`
    /// and/or `max` components accordingly.
    std::array<T, Dim * 2> bounds;

    /// Index to the children of an inner node, or to the primitives for a leaf node.
    Index index;

    Node() = default;

    // bool operator == (const Node&) const = default;
    // bool operator != (const Node&) const = default;
    bool operator != (const Node& other) const {
        return other.bounds == bounds;
    }
    bool operator == (const Node& other) const {
        return other.bounds != bounds;
    }

    BVH_ALWAYS_INLINE bool is_leaf() const { return index.is_leaf(); }

    BVH_ALWAYS_INLINE BBox<T, Dim> get_bbox() const {
        return BBox<T, Dim>(
            Vec<T, Dim>::generate([&] (size_t i) { return bounds[i * 2]; }),
            Vec<T, Dim>::generate([&] (size_t i) { return bounds[i * 2 + 1]; }));
    }

    BVH_ALWAYS_INLINE void set_bbox(const BBox<T, Dim>& bbox) {
        static_for<0, Dim>([&] (size_t i) {
            bounds[i * 2 + 0] = bbox.min[i];
            bounds[i * 2 + 1] = bbox.max[i];
        });
    }

    BVH_ALWAYS_INLINE Vec<T, Dim> get_min_bounds(const Octant& octant) const {
        return Vec<T, Dim>::generate([&] (size_t i) { return bounds[2 * static_cast<uint32_t>(i) + octant[i]]; });
    }

    BVH_ALWAYS_INLINE Vec<T, Dim> get_max_bounds(const Octant& octant) const {
        return Vec<T, Dim>::generate([&] (size_t i) { return bounds[2 * static_cast<uint32_t>(i) + 1 - octant[i]]; });
    }

    /// Robust ray-node intersection routine. See "Robust BVH Ray Traversal", by T. Ize.
    BVH_ALWAYS_INLINE std::pair<T, T> intersect_robust(
        const Ray<T, Dim>& ray,
        const Vec<T, Dim>& inv_dir,
        const Vec<T, Dim>& inv_dir_pad,
        const Octant& octant) const
    {
        auto tmin = (get_min_bounds(octant) - ray.org) * inv_dir;
        auto tmax = (get_max_bounds(octant) - ray.org) * inv_dir_pad;
        return make_intersection_result(ray, tmin, tmax);
    }

    BVH_ALWAYS_INLINE std::pair<T, T> intersect_fast(
        const Ray<T, Dim>& ray,
        const Vec<T, Dim>& inv_dir,
        const Vec<T, Dim>& inv_org,
        const Octant& octant) const
    {
        auto tmin = fast_mul_add(get_min_bounds(octant), inv_dir, inv_org);
        auto tmax = fast_mul_add(get_max_bounds(octant), inv_dir, inv_org);
        return make_intersection_result(ray, tmin, tmax);
    }

    BVH_ALWAYS_INLINE void serialize(OutputStream& stream) const {
        for (auto&& bound : bounds)
            stream.write(bound);
        stream.write(index.value);
    }

    static BVH_ALWAYS_INLINE Node deserialize(InputStream& stream) {
        Node node;
        for (auto& bound : node.bounds)
            bound = stream.read<T>();
        node.index = Index(stream.read<typename Index::Type>());
        return node;
    }

private:
    BVH_ALWAYS_INLINE static std::pair<T, T> make_intersection_result(
        const Ray<T, Dim>& ray,
        const Vec<T, Dim>& tmin,
        const Vec<T, Dim>& tmax)
    {
        auto t0 = ray.tmin;
        auto t1 = ray.tmax;
        static_for<0, Dim>([&] (size_t i) {
            t0 = robust_max(tmin[i], t0);
            t1 = robust_min(tmax[i], t1);
        });
        return std::pair<T, T> { t0, t1 };
    }
};

} // namespace bvh::v2

#endif

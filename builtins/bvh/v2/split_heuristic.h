#ifndef BVH_V2_SPLIT_HEURISTIC_H
#define BVH_V2_SPLIT_HEURISTIC_H

#include "bvh/v2/bbox.h"
#include "bvh/v2/utils.h"

#include <cstddef>

namespace bvh::v2 {

template <typename T>
class SplitHeuristic {
public:
    /// Creates an SAH evaluator object, used by top-down builders to determine where to split.
    /// The two parameters are the log of the size of primitive clusters in base 2, and the ratio of
    /// the cost of intersecting a node (a ray-box intersection) over the cost of intersecting a
    /// primitive.
    BVH_ALWAYS_INLINE SplitHeuristic(
        size_t log_cluster_size = 0,
        T cost_ratio = static_cast<T>(1.))
        : log_cluster_size_(log_cluster_size)
        , prim_offset_(make_bitmask<size_t>(log_cluster_size))
        , cost_ratio_(cost_ratio)
    {}

    BVH_ALWAYS_INLINE size_t get_prim_count(size_t size) const {
        return (size + prim_offset_) >> log_cluster_size_;
    }

    template <size_t N>
    BVH_ALWAYS_INLINE T get_leaf_cost(size_t begin, size_t end, const BBox<T, N>& bbox) const {
        return bbox.get_half_area() * static_cast<T>(get_prim_count(end - begin));
    }

    template <size_t N>
    BVH_ALWAYS_INLINE T get_non_split_cost(size_t begin, size_t end, const BBox<T, N>& bbox) const {
        return bbox.get_half_area() * (static_cast<T>(get_prim_count(end - begin)) - cost_ratio_);
    }

private:
    size_t log_cluster_size_;
    size_t prim_offset_;
    T cost_ratio_;
};

} // namespace bvh::v2

#endif

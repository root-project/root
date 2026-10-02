#ifndef BVH_V2_SWEEP_SAH_BUILDER_H
#define BVH_V2_SWEEP_SAH_BUILDER_H

#include "bvh/v2/top_down_sah_builder.h"

#include <stack>
#include <tuple>
#include <algorithm>
#include <optional>
#include <numeric>
#include <cassert>

namespace bvh::v2 {

/// Single-threaded top-down builder that partitions primitives based on the Surface
/// Area Heuristic (SAH). Primitives are only sorted once along each axis.
template <typename Node>
class SweepSahBuilder : public TopDownSahBuilder<Node> {
    using typename TopDownSahBuilder<Node>::Scalar;
    using typename TopDownSahBuilder<Node>::Vec;
    using typename TopDownSahBuilder<Node>::BBox;

    using TopDownSahBuilder<Node>::build;
    using TopDownSahBuilder<Node>::config_;
    using TopDownSahBuilder<Node>::bboxes_;

public:
    using typename TopDownSahBuilder<Node>::Config;

    BVH_ALWAYS_INLINE static Bvh<Node> build(
        std::span<const BBox> bboxes,
        std::span<const Vec> centers,
        const Config& config = {})
    {
        return SweepSahBuilder(bboxes, centers, config).build();
    }

protected:
    struct Split {
        size_t pos;
        Scalar cost;
        size_t axis;
    };

    std::vector<bool> marks_;
    std::vector<Scalar> accum_;
    std::vector<size_t> prim_ids_[Node::dimension];

    BVH_ALWAYS_INLINE SweepSahBuilder(
        std::span<const BBox> bboxes,
        std::span<const Vec> centers,
        const Config& config)
        : TopDownSahBuilder<Node>(bboxes, centers, config)
    {
        marks_.resize(bboxes.size());
        accum_.resize(bboxes.size());
        for (size_t axis = 0; axis < Node::dimension; ++axis) {
            prim_ids_[axis].resize(bboxes.size());
            std::iota(prim_ids_[axis].begin(), prim_ids_[axis].end(), 0);
            std::sort(prim_ids_[axis].begin(), prim_ids_[axis].end(), [&] (size_t i, size_t j) {
                return centers[i][axis] < centers[j][axis];
            });
        }
    }

    std::vector<size_t>& get_prim_ids() override { return prim_ids_[0]; }

    void find_best_split(size_t axis, size_t begin, size_t end, Split& best_split) {
        size_t first_right = begin;

        // Sweep from the right to the left, computing the partial SAH cost
        auto right_bbox = BBox::make_empty();
        for (size_t i = end - 1; i > begin;) {
            static constexpr size_t chunk_size = 32;
            size_t next = i - std::min(i - begin, chunk_size);
            auto right_cost = static_cast<Scalar>(0.);
            for (; i > next; --i) {
                right_bbox.extend(bboxes_[prim_ids_[axis][i]]);
                accum_[i] = right_cost = config_.sah.get_leaf_cost(i, end, right_bbox);
            }
            // Every `chunk_size` elements, check that we are not above the maximum cost
            if (right_cost > best_split.cost) {
                first_right = i;
                break;
            }
        }

        // Sweep from the left to the right, computing the full cost
        auto left_bbox = BBox::make_empty();
        for (size_t i = begin; i < first_right; ++i)
            left_bbox.extend(bboxes_[prim_ids_[axis][i]]);
        for (size_t i = first_right; i < end - 1; ++i) {
            left_bbox.extend(bboxes_[prim_ids_[axis][i]]);
            auto left_cost = config_.sah.get_leaf_cost(begin, i + 1, left_bbox);
            auto cost = left_cost + accum_[i + 1];
            if (cost < best_split.cost)
                best_split = Split { i + 1, cost, axis };
            else if (left_cost > best_split.cost)
                break;
        }
    }

    BVH_ALWAYS_INLINE void mark_primitives(size_t axis, size_t begin, size_t split_pos, size_t end) {
        for (size_t i = begin; i < split_pos; ++i) marks_[prim_ids_[axis][i]] = true;
        for (size_t i = split_pos; i < end; ++i)   marks_[prim_ids_[axis][i]] = false;
    }

    std::optional<size_t> try_split(const BBox& bbox, size_t begin, size_t end) override {
        // Find the best split over all axes
        auto leaf_cost = config_.sah.get_non_split_cost(begin, end, bbox);
        auto best_split = Split { (begin + end + 1) / 2, leaf_cost, 0 };
        for (size_t axis = 0; axis < Node::dimension; ++axis)
            find_best_split(axis, begin, end, best_split);

        // Make sure that the split is good before proceeding with it
        if (best_split.cost >= leaf_cost) {
            if (end - begin <= config_.max_leaf_size)
                return std::nullopt;

            // If the number of primitives is too high, fallback on a split at the
            // median on the largest axis.
            best_split.pos = (begin + end + 1) / 2;
            best_split.axis = bbox.get_diagonal().get_largest_axis();
        }

        // Partition primitives (keeping the order intact so that the next recursive calls do not
        // need to sort primitives again).
        mark_primitives(best_split.axis, begin, best_split.pos, end);
        for (size_t axis = 0; axis < Node::dimension; ++axis) {
            if (axis == best_split.axis)
                continue;
            std::stable_partition(
                prim_ids_[axis].begin() + begin,
                prim_ids_[axis].begin() + end,
                [&] (size_t i) { return marks_[i]; });
        }

        return std::make_optional(best_split.pos);
    }
};

} // namespace bvh::v2

#endif

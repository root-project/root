#ifndef BVH_V2_BINNED_SAH_BUILDER_H
#define BVH_V2_BINNED_SAH_BUILDER_H

#include "bvh/v2/top_down_sah_builder.h"

#include <stack>
#include <tuple>
#include <algorithm>
#include <optional>
#include <numeric>
#include <cassert>

namespace bvh::v2 {

/// Single-threaded top-down builder that partitions primitives based on a binned approximation of
/// the Surface Area Heuristic (SAH). This builder is inspired by
/// "On Fast Construction of SAH-based Bounding Volume Hierarchies", by I. Wald.
template <typename Node, size_t BinCount = 8>
class BinnedSahBuilder : public TopDownSahBuilder<Node> {
    using typename TopDownSahBuilder<Node>::Scalar;
    using typename TopDownSahBuilder<Node>::Vec;
    using typename TopDownSahBuilder<Node>::BBox;

    using TopDownSahBuilder<Node>::build;
    using TopDownSahBuilder<Node>::config_;
    using TopDownSahBuilder<Node>::bboxes_;
    using TopDownSahBuilder<Node>::centers_;

public:
    using typename TopDownSahBuilder<Node>::Config;

    BVH_ALWAYS_INLINE static Bvh<Node> build(
        std::span<const BBox> bboxes,
        std::span<const Vec> centers,
        const Config& config = {})
    {
        return BinnedSahBuilder(bboxes, centers, config).build();
    }

protected:
    struct Split {
        size_t bin_id;
        Scalar cost;
        size_t axis;
    };

    struct Bin {
        BBox bbox = BBox::make_empty();
        size_t prim_count = 0;

        Bin() = default;

        BVH_ALWAYS_INLINE Scalar get_cost(const SplitHeuristic<Scalar>& sah) const {
            return sah.get_leaf_cost(0, prim_count, bbox);
        }

        BVH_ALWAYS_INLINE void add(const BBox& bbox, size_t prim_count = 1) {
            this->bbox.extend(bbox);
            this->prim_count += prim_count;
        }

        BVH_ALWAYS_INLINE void add(const Bin& bin) { add(bin.bbox, bin.prim_count); }
    };

    using Bins = std::array<Bin, BinCount>;
    using PerAxisBins = std::array<Bins, Node::dimension>;

    std::vector<size_t> prim_ids_;

    BVH_ALWAYS_INLINE BinnedSahBuilder(
        std::span<const BBox> bboxes,
        std::span<const Vec> centers,
        const Config& config)
        : TopDownSahBuilder<Node>(bboxes, centers, config)
        , prim_ids_(bboxes.size())
    {
        std::iota(prim_ids_.begin(), prim_ids_.end(), 0);
    }

    std::vector<size_t>& get_prim_ids() override { return prim_ids_; }

    BVH_ALWAYS_INLINE void fill_bins(
        PerAxisBins& per_axis_bins,
        const BBox& bbox,
        size_t begin,
        size_t end)
    {
        auto bin_scale = Vec(BinCount) / bbox.get_diagonal();
        auto bin_offset = -bbox.min * bin_scale;

        for (size_t i = begin; i < end; ++i) {
            auto pos = fast_mul_add(centers_[prim_ids_[i]], bin_scale, bin_offset);
            static_for<0, Node::dimension>([&] (size_t axis) {
                size_t index = std::min(BinCount - 1,
                    static_cast<size_t>(robust_max(pos[axis], static_cast<Scalar>(0.))));
                per_axis_bins[axis][index].add(bboxes_[prim_ids_[i]]);
            });
        }
    }

    void find_best_split(size_t axis, const Bins& bins, Split& best_split) {
        Bin right_accum;
        std::array<Scalar, BinCount> right_costs;
        for (size_t i = BinCount - 1; i > 0; --i) {
            right_accum.add(bins[i]);
            right_costs[i] = right_accum.get_cost(config_.sah);
        }

        Bin left_accum;
        for (size_t i = 0; i < BinCount - 1; ++i) {
            left_accum.add(bins[i]);
            auto cost = left_accum.get_cost(config_.sah) + right_costs[i + 1];
            if (cost < best_split.cost)
                best_split = Split { i + 1, cost, axis };
        }
    }

    size_t fallback_split(size_t axis, size_t begin, size_t end) {
        size_t mid = (begin + end + 1) / 2;
        std::partial_sort(
            prim_ids_.begin() + begin,
            prim_ids_.begin() + mid,
            prim_ids_.begin() + end,
            [&] (size_t i, size_t j) { return centers_[i][axis] < centers_[j][axis]; });
        return mid;
    }

    std::optional<size_t> try_split(const BBox& bbox, size_t begin, size_t end) override {
        PerAxisBins per_axis_bins;
        fill_bins(per_axis_bins, bbox, begin, end);

        auto largest_axis = bbox.get_diagonal().get_largest_axis();
        auto best_split = Split { BinCount / 2, std::numeric_limits<Scalar>::max(), largest_axis };
        for (size_t axis = 0; axis < Node::dimension; ++axis)
            find_best_split(axis, per_axis_bins[axis], best_split);

        // Make sure that the split is good before proceeding with it
        auto leaf_cost = config_.sah.get_non_split_cost(begin, end, bbox);
        if (best_split.cost >= leaf_cost) {
            if (end - begin <= config_.max_leaf_size)
                return std::nullopt;
            return fallback_split(largest_axis, begin, end);
        }

        auto split_pos = fast_mul_add(
            bbox.get_diagonal()[best_split.axis] / static_cast<Scalar>(BinCount),
            static_cast<Scalar>(best_split.bin_id),
            bbox.min[best_split.axis]);

        size_t index = std::partition(prim_ids_.begin() + begin, prim_ids_.begin() + end,
            [&] (size_t i) { return centers_[i][best_split.axis] < split_pos; }) - prim_ids_.begin();
        if (index == begin || index == end)
            return fallback_split(largest_axis, begin, end);

        return std::make_optional(index);
    }
};

} // namespace bvh::v2

#endif

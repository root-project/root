#ifndef BVH_V2_TOP_DOWN_SAH_BUILDER_H
#define BVH_V2_TOP_DOWN_SAH_BUILDER_H

#include "bvh/v2/bvh.h"
#include "bvh/v2/vec.h"
#include "bvh/v2/bbox.h"
#include "bvh/v2/split_heuristic.h"
#include <stack>
#if __has_include(<span>)
#include <span>
#else
// Falling back to ROOT span
#include "ROOT/span.hxx"
#endif
#include <algorithm>
#include <optional>
#include <numeric>
#include <cassert>

namespace bvh::v2 {

/// Base class for all SAH-based, top-down builders.
template <typename Node>
class TopDownSahBuilder {
protected:
    using Scalar = typename Node::Scalar;
    using Vec  = bvh::v2::Vec<Scalar, Node::dimension>;
    using BBox = bvh::v2::BBox<Scalar, Node::dimension>;

public:
    struct Config {
        /// SAH heuristic parameters that control how primitives are partitioned.
        SplitHeuristic<Scalar> sah;

        /// Nodes containing less than this amount of primitives will not be split.
        /// This is mostly to speed up BVH construction, and using large values may lead to lower
        /// quality BVHs.
        size_t min_leaf_size = 1;

        /// Nodes that cannot be split based on the SAH and have a number of primitives larger than
        /// this will be split using a fallback strategy. This should not happen often, but may
        /// happen in worst-case scenarios or poorly designed scenes.
        size_t max_leaf_size = 8;
    };

protected:
    struct WorkItem {
        size_t node_id;
        size_t begin;
        size_t end;

        BVH_ALWAYS_INLINE size_t size() const { return end - begin; }
    };

    std::span<const BBox> bboxes_;
    std::span<const Vec> centers_;
    const Config& config_;

    BVH_ALWAYS_INLINE TopDownSahBuilder(
        std::span<const BBox> bboxes,
        std::span<const Vec> centers,
        const Config& config)
        : bboxes_(bboxes)
        , centers_(centers)
        , config_(config)
    {
        assert(bboxes.size() == centers.size());
        assert(config.min_leaf_size <= config.max_leaf_size);
    }

    virtual std::vector<size_t>& get_prim_ids() = 0;
    virtual std::optional<size_t> try_split(const BBox& bbox, size_t begin, size_t end) = 0;

    BVH_ALWAYS_INLINE const std::vector<size_t>& get_prim_ids() const {
        return const_cast<TopDownSahBuilder*>(this)->get_prim_ids();
    }

    Bvh<Node> build() {
        const auto prim_count = bboxes_.size();

        Bvh<Node> bvh;
        bvh.nodes.reserve((2 * prim_count) / config_.min_leaf_size);
        bvh.nodes.emplace_back();
        bvh.nodes.back().set_bbox(compute_bbox(0, prim_count));

        std::stack<WorkItem> stack;
        stack.push(WorkItem { 0, 0, prim_count });
        while (!stack.empty()) {
            auto item = stack.top();
            stack.pop();

            auto& node = bvh.nodes[item.node_id];
            if (item.size() > config_.min_leaf_size) {
                if (auto split_pos = try_split(node.get_bbox(), item.begin, item.end)) {
                    auto first_child = bvh.nodes.size();
                    node.index = Node::Index::make_inner(first_child);

                    bvh.nodes.resize(first_child + 2);

                    auto first_bbox   = compute_bbox(item.begin, *split_pos);
                    auto second_bbox  = compute_bbox(*split_pos, item.end);
                    auto first_range  = std::make_pair(item.begin, *split_pos);
                    auto second_range = std::make_pair(*split_pos, item.end);

                    // For "any-hit" queries, the left child is chosen first, so we make sure that
                    // it is the child with the largest area, as it is more likely to contain an
                    // an occluder. See "SATO: Surface Area Traversal Order for Shadow Ray Tracing",
                    // by J. Nah and D. Manocha.
                    if (first_bbox.get_half_area() < second_bbox.get_half_area()) {
                        std::swap(first_bbox, second_bbox);
                        std::swap(first_range, second_range);
                    }

                    auto first_item  = WorkItem { first_child + 0, first_range.first, first_range.second };
                    auto second_item = WorkItem { first_child + 1, second_range.first, second_range.second };
                    bvh.nodes[first_child + 0].set_bbox(first_bbox);
                    bvh.nodes[first_child + 1].set_bbox(second_bbox);

                    // Process the largest child item first, in order to minimize the stack size.
                    if (first_item.size() < second_item.size())
                        std::swap(first_item, second_item);

                    stack.push(first_item);
                    stack.push(second_item);
                    continue;
                }
            }

            node.index = Node::Index::make_leaf(item.begin, item.size());
        }

        bvh.prim_ids = std::move(get_prim_ids());
        bvh.nodes.shrink_to_fit();
        return bvh;
    }

    BVH_ALWAYS_INLINE BBox compute_bbox(size_t begin, size_t end) const {
        const auto& prim_ids = get_prim_ids();
        auto bbox = BBox::make_empty();
        for (size_t i = begin; i < end; ++i)
            bbox.extend(bboxes_[prim_ids[i]]);
        return bbox;
    }
};

} // namespace bvh::v2

#endif

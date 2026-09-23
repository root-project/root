#ifndef BVH_V2_REINSERTION_OPTIMIZER_H
#define BVH_V2_REINSERTION_OPTIMIZER_H

#include "bvh/v2/bvh.h"
#include "bvh/v2/thread_pool.h"
#include "bvh/v2/executor.h"

#include <vector>
#include <algorithm>

namespace bvh::v2 {

template <typename Node>
class ReinsertionOptimizer {
    using Scalar = typename Node::Scalar;
    using BBox = bvh::v2::BBox<Scalar, Node::dimension>;

public:
    struct Config {
        /// Fraction of the number of nodes to optimize per iteration.
        Scalar batch_size_ratio = static_cast<Scalar>(0.05);

        /// Maximum number of iterations.
        size_t max_iter_count = 3;
    };

    static void optimize(ThreadPool& thread_pool, Bvh<Node>& bvh, const Config& config = {}) {
        ParallelExecutor executor(thread_pool);
        optimize(executor, bvh, config);
    }

    static void optimize(Bvh<Node>& bvh, const Config& config = {}) {
        SequentialExecutor executor;
        optimize(executor, bvh, config);
    }

private:
    struct Candidate {
        size_t node_id = 0;
        Scalar cost = -std::numeric_limits<Scalar>::max();

        BVH_ALWAYS_INLINE bool operator > (const Candidate& other) const {
            return cost > other.cost;
        }
    };

    struct Reinsertion {
        size_t from = 0;
        size_t to = 0;
        Scalar area_diff = static_cast<Scalar>(0);

        BVH_ALWAYS_INLINE bool operator > (const Reinsertion& other) const {
            return area_diff > other.area_diff;
        }
    };

    Bvh<Node>& bvh_;
    std::vector<size_t> parents_;

    ReinsertionOptimizer(Bvh<Node>& bvh, std::vector<size_t>&& parents)
        : bvh_(bvh)
        , parents_(std::move(parents))
    {}

    template <typename Derived>
    static void optimize(Executor<Derived>& executor, Bvh<Node>& bvh, const Config& config) {
        auto parents = compute_parents(executor, bvh);
        ReinsertionOptimizer<Node>(bvh, std::move(parents)).optimize(executor, config);
    }

    template <typename Derived>
    static std::vector<size_t> compute_parents(Executor<Derived>& executor, const Bvh<Node>& bvh) {
        std::vector<size_t> parents(bvh.nodes.size());
        parents[0] = 0;
        executor.for_each(0, bvh.nodes.size(),
            [&] (size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    auto& node = bvh.nodes[i];
                    if (!node.is_leaf()) {
                        parents[node.index.first_id() + 0] = i;
                        parents[node.index.first_id() + 1] = i;
                    }
                }
            });
        return parents;
    }

    BVH_ALWAYS_INLINE std::vector<Candidate> find_candidates(size_t target_count) {
        // Gather the `target_count` nodes that have the highest cost.
        // Note that this may produce fewer nodes if the BVH has fewer than `target_count` nodes.
        const auto node_count = std::min(bvh_.nodes.size(), target_count + 1);
        std::vector<Candidate> candidates;
        for (size_t i = 1; i < node_count; ++i)
            candidates.push_back(Candidate { i, bvh_.nodes[i].get_bbox().get_half_area() });
        std::make_heap(candidates.begin(), candidates.end(), std::greater<>{});
        for (size_t i = node_count; i < bvh_.nodes.size(); ++i) {
            auto cost = bvh_.nodes[i].get_bbox().get_half_area();
            if (candidates.front().cost < cost) {
                std::pop_heap(candidates.begin(), candidates.end(), std::greater<>{});
                candidates.back() = Candidate { i, cost };
                std::push_heap(candidates.begin(), candidates.end(), std::greater<>{});
            }
        }
        return candidates;
    }

    Reinsertion find_reinsertion(size_t node_id) {
        assert(node_id != 0);

        /*
         * Here is an example that explains how the cost of a reinsertion is computed. For the
         * reinsertion from A to C, in the figure below, we need to remove P1, replace it by B,
         * and create a node that holds A and C and place it where C was.
         *
         *             R
         *            / \
         *          Pn   Q1
         *          /     \
         *        ...     ...
         *        /         \
         *       P1          C
         *      / \
         *     A   B
         *
         * The resulting area *decrease* is (SA(x) means the surface area of x):
         *
         *     SA(P1) +                                                : P1 was removed
         *     SA(P2) - SA(B) +                                        : P2 now only contains B
         *     SA(P3) - SA(B U sibling(P2)) +                          : Same but for P3
         *     ... +
         *     SA(Pn) - SA(B U sibling(P2) U ... U sibling(P(n - 1)) + : Same but for Pn
         *     0 +                                                     : R does not change
         *     SA(Q1) - SA(Q1 U A) +                                   : Q1 now contains A
         *     SA(Q2) - SA(Q2 U A) +                                   : Q2 now contains A
         *     ... +
         *     -SA(A U C)                                              : For the parent of A and C
         */

        Reinsertion best_reinsertion { .from = node_id };
        auto node_area   = bvh_.nodes[node_id].get_bbox().get_half_area();
        auto parent_area = bvh_.nodes[parents_[node_id]].get_bbox().get_half_area();
        auto area_diff = parent_area;
        auto sibling_id = Bvh<Node>::get_sibling_id(node_id);
        auto pivot_bbox = bvh_.nodes[sibling_id].get_bbox();
        auto parent_id = parents_[node_id];
        auto pivot_id = parent_id;

        std::vector<std::pair<Scalar, size_t>> stack;
        do {
            // Try to find a reinsertion in the sibling at the current level of the tree
            stack.emplace_back(area_diff, sibling_id);
            while (!stack.empty()) {
                auto top = stack.back();
                stack.pop_back();
                if (top.first - node_area <= best_reinsertion.area_diff)
                    continue;

                auto& dst_node = bvh_.nodes[top.second];
                auto merged_area = dst_node.get_bbox().extend(bvh_.nodes[node_id].get_bbox()).get_half_area();
                auto reinsert_area = top.first - merged_area;
                if (reinsert_area > best_reinsertion.area_diff) {
                    best_reinsertion.to = top.second;
                    best_reinsertion.area_diff = reinsert_area;
                }

                if (!dst_node.is_leaf()) {
                    auto child_area = reinsert_area + dst_node.get_bbox().get_half_area();
                    stack.emplace_back(child_area, dst_node.index.first_id() + 0);
                    stack.emplace_back(child_area, dst_node.index.first_id() + 1);
                }
            }

            // Compute the bounding box on the path from the node to the root, and record the
            // corresponding decrease in area.
            if (pivot_id != parent_id) {
                pivot_bbox.extend(bvh_.nodes[sibling_id].get_bbox());
                area_diff += bvh_.nodes[pivot_id].get_bbox().get_half_area() - pivot_bbox.get_half_area();
            }

            sibling_id = Bvh<Node>::get_sibling_id(pivot_id);
            pivot_id = parents_[pivot_id];
        } while (pivot_id != 0);

        if (best_reinsertion.to == Bvh<Node>::get_sibling_id(best_reinsertion.from) ||
            best_reinsertion.to == parents_[best_reinsertion.from])
            best_reinsertion = {};
        return best_reinsertion;
    }

    BVH_ALWAYS_INLINE void reinsert_node(size_t from, size_t to) {
        auto sibling_id = Bvh<Node>::get_sibling_id(from);
        auto parent_id  = parents_[from];
        auto sibling_node = bvh_.nodes[sibling_id];
        auto dst_node     = bvh_.nodes[to];

        bvh_.nodes[to].index = Node::Index::make_inner(Bvh<Node>::get_left_sibling_id(from));
        bvh_.nodes[sibling_id] = dst_node;
        bvh_.nodes[parent_id] = sibling_node;

        if (!dst_node.is_leaf()) {
            parents_[dst_node.index.first_id() + 0] = sibling_id;
            parents_[dst_node.index.first_id() + 1] = sibling_id;
        }
        if (!sibling_node.is_leaf()) {
            parents_[sibling_node.index.first_id() + 0] = parent_id;
            parents_[sibling_node.index.first_id() + 1] = parent_id;
        }

        parents_[sibling_id] = to;
        parents_[from] = to;
        refit_from(to);
        refit_from(parent_id);
    }

    BVH_ALWAYS_INLINE void refit_from(size_t index) {
        do {
            auto& node = bvh_.nodes[index];
            if (!node.is_leaf()) {
                node.set_bbox(
                    bvh_.nodes[node.index.first_id() + 0].get_bbox().extend(
                    bvh_.nodes[node.index.first_id() + 1].get_bbox()));
            }
            index = parents_[index];
        } while (index != 0);
    }

    BVH_ALWAYS_INLINE auto get_conflicts(size_t from, size_t to) {
        return std::array<size_t, 5> {
            to, from,
            Bvh<Node>::get_sibling_id(from),
            parents_[to],
            parents_[from]
        };
    }

    template <typename Derived>
    void optimize(Executor<Derived>& executor, const Config& config) {
        auto batch_size = std::max(size_t{1},
            static_cast<size_t>(static_cast<Scalar>(bvh_.nodes.size()) * config.batch_size_ratio));
        std::vector<Reinsertion> reinsertions;
        std::vector<bool> touched(bvh_.nodes.size());

        for (size_t iter = 0; iter < config.max_iter_count; ++iter) {
            auto candidates = find_candidates(batch_size);

            std::fill(touched.begin(), touched.end(), false);
            reinsertions.resize(candidates.size());
            executor.for_each(0, candidates.size(),
                [&] (size_t begin, size_t end) {
                    for (size_t i = begin; i < end; ++i)
                        reinsertions[i] = find_reinsertion(candidates[i].node_id);
                });

            reinsertions.erase(std::remove_if(reinsertions.begin(), reinsertions.end(),
                [] (auto& r) { return r.area_diff <= 0; }), reinsertions.end());
            std::sort(reinsertions.begin(), reinsertions.end(), std::greater<>{});

            for (auto& reinsertion : reinsertions) {
                auto conflicts = get_conflicts(reinsertion.from, reinsertion.to);
                if (std::any_of(conflicts.begin(), conflicts.end(), [&] (size_t i) { return touched[i]; }))
                    continue;
                for (auto conflict : conflicts)
                    touched[conflict] = true;
                reinsert_node(reinsertion.from, reinsertion.to);
            }
        }
    }
};

} // namespace bvh::v2

#endif

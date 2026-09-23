#ifndef BVH_V2_MINI_TREE_BUILDER_H
#define BVH_V2_MINI_TREE_BUILDER_H

#include "bvh/v2/sweep_sah_builder.h"
#include "bvh/v2/binned_sah_builder.h"
#include "bvh/v2/thread_pool.h"
#include "bvh/v2/executor.h"

#include <stack>
#include <tuple>
#include <algorithm>
#include <optional>
#include <numeric>
#include <cassert>

namespace bvh::v2 {

/// Multi-threaded top-down builder that partitions primitives using a grid. Multiple instances
/// of a single-threaded builder are run in parallel on that partition, generating many small
/// trees. Finally, a top-level tree is built on these smaller trees to form the final BVH.
/// This builder is inspired by
/// "Rapid Bounding Volume Hierarchy Generation using Mini Trees", by P. Ganestam et al.
template <typename Node, typename MortonCode = uint32_t>
class MiniTreeBuilder {
    using Scalar = typename Node::Scalar;
    using Vec  = bvh::v2::Vec<Scalar, Node::dimension>;
    using BBox = bvh::v2::BBox<Scalar, Node::dimension>;

public:
    struct Config : TopDownSahBuilder<Node>::Config {
        /// Flag that turns on/off mini-tree pruning.
        bool enable_pruning = true;

        /// Threshold on the area of a mini-tree node above which it is pruned, expressed in
        /// fraction of the area of bounding box around the entire set of primitives.
        Scalar pruning_area_ratio = static_cast<Scalar>(0.01);

        /// Minimum number of primitives per parallel task.
        size_t parallel_threshold = 1024;

        /// Log of the dimension of the grid used to split the workload horizontally.
        size_t log2_grid_dim = 4;
    };

    /// Starts building a BVH with the given primitive data. The build algorithm is multi-threaded,
    /// and runs on the given thread pool.
    BVH_ALWAYS_INLINE static Bvh<Node> build(
        ThreadPool& thread_pool,
        std::span<const BBox> bboxes,
        std::span<const Vec> centers,
        const Config& config = {})
    {
        MiniTreeBuilder builder(thread_pool, bboxes, centers, config);
        auto mini_trees = builder.build_mini_trees();
        if (config.enable_pruning)
            mini_trees = builder.prune_mini_trees(std::move(mini_trees));
        return builder.build_top_bvh(mini_trees);
    }

private:
    friend struct BuildTask;

    struct Bin {
        std::vector<size_t> ids;

        BVH_ALWAYS_INLINE void add(size_t id) { ids.push_back(id); }

        BVH_ALWAYS_INLINE void merge(Bin&& other) {
            if (ids.empty())
                ids = std::move(other.ids);
            else {
                ids.insert(ids.end(), other.ids.begin(), other.ids.end());
                other.ids.clear();
            }
        }
    };

    struct LocalBins {
        std::vector<Bin> bins;

        BVH_ALWAYS_INLINE Bin& operator [] (size_t i) { return bins[i]; }
        BVH_ALWAYS_INLINE const Bin& operator [] (size_t i) const { return bins[i]; }

        BVH_ALWAYS_INLINE void merge_small_bins(size_t threshold) {
            for (size_t i = 0; i < bins.size();) {
                size_t j = i + 1;
                for (; j < bins.size() && bins[j].ids.size() + bins[i].ids.size() <= threshold; ++j)
                    bins[i].merge(std::move(bins[j]));
                i = j;
            }
        }

        BVH_ALWAYS_INLINE void remove_empty_bins() {
            bins.resize(std::remove_if(bins.begin(), bins.end(),
                [] (const Bin& bin) { return bin.ids.empty(); }) - bins.begin());
        }

        BVH_ALWAYS_INLINE void merge(LocalBins&& other) {
            bins.resize(std::max(bins.size(), other.bins.size()));
            for (size_t i = 0, n = std::min(bins.size(), other.bins.size()); i < n; ++i)
                bins[i].merge(std::move(other[i]));
        }
    };

    struct BuildTask {
        MiniTreeBuilder* builder;
        Bvh<Node>& bvh;
        std::vector<size_t> prim_ids;

        std::vector<BBox> bboxes;
        std::vector<Vec> centers;

        BuildTask(
            MiniTreeBuilder* builder,
            Bvh<Node>& bvh,
            std::vector<size_t>&& prim_ids)
            : builder(builder)
            , bvh(bvh)
            , prim_ids(std::move(prim_ids))
        {}

        BVH_ALWAYS_INLINE void run() {
            // Make sure that rebuilds produce the same BVH
            std::sort(prim_ids.begin(), prim_ids.end());

            // Extract bounding boxes and centers for this set of primitives
            bboxes.resize(prim_ids.size());
            centers.resize(prim_ids.size());
            for (size_t i = 0; i < prim_ids.size(); ++i) {
                bboxes[i] = builder->bboxes_[prim_ids[i]];
                centers[i] = builder->centers_[prim_ids[i]];
            }

            bvh = BinnedSahBuilder<Node>::build(bboxes, centers, builder->config_);

            // Permute primitive indices so that they index the proper set of primitives
            for (size_t i = 0; i < bvh.prim_ids.size(); ++i)
                bvh.prim_ids[i] = prim_ids[bvh.prim_ids[i]];
        }
    };

    ParallelExecutor executor_;
    std::span<const BBox> bboxes_;
    std::span<const Vec> centers_;
    const Config& config_;

    BVH_ALWAYS_INLINE MiniTreeBuilder(
        ThreadPool& thread_pool,
        std::span<const BBox> bboxes,
        std::span<const Vec> centers,
        const Config& config)
        : executor_(thread_pool)
        , bboxes_(bboxes)
        , centers_(centers)
        , config_(config)
    {
        assert(bboxes.size() == centers.size());
    }

    std::vector<Bvh<Node>> build_mini_trees() {
        // Compute the bounding box of all centers
        auto center_bbox = executor_.reduce(0, bboxes_.size(), BBox::make_empty(),
            [this] (BBox& bbox, size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i)
                    bbox.extend(centers_[i]);
            },
            [] (BBox& bbox, const BBox& other) { bbox.extend(other); });

        assert(config_.log2_grid_dim <= std::numeric_limits<MortonCode>::digits / Node::dimension);
        auto bin_count = size_t{1} << (config_.log2_grid_dim * Node::dimension);
        auto grid_dim = size_t{1} << config_.log2_grid_dim;
        auto grid_scale = Vec(static_cast<Scalar>(grid_dim)) * safe_inverse(center_bbox.get_diagonal());
        auto grid_offset = -center_bbox.min * grid_scale;

        // Place primitives in bins
        auto final_bins = executor_.reduce(0, bboxes_.size(), LocalBins {},
            [&] (LocalBins& local_bins, size_t begin, size_t end) {
                local_bins.bins.resize(bin_count);
                for (size_t i = begin; i < end; ++i) {
                    auto p = robust_max(fast_mul_add(centers_[i], grid_scale, grid_offset), Vec(0));
                    auto x = std::min(grid_dim - 1, static_cast<size_t>(p[0]));
                    auto y = std::min(grid_dim - 1, static_cast<size_t>(p[1]));
                    auto z = std::min(grid_dim - 1, static_cast<size_t>(p[2]));
                    local_bins[morton_encode(x, y, z) & (bin_count - 1)].add(i);
                }
            },
            [&] (LocalBins& result, LocalBins&& other) { result.merge(std::move(other)); });

        // Note: Merging small bins will deteriorate the quality of the top BVH if there is no
        // pruning, since it will then produce larger mini-trees. For this reason, it is only enabled
        // when mini-tree pruning is enabled.
        if (config_.enable_pruning)
            final_bins.merge_small_bins(config_.parallel_threshold);
        final_bins.remove_empty_bins();

        // Iterate over bins to collect groups of primitives and build BVHs over them in parallel
        std::vector<Bvh<Node>> mini_trees(final_bins.bins.size());
        for (size_t i = 0; i < final_bins.bins.size(); ++i) {
            auto task = new BuildTask(this, mini_trees[i], std::move(final_bins[i].ids));
            executor_.thread_pool.push([task] (size_t) { task->run(); delete task; });
        }
        executor_.thread_pool.wait();

        return mini_trees;
    }

    std::vector<Bvh<Node>> prune_mini_trees(std::vector<Bvh<Node>>&& mini_trees) {
        // Compute the area threshold based on the area of the entire set of primitives
        auto avg_area = static_cast<Scalar>(0.);
        for (auto& mini_tree : mini_trees)
            avg_area += mini_tree.get_root().get_bbox().get_half_area();
        avg_area /= static_cast<Scalar>(mini_trees.size());
        auto threshold = avg_area * config_.pruning_area_ratio;

        // Cull nodes whose area is above the threshold
        std::stack<size_t> stack;
        std::vector<std::pair<size_t, size_t>> pruned_roots;
        for (size_t i = 0; i < mini_trees.size(); ++i) {
            stack.push(0);
            auto& mini_tree = mini_trees[i];
            while (!stack.empty()) {
                auto node_id = stack.top();
                auto& node = mini_tree.nodes[node_id];
                stack.pop();
                if (node.get_bbox().get_half_area() < threshold || node.is_leaf()) {
                    pruned_roots.emplace_back(i, node_id);
                } else {
                    stack.push(node.index.first_id());
                    stack.push(node.index.first_id() + 1);
                }
            }
        }

        // Extract the BVHs rooted at the previously computed indices
        std::vector<Bvh<Node>> pruned_trees(pruned_roots.size());
        executor_.for_each(0, pruned_roots.size(),
            [&] (size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    if (pruned_roots[i].second == 0)
                        pruned_trees[i] = std::move(mini_trees[pruned_roots[i].first]);
                    else
                        pruned_trees[i] = mini_trees[pruned_roots[i].first]
                            .extract_bvh(pruned_roots[i].second);
                }
            });
        return pruned_trees;
    }

    Bvh<Node> build_top_bvh(std::vector<Bvh<Node>>& mini_trees) {
        // Build a BVH using the mini trees as leaves
        std::vector<Vec> centers(mini_trees.size());
        std::vector<BBox> bboxes(mini_trees.size());
        for (size_t i = 0; i < mini_trees.size(); ++i) {
            bboxes[i] = mini_trees[i].get_root().get_bbox();
            centers[i] = bboxes[i].get_center();
        }

        typename SweepSahBuilder<Node>::Config config = config_;
        config.max_leaf_size = config.min_leaf_size = 1; // Needs to have only one mini-tree in each leaf
        auto bvh = SweepSahBuilder<Node>::build(bboxes, centers, config);

        // Compute the offsets to apply to primitive and node indices
        std::vector<size_t> node_offsets(mini_trees.size());
        std::vector<size_t> prim_offsets(mini_trees.size());
        size_t node_count = bvh.nodes.size();
        size_t prim_count = 0;
        for (size_t i = 0; i < mini_trees.size(); ++i) {
            node_offsets[i] = node_count - 1; // Skip root node
            prim_offsets[i] = prim_count;
            node_count += mini_trees[i].nodes.size() - 1; // idem
            prim_count += mini_trees[i].prim_ids.size();
        }

        // Helper function to copy and fix the child/primitive index of a node
        auto copy_node = [&] (size_t i, Node& dst_node, const Node& src_node) {
            dst_node = src_node;
            dst_node.index.set_first_id(dst_node.index.first_id() +
                (src_node.is_leaf() ? prim_offsets[i] : node_offsets[i]));
        };

        // Make the leaves of the top BVH point to the right internal nodes
        for (auto& node : bvh.nodes) {
            if (!node.is_leaf())
                continue;
            assert(node.index.prim_count() == 1);
            size_t tree_id = bvh.prim_ids[node.index.first_id()];
            copy_node(tree_id, node, mini_trees[tree_id].get_root());
        }

        bvh.nodes.resize(node_count);
        bvh.prim_ids.resize(prim_count);
        executor_.for_each(0, mini_trees.size(),
            [&] (size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    auto& mini_tree = mini_trees[i];

                    // Copy the nodes of the mini tree with the offsets applied, without copying
                    // the root node (since it is already copied to the top-level part of the BVH).
                    for (size_t j = 1; j < mini_tree.nodes.size(); ++j)
                        copy_node(i, bvh.nodes[node_offsets[i] + j], mini_tree.nodes[j]);

                    std::copy(
                        mini_tree.prim_ids.begin(),
                        mini_tree.prim_ids.end(),
                        bvh.prim_ids.begin() + prim_offsets[i]);
                }
            });

        return bvh;
    }
};

} // namespace bvh::v2

#endif

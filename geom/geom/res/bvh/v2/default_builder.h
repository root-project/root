#ifndef BVH_V2_DEFAULT_BUILDER_H
#define BVH_V2_DEFAULT_BUILDER_H

#include "bvh/v2/mini_tree_builder.h"
#include "bvh/v2/sweep_sah_builder.h"
#include "bvh/v2/binned_sah_builder.h"
#include "bvh/v2/reinsertion_optimizer.h"
#include "bvh/v2/thread_pool.h"

namespace bvh::v2 {

/// This builder is only a wrapper around all the other builders, which selects the best builder
/// depending on the desired BVH quality and whether a multi-threaded build is desired.
template <typename Node>
class DefaultBuilder {
    using Scalar = typename Node::Scalar;
    using Vec  = bvh::v2::Vec<Scalar, Node::dimension>;
    using BBox = bvh::v2::BBox<Scalar, Node::dimension>;

public:
    enum class Quality { Low, Medium, High };

    struct Config : TopDownSahBuilder<Node>::Config {
        /// The quality of the BVH produced by the builder. The higher the quality the faster the
        /// BVH is to traverse, but the slower it is to build.
        Quality quality = Quality::High;

        /// Threshold, in number of primitives, under which the builder operates in a single-thread.
        size_t parallel_threshold = 1024;
    };

    /// Build a BVH in parallel using the given thread pool.
    BVH_ALWAYS_INLINE static Bvh<Node> build(
        ThreadPool& thread_pool,
        std::span<const BBox> bboxes,
        std::span<const Vec> centers,
        const Config& config = {})
    {
        if (bboxes.size() < config.parallel_threshold)
            return build(bboxes, centers, config);
        auto bvh = MiniTreeBuilder<Node>::build(
            thread_pool, bboxes, centers, make_mini_tree_config(config));
        if (config.quality == Quality::High)
            ReinsertionOptimizer<Node>::optimize(thread_pool, bvh);
        return bvh;
    }

    /// Build a BVH in a single-thread.
    BVH_ALWAYS_INLINE static Bvh<Node> build(
        std::span<const BBox>  bboxes,
        std::span<const Vec> centers,
        const Config& config = {})
    {
        if (config.quality == Quality::Low)
            return BinnedSahBuilder<Node>::build(bboxes, centers, config);
        else {
            auto bvh = SweepSahBuilder<Node>::build(bboxes, centers, config);
            if (config.quality == Quality::High)
                ReinsertionOptimizer<Node>::optimize(bvh);
            return bvh;
        }
    }

private:
    BVH_ALWAYS_INLINE static auto make_mini_tree_config(const Config& config) {
        typename MiniTreeBuilder<Node>::Config mini_tree_config;
        static_cast<typename TopDownSahBuilder<Node>::Config&>(mini_tree_config) = config;
        mini_tree_config.enable_pruning = config.quality == Quality::Low ? false : true;
        mini_tree_config.pruning_area_ratio =
            config.quality == Quality::High ? static_cast<Scalar>(0.01) : static_cast<Scalar>(0.1);
        mini_tree_config.parallel_threshold = config.parallel_threshold;
        return mini_tree_config;
    }
};

} // namespace bvh::v2

#endif

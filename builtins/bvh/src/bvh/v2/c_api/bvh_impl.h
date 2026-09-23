#ifndef BVH_V2_C_API_BVH_IMPL_H
#define BVH_V2_C_API_BVH_IMPL_H

#include <bvh/v2/stack.h>
#include <bvh/v2/default_builder.h>
#include <bvh/v2/reinsertion_optimizer.h>

namespace bvh::v2::c_api {

template <typename T, size_t Dim>
struct BvhTypes {};

template <typename T>
struct BvhCallback {};

template <> struct BvhCallback<float>  { using Type = bvh_intersect_callbackf; };
template <> struct BvhCallback<double> { using Type = bvh_intersect_callbackd; };

template <typename T, size_t Dim>
static auto translate(enum bvh_build_quality quality) {
    switch (quality) {
#ifndef BVH_C_API_UNSAFE_CASTS
        case BVH_BUILD_QUALITY_LOW:
            return bvh::v2::DefaultBuilder<bvh::v2::Node<T, Dim>>::Quality::Low;
        case BVH_BUILD_QUALITY_MEDIUM:
            return bvh::v2::DefaultBuilder<bvh::v2::Node<T, Dim>>::Quality::Medium;
        case BVH_BUILD_QUALITY_HIGH:
            return bvh::v2::DefaultBuilder<bvh::v2::Node<T, Dim>>::Quality::High;
#endif
        default:
            return static_cast<typename bvh::v2::DefaultBuilder<bvh::v2::Node<T, Dim>>::Quality>(quality);
    }
}

template <typename T, size_t Dim>
static auto translate(const bvh_build_config* config) {
    auto translated_config = typename bvh::v2::DefaultBuilder<bvh::v2::Node<T, Dim>>::Config {};
    if (config) {
        translated_config.quality = translate<T, Dim>(config->quality);
        translated_config.min_leaf_size = config->min_leaf_size;
        translated_config.max_leaf_size = config->max_leaf_size;
        translated_config.parallel_threshold = config->parallel_threshold;
    }
    return translated_config;
}

template <typename T, size_t Dim>
static auto translate(const typename BvhTypes<T, Dim>::Vec& vec) {
    static_assert(Dim == 2 || Dim == 3);
    if constexpr (Dim == 2) {
        return bvh::v2::Vec<T, 2> { vec.x, vec.y };
    } else {
        return bvh::v2::Vec<T, Dim> { vec.x, vec.y, vec.z };
    }
}

template <typename T, size_t Dim>
static auto translate(const bvh::v2::Vec<T, Dim>& vec) {
    static_assert(Dim == 2 || Dim == 3);
    if constexpr (Dim == 2) {
        return typename BvhTypes<T, Dim>::Vec { vec[0], vec[1] };
    } else {
        return typename BvhTypes<T, Dim>::Vec { vec[0], vec[1], vec[2] };
    }
}

template <typename T, size_t Dim>
static auto translate(const typename BvhTypes<T, Dim>::BBox& bbox) {
    return bvh::v2::BBox<T, Dim> { translate<T, Dim>(bbox.min), translate<T, Dim>(bbox.max) };
}

template <typename T, size_t Dim>
static auto translate(const bvh::v2::BBox<T, Dim>& bbox) {
    return typename BvhTypes<T, Dim>::BBox { translate<T, Dim>(bbox.min), translate<T, Dim>(bbox.max) };
}

template <typename T, size_t Dim>
static auto translate(const typename BvhTypes<T, Dim>::Ray& ray) {
    return bvh::v2::Ray<T, Dim> { translate<T, Dim>(ray.org), translate<T, Dim>(ray.dir), ray.tmin, ray.tmax };
}

template <typename T, size_t Dim>
static typename BvhTypes<T, Dim>::Bvh* bvh_build(
    bvh_thread_pool* thread_pool,
    const typename BvhTypes<T, Dim>::BBox* bboxes,
    const typename BvhTypes<T, Dim>::Vec* centers,
    size_t prim_count,
    const bvh_build_config* config)
{
    bvh::v2::BBox<T, Dim>* translated_bboxes = nullptr;
    bvh::v2::Vec<T, Dim>* translated_centers = nullptr;
#ifdef BVH_C_API_UNSAFE_CASTS
    translated_bboxes  = reinterpret_cast<decltype(translated_bboxes)>(bboxes);
    translated_centers = reinterpret_cast<decltype(translated_centers)>(centers);
#else
    std::vector<bvh::v2::BBox<T, Dim>> bbox_buf(prim_count);
    std::vector<bvh::v2::Vec<T, Dim>> center_buf(prim_count);
    for (size_t i = 0; i < prim_count; ++i) {
        bbox_buf[i]   = translate<T, Dim>(bboxes[i]);
        center_buf[i] = translate<T, Dim>(centers[i]);
    }
    translated_bboxes = bbox_buf.data();
    translated_centers = center_buf.data();
#endif
    auto bvh = thread_pool
        ? bvh::v2::DefaultBuilder<bvh::v2::Node<T, Dim>>::build(
            *reinterpret_cast<bvh::v2::ThreadPool*>(thread_pool),
            std::span { translated_bboxes,  translated_bboxes + prim_count },
            std::span { translated_centers, translated_centers + prim_count },
            translate<T, Dim>(config))
        : bvh::v2::DefaultBuilder<bvh::v2::Node<T, Dim>>::build(
            std::span { translated_bboxes,  translated_bboxes + prim_count },
            std::span { translated_centers, translated_centers + prim_count },
            translate<T, Dim>(config));
    return reinterpret_cast<typename BvhTypes<T, Dim>::Bvh*>(new decltype(bvh) { std::move(bvh) });
}

template <typename T, size_t Dim>
static void bvh_save(const typename BvhTypes<T, Dim>::Bvh* bvh, FILE* file) {
    struct FileOutputStream : bvh::v2::OutputStream {
        FILE* file;
        FileOutputStream(FILE* file)
            : file(file)
        {}
        bool write_raw(const void* data, size_t size) override {
            return fwrite(data, 1, size, file) == size;
        }
    };
    FileOutputStream stream { file };
    reinterpret_cast<const bvh::v2::Bvh<bvh::v2::Node<T, Dim>>*>(bvh)->serialize(stream);
}

template <typename T, size_t Dim>
static typename BvhTypes<T, Dim>::Bvh* bvh_load(FILE* file) {
    struct FileInputStream : bvh::v2::InputStream {
        FILE* file;
        FileInputStream(FILE* file)
            : file(file)
        {}
        size_t read_raw(void* data, size_t size) override {
            return fread(data, 1, size, file);
        }
    };
    FileInputStream stream { file };
    return reinterpret_cast<typename BvhTypes<T, Dim>::Bvh*>(
        new bvh::v2::Bvh<bvh::v2::Node<T, Dim>>(bvh::v2::Bvh<bvh::v2::Node<T, Dim>>::deserialize(stream)));
}

template <typename T, size_t Dim>
static typename BvhTypes<T, Dim>::Node* bvh_get_node(typename BvhTypes<T, Dim>::Bvh* bvh, size_t node_id) {
    auto& nodes = reinterpret_cast<bvh::v2::Bvh<bvh::v2::Node<T, Dim>>*>(bvh)->nodes;
    assert(node_id < nodes.size());
    return reinterpret_cast<typename BvhTypes<T, Dim>::Node*>(&nodes[node_id]);
}

template <typename T, size_t Dim>
static size_t bvh_get_prim_id(const typename BvhTypes<T, Dim>::Bvh* bvh, size_t i) {
    auto& prim_ids = reinterpret_cast<const bvh::v2::Bvh<bvh::v2::Node<T, Dim>>*>(bvh)->prim_ids;
    assert(i < prim_ids.size());
    return prim_ids[i];
}

template <typename T, size_t Dim>
static size_t bvh_get_prim_count(const typename BvhTypes<T, Dim>::Bvh* bvh) {
    return reinterpret_cast<const bvh::v2::Bvh<bvh::v2::Node<T, Dim>>*>(bvh)->prim_ids.size();
}

template <typename T, size_t Dim>
static size_t bvh_get_node_count(const typename BvhTypes<T, Dim>::Bvh* bvh) {
    return reinterpret_cast<const bvh::v2::Bvh<bvh::v2::Node<T, Dim>>*>(bvh)->nodes.size();
}

template <typename T, size_t Dim>
static bool bvh_node_is_leaf(const typename BvhTypes<T, Dim>::Node* node) {
    return reinterpret_cast<const bvh::v2::Node<T, Dim>*>(node)->is_leaf();
}

template <typename T, size_t Dim>
static size_t bvh_node_get_prim_count(const typename BvhTypes<T, Dim>::Node* node) {
    return reinterpret_cast<const bvh::v2::Node<T, Dim>*>(node)->index.prim_count();
}

template <typename T, size_t Dim>
static void bvh_node_set_prim_count(typename BvhTypes<T, Dim>::Node* node, size_t prim_count) {
    return reinterpret_cast<bvh::v2::Node<T, Dim>*>(node)->index.set_prim_count(prim_count);
}

template <typename T, size_t Dim>
static size_t bvh_node_get_first_id(const typename BvhTypes<T, Dim>::Node* node) {
    return reinterpret_cast<const bvh::v2::Node<T, Dim>*>(node)->index.first_id();
}

template <typename T, size_t Dim>
static void bvh_node_set_first_id(typename BvhTypes<T, Dim>::Node* node, size_t first_id) {
    reinterpret_cast<bvh::v2::Node<T, Dim>*>(node)->index.set_first_id(first_id);
}

template <typename T, size_t Dim>
static typename BvhTypes<T, Dim>::BBox bvh_node_get_bbox(const typename BvhTypes<T, Dim>::Node* node) {
    return translate(reinterpret_cast<const bvh::v2::Node<T, Dim>*>(node)->get_bbox());
}

template <typename T, size_t Dim>
static void bvh_node_set_bbox(typename BvhTypes<T, Dim>::Node* node, const typename BvhTypes<T, Dim>::BBox* bbox) {
    reinterpret_cast<bvh::v2::Node<T, Dim>*>(node)->set_bbox(translate<T, Dim>(*bbox));
}

template <typename T, size_t Dim>
static void bvh_append_node(typename BvhTypes<T, Dim>::Bvh* bvh) {
    reinterpret_cast<bvh::v2::Bvh<bvh::v2::Node<T, Dim>>*>(bvh)->nodes.emplace_back();
}

template <typename T, size_t Dim>
static void bvh_remove_last_node(typename BvhTypes<T, Dim>::Bvh* bvh) {
    reinterpret_cast<bvh::v2::Bvh<bvh::v2::Node<T, Dim>>*>(bvh)->nodes.pop_back();
}

template <typename T, size_t Dim>
static void bvh_refit(typename BvhTypes<T, Dim>::Bvh* bvh) {
    reinterpret_cast<bvh::v2::Bvh<bvh::v2::Node<T, Dim>>*>(bvh)->refit();
}

template <typename T, size_t Dim>
static void bvh_optimize(bvh_thread_pool* thread_pool, typename BvhTypes<T, Dim>::Bvh* bvh) {
    if (thread_pool) {
        bvh::v2::ReinsertionOptimizer<bvh::v2::Node<T, Dim>>::optimize(
            *reinterpret_cast<bvh::v2::ThreadPool*>(thread_pool),
            *reinterpret_cast<bvh::v2::Bvh<bvh::v2::Node<T, Dim>>*>(bvh));
    } else {
        bvh::v2::ReinsertionOptimizer<bvh::v2::Node<T, Dim>>::optimize(
            *reinterpret_cast<bvh::v2::Bvh<bvh::v2::Node<T, Dim>>*>(bvh));
    }
}

template <typename T, size_t Dim, bool AnyHit, bool RobustTraversal>
static void bvh_intersect_ray(
    const typename BvhTypes<T, Dim>::Bvh* bvh,
    const typename BvhTypes<T, Dim>::Ray* ray,
    const typename BvhCallback<T>::Type* callback)
{
    static constexpr size_t stack_size = 64;
    auto translated_ray = translate<T, Dim>(*ray);
    auto& translated_bvh = *reinterpret_cast<const bvh::v2::Bvh<bvh::v2::Node<T, Dim>>*>(bvh);
    bvh::v2::SmallStack<typename bvh::v2::Node<T, Dim>::Index, stack_size> stack;
    translated_bvh.template intersect<AnyHit, RobustTraversal>(
        translated_ray, translated_bvh.get_root().index, stack,
        [&] (size_t begin, size_t end) {
            return callback->user_fn(callback->user_data, &translated_ray.tmax, begin, end);
        });
}

#define BVH_TYPES(T, Dim, suffix) \
    template <> \
    struct BvhTypes<T, Dim> { \
        using Bvh  = bvh##suffix; \
        using Vec  = bvh_vec##suffix; \
        using BBox = bvh_bbox##suffix; \
        using Node = bvh_node##suffix; \
        using Ray  = bvh_ray##suffix; \
    };

#define BVH_IMPL(T, Dim, suffix) \
    BVH_EXPORT typename BvhTypes<T, Dim>::Bvh* bvh##suffix##_build( \
        bvh_thread_pool* thread_pool, \
        const typename BvhTypes<T, Dim>::BBox* bboxes, \
        const typename BvhTypes<T, Dim>::Vec* centers, \
        size_t prim_count, \
        const bvh_build_config* config) \
    { \
        return bvh_build<T, Dim>(thread_pool, bboxes, centers, prim_count, config); \
    } \
    BVH_EXPORT void bvh##suffix##_destroy(typename BvhTypes<T, Dim>::Bvh* bvh) { \
        delete reinterpret_cast<bvh::v2::Bvh<bvh::v2::Node<T, Dim>>*>(bvh); \
    } \
    BVH_EXPORT void bvh##suffix##_save(const typename BvhTypes<T, Dim>::Bvh* bvh, FILE* file) { \
        bvh_save<T, Dim>(bvh, file); \
    } \
    BVH_EXPORT typename BvhTypes<T, Dim>::Bvh* bvh##suffix##_load(FILE* file) { \
        return bvh_load<T, Dim>(file); \
    } \
    BVH_EXPORT typename BvhTypes<T, Dim>::Node* bvh##suffix##_get_node(typename BvhTypes<T, Dim>::Bvh* bvh, size_t node_id) { \
        return bvh_get_node<T, Dim>(bvh, node_id); \
    } \
    BVH_EXPORT size_t bvh##suffix##_get_prim_id(const typename BvhTypes<T, Dim>::Bvh* bvh, size_t i) { \
        return bvh_get_prim_id<T, Dim>(bvh, i); \
    } \
    BVH_EXPORT size_t bvh##suffix##_get_prim_count(const typename BvhTypes<T, Dim>::Bvh* bvh) { \
        return bvh_get_prim_count<T, Dim>(bvh); \
    } \
    BVH_EXPORT size_t bvh##suffix##_get_node_count(const typename BvhTypes<T, Dim>::Bvh* bvh) { \
        return bvh_get_node_count<T, Dim>(bvh); \
    } \
    BVH_EXPORT bool bvh_node##suffix##_is_leaf(const typename BvhTypes<T, Dim>::Node* node) { \
        return bvh_node_is_leaf<T, Dim>(node); \
    } \
    BVH_EXPORT size_t bvh_node##suffix##_get_prim_count(const typename BvhTypes<T, Dim>::Node* node) { \
        return bvh_node_get_prim_count<T, Dim>(node); \
    } \
    BVH_EXPORT void bvh_node##suffix##_set_prim_count(typename BvhTypes<T, Dim>::Node* node, size_t prim_count) { \
        bvh_node_set_prim_count<T, Dim>(node, prim_count); \
    } \
    BVH_EXPORT size_t bvh_node##suffix##_get_first_id(const typename BvhTypes<T, Dim>::Node* node) { \
        return bvh_node_get_first_id<T, Dim>(node); \
    } \
    BVH_EXPORT void bvh_node##suffix##_set_first_id(typename BvhTypes<T, Dim>::Node* node, size_t first_id) { \
        bvh_node_set_first_id<T, Dim>(node, first_id); \
    } \
    BVH_EXPORT typename BvhTypes<T, Dim>::BBox bvh_node##suffix##_get_bbox(const typename BvhTypes<T, Dim>::Node* node) { \
        return bvh_node_get_bbox<T, Dim>(node); \
    } \
    BVH_EXPORT void bvh_node##suffix##_set_bbox(typename BvhTypes<T, Dim>::Node* node, const typename BvhTypes<T, Dim>::BBox* bbox) { \
        bvh_node_set_bbox<T, Dim>(node, bbox); \
    } \
    BVH_EXPORT void bvh##suffix##_append_node(typename BvhTypes<T, Dim>::Bvh* bvh) { \
        bvh_append_node<T, Dim>(bvh); \
    } \
    BVH_EXPORT void bvh##suffix##_remove_last_node(typename BvhTypes<T, Dim>::Bvh* bvh) { \
        bvh_remove_last_node<T, Dim>(bvh); \
    } \
    BVH_EXPORT void bvh##suffix##_refit(typename BvhTypes<T, Dim>::Bvh* bvh) { \
        bvh_refit<T, Dim>(bvh); \
    } \
    BVH_EXPORT void bvh##suffix##_optimize(bvh_thread_pool* thread_pool, typename BvhTypes<T, Dim>::Bvh* bvh) { \
        bvh_optimize<T, Dim>(thread_pool, bvh); \
    } \
    BVH_EXPORT void bvh##suffix##_intersect_ray_any( \
        const typename BvhTypes<T, Dim>::Bvh* bvh, \
        const typename BvhTypes<T, Dim>::Ray* ray, \
        const typename BvhCallback<T>::Type* callback) \
    { \
        bvh_intersect_ray<T, Dim, true, false>(bvh, ray, callback); \
    } \
    BVH_EXPORT void bvh##suffix##_intersect_ray_any_robust( \
        const typename BvhTypes<T, Dim>::Bvh* bvh, \
        const typename BvhTypes<T, Dim>::Ray* ray, \
        const typename BvhCallback<T>::Type* callback) \
    { \
        bvh_intersect_ray<T, Dim, true, true>(bvh, ray, callback); \
    } \
    BVH_EXPORT void bvh##suffix##_intersect_ray( \
        const typename BvhTypes<T, Dim>::Bvh* bvh, \
        const typename BvhTypes<T, Dim>::Ray* ray, \
        const typename BvhCallback<T>::Type* callback) \
    { \
        bvh_intersect_ray<T, Dim, false, false>(bvh, ray, callback); \
    } \
    BVH_EXPORT void bvh##suffix##_intersect_ray_robust( \
        const typename BvhTypes<T, Dim>::Bvh* bvh, \
        const typename BvhTypes<T, Dim>::Ray* ray, \
        const typename BvhCallback<T>::Type* callback) \
    { \
        bvh_intersect_ray<T, Dim, false, true>(bvh, ray, callback); \
    }

} // namespace bvh::v2::c_api

#endif

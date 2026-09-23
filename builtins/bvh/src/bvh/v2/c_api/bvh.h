#ifndef BVH_V2_C_API_BVH_H
#define BVH_V2_C_API_BVH_H

// C API to the BVH library, providing high-level access to BVH construction and traversal. This API
// may have an overhead compared to using the C++ API directly, as it may have to translate types
// between the C interface and the C++ one, and because callbacks are not going to be inlined at the
// API boundary. To mitigate that problem, specialized functions are provided which provide faster
// operation for a small set of situations.

#include <stddef.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
#define BVH_EXPORT __declspec(dllexport)
#define BVH_IMPORT __declspec(dllimport)
#else
#define BVH_EXPORT __attribute__((visibility("default")))
#define BVH_IMPORT BVH_EXPORT
#endif

#ifdef BVH_BUILD_API
#define BVH_API BVH_EXPORT
#else
#define BVH_API BVH_IMPORT
#endif

#define BVH_ROOT_INDEX 0
#define BVH_INVALID_PRIM_ID SIZE_MAX

struct bvh2f;
struct bvh3f;
struct bvh_node2f;
struct bvh_node3f;

struct bvh2d;
struct bvh3d;
struct bvh_node2d;
struct bvh_node3d;

struct bvh_thread_pool;

enum bvh_build_quality {
    BVH_BUILD_QUALITY_LOW,
    BVH_BUILD_QUALITY_MEDIUM,
    BVH_BUILD_QUALITY_HIGH
};

struct bvh_build_config {
    enum bvh_build_quality quality;
    size_t min_leaf_size;
    size_t max_leaf_size;
    size_t parallel_threshold;
};

struct bvh_vec2f { float x, y; };
struct bvh_vec3f { float x, y, z; };
struct bvh_vec2d { double x, y; };
struct bvh_vec3d { double x, y, z; };

struct bvh_bbox2f { struct bvh_vec2f min, max; };
struct bvh_bbox3f { struct bvh_vec3f min, max; };
struct bvh_bbox2d { struct bvh_vec2d min, max; };
struct bvh_bbox3d { struct bvh_vec3d min, max; };

struct bvh_ray2f { struct bvh_vec2f org, dir; float tmin, tmax; };
struct bvh_ray3f { struct bvh_vec3f org, dir; float tmin, tmax; };
struct bvh_ray2d { struct bvh_vec2d org, dir; double tmin, tmax; };
struct bvh_ray3d { struct bvh_vec3d org, dir; double tmin, tmax; };

struct bvh_intersect_callbackf {
    void* user_data;
    bool (*user_fn)(void*, float*, size_t begin, size_t end);
};

struct bvh_intersect_callbackd {
    void* user_data;
    bool (*user_fn)(void*, double*, size_t begin, size_t end);
};

// Thread Pool ------------------------------------------------------------------------------------

// A thread count of zero instructs the thread pool to detect the number of threads available on the
// machine.

BVH_API struct bvh_thread_pool* bvh_thread_pool_create(size_t thread_count);
BVH_API void bvh_thread_pool_destroy(struct bvh_thread_pool*);

// BVH Construction -------------------------------------------------------------------------------

// These construction functions can be called with a `NULL` thread pool, in which case the BVH is
// constructed serially. The configuration can also be `NULL`, in which case the default
// configuration is used.

BVH_API struct bvh2f* bvh2f_build(
    struct bvh_thread_pool* thread_pool,
    const struct bvh_bbox2f* bboxes,
    const struct bvh_vec2f* centers,
    size_t prim_count,
    const struct bvh_build_config* config);

BVH_API struct bvh3f* bvh3f_build(
    struct bvh_thread_pool* thread_pool,
    const struct bvh_bbox3f* bboxes,
    const struct bvh_vec3f* centers,
    size_t prim_count,
    const struct bvh_build_config* config);

BVH_API struct bvh2d* bvh2d_build(
    struct bvh_thread_pool* thread_pool,
    const struct bvh_bbox2d* bboxes,
    const struct bvh_vec2d* centers,
    size_t prim_count,
    const struct bvh_build_config* config);

BVH_API struct bvh3d* bvh3d_build(
    struct bvh_thread_pool* thread_pool,
    const struct bvh_bbox3d* bboxes,
    const struct bvh_vec3d* centers,
    size_t prim_count,
    const struct bvh_build_config* config);

// BVH Destruction --------------------------------------------------------------------------------

BVH_API void bvh2f_destroy(struct bvh2f*);
BVH_API void bvh3f_destroy(struct bvh3f*);
BVH_API void bvh2d_destroy(struct bvh2d*);
BVH_API void bvh3d_destroy(struct bvh3d*);

// Serialization/Deserialization to Files ---------------------------------------------------------

BVH_API void bvh2f_save(const struct bvh2f*, FILE*);
BVH_API void bvh3f_save(const struct bvh3f*, FILE*);
BVH_API void bvh2d_save(const struct bvh2d*, FILE*);
BVH_API void bvh3d_save(const struct bvh3d*, FILE*);

BVH_API struct bvh2f* bvh2f_load(FILE*);
BVH_API struct bvh3f* bvh3f_load(FILE*);
BVH_API struct bvh2d* bvh2d_load(FILE*);
BVH_API struct bvh3d* bvh3d_load(FILE*);

// Accessing Nodes and Primitive Indices ----------------------------------------------------------

BVH_API struct bvh_node2f* bvh2f_get_node(struct bvh2f*, size_t);
BVH_API struct bvh_node3f* bvh3f_get_node(struct bvh3f*, size_t);
BVH_API struct bvh_node2d* bvh2d_get_node(struct bvh2d*, size_t);
BVH_API struct bvh_node3d* bvh3d_get_node(struct bvh3d*, size_t);

BVH_API size_t bvh2f_get_prim_id(const struct bvh2f*, size_t);
BVH_API size_t bvh3f_get_prim_id(const struct bvh3f*, size_t);
BVH_API size_t bvh2d_get_prim_id(const struct bvh2d*, size_t);
BVH_API size_t bvh3d_get_prim_id(const struct bvh3d*, size_t);

BVH_API size_t bvh2f_get_prim_count(const struct bvh2f*);
BVH_API size_t bvh3f_get_prim_count(const struct bvh3f*);
BVH_API size_t bvh2d_get_prim_count(const struct bvh2d*);
BVH_API size_t bvh3d_get_prim_count(const struct bvh3d*);

BVH_API size_t bvh2f_get_node_count(const struct bvh2f*);
BVH_API size_t bvh3f_get_node_count(const struct bvh3f*);
BVH_API size_t bvh2d_get_node_count(const struct bvh2d*);
BVH_API size_t bvh3d_get_node_count(const struct bvh3d*);

// Accessing and Modifying Node properties --------------------------------------------------------

BVH_API bool bvh_node2f_is_leaf(const struct bvh_node2f*);
BVH_API bool bvh_node3f_is_leaf(const struct bvh_node3f*);
BVH_API bool bvh_node2d_is_leaf(const struct bvh_node2d*);
BVH_API bool bvh_node3d_is_leaf(const struct bvh_node3d*);

BVH_API size_t bvh_node2f_get_prim_count(const struct bvh_node2f*);
BVH_API size_t bvh_node3f_get_prim_count(const struct bvh_node3f*);
BVH_API size_t bvh_node2d_get_prim_count(const struct bvh_node2d*);
BVH_API size_t bvh_node3d_get_prim_count(const struct bvh_node3d*);

BVH_API void bvh_node2f_set_prim_count(struct bvh_node2f*, size_t);
BVH_API void bvh_node3f_set_prim_count(struct bvh_node3f*, size_t);
BVH_API void bvh_node2d_set_prim_count(struct bvh_node2d*, size_t);
BVH_API void bvh_node3d_set_prim_count(struct bvh_node3d*, size_t);

BVH_API size_t bvh_node2f_get_first_id(const struct bvh_node2f*);
BVH_API size_t bvh_node3f_get_first_id(const struct bvh_node3f*);
BVH_API size_t bvh_node2d_get_first_id(const struct bvh_node2d*);
BVH_API size_t bvh_node3d_get_first_id(const struct bvh_node3d*);

BVH_API void bvh_node2f_set_first_id(struct bvh_node2f*, size_t);
BVH_API void bvh_node3f_set_first_id(struct bvh_node3f*, size_t);
BVH_API void bvh_node2d_set_first_id(struct bvh_node2d*, size_t);
BVH_API void bvh_node3d_set_first_id(struct bvh_node3d*, size_t);

BVH_API struct bvh_bbox2f bvh_node2f_get_bbox(const struct bvh_node2f*);
BVH_API struct bvh_bbox3f bvh_node3f_get_bbox(const struct bvh_node3f*);
BVH_API struct bvh_bbox2d bvh_node2d_get_bbox(const struct bvh_node2d*);
BVH_API struct bvh_bbox3d bvh_node3d_get_bbox(const struct bvh_node3d*);

BVH_API void bvh_node2f_set_bbox(struct bvh_node2f*, const struct bvh_bbox2f*);
BVH_API void bvh_node3f_set_bbox(struct bvh_node3f*, const struct bvh_bbox3f*);
BVH_API void bvh_node2d_set_bbox(struct bvh_node2d*, const struct bvh_bbox2d*);
BVH_API void bvh_node3d_set_bbox(struct bvh_node3d*, const struct bvh_bbox3d*);

// Refitting and BVH Modification -----------------------------------------------------------------

// Refitting functions resize the bounding boxes of the BVH in a bottom-up fashion. This can be
// combined with the optimization function below to allow updating the BVH in an incremental manner.
// IMPORTANT: Appending a node to the BVH invalidates all the node pointers.

BVH_API void bvh2f_append_node(struct bvh2f*);
BVH_API void bvh3f_append_node(struct bvh3f*);
BVH_API void bvh2d_append_node(struct bvh2d*);
BVH_API void bvh3d_append_node(struct bvh3d*);

BVH_API void bvh2f_remove_last_node(struct bvh2f*);
BVH_API void bvh3f_remove_last_node(struct bvh3f*);
BVH_API void bvh2d_remove_last_node(struct bvh2d*);
BVH_API void bvh3d_remove_last_node(struct bvh3d*);

BVH_API void bvh2f_refit(struct bvh2f*);
BVH_API void bvh3f_refit(struct bvh3f*);
BVH_API void bvh2d_refit(struct bvh2d*);
BVH_API void bvh3d_refit(struct bvh3d*);

BVH_API void bvh2f_optimize(struct bvh_thread_pool*, struct bvh2f*);
BVH_API void bvh3f_optimize(struct bvh_thread_pool*, struct bvh3f*);
BVH_API void bvh2d_optimize(struct bvh_thread_pool*, struct bvh2d*);
BVH_API void bvh3d_optimize(struct bvh_thread_pool*, struct bvh3d*);

// BVH Intersection -------------------------------------------------------------------------------

// Intersection routines: Intersects the BVH with a ray using a callback to intersect the primitives
// contained in the leaves.
//
// The callback takes a pointer to user data, a pointer to the current distance along the ray, and a
// range of primitives. It returns true when an intersection is found, in which case it writes the
// intersection distance in the provided pointer, otherwise, if no intersection is found, it returns
// false and leaves the intersection distance unchanged. The given range is expressed in terms of
// *BVH primitives*, which means that it does not correspond to a range in the original set of
// primitives. Instead, the user has two options: pre-permute the primitives according to the BVH
// primitive indices, or perform an indirection everytime a primitive is accessed.
//
// Here is a basic example of an intersection callback:
//
//     struct my_user_data {
//         struct bvh3f* bvh;
//         struct bvh_ray3f* ray;
//         struct my_prim* prims;
//     };
//
//     struct my_prim_hit {
//         float t;
//     };
//
//     // Declared & defined somewhere else.
//     bool intersect_prim(const struct my_prim*, const struct bvh_ray3f*, struct my_prim_hit*);
//
//     bool my_user_fn(void* user_data, float* t, size_t begin, size_t end) {
//         struct my_user_data* my_user_data = (my_user_data)data;
//         bool was_hit = false;
//         for (size_t i = begin; i < end; ++i) {
//             const size_t prim_id = bvh3f_get_prim_id(my_user_data->bvh, i);
//             struct my_prim_hit hit;
//             if (intersect_prim(&my_user_data->prims[i], my_user_data->ray, &hit)) {
//                 // Note: It is important to remember the maximum distance so that the BVH
//                 // traversal routine can cull nodes that are too far away, and so that the primitive
//                 // intersection routine can exit earlier when that is possible.
//                 *t = my_user_data->ray->tmax = hit.t;
//                 was_hit = true;
//             }
//         }
//         return was_hit;
//     }
//

BVH_API void bvh2f_intersect_ray_any(const struct bvh2f*, const struct bvh_ray2f*, const struct bvh_intersect_callbackf*);
BVH_API void bvh3f_intersect_ray_any(const struct bvh3f*, const struct bvh_ray3f*, const struct bvh_intersect_callbackf*);
BVH_API void bvh2d_intersect_ray_any(const struct bvh2d*, const struct bvh_ray2d*, const struct bvh_intersect_callbackd*);
BVH_API void bvh3d_intersect_ray_any(const struct bvh3d*, const struct bvh_ray3d*, const struct bvh_intersect_callbackd*);

BVH_API void bvh2f_intersect_ray_any_robust(const struct bvh2f*, const struct bvh_ray2f*, const struct bvh_intersect_callbackf*);
BVH_API void bvh3f_intersect_ray_any_robust(const struct bvh3f*, const struct bvh_ray3f*, const struct bvh_intersect_callbackf*);
BVH_API void bvh2d_intersect_ray_any_robust(const struct bvh2d*, const struct bvh_ray2d*, const struct bvh_intersect_callbackd*);
BVH_API void bvh3d_intersect_ray_any_robust(const struct bvh3d*, const struct bvh_ray3d*, const struct bvh_intersect_callbackd*);

BVH_API void bvh2f_intersect_ray(const struct bvh2f*, const struct bvh_ray2f*, const struct bvh_intersect_callbackf*);
BVH_API void bvh3f_intersect_ray(const struct bvh3f*, const struct bvh_ray3f*, const struct bvh_intersect_callbackf*);
BVH_API void bvh2d_intersect_ray(const struct bvh2d*, const struct bvh_ray2d*, const struct bvh_intersect_callbackd*);
BVH_API void bvh3d_intersect_ray(const struct bvh3d*, const struct bvh_ray3d*, const struct bvh_intersect_callbackd*);

BVH_API void bvh2f_intersect_ray_robust(const struct bvh2f*, const struct bvh_ray2f*, const struct bvh_intersect_callbackf*);
BVH_API void bvh3f_intersect_ray_robust(const struct bvh3f*, const struct bvh_ray3f*, const struct bvh_intersect_callbackf*);
BVH_API void bvh2d_intersect_ray_robust(const struct bvh2d*, const struct bvh_ray2d*, const struct bvh_intersect_callbackd*);
BVH_API void bvh3d_intersect_ray_robust(const struct bvh3d*, const struct bvh_ray3d*, const struct bvh_intersect_callbackd*);

#ifdef __cplusplus
}
#endif

#endif

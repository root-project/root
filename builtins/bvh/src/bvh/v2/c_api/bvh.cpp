#include <bvh/v2/c_api/bvh.h>
#include <bvh/v2/c_api/bvh_impl.h>
#include <bvh/v2/thread_pool.h>

namespace bvh::v2::c_api {

BVH_TYPES(float, 2, 2f)
BVH_TYPES(float, 3, 3f)
BVH_TYPES(double, 2, 2d)
BVH_TYPES(double, 3, 3d)

extern "C" {

BVH_EXPORT struct bvh_thread_pool* bvh_thread_pool_create(size_t thread_count) {
    return reinterpret_cast<bvh_thread_pool*>(new bvh::v2::ThreadPool(thread_count));
}

BVH_EXPORT void bvh_thread_pool_destroy(bvh_thread_pool* thread_pool) {
    return delete reinterpret_cast<bvh::v2::ThreadPool*>(thread_pool);
}

BVH_IMPL(float, 2, 2f)
BVH_IMPL(float, 3, 3f)
BVH_IMPL(double, 2, 2d)
BVH_IMPL(double, 3, 3d)

} // extern "C"
} // namespace bvh::v2::c_api

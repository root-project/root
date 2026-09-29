#ifndef BVH_V2_SPHERE_H
#define BVH_V2_SPHERE_H

#include "bvh/v2/vec.h"
#include "bvh/v2/ray.h"

#include <limits>
#include <utility>
#include <optional>

namespace bvh::v2 {

/// Sphere primitive defined by a center and a radius.
template <typename T, size_t N>
struct Sphere  {
    Vec<T, N> center;
    T radius;

    BVH_ALWAYS_INLINE Sphere() = default;
    BVH_ALWAYS_INLINE Sphere(const Vec<T, N>& center, T radius)
        : center(center), radius(radius)
    {}

    BVH_ALWAYS_INLINE Vec<T, N> get_center() const { return center; }
    BVH_ALWAYS_INLINE BBox<T, N> get_bbox() const {
        return BBox<T, N>(center - Vec<T, N>(radius), center + Vec<T, N>(radius));
    }

    /// Intersects a ray with the sphere. If the ray is normalized, a dot product can be saved by
    /// setting `AssumeNormalized` to true.
    template <bool AssumeNormalized = false>
    [[nodiscard]] BVH_ALWAYS_INLINE std::optional<std::pair<T, T>> intersect(const Ray<T, N>& ray) const {
        auto oc = ray.org - center;
        auto a = AssumeNormalized ? static_cast<T>(1.) : dot(ray.dir, ray.dir);
        auto b = static_cast<T>(2.) * dot(ray.dir, oc);
        auto c = dot(oc, oc) - radius * radius;

        auto delta = b * b - static_cast<T>(4.) * a * c;
        if (delta >= 0) {
            auto inv = -static_cast<T>(0.5) / a;
            auto sqrt_delta = std::sqrt(delta);
            auto t0 = robust_max((b + sqrt_delta) * inv, ray.tmin);
            auto t1 = robust_min((b - sqrt_delta) * inv, ray.tmax);
            if (t0 <= t1)
                return std::make_optional(std::make_pair(t0, t1));
        }

        return std::nullopt;
    }
};

} // namespace bvh::v2

#endif

#ifndef BVH_V2_RAY_H
#define BVH_V2_RAY_H

#include "bvh/v2/vec.h"

namespace bvh::v2 {

struct Octant {
    uint32_t value = 0;
    static constexpr size_t max_dim = sizeof(value) * CHAR_BIT;

    uint32_t operator [] (size_t i) const { return (value >> i) & uint32_t{1}; }
};

template <typename T, size_t N>
struct Ray {
    Vec<T, N> org, dir;
    T tmin, tmax;

    Ray() = default;
    BVH_ALWAYS_INLINE Ray(
        const Vec<T, N>& org,
        const Vec<T, N>& dir,
        T tmin = 0,
        T tmax = std::numeric_limits<T>::max())
        : org(org), dir(dir), tmin(tmin), tmax(tmax)
    {}

    template <bool SafeInverse = false>
    BVH_ALWAYS_INLINE Vec<T, N> get_inv_dir() const {
        return Vec<T, N>::generate([&] (size_t i) {
            return SafeInverse ? safe_inverse(dir[i]) : static_cast<T>(1) / dir[i];
        });
    }

    BVH_ALWAYS_INLINE Octant get_octant() const {
        static_assert(N <= Octant::max_dim);
        Octant octant;
        static_for<0, N>([&] (size_t i) {
            octant.value |= std::signbit(dir[i]) * (uint32_t{1} << i);
        });
        return octant;
    }

    // Pads the inverse direction according to T. Ize's "Robust BVH ray traversal"
    BVH_ALWAYS_INLINE static Vec<T, N> pad_inv_dir(const Vec<T, N>& inv_dir) {
        return Vec<T, N>::generate([&] (size_t i) { return add_ulp_magnitude(inv_dir[i], 2); });
    }
};

} // namespace bvh::v2

#endif

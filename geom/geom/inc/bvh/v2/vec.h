#ifndef BVH_V2_VEC_H
#define BVH_V2_VEC_H

#include "bvh/v2/utils.h"

#include <cstddef>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace bvh::v2 {

template <typename T, size_t N>
struct Vec {
    T values[N];

    Vec() = default;
    template <typename... Args>
    BVH_ALWAYS_INLINE Vec(T x, T y, Args&&... args) : values { x, y, static_cast<T>(std::forward<Args>(args))... } {}
    BVH_ALWAYS_INLINE explicit Vec(T x) { std::fill(values, values + N, x); }

    template <typename Compare>
    BVH_ALWAYS_INLINE size_t get_best_axis(Compare&& compare) const {
        size_t axis = 0;
        static_for<1, N>([&] (size_t i) { 
            if (compare(values[i], values[axis]))
                axis = i;
        });
        return axis;
    }

    // Note: These functions are designed to be robust to NaNs
    BVH_ALWAYS_INLINE size_t get_largest_axis() const { return get_best_axis(std::greater<T>()); }
    BVH_ALWAYS_INLINE size_t get_smallest_axis() const { return get_best_axis(std::less<T>()); }

    BVH_ALWAYS_INLINE T& operator [] (size_t i) { return values[i]; }
    BVH_ALWAYS_INLINE T operator [] (size_t i) const { return values[i]; }

    template <typename F>
    BVH_ALWAYS_INLINE static Vec<T, N> generate(F&& f) {
        Vec<T, N> v;
        static_for<0, N>([&] (size_t i) { v[i] = f(i); });
        return v;
    }
};

template <typename T, size_t N>
BVH_ALWAYS_INLINE Vec<T, N> operator + (const Vec<T, N>& a, const Vec<T, N>& b) {
    return Vec<T, N>::generate([&] (size_t i) { return a[i] + b[i]; });
}

template <typename T, size_t N>
BVH_ALWAYS_INLINE Vec<T, N> operator - (const Vec<T, N>& a, const Vec<T, N>& b) {
    return Vec<T, N>::generate([&] (size_t i) { return a[i] - b[i]; });
}

template <typename T, size_t N>
BVH_ALWAYS_INLINE Vec<T, N> operator - (const Vec<T, N>& a) {
    return Vec<T, N>::generate([&] (size_t i) { return -a[i]; });
}

template <typename T, size_t N>
BVH_ALWAYS_INLINE Vec<T, N> operator * (const Vec<T, N>& a, const Vec<T, N>& b) {
    return Vec<T, N>::generate([&] (size_t i) { return a[i] * b[i]; });
}

template <typename T, size_t N>
BVH_ALWAYS_INLINE Vec<T, N> operator / (const Vec<T, N>& a, const Vec<T, N>& b) {
    return Vec<T, N>::generate([&] (size_t i) { return a[i] / b[i]; });
}

template <typename T, size_t N>
BVH_ALWAYS_INLINE Vec<T, N> operator * (const Vec<T, N>& a, T b) {
    return Vec<T, N>::generate([&] (size_t i) { return a[i] * b; });
}

template <typename T, size_t N>
BVH_ALWAYS_INLINE Vec<T, N> operator * (T a, const Vec<T, N>& b) {
    return b * a;
}

template <typename T, size_t N>
BVH_ALWAYS_INLINE Vec<T, N> operator / (T a, const Vec<T, N>& b) {
    return Vec<T, N>::generate([&] (size_t i) { return a / b[i]; });
}

template <typename T, size_t N>
BVH_ALWAYS_INLINE Vec<T, N> robust_min(const Vec<T, N>& a, const Vec<T, N>& b) {
    return Vec<T, N>::generate([&] (size_t i) { return robust_min(a[i], b[i]); });
}

template <typename T, size_t N>
BVH_ALWAYS_INLINE Vec<T, N> robust_max(const Vec<T, N>& a, const Vec<T, N>& b) {
    return Vec<T, N>::generate([&] (size_t i) { return robust_max(a[i], b[i]); });
}

template <typename T, size_t N>
BVH_ALWAYS_INLINE T dot(const Vec<T, N>& a, const Vec<T, N>& b) {
    // return std::transform_reduce(a.values, a.values + N, b.values, T(0));
    return std::inner_product(a.values, a.values + N, b.values, T(0));
}

template <typename T>
BVH_ALWAYS_INLINE Vec<T, 3> cross(const Vec<T, 3>& a, const Vec<T, 3>& b) {
    return Vec<T, 3>(
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0]);
}

template <typename T, size_t N>
BVH_ALWAYS_INLINE Vec<T, N> fast_mul_add(const Vec<T, N>& a, const Vec<T, N>& b, const Vec<T, N>& c) {
    return Vec<T, N>::generate([&] (size_t i) { return fast_mul_add(a[i], b[i], c[i]); });
}

template <typename T, size_t N>
BVH_ALWAYS_INLINE Vec<T, N> safe_inverse(const Vec<T, N>& v) {
    return Vec<T, N>::generate([&] (size_t i) { return safe_inverse(v[i]); });
}

template <typename T, size_t N>
BVH_ALWAYS_INLINE T length(const Vec<T, N>& v) {
    return std::sqrt(dot(v, v));
}

template <typename T, size_t N>
BVH_ALWAYS_INLINE Vec<T, N> normalize(const Vec<T, N>& v) {
    return v * (static_cast<T>(1.) / length(v));
}

} // namespace bvh::v2

#endif

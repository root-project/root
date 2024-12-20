#ifndef BVH_V2_UTILS_H
#define BVH_V2_UTILS_H

#include "bvh/v2/platform.h"

#include <limits>
#include <climits>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <atomic>

namespace bvh::v2 {

/// Helper type that gives an unsigned integer type with the given number of bits.
template <size_t Bits>
struct UnsignedInt {};

template <> struct UnsignedInt< 8> { using Type = uint8_t ; };
template <> struct UnsignedInt<16> { using Type = uint16_t; };
template <> struct UnsignedInt<32> { using Type = uint32_t; };
template <> struct UnsignedInt<64> { using Type = uint64_t; };

template <size_t Bits>
using UnsignedIntType = typename UnsignedInt<Bits>::Type;

/// Helper callable object that just ignores its arguments and returns nothing.
struct IgnoreArgs {
    template <typename... Args>
    void operator () (Args&&...) const {}
};

/// Generates a bitmask with the given number of bits.
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
BVH_ALWAYS_INLINE constexpr T make_bitmask(size_t bits) {
    return bits >= std::numeric_limits<T>::digits ? static_cast<T>(-1) : (static_cast<T>(1) << bits) - 1;
}

// These two functions are designed to return the second argument if the first one is a NaN.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
BVH_ALWAYS_INLINE T robust_min(T a, T b) { return a < b ? a : b; }
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
BVH_ALWAYS_INLINE T robust_max(T a, T b) { return a > b ? a : b; }

/// Adds the given number of ULPs (Units in the Last Place) to the given floating-point number.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
BVH_ALWAYS_INLINE T add_ulp_magnitude(T t, unsigned ulp) {
    if (!std::isfinite(t))
        return t;
    UnsignedIntType<sizeof(T) * CHAR_BIT> u;
    std::memcpy(&u, &t, sizeof(T));
    u += ulp;
    std::memcpy(&t, &u, sizeof(T));
    return t;
}

/// Computes the inverse of the given value, always returning a finite value.
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
BVH_ALWAYS_INLINE T safe_inverse(T x) {
    return std::fabs(x) <= std::numeric_limits<T>::epsilon()
        ? std::copysign(std::numeric_limits<T>::max(), x)
        : static_cast<T>(1.) / x;
}

/// Fast multiply-add operation. Should translate into an FMA for architectures that support it. On
/// architectures which do not support FMA in hardware, or on which FMA is slow, this defaults to a
/// regular multiplication followed by an addition.
#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(push)
#pragma float_control(precise, off)
#pragma fp_contract(on)
#endif
template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
BVH_ALWAYS_INLINE T fast_mul_add(T a, T b, T c) {
#ifdef FP_FAST_FMAF
    return std::fma(a, b, c);
#elif defined(__clang__)
    BVH_CLANG_ENABLE_FP_CONTRACT
#endif
    return a * b + c;
}
#if defined(_MSC_VER) && !defined(__clang__)
#pragma float_control(pop)
#endif

/// Executes the given function once for every integer in the range `[Begin, End)`.
template <size_t Begin, size_t End, typename F>
BVH_ALWAYS_INLINE void static_for(F&& f) {
    if constexpr (Begin < End) {
        f(Begin);
        static_for<Begin + 1, End>(std::forward<F>(f));
    }
}

/// Computes the (rounded-up) compile-time log in base-2 of an unsigned integer.
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
inline constexpr T round_up_log2(T i, T p = 0) {
    return (static_cast<T>(1) << p) >= i ? p : round_up_log2(i, p + 1);
}

/// Split an unsigned integer such that its bits are spaced by 2 zeros.
/// For instance, split_bits(0b00110010) = 0b000000001001000000001000.
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
BVH_ALWAYS_INLINE T split_bits(T x) {
    constexpr size_t bit_count = sizeof(T) * CHAR_BIT;
    constexpr size_t log_bits = round_up_log2(bit_count);
    auto mask = static_cast<T>(-1) >> (bit_count / 2);
    x &= mask;
    for (size_t i = log_bits - 1, n = size_t{1} << i; i > 0; --i, n >>= 1) {
        mask = (mask | (mask << n)) & ~(mask << (n / 2));
        x = (x | (x << n)) & mask;
    }
    return x;
}

/// Morton-encode three unsigned integers into one.
template <typename T, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
BVH_ALWAYS_INLINE T morton_encode(T x, T y, T z) {
    return split_bits(x) | (split_bits(y) << 1) | (split_bits(z) << 2);
}

/// Computes the maximum between an atomic variable and a value, and returns the value previously
/// held by the atomic variable.
template <typename T>
BVH_ALWAYS_INLINE T atomic_max(std::atomic<T>& atomic, const T& value) {
    auto prev_value = atomic;
    while (prev_value < value && !atomic.compare_exchange_weak(prev_value, value));
    return prev_value;
}

} // namespace bvh::v2

#endif

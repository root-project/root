/// \file
/// \warning This file contains implementation details that will change without notice. User code should never include
/// this header directly.

#ifndef ROOT_RHistUtils
#define ROOT_RHistUtils

#include <type_traits>

#ifdef _MSC_VER
#include <intrin.h>
#endif

namespace ROOT {
namespace Experimental {
namespace Internal {

template <typename T, typename... Ts>
struct LastType : LastType<Ts...> {};
template <typename T>
struct LastType<T> {
   using type = T;
};

#ifdef _MSC_VER
namespace MSVC {
template <std::size_t N>
struct AtomicOps {};

template <>
struct AtomicOps<1> {
   static void Load(const void *ptr, void *ret)
   {
      *static_cast<char *>(ret) = __iso_volatile_load8(static_cast<const char *>(ptr));
   }
   static void Add(void *ptr, const void *val)
   {
      _InterlockedExchangeAdd8(static_cast<char *>(ptr), *static_cast<const char *>(val));
   }
   static bool CompareExchange(void *ptr, void *expected, const void *desired)
   {
      // MSVC functions have the arguments reversed...
      const char expectedVal = *static_cast<char *>(expected);
      const char desiredVal = *static_cast<const char *>(desired);
      const char previous = _InterlockedCompareExchange8(static_cast<char *>(ptr), desiredVal, expectedVal);
      if (previous == expectedVal) {
         return true;
      }
      *static_cast<char *>(expected) = previous;
      return false;
   }
};

template <>
struct AtomicOps<2> {
   static void Load(const void *ptr, void *ret)
   {
      *static_cast<short *>(ret) = __iso_volatile_load16(static_cast<const short *>(ptr));
   }
   static void Add(void *ptr, const void *val)
   {
      _InterlockedExchangeAdd16(static_cast<short *>(ptr), *static_cast<const short *>(val));
   }
   static bool CompareExchange(void *ptr, void *expected, const void *desired)
   {
      // MSVC functions have the arguments reversed...
      const short expectedVal = *static_cast<short *>(expected);
      const short desiredVal = *static_cast<const short *>(desired);
      const short previous = _InterlockedCompareExchange16(static_cast<short *>(ptr), desiredVal, expectedVal);
      if (previous == expectedVal) {
         return true;
      }
      *static_cast<short *>(expected) = previous;
      return false;
   }
};

template <>
struct AtomicOps<4> {
   static void Load(const void *ptr, void *ret)
   {
      *static_cast<int *>(ret) = __iso_volatile_load32(static_cast<const int *>(ptr));
   }
   static void Add(void *ptr, const void *val)
   {
      _InterlockedExchangeAdd(static_cast<long *>(ptr), *static_cast<const long *>(val));
   }
   static bool CompareExchange(void *ptr, void *expected, const void *desired)
   {
      // MSVC functions have the arguments reversed...
      const long expectedVal = *static_cast<long *>(expected);
      const long desiredVal = *static_cast<const long *>(desired);
      const long previous = _InterlockedCompareExchange(static_cast<long *>(ptr), desiredVal, expectedVal);
      if (previous == expectedVal) {
         return true;
      }
      *static_cast<long *>(expected) = previous;
      return false;
   }
};

template <>
struct AtomicOps<8> {
   static void Load(const void *ptr, void *ret)
   {
      *static_cast<__int64 *>(ret) = __iso_volatile_load64(static_cast<const __int64 *>(ptr));
   }
   static void Add(void *ptr, const void *val);
   static bool CompareExchange(void *ptr, void *expected, const void *desired)
   {
      // MSVC functions have the arguments reversed...
      const __int64 expectedVal = *static_cast<__int64 *>(expected);
      const __int64 desiredVal = *static_cast<const __int64 *>(desired);
      const __int64 previous = _InterlockedCompareExchange64(static_cast<__int64 *>(ptr), desiredVal, expectedVal);
      if (previous == expectedVal) {
         return true;
      }
      *static_cast<__int64 *>(expected) = previous;
      return false;
   }
};
} // namespace MSVC
#endif

template <typename T>
void AtomicLoad(const T *ptr, T *ret)
{
#ifndef _MSC_VER
   __atomic_load(ptr, ret, __ATOMIC_RELAXED);
#else
   MSVC::AtomicOps<sizeof(T)>::Load(ptr, ret);
#endif
}

template <typename T>
bool AtomicCompareExchange(T *ptr, T *expected, T *desired)
{
#ifndef _MSC_VER
   return __atomic_compare_exchange(ptr, expected, desired, /*weak=*/false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);
#else
   return MSVC::AtomicOps<sizeof(T)>::CompareExchange(ptr, expected, desired);
#endif
}

template <typename T>
void AtomicAddCompareExchangeLoop(T *ptr, T val)
{
   T expected;
   AtomicLoad(ptr, &expected);
   T desired = expected + val;
   while (!AtomicCompareExchange(ptr, &expected, &desired)) {
      // expected holds the new value; try again.
      desired = expected + val;
   }
}

#ifdef _MSC_VER
namespace MSVC {
inline void AtomicOps<8>::Add(void *ptr, const void *val)
{
#if _WIN64
   _InterlockedExchangeAdd64(static_cast<__int64 *>(ptr), *static_cast<const __int64 *>(val));
#else
   AtomicAddCompareExchangeLoop(static_cast<__int64 *>(ptr), *static_cast<const __int64 *>(val));
#endif
}
} // namespace MSVC
#endif

template <typename T>
std::enable_if_t<std::is_integral_v<T>> AtomicAdd(T *ptr, T val)
{
#ifndef _MSC_VER
   __atomic_fetch_add(ptr, val, __ATOMIC_RELAXED);
#else
   MSVC::AtomicOps<sizeof(T)>::Add(ptr, &val);
#endif
}

template <typename T>
std::enable_if_t<std::is_floating_point_v<T>> AtomicAdd(T *ptr, T val)
{
   AtomicAddCompareExchangeLoop(ptr, val);
}

// For adding a double-precision weight to a single-precision bin content type, cast the argument once before the
// compare-exchange loop.
static inline void AtomicAdd(float *ptr, double val)
{
   AtomicAdd(ptr, static_cast<float>(val));
}

template <typename T>
std::enable_if_t<std::is_arithmetic_v<T>> AtomicInc(T *ptr)
{
   AtomicAdd(ptr, static_cast<T>(1));
}

template <typename T, typename U>
auto AtomicAdd(T *ptr, const U &add) -> decltype(ptr->AtomicAdd(add))
{
   return ptr->AtomicAdd(add);
}

template <typename T>
auto AtomicInc(T *ptr) -> decltype(ptr->AtomicInc())
{
   return ptr->AtomicInc();
}

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif

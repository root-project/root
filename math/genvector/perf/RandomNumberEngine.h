
#include <random>
#include <type_traits>

namespace fallbacks {
template <typename T>
using Allocator = std::allocator<T>;
}
namespace Vc_1 {
// Tell ADL to resolve missing Vc entities from fallbacks namespace.
using namespace fallbacks;
} // namespace Vc_1
namespace Vc = Vc_1;

namespace ROOT {
namespace Benchmark {
static std::mt19937 ggen(0);

// FIXME: Those should be taking T, so that we could test against floats.
static std::uniform_real_distribution<double> p_x(1, 2), p_y(2, 3), p_z(3, 4);
static std::uniform_real_distribution<double> d_x(1, 2), d_y(2, 3), d_z(3, 4);
static std::uniform_real_distribution<double> c_x(1, 2), c_y(2, 3), c_z(3, 4);
static std::uniform_real_distribution<double> p0(-0.002, 0.002), p1(-0.2, 0.2), p2(0.97, 0.99), p3(-1300, 1300);

// Make this a weak symbol in order to keep it in the header.
// FIXME: Move in a source file if we have more utility functions like this.
inline void SetSeed(int N)
{
   ggen.seed(N);
   // We do not want our distributions to depend on previous state.
   p_x.reset();
   p_y.reset();
   p_z.reset();
   d_x.reset();
   d_y.reset();
   d_z.reset();
   c_x.reset();
   c_y.reset();
   c_z.reset();
   p0.reset();
   p1.reset();
   p2.reset();
   p3.reset();
}

template <typename T>
class TypeSize {
   template <typename C>
   static constexpr size_t Get(decltype(&C::Size))
   {
      return C::Size;
   }
   template <typename C>
   static constexpr size_t Get(typename std::enable_if<std::is_arithmetic<C>::value>::type * = 0)
   {
      return 1;
   }
   // static constexpr size_t Get(...) { return sizeof(C); }
   // template <typename C>
   // static constexpr size_t Get(...) { static_assert (0, "Works on vector and arithmetic types!"); return -1; }
public:
   static constexpr size_t Get() { return Get<T>(0); }
};

template <typename T>
struct VecTraits {
   // FIXME: Deduce the underlying vector type instead of hardcoding it to double.
   static constexpr double Sum(double value) { return value; }
   template <typename C = T, typename = typename std::enable_if<!std::is_arithmetic<C>::value>::type>
   static double Sum(C value)
   {
      double sum = 0.;
      for (size_t i = 0, e = TypeSize<T>::Get(); i < e; ++i)
         sum += value[i];
      return sum;
   }
   bool is_aligned(const void *__restrict__ ptr, size_t align) { return (uintptr_t)ptr % align == 0; }
};

// FIXME: Generalize this to accomodate N elements, this way we could use it as *the* data token for benchmarking.
template <typename T>
struct Data {
   T X;
   T Y;
   T Z;
   typedef std::vector<Data, Vc::Allocator<Data>> Vector;
   // Fill with random data.
   template <typename U = T>
   Data(typename std::enable_if<std::is_arithmetic<U>::value>::type * = 0) : X(p_x(ggen)), Y(p_y(ggen)), Z(p_z(ggen))
   {
   }
   // FIXME: We cannot use std::is_array but we could check if there is subscript operator
   template <typename U = T>
   Data(typename std::enable_if<!std::is_arithmetic<U>::value>::type * = 0)
   {
      for (size_t i = 0, e = TypeSize<T>::Get(); i < e; ++i) {
         X[i] = p_x(ggen);
         Y[i] = p_y(ggen);
         Z[i] = p_z(ggen);
      }
   }
};
} // namespace Benchmark
} // namespace ROOT

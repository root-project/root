
#include <random>

namespace ROOT {
namespace Benchmark {
static std::default_random_engine ggen;

// FIXME: Those should be taking T, so that we could test against floats.
static std::uniform_real_distribution<double> p_x(1, 2), p_y(2, 3), p_z(3, 4);
static std::uniform_real_distribution<double> d_x(1, 2), d_y(2, 3), d_z(3, 4);
static std::uniform_real_distribution<double> c_x(1, 2), c_y(2, 3), c_z(3, 4);
static std::uniform_real_distribution<double> p0(-0.002, 0.002), p1(-0.2, 0.2), p2(0.97, 0.99), p3(-1300, 1300);

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

// FIXME: Generalize this to accomodate N elements, this way we could use it as *the* data token for benchmarking.
template <typename T>
struct Data {
   T X;
   T Y;
   T Z;
   typedef std::vector<Data, Vc::Allocator<Data>> Vector;
   // Fill with random data.
   Data() : X(p_x(ggen)), Y(p_y(ggen)), Z(p_z(ggen)) {}
};
// template <typename INDATA, typename OUTDATA>
// void clone(const INDATA &in, OUTDATA &out)
// {
//    out.clear();
//    out.reserve(in.size());
//    for (const auto &i : in) {
//       out.emplace_back(i);
//    }
// }
} // namespace Benchmark
} // namespace ROOT

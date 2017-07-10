#include "Math/Cartesian3D.h"
#include "Math/Point3D.h"
#include "Math/Vector3D.h"
#include "Math/Vector4D.h"
#include "Math/GenVector/Rotation3D.h"
#include "Math/GenVector/EulerAngles.h"
#include "Math/GenVector/AxisAngle.h"
#include "Math/GenVector/Quaternion.h"
#include "Math/GenVector/RotationX.h"
#include "Math/GenVector/RotationY.h"
#include "Math/GenVector/RotationZ.h"
#include "Math/GenVector/RotationZYX.h"
#include "Math/GenVector/LorentzRotation.h"
#include "Math/GenVector/Boost.h"
#include "Math/GenVector/BoostX.h"
#include "Math/GenVector/BoostY.h"
#include "Math/GenVector/BoostZ.h"
#include "Math/GenVector/Transform3D.h"
#include "Math/GenVector/Plane3D.h"
#include "Math/GenVector/VectorUtil.h"

#include "Math/Math_vectypes.hxx"

#include "benchmark/benchmark.h"

#include <random>

static bool is_aligned(const void *__restrict__ ptr, size_t align)
{
   return (uintptr_t)ptr % align == 0;
}

template <typename T>
using Point = ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<T>, ROOT::Math::DefaultCoordinateSystemTag>;
template <typename T>
using Vector = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<T>, ROOT::Math::DefaultCoordinateSystemTag>;
template <typename T>
using Plane = ROOT::Math::Impl::Plane3D<T>;

static std::default_random_engine ggen;

static std::uniform_real_distribution<double> p_x(1, 2), p_y(2, 3), p_z(3, 4);
static std::uniform_real_distribution<double> d_x(1, 2), d_y(2, 3), d_z(3, 4);
static std::uniform_real_distribution<double> c_x(1, 2), c_y(2, 3), c_z(3, 4);
static std::uniform_real_distribution<double> p0(-0.002, 0.002), p1(-0.2, 0.2), p2(0.97, 0.99), p3(-1300, 1300);

static void BM_Cartesian3D_CreateEmpty(benchmark::State &state)
{
   while (state.KeepRunning()) ROOT::Math::Cartesian3D<ROOT::Double_v> c;
}
BENCHMARK(BM_Cartesian3D_CreateEmpty);

template <typename T>
static void BM_Cartesian3D_Theta(benchmark::State &state)
{
   ROOT::Math::Cartesian3D<T> c(1., 2., 3.);
   // std::cout << is_aligned(&c, 16) << std::endl;
   while (state.KeepRunning()) c.Theta();
}
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, double)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, ROOT::Double_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, float)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Theta, ROOT::Float_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);

template <typename T>
static void BM_Cartesian3D_Phi(benchmark::State &state)
{
   ROOT::Math::Cartesian3D<T> c(1., 2., 3.);
   // std::cout << is_aligned(&c, 16) << std::endl;
   while (state.KeepRunning()) c.Phi();
   state.SetComplexityN(state.range(0));
}
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, double)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, ROOT::Double_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, float)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Phi, ROOT::Float_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);

template <typename T>
static void BM_Cartesian3D_Mag2(benchmark::State &state)
{

   ROOT::Math::Cartesian3D<T> c(1., 2., 3.);
   // std::cout << is_aligned(&c, 16) << std::endl;
   while (state.KeepRunning()) c.Mag2();
}
BENCHMARK_TEMPLATE(BM_Cartesian3D_Mag2, double)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Cartesian3D_Mag2, ROOT::Double_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);

template <typename T>
static void BM_Point3D(benchmark::State &state)
{
   while (state.KeepRunning()) Point<T> sp1, sp2, sp3, sp4, sp5, sp6;
}

BENCHMARK_TEMPLATE(BM_Point3D, double)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Point3D, ROOT::Double_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);

template <typename T>
static void BM_Point3D_Gen(benchmark::State &state)
{
   while (state.KeepRunning()) {
      Point<T> sp1(p_x(ggen), p_y(ggen), p_z(ggen));
   }
}
BENCHMARK_TEMPLATE(BM_Point3D_Gen, double)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Point3D_Gen, ROOT::Double_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);

template <typename T>
static void BM_Transform3D(benchmark::State &state)
{
   Point<T> sp1(p_x(ggen), p_y(ggen), p_z(ggen));
   Point<T> sp2(p_x(ggen), p_y(ggen), p_z(ggen));
   Point<T> sp3(p_x(ggen), p_y(ggen), p_z(ggen));
   Point<T> sp4(p_x(ggen), p_y(ggen), p_z(ggen));
   Point<T> sp5(p_x(ggen), p_y(ggen), p_z(ggen));
   Point<T> sp6(p_x(ggen), p_y(ggen), p_z(ggen));
   while (state.KeepRunning()) {
      ROOT::Math::Impl::Transform3D<T> st(sp1, sp2, sp3, sp4, sp5, sp6);
      st.Translation();
   }
}
// BENCHMARK_TEMPLATE(BM_Transform3D, double)->Range(8, 8<<10)->Complexity(benchmark::o1);
// BENCHMARK_TEMPLATE(BM_Transform3D, float)->Range(8, 8<<10)->Complexity(benchmark::o1);
// BENCHMARK_TEMPLATE(BM_Transform3D, ROOT::Double_v)->Range(8, 8<<10)->Complexity(benchmark::o1);
// BENCHMARK_TEMPLATE(BM_Transform3D, ROOT::Float_v)->Range(8, 8<<10)->Complexity(benchmark::o1);

template <typename T>
static void BM_Plane3D(benchmark::State &state)
{
   const double a(p0(ggen)), b(p1(ggen)), c(p2(ggen)), d(p3(ggen));
   while (state.KeepRunning()) {
      Plane<T> sc_plane(a, b, c, d);
   }
}
BENCHMARK_TEMPLATE(BM_Plane3D, double)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Plane3D, ROOT::Double_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Plane3D, ROOT::Float_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);

template <typename T>
static void BM_Plane3D_Hessian(benchmark::State &state)
{
   const double a(p0(ggen)), b(p1(ggen)), c(p2(ggen)), d(p3(ggen));
   while (state.KeepRunning()) {
      Plane<T> sc_plane(a, b, c, d);
      sc_plane.HesseDistance();
   }
}
BENCHMARK_TEMPLATE(BM_Plane3D_Hessian, double)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Plane3D_Hessian, ROOT::Double_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);

template <typename T>
static void BM_Plane3D_Normal(benchmark::State &state)
{
   const double a(p0(ggen)), b(p1(ggen)), c(p2(ggen)), d(p3(ggen));
   while (state.KeepRunning()) {
      Plane<T> sc_plane(a, b, c, d);
      sc_plane.Normal();
   }
}
BENCHMARK_TEMPLATE(BM_Plane3D_Normal, double)->Range(8, 8 << 10)->Complexity(benchmark::o1);
;
BENCHMARK_TEMPLATE(BM_Plane3D_Normal, ROOT::Double_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Plane3D_Normal, ROOT::Float_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);

// Define our main.
BENCHMARK_MAIN();

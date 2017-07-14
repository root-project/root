#include "Math/GenVector/Transform3D.h"

#include "Math/Math_vectypes.hxx"

#include "RandomNumberEngine.h"

#include "benchmark/benchmark.h"

template <typename T>
using Point = ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<T>, ROOT::Math::DefaultCoordinateSystemTag>;
template <typename T>
using Vector = ROOT::Math::DisplacementVector3D<ROOT::Math::Cartesian3D<T>, ROOT::Math::DefaultCoordinateSystemTag>;
template <typename T>
using Plane = ROOT::Math::Impl::Plane3D<T>;

Point<T> sp1(p_x(ggen), p_y(ggen), p_z(ggen));
Point<T> sp2(p_x(ggen), p_y(ggen), p_z(ggen));
Point<T> sp3(p_x(ggen), p_y(ggen), p_z(ggen));
Point<T> sp4(p_x(ggen), p_y(ggen), p_z(ggen));
Point<T> sp5(p_x(ggen), p_y(ggen), p_z(ggen));
Point<T> sp6(p_x(ggen), p_y(ggen), p_z(ggen));

template <typename T>
static void BM_Transform3D(benchmark::State &state)
{
   while (state.KeepRunning()) {
      ROOT::Math::Impl::Transform3D<T> st(sp1, sp2, sp3, sp4, sp5, sp6);
      st.Translation();
   }
}
BENCHMARK_TEMPLATE(BM_Transform3D, double)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Transform3D, float)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Transform3D, ROOT::Double_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);
BENCHMARK_TEMPLATE(BM_Transform3D, ROOT::Float_v)->Range(8, 8 << 10)->Complexity(benchmark::o1);

template <typename T>
using Point = ROOT::Math::PositionVector3D<ROOT::Math::Cartesian3D<T>, ROOT::Math::DefaultCoordinateSystemTag>;
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

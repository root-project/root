// A unit test for TGeoTessellated, ported from existing test in VecGeom
// Sandro Wenzel

#include <gtest/gtest.h>

#include <TGeoTessellated.h>
#include <TGeoShape.h>

#include <array>
#include <cmath>
#include <iostream>
#include <memory>

namespace {

using Vtx = TGeoTessellated::Vertex_t;

constexpr double kTol = 1e-12;

// ----------------------------- helpers ----------------------------------------

inline std::array<double, 3> P(double x, double y, double z)
{
   return {x, y, z};
}

inline std::array<double, 3> Unit(const std::array<double, 3> &v)
{
   const double n = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
   if (n == 0.0)
      return {0.0, 0.0, 0.0};
   return {v[0] / n, v[1] / n, v[2] / n};
}

inline void ExpectNearVec3(const std::array<double, 3> &a, const std::array<double, 3> &b, double tol = kTol)
{
   EXPECT_NEAR(a[0], b[0], tol);
   EXPECT_NEAR(a[1], b[1], tol);
   EXPECT_NEAR(a[2], b[2], tol);
}

inline std::array<double, 3>
NormalAt(const TGeoShape &s, const std::array<double, 3> &p, const std::array<double, 3> &dir_hint)
{
   double n[3] = {0.0, 0.0, 0.0};
   s.ComputeNormal(p.data(), dir_hint.data(), n);
   return {n[0], n[1], n[2]};
}

// mapping VecGeom style DistanceToOut(p,v) ~ ROOT/TGeo DistFromInside(p,v,...)
inline double
DistanceToOut(const TGeoShape &s, const std::array<double, 3> &p, const std::array<double, 3> &v, int iact = 3)
{
   return s.DistFromInside(p.data(), v.data(), iact, /*step=*/TGeoShape::Big(), /*safe=*/nullptr);
}

// mapping VecGeom style DistanceToIn(p,v) ~ ROOT/TGeo DistFromOutside(p,v,...)
inline double
DistanceToIn(const TGeoShape &s, const std::array<double, 3> &p, const std::array<double, 3> &v, int iact = 3)
{
   return s.DistFromOutside(p.data(), v.data(), iact, /*step=*/TGeoShape::Big(), /*safe=*/nullptr);
}

// Add a quad facet as 2 triangles preserving the winding of (A,B,C,D)
inline void AddQuadAsTriangles(TGeoTessellated &tsl, const Vtx &A, const Vtx &B, const Vtx &C, const Vtx &D)
{
   // Split along diagonal A-C:
   //   (A,B,C) and (A,C,D) keeps the same rotation as the original quad order.
   EXPECT_TRUE(tsl.AddFacet(A, B, C));
   EXPECT_TRUE(tsl.AddFacet(A, C, D));
}

// ------------------------- test case geometry builder -------------------------------
TGeoTessellated *
CreateTrdLikeTessellated_Triangles(const char *name, double x1, double x2, double y1, double y2, double z)
{
   auto *tsl = new TGeoTessellated(name);

   const Vtx tA(-x2, y2, z), tB(-x2, -y2, z), tC(x2, -y2, z), tD(x2, y2, z);
   const Vtx bA(-x1, y1, -z), bB(x1, y1, -z), bC(x1, -y1, -z), bD(-x1, -y1, -z);

   // Top (+z)
   AddQuadAsTriangles(*tsl, tA, tB, tC, tD);
   // Bottom (-z)
   AddQuadAsTriangles(*tsl, bA, bB, bC, bD);
   // Front (-y)
   AddQuadAsTriangles(*tsl, tB, bD, bC, tC);
   // Right (+x)
   AddQuadAsTriangles(*tsl, tC, bC, bB, tD);
   // Back (+y)
   AddQuadAsTriangles(*tsl, tD, bB, bA, tA);
   // Left (-x)
   AddQuadAsTriangles(*tsl, tA, bA, bD, tB);

   tsl->CloseShape(/*check=*/true, /*fixFlipped=*/true, /*verbose=*/false);
   return tsl;
}

} // namespace

TEST(TGeoTessellated, TrdLike_CoreNavigation)
{
   // Representative points (ported)
   const auto pzero = P(0, 0, 0);

   const auto pbigx = P(100, 0, 0);
   const auto pbigy = P(0, 100, 0);
   const auto pbigz = P(0, 0, 100);
   const auto pbigmx = P(-100, 0, 0);
   const auto pbigmy = P(0, -100, 0);
   const auto pbigmz = P(0, 0, -100);

   const auto vx = P(1, 0, 0);
   const auto vy = P(0, 1, 0);
   const auto vz = P(0, 0, 1);
   const auto vmx = P(-1, 0, 0);
   const auto vmy = P(0, -1, 0);
   const auto vmz = P(0, 0, -1);

   const auto vxy = P(1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);

   // Solids (ported)
   std::unique_ptr<TGeoTessellated> tsl1(CreateTrdLikeTessellated_Triangles("Test Box #1", 20, 20, 30, 30, 40));
   std::unique_ptr<TGeoTessellated> tsl2(CreateTrdLikeTessellated_Triangles("Test Trd", 10, 30, 20, 40, 40));
   std::unique_ptr<TGeoTessellated> tsl3(CreateTrdLikeTessellated_Triangles("BABAR Trd", 0.14999999999999999,
                                                                            0.14999999999999999, 24.707000000000001,
                                                                            24.707000000000001, 22.699999999999999));

   ASSERT_TRUE(tsl1->IsDefined());
   ASSERT_TRUE(tsl2->IsDefined());
   ASSERT_TRUE(tsl3->IsDefined());

   // 6 quads -> 12 triangles
   EXPECT_EQ(tsl1->GetNfacets(), 12);
   EXPECT_EQ(tsl2->GetNfacets(), 12);

   // Volume
   EXPECT_NEAR(tsl1->Capacity(), 8.0 * 20.0 * 30.0 * 40.0, 1e-9);

   // "Inside" checks using TGeo convention: Contains(p) -> bool
   EXPECT_TRUE(tsl1->Contains(pzero.data()));
   EXPECT_FALSE(tsl1->Contains(pbigz.data()));

   EXPECT_TRUE(tsl2->Contains(pzero.data()));
   EXPECT_FALSE(tsl2->Contains(pbigz.data()));

   // Surface normals (ComputeNormal) at face centers to avoid edge/corner ambiguity
   {
      const auto px = P(20, 0, 0);
      const auto nx = P(-20, 0, 0);
      const auto py = P(0, 30, 0);
      const auto ny = P(0, -30, 0);
      const auto pz = P(0, 0, 40);
      const auto nz = P(0, 0, -40);

      ExpectNearVec3(NormalAt(*tsl1, px, vx), vx);
      ExpectNearVec3(NormalAt(*tsl1, nx, vmx), vmx);
      ExpectNearVec3(NormalAt(*tsl1, py, vy), vy);
      ExpectNearVec3(NormalAt(*tsl1, ny, vmy), vmy);
      ExpectNearVec3(NormalAt(*tsl1, pz, vz), vz);
      ExpectNearVec3(NormalAt(*tsl1, nz, vmz), vmz);
   }

   const double cosa = 4.0 / std::sqrt(17.0);
   const double sina = 1.0 / std::sqrt(17.0);
   {
      // Use points on the side planes (face centers) matching the tsl1-style coordinates.
      const auto px = P(20, 0, 0);
      const auto nx = P(-20, 0, 0);
      const auto py = P(0, 30, 0);
      const auto ny = P(0, -30, 0);

      ExpectNearVec3(NormalAt(*tsl2, px, vx), P(cosa, 0.0, -sina));
      ExpectNearVec3(NormalAt(*tsl2, nx, vmx), P(-cosa, 0.0, -sina));
      ExpectNearVec3(NormalAt(*tsl2, py, vy), P(0.0, cosa, -sina));
      ExpectNearVec3(NormalAt(*tsl2, ny, vmy), P(0.0, -cosa, -sina));

      // Top/bottom should remain axis-aligned
      const auto pz = P(0, 0, 40);
      const auto nz = P(0, 0, -40);
      ExpectNearVec3(NormalAt(*tsl2, pz, vz), vz);
      ExpectNearVec3(NormalAt(*tsl2, nz, vmz), vmz);
   }

   // DistanceToOut from inside points (TGeo: DistFromInside)
   {
      const double d = DistanceToOut(*tsl1, pzero, vx);
      EXPECT_NEAR(d, 20.0, 1e-9);
      ExpectNearVec3(NormalAt(*tsl1, P(pzero[0] + d * vx[0], pzero[1] + d * vx[1], pzero[2] + d * vx[2]), vx), vx);
   }
   {
      const double d = DistanceToOut(*tsl1, pzero, vy);
      EXPECT_NEAR(d, 30.0, 1e-9);
      ExpectNearVec3(NormalAt(*tsl1, P(pzero[0] + d * vy[0], pzero[1] + d * vy[1], pzero[2] + d * vy[2]), vy), vy);
   }
   {
      const double d = DistanceToOut(*tsl1, pzero, vz);
      EXPECT_NEAR(d, 40.0, 1e-9);
      ExpectNearVec3(NormalAt(*tsl1, P(pzero[0] + d * vz[0], pzero[1] + d * vz[1], pzero[2] + d * vz[2]), vz), vz);
   }
   EXPECT_NEAR(DistanceToOut(*tsl1, pzero, vxy), std::sqrt(800.0), 1e-9);

   // tsl2 (ported expectations)
   {
      const double d = DistanceToOut(*tsl2, pzero, vx);
      EXPECT_NEAR(d, 20.0, 1e-9);
      ExpectNearVec3(NormalAt(*tsl2, P(pzero[0] + d * vx[0], pzero[1] + d * vx[1], pzero[2] + d * vx[2]), vx),
                     P(cosa, 0.0, -sina));
   }
   {
      const double d = DistanceToOut(*tsl2, pzero, vy);
      EXPECT_NEAR(d, 30.0, 1e-9);
      ExpectNearVec3(NormalAt(*tsl2, P(pzero[0] + d * vy[0], pzero[1] + d * vy[1], pzero[2] + d * vy[2]), vy),
                     P(0.0, cosa, -sina));
   }
   {
      const double d = DistanceToOut(*tsl2, pzero, vz);
      EXPECT_NEAR(d, 40.0, 1e-9);
      ExpectNearVec3(NormalAt(*tsl2, P(pzero[0] + d * vz[0], pzero[1] + d * vz[1], pzero[2] + d * vz[2]), vz), vz);
   }
   EXPECT_NEAR(DistanceToOut(*tsl2, pzero, vxy), std::sqrt(800.0), 1e-9);

   // DistanceToIn from outside points (TGeo: DistFromOutside)
   const double kInf = TGeoShape::Big();

   EXPECT_NEAR(DistanceToIn(*tsl1, pbigx, vmx), 80.0, 1e-9);
   EXPECT_NEAR(DistanceToIn(*tsl1, pbigmx, vx), 80.0, 1e-9);
   EXPECT_NEAR(DistanceToIn(*tsl1, pbigy, vmy), 70.0, 1e-9);
   EXPECT_NEAR(DistanceToIn(*tsl1, pbigmy, vy), 70.0, 1e-9);
   EXPECT_NEAR(DistanceToIn(*tsl1, pbigz, vmz), 60.0, 1e-9);
   EXPECT_NEAR(DistanceToIn(*tsl1, pbigmz, vz), 60.0, 1e-9);

   // Miss cases (ported intent)
   EXPECT_DOUBLE_EQ(DistanceToIn(*tsl1, pbigx, vxy), kInf);
   EXPECT_DOUBLE_EQ(DistanceToIn(*tsl1, pbigmx, vxy), kInf);

   EXPECT_NEAR(DistanceToIn(*tsl2, pbigx, vmx), 80.0, 1e-9);
   EXPECT_NEAR(DistanceToIn(*tsl2, pbigmx, vx), 80.0, 1e-9);
   EXPECT_NEAR(DistanceToIn(*tsl2, pbigy, vmy), 70.0, 1e-9);
   EXPECT_NEAR(DistanceToIn(*tsl2, pbigmy, vy), 70.0, 1e-9);
   EXPECT_NEAR(DistanceToIn(*tsl2, pbigz, vmz), 60.0, 1e-9);
   EXPECT_NEAR(DistanceToIn(*tsl2, pbigmz, vz), 60.0, 1e-9);

   EXPECT_DOUBLE_EQ(DistanceToIn(*tsl2, pbigx, vxy), kInf);
   EXPECT_DOUBLE_EQ(DistanceToIn(*tsl2, pbigmx, vxy), kInf);

   // BABAR regression (ported)
   {
      const auto pb = P(0.15000000000000185, -22.048743592955137, 2.4268539333219472);
      const auto vb = Unit(P(-0.76165597579890043, 0.64364445891356026, -0.074515708658524193));
      const double d = DistanceToIn(*tsl3, pb, vb);
      EXPECT_NEAR(d, 0.0, 1e-9);
   }

   // Extent / cached bbox checks (TGeo accessors)
   {
      const double *o = tsl1->GetOrigin();
      EXPECT_NEAR(o[0] - tsl1->GetDX(), -20.0, 1e-9);
      EXPECT_NEAR(o[0] + tsl1->GetDX(), 20.0, 1e-9);
      EXPECT_NEAR(o[1] - tsl1->GetDY(), -30.0, 1e-9);
      EXPECT_NEAR(o[1] + tsl1->GetDY(), 30.0, 1e-9);
      EXPECT_NEAR(o[2] - tsl1->GetDZ(), -40.0, 1e-9);
      EXPECT_NEAR(o[2] + tsl1->GetDZ(), 40.0, 1e-9);
   }
   {
      const double *o = tsl2->GetOrigin();
      EXPECT_NEAR(o[0] - tsl2->GetDX(), -30.0, 1e-9);
      EXPECT_NEAR(o[0] + tsl2->GetDX(), 30.0, 1e-9);
      EXPECT_NEAR(o[1] - tsl2->GetDY(), -40.0, 1e-9);
      EXPECT_NEAR(o[1] + tsl2->GetDY(), 40.0, 1e-9);
      EXPECT_NEAR(o[2] - tsl2->GetDZ(), -40.0, 1e-9);
      EXPECT_NEAR(o[2] + tsl2->GetDZ(), 40.0, 1e-9);
   }
}

int main(int argc, char **argv)
{
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}

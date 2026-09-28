#include "TGeoManager.h"
#include "TGeoPgon.h"
#include "TMath.h"

#include <gtest/gtest.h>

#include <cmath>

namespace {

// ROOT orients normals towards the direction. Check parallelism independently of sign.
void CheckNormal(const TGeoPgon &shape, const double *point, const double *expected)
{
   for (double sign : {-1., 1.}) {
      const double direction[] = {sign * expected[0], sign * expected[1], sign * expected[2]};
      double normal[3];
      shape.ComputeNormal(point, direction, normal);
      EXPECT_NEAR(normal[1] * expected[2] - normal[2] * expected[1], 0., 1.e-12);
      EXPECT_NEAR(normal[2] * expected[0] - normal[0] * expected[2], 0., 1.e-12);
      EXPECT_NEAR(normal[0] * expected[1] - normal[1] * expected[0], 0., 1.e-12);
      EXPECT_NEAR(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2], 1., 1.e-12);
      EXPECT_GT(normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2], 0.);
   }
}

TEST(TGeoPgon, NormalNearPhiCut)
{
   const int verbose = TGeoManager::GetVerboseLevel();
   TGeoManager::SetVerboseLevel(0);
   TGeoManager manager("pgon_normal", "polygon normal tests");
   TGeoManager::SetVerboseLevel(verbose);
   TGeoPgon shape(15., 270., 8, 2);
   shape.DefineSection(0, -15., 3., 10.);
   shape.DefineSection(1, 15., 3., 10.);

   // A recorded radial exit lies about 6.4e-6 cm from the phi plane.
   const double exit[] = {3.0281659171434625, .81140121787309871, 8.3339698762290197};
   const double innerNormal[] = {std::cos(31.875 * TMath::DegToRad()), std::sin(31.875 * TMath::DegToRad()), 0.};
   CheckNormal(shape, exit, innerNormal);

   for (double phi : {15., 285.}) {
      const double angle = phi * TMath::DegToRad();
      const double faceAngle = (phi == 15. ? 31.875 : 268.125) * TMath::DegToRad();
      const double radialNormal[] = {std::cos(faceAngle), std::sin(faceAngle), 0.};
      const double phiNormal[] = {-std::sin(angle), std::cos(angle), 0.};
      for (double offset : {1.e-8, 1.e-6}) {
         SCOPED_TRACE(testing::Message() << "phi=" << phi << ", offset=" << offset);
         const double nearAngle = angle + (phi == 15. ? offset : -offset);
         for (double radius : {3., 10.}) {
            const double r = radius / std::cos(nearAngle - faceAngle);
            const double point[] = {r * std::cos(nearAngle), r * std::sin(nearAngle), 0.};
            CheckNormal(shape, point, radialNormal);
         }
         // A displaced phi point must still choose phi when radial faces are farther away.
         const double phiPoint[] = {6. * std::cos(nearAngle), 6. * std::sin(nearAngle), 0.};
         CheckNormal(shape, phiPoint, phiNormal);
      }
      const double phiPoint[] = {6. * std::cos(angle), 6. * std::sin(angle), 0.};
      CheckNormal(shape, phiPoint, phiNormal);
      const double outsideAngle = angle + (phi == 15. ? -1.e-8 : 1.e-8);
      const double outsidePoint[] = {6. * std::cos(outsideAngle), 6. * std::sin(outsideAngle), 0.};
      CheckNormal(shape, outsidePoint, phiNormal);
   }
}

TEST(TGeoPgon, NormalNearEndCap)
{
   const int verbose = TGeoManager::GetVerboseLevel();
   TGeoManager::SetVerboseLevel(0);
   TGeoManager manager("pgon_cap_normal", "polygon cap normal tests");
   TGeoManager::SetVerboseLevel(verbose);
   TGeoPgon shape(15., 270., 8, 2);
   shape.DefineSection(0, -15., 3., 10.);
   shape.DefineSection(1, 15., 3., 10.);
   const double angle = 15. * TMath::DegToRad();
   const double phiNormal[] = {-std::sin(angle), std::cos(angle), 0.};
   const double capNormal[] = {0., 0., 1.};
   const double faceAngle = 31.875 * TMath::DegToRad();
   const double radialNormal[] = {std::cos(faceAngle), std::sin(faceAngle), 0.};
   for (double sign : {-1., 1.}) {
      const double radialPoint[] = {10. * std::cos(faceAngle), 10. * std::sin(faceAngle), sign * (15. - 1.e-6)};
      CheckNormal(shape, radialPoint, radialNormal);
      const double phiPoint[] = {6. * std::cos(angle), 6. * std::sin(angle), sign * (15. - 1.e-6)};
      CheckNormal(shape, phiPoint, phiNormal);
      for (double gap : {0., 1.e-8}) {
         const double capPoint[] = {6. * std::cos(angle + 1.e-6), 6. * std::sin(angle + 1.e-6), sign * (15. - gap)};
         CheckNormal(shape, capPoint, capNormal);
      }
   }
}

TEST(TGeoPgon, NormalNearSlopedFace)
{
   const int verbose = TGeoManager::GetVerboseLevel();
   TGeoManager::SetVerboseLevel(0);
   TGeoManager manager("pgon_slope_normal", "polygon sloped normal tests");
   TGeoManager::SetVerboseLevel(verbose);
   TGeoPgon shape(15., 270., 8, 2);
   shape.DefineSection(0, -15., 3., 10.);
   shape.DefineSection(1, 15., 3., 40.);
   const double angle = 15. * TMath::DegToRad();
   const double faceAngle = 31.875 * TMath::DegToRad();
   const double radialNormal[] = {std::cos(faceAngle) * std::sqrt(0.5), std::sin(faceAngle) * std::sqrt(0.5),
                                  -std::sqrt(0.5)};
   const double phiNormal[] = {-std::sin(angle), std::cos(angle), 0.};
   // Compare perpendicular distances: the radial separation must include the slope.
   for (double gap : {3.e-6, 6.e-6}) {
      const double r = (25. - gap) / std::cos(angle + 1.e-7 - faceAngle);
      const double point[] = {r * std::cos(angle + 1.e-7), r * std::sin(angle + 1.e-7), 0.};
      CheckNormal(shape, point, gap == 3.e-6 ? radialNormal : phiNormal);
   }
}

TEST(TGeoPgon, NormalAtZDiscontinuity)
{
   const int verbose = TGeoManager::GetVerboseLevel();
   TGeoManager::SetVerboseLevel(0);
   TGeoManager manager("pgon_step_normal", "polygon step normal tests");
   TGeoManager::SetVerboseLevel(verbose);
   TGeoPgon shape(0., 360., 4, 4);
   shape.DefineSection(0, -15., 0., 10.);
   shape.DefineSection(1, 0., 0., 10.);
   shape.DefineSection(2, 0., 0., 6.);
   shape.DefineSection(3, 15., 0., 6.);
   const double radialNormal[] = {std::sqrt(0.5), std::sqrt(0.5), 0.};
   const double capNormal[] = {0., 0., 1.};
   for (double z : {-1.e-6, 0., 1.e-6}) {
      const double ledgePoint[] = {8. * radialNormal[0], 8. * radialNormal[1], z};
      CheckNormal(shape, ledgePoint, capNormal);
      // Within the common cross section, the internal z plane is not a face.
      const double commonPoint[] = {5. * radialNormal[0], 5. * radialNormal[1], z};
      CheckNormal(shape, commonPoint, radialNormal);
   }
}

} // namespace

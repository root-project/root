#include "TGeoArb8.h"
#include "TGeoEltu.h"
#include "TGeoManager.h"
#include "TGeoSphere.h"
#include "TGeoXtru.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <memory>

namespace {

class ShapeSafety : public testing::Test {
protected:
   int fVerbose = TGeoManager::GetVerboseLevel();
   std::unique_ptr<TGeoManager> fManager;

   void SetUp() override
   {
      TGeoManager::SetVerboseLevel(0);
      fManager = std::make_unique<TGeoManager>("safety", "shape safety tests");
   }

   void TearDown() override
   {
      fManager.reset();
      TGeoManager::SetVerboseLevel(fVerbose);
   }
};

void CheckEntrySafety(const TGeoShape &shape, const double *point, const double *direction)
{
   ASSERT_FALSE(shape.Contains(point));
   const double expected = shape.Safety(point, false);
   ASSERT_GT(expected, 0.);
   const double distance = shape.DistFromOutside(point, direction, 3);
   ASSERT_LT(distance, TGeoShape::Big());
   for (int action : {0, 1, 2}) {
      SCOPED_TRACE(action);
      double safety = -1.;
      const double result = shape.DistFromOutside(point, direction, action, TGeoShape::Big(), &safety);
      EXPECT_GE(safety, 0.);
      EXPECT_DOUBLE_EQ(safety, expected);
      EXPECT_LE(safety, distance + 1.e-12);
      EXPECT_DOUBLE_EQ(result, action == 0 ? TGeoShape::Big() : distance);
   }
   double safety = -1.;
   EXPECT_EQ(shape.DistFromOutside(point, direction, 1, 0.5 * expected, &safety), TGeoShape::Big());
   EXPECT_DOUBLE_EQ(safety, expected);
}

TEST_F(ShapeSafety, EltuOutsideEndCaps)
{
   TGeoEltu shape(5., 10., 15.);
   const std::array<std::array<double, 3>, 4> points = {
      {{0., 0., 20.}, {3.741789718811936, -5.653856736807612, -20.911110224363568}, {4., 4., 20.}, {-4., -4., -20.}}};
   for (const auto &point : points) {
      SCOPED_TRACE(testing::PrintToString(point));
      ASSERT_FALSE(shape.Contains(point.data()));
      const double direction[] = {0., 0., point[2] > 0. ? -1. : 1.};
      const double capDistance = std::abs(point[2]) - 15.;
      EXPECT_DOUBLE_EQ(shape.Safety(point.data(), false), capDistance);
      EXPECT_DOUBLE_EQ(shape.DistFromOutside(point.data(), direction, 3), capDistance);
   }
}

TEST_F(ShapeSafety, EltuExitNearLateralSurface)
{
   // A circular section provides an exact radial distance as an independent upper bound.
   for (double b : {5., 10.}) {
      TGeoEltu shape(5., b, 15.);
      for (double gap : {1.e-6, 1.e-4, 0.1}) {
         SCOPED_TRACE(testing::Message() << "b=" << b << ", gap=" << gap);
         const double point[] = {0.6 * (5. - gap), 0.8 * b * (1. - gap / 5.), 0.};
         const double direction[] = {0.6, 0.8, 0.};
         ASSERT_TRUE(shape.Contains(point));
         const double distance = shape.DistFromInside(point, direction, 3);
         if (b == 5.)
            EXPECT_NEAR(distance, gap, 1.e-12);
         const double expected = shape.Safety(point, true);
         ASSERT_GT(expected, 0.);
         for (int action : {0, 1, 2}) {
            SCOPED_TRACE(action);
            double safety = -1.;
            const double result = shape.DistFromInside(point, direction, action, TGeoShape::Big(), &safety);
            EXPECT_GE(safety, 0.);
            EXPECT_LE(safety, distance + 1.e-12);
            EXPECT_DOUBLE_EQ(safety, expected);
            EXPECT_DOUBLE_EQ(result, action == 0 ? TGeoShape::Big() : distance);
         }
         double safety = -1.;
         EXPECT_EQ(shape.DistFromInside(point, direction, 1, 0.5 * expected, &safety), TGeoShape::Big());
         EXPECT_DOUBLE_EQ(safety, expected);
      }
   }
}

TEST_F(ShapeSafety, GtraEntry)
{
   TGeoGtra shape(15., 0., 0., 15., 8., 5., 10., 0., 8., 5., 10., 0.);
   for (double sign : {-1., 1.}) {
      const double point[] = {0., 0., sign * 20.}, direction[] = {0., 0., -sign};
      EXPECT_DOUBLE_EQ(shape.Safety(point, false), 5.);
      CheckEntrySafety(shape, point, direction);
   }
}

TEST_F(ShapeSafety, XtruEntry)
{
   TGeoXtru shape(2);
   const double x[] = {-5., 5., 5., -5.}, y[] = {-10., -10., 10., 10.};
   shape.DefinePolygon(4, x, y);
   shape.DefineSection(0, -15.);
   shape.DefineSection(1, 15.);
   const std::array<std::array<double, 3>, 3> points = {{{0., 0., 20.}, {0., 0., -20.}, {8., 0., 0.}}};
   for (const auto &point : points) {
      const double direction[] = {point[0] > 0. ? -1. : 0., 0., point[2] == 0. ? 0. : -std::copysign(1., point[2])};
      CheckEntrySafety(shape, point.data(), direction);
   }
}

TEST_F(ShapeSafety, SphereEntryThroughThetaCut)
{
   TGeoSphere shape(0., 10., 20., 150., 15., 285.);
   // Outside theta1 but inside the radial and phi ranges. The old optional safety
   // used the sine of a phi separation greater than 180 degrees and became negative.
   const double point[] = {0.5638675685705744, 3.1849154358102036, 9.116139441428437};
   const double direction[] = {-0.7131596062860449, 0.23833579772843216, -0.6592415516964061};
   CheckEntrySafety(shape, point, direction);
}

TEST_F(ShapeSafety, SphereEntryThroughPhiCut)
{
   TGeoSphere shape(0., 10., 0., 180., 15., 285.);
   const double point[] = {5., 0., 0.}, direction[] = {0., 1., 0.};
   CheckEntrySafety(shape, point, direction);
}

TEST_F(ShapeSafety, SphereEntryThroughRadialSurfaces)
{
   TGeoSphere shape(2., 10.);
   const double innerPoint[] = {0., 0., 1.}, innerDirection[] = {0., 0., 1.};
   CheckEntrySafety(shape, innerPoint, innerDirection);
   // Inside the bounding box, so its rejection does not skip the optional safety.
   const double outerPoint[] = {8., 8., 0.}, outerDirection[] = {-std::sqrt(0.5), -std::sqrt(0.5), 0.};
   CheckEntrySafety(shape, outerPoint, outerDirection);
}

} // namespace

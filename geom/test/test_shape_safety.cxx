#include "TGeoEltu.h"
#include "TGeoManager.h"

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

} // namespace

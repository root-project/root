// TGeoVGShape.h used without VecGeom headers, as by clients of libConverterVG: it must declare the VecGeom types the
// library was built with.

#include "TGeoBBox.h"
#include "TGeoVGShape.h"

#include "gtest/gtest.h"

#include <type_traits>
#include <utility>

#if ROOT_TEST_VECGEOM_2_3
// Since VecGeom 2.3 the volume classes are templates on the memory space (host = 0)
static_assert(std::is_same_v<decltype(std::declval<TGeoVGShape const &>().GetVGShape()),
                             vecgeom::VPlacedVolumeT<static_cast<vecgeom::MemSpace>(0)> *>,
              "TGeoVGShape.h does not declare the VecGeom 2.3 placed volume type");
#endif

TEST(VecGeomBridge, ShapeWithoutVecGeomHeaders)
{
   TGeoBBox box("box", 2, 3, 4);
   TGeoVGShape *shape = TGeoVGShape::Create(&box);
   ASSERT_NE(shape, nullptr);
   EXPECT_NE(shape->GetVGShape(), nullptr);
   double origin[3] = {0, 0, 0};
   double outside[3] = {8, 0, 0};
   EXPECT_TRUE(shape->Contains(origin));
   EXPECT_FALSE(shape->Contains(outside));
   delete shape;
}

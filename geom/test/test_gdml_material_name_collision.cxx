#include <gtest/gtest.h>

#include <TGeoElement.h>
#include <TGeoManager.h>
#include <TGeoMaterial.h>

TEST(Geometry, GDMLMaterialElementNameCollision)
{
   auto geom = TGeoManager::Import("material_name_collision.gdml");
   ASSERT_NE(geom, nullptr);

   auto iron = geom->GetMaterial("Iron");
   ASSERT_NE(iron, nullptr);
   ASSERT_TRUE(iron->IsMixture());
   auto ironMixture = static_cast<TGeoMixture *>(iron);
   ASSERT_EQ(iron->GetNelements(), 1);
   EXPECT_STREQ(iron->GetElement(0)->GetName(), "FE");
   EXPECT_DOUBLE_EQ(ironMixture->GetWmixt()[0], 1.0);

   auto alloy = geom->GetMaterial("Alloy");
   ASSERT_NE(alloy, nullptr);
   ASSERT_TRUE(alloy->IsMixture());
   auto alloyMixture = static_cast<TGeoMixture *>(alloy);
   ASSERT_EQ(alloy->GetNelements(), 2);
   EXPECT_STREQ(alloy->GetElement(0)->GetName(), "FE");
   EXPECT_STREQ(alloy->GetElement(1)->GetName(), "NI");
   EXPECT_DOUBLE_EQ(alloyMixture->GetWmixt()[0], 0.5);
   EXPECT_DOUBLE_EQ(alloyMixture->GetWmixt()[1], 0.5);
}

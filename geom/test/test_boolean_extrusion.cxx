#include <gtest/gtest.h>

#include <TGeoManager.h>
#include <TROOT.h>

/**
   Test to verify that child volumes spanning over more than one component of their Boolean parent
   do not give false positives with the overlap checker
   \author Andrei Gheata
*/
TEST(Geometry, NoExtrusionInUnionSpan)
{
   // import a GDML geometry with a policone inside a union of two polycones and a tube
   auto geom = TGeoManager::Import("no_extrusion.gdml");
   EXPECT_NE(geom, nullptr);
   if (!geom) {
      std::cout << "Could not load geometry file no_extrusion.gdml\n";
      return;
   }
   geom->CheckOverlaps(0.001);
   auto num_overlaps = gGeoManager->GetListOfOverlaps()->GetEntries();
   EXPECT_EQ(num_overlaps, 0);
}

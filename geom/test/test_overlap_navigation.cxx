#include <gtest/gtest.h>

#include <TGeoManager.h>
#include <TGeoMaterial.h>
#include <TGeoMedium.h>
#include <TGeoNode.h>
#include <TGeoVolume.h>

#include <memory>

TEST(TGeoNavigator, ManyOverlapIsResolvedFromOnlyDaughter)
{
   auto geom = std::make_unique<TGeoManager>("many_overlap", "MANY overlap navigation test");

   auto *matAir = new TGeoMaterial("Air");
   auto *matAl = new TGeoMaterial("Al");
   auto *matPb = new TGeoMaterial("Pb");
   auto *air = new TGeoMedium("air", 1, matAir);
   auto *al = new TGeoMedium("al", 2, matAl);
   auto *pb = new TGeoMedium("pb", 3, matPb);

   auto *top = geom->MakeBox("TOP", air, 50., 50., 50.);
   auto *mother = geom->MakeBox("M", air, 20., 20., 20.);
   auto *many = geom->MakeBox("A", al, 5., 5., 5.);
   auto *daughter = geom->MakeBox("D", air, 2., 2., 5.);
   auto *onlySibling = geom->MakeBox("B", pb, 2., 2., 3.);

   // M contains overlapping siblings A (MANY) and B (ONLY), while A contains
   // the ONLY daughter D. D covers the full region where B overlaps A:
   //
   // M
   // |-- A (MANY)
   // |   `-- D (ONLY)
   // `-- B (ONLY)
   geom->SetTopVolume(top);
   top->AddNode(mother, 1);
   mother->AddNodeOverlap(many, 1);
   mother->AddNode(onlySibling, 1);
   many->AddNode(daughter, 1);
   geom->CloseGeometry();

   // Point location establishes the expected priority: B wins over A and D.
   ASSERT_EQ(geom->FindNode(0., 0., 0.)->GetVolume(), onlySibling);

   // The first step enters D at z = -5. The non-stepping query must retain
   // that current path while finding B at z = -3 as the next boundary.
   geom->InitTrack(0., 0., -10., 0., 0., 1.);
   ASSERT_STREQ(geom->GetCurrentNode()->GetName(), "M_1");

   geom->FindNextBoundaryAndStep();
   EXPECT_DOUBLE_EQ(geom->GetStep(), 5.);
   ASSERT_STREQ(geom->GetCurrentNode()->GetName(), "D_1");

   auto *next = geom->FindNextBoundary();
   EXPECT_DOUBLE_EQ(geom->GetStep(), 2.);
   ASSERT_NE(next, nullptr);
   EXPECT_EQ(next->GetVolume(), onlySibling);

   // Repeat with the combined API: its second step must commit the sibling
   // transition from D to B rather than crossing all of D.
   geom->InitTrack(0., 0., -10., 0., 0., 1.);
   geom->FindNextBoundaryAndStep();
   ASSERT_STREQ(geom->GetCurrentNode()->GetName(), "D_1");

   geom->FindNextBoundaryAndStep();
   EXPECT_DOUBLE_EQ(geom->GetStep(), 2.);
   EXPECT_STREQ(geom->GetCurrentNode()->GetName(), "B_1");
}

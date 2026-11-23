#include <gtest/gtest.h>

#include <TGeoBBox.h>
#include <TGeoChecker.h>
#include <TGeoCompositeShape.h>
#include <TGeoManager.h>
#include <TGeoMaterial.h>
#include <TGeoMatrix.h>
#include <TGeoMedium.h>
#include <TGeoVolume.h>
#include <TObjArray.h>

#include <memory>
#include <tuple>

class OverlapQuadrant : public testing::TestWithParam<std::tuple<int, int, bool, bool>> {};

TEST_P(OverlapQuadrant, ContainedSibling)
{
   const auto [sx, sy, reverse, translate] = GetParam();
   auto geom = std::make_unique<TGeoManager>("q", "quadrant overlap regression");
   auto *med = new TGeoMedium("vac", 1, new TGeoMaterial("vac", 0, 0, 0));
   auto *top = geom->MakeBox("TOP", med, 200, 200, 200);
   geom->SetTopVolume(top);
   auto *b = geom->MakeBox("B", med, 100, 100, 100);
   double origin[3] = {translate ? 0. : 15. * sx, translate ? 0. : 15. * sy, 0.};
   auto *a = new TGeoVolume("A", new TGeoBBox(5, 5, 5, origin), med);
   auto *placement = new TGeoTranslation(translate ? 15. * sx : 0., translate ? 15. * sy : 0., 0.);
   if (reverse) {
      top->AddNode(a, 1, placement);
      top->AddNode(b, 1);
   } else {
      top->AddNode(b, 1);
      top->AddNode(a, 1, placement);
   }
   geom->CloseGeometry();
   geom->CheckOverlaps(0.1);
   EXPECT_EQ(geom->GetListOfOverlaps()->GetEntriesFast(), 1);

   // Force both candidate orders independently of voxel candidate ordering.
   TGeoOverlapCandidate candidate;
   candidate.fVol1 = reverse ? a : b;
   candidate.fVol2 = reverse ? b : a;
   (reverse ? candidate.fMat1 : candidate.fMat2) = *placement;
   candidate.fIsOverlap = true;
   candidate.fOvlp = 0.1;
   TGeoOverlapResult result;
   TGeoChecker checker(geom.get());
   checker.BuildMeshPointsCache({candidate});
   EXPECT_TRUE(checker.ComputeOverlap(candidate, result));
   EXPECT_FALSE(result.fPoints.empty());
}

INSTANTIATE_TEST_SUITE_P(Quadrants, OverlapQuadrant,
                         testing::Combine(testing::Values(-1, 1), testing::Values(-1, 1), testing::Bool(),
                                          testing::Bool()));

class ExtrusionQuadrant : public testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(ExtrusionQuadrant, MotherHoleInsideDaughter)
{
   const auto [sx, sy] = GetParam();
   auto geom = std::make_unique<TGeoManager>("q", "mother mesh extrusion regression");
   auto *med = new TGeoMedium("vac", 1, new TGeoMaterial("vac", 0, 0, 0));
   new TGeoBBox("outer", 100, 100, 100);
   double origin[3] = {15. * sx, 15. * sy, 0.};
   new TGeoBBox("hole", 5, 5, 5, origin);
   auto *mother = new TGeoVolume("M", new TGeoCompositeShape("outer-hole"), med);
   auto *daughter = new TGeoVolume("D", new TGeoBBox(10, 10, 10, origin), med);
   geom->SetTopVolume(mother);
   mother->AddNode(daughter, 1);
   geom->CloseGeometry();
   // Daughter vertices are all inside the mother material. Only the mother's
   // hole vertices reveal the extrusion into the excluded cavity.
   geom->CheckOverlaps(0.1);
   EXPECT_EQ(geom->GetListOfOverlaps()->GetEntriesFast(), 1);
}

INSTANTIATE_TEST_SUITE_P(Quadrants, ExtrusionQuadrant,
                         testing::Combine(testing::Values(-1, 1), testing::Values(-1, 1)));

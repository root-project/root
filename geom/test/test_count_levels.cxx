#include <gtest/gtest.h>

#include <TGeoBBox.h>
#include <TGeoManager.h>
#include <TGeoMaterial.h>
#include <TGeoMatrix.h>
#include <TGeoMedium.h>
#include <TGeoNode.h>
#include <TGeoVolume.h>
#include <TGeoXtru.h>
#include <TRandom3.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

struct Counts {
   Int_t maxLevel;
   Int_t maxDaughters;
   Int_t maxXtruVert;
};

// Reference: the values TGeoManager::CountLevels() derives, computed by visiting every node of the expanded
// tree (every placement path) with TGeoIterator.
Counts CountByIteration(TGeoVolume *top)
{
   Counts c{1, top->GetNdaughters(), 1};
   TGeoIterator next(top);
   while (TGeoNode *node = next()) {
      c.maxDaughters = std::max(c.maxDaughters, node->GetNdaughters());
      c.maxLevel = std::max(c.maxLevel, next.GetLevel());
      if (auto xtru = dynamic_cast<TGeoXtru *>(node->GetVolume()->GetShape()))
         c.maxXtruVert = std::max(c.maxXtruVert, xtru->GetNvert());
   }
   return c;
}

Counts CountByManager()
{
   return {TGeoManager::GetMaxLevels(), TGeoManager::GetMaxDaughters(), TGeoManager::GetMaxXtruVert()};
}

TGeoVolume *MakeXtru(const char *name, int nvert, double size, TGeoMedium *medium)
{
   std::vector<double> x(nvert), y(nvert);
   for (int i = 0; i < nvert; ++i) {
      x[i] = size * std::cos(2 * M_PI * i / nvert);
      y[i] = size * std::sin(2 * M_PI * i / nvert);
   }
   auto *xtru = new TGeoXtru(2);
   xtru->DefinePolygon(nvert, x.data(), y.data());
   xtru->DefineSection(0, -size);
   xtru->DefineSection(1, size);
   return new TGeoVolume(name, xtru, medium);
}

TGeoMedium *MakeGeometry(const char *name)
{
   delete gGeoManager;
   new TGeoManager(name, "CountLevels test");
   auto *material = new TGeoMaterial("Vacuum", 0., 0., 0.);
   return new TGeoMedium("Vacuum", 1, material);
}

} // namespace

// Volumes placed several times, at several depths: the counts only depend on the volume tree.
TEST(TGeoManager, CountLevelsRepeatedPlacements)
{
   TGeoManager::SetVerboseLevel(0);
   auto *medium = MakeGeometry("repeated");
   auto *top = gGeoManager->MakeBox("top", medium, 100., 100., 100.);
   auto *a = gGeoManager->MakeBox("a", medium, 10., 10., 10.);
   auto *b = gGeoManager->MakeBox("b", medium, 2., 2., 2.);
   auto *c = gGeoManager->MakeBox("c", medium, 0.1, 0.1, 0.1);
   auto *x = MakeXtru("x", 7, 1., medium);
   for (int i = 0; i < 3; ++i)
      top->AddNode(a, i, new TGeoTranslation(-50. + 25. * i, 0., 0.));
   top->AddNode(b, 10, new TGeoTranslation(50., 0., 0.)); // b also directly below top
   a->AddNode(b, 0, new TGeoTranslation(-5., 0., 0.));
   a->AddNode(b, 1, new TGeoTranslation(5., 0., 0.));
   a->AddNode(x, 2, new TGeoTranslation(0., 5., 0.));
   for (int i = 0; i < 4; ++i)
      b->AddNode(c, i, new TGeoTranslation(-1.5 + i, 0., 0.));
   gGeoManager->SetTopVolume(top);
   gGeoManager->CloseGeometry();

   const Counts counts = CountByManager();
   EXPECT_EQ(counts.maxLevel, 3);
   EXPECT_EQ(counts.maxDaughters, 4);
   EXPECT_EQ(counts.maxXtruVert, 7);
   delete gGeoManager;
}

// Random volume trees with shared volumes, compared with a walk over every placement path.
TEST(TGeoManager, CountLevelsMatchesFullIteration)
{
   TGeoManager::SetVerboseLevel(0);
   TRandom3 rng(1234);
   for (int iteration = 0; iteration < 200; ++iteration) {
      auto *medium = MakeGeometry(TString::Format("random_%d", iteration));
      const int nvol = 1 + rng.Integer(10);
      // volume i has size 4^-i, so that up to 3 daughters fit side by side in any mother
      std::vector<TGeoVolume *> volumes(nvol);
      std::vector<double> sizes(nvol);
      for (int i = 0; i < nvol; ++i) {
         TString name = TString::Format("v%d", i);
         sizes[i] = std::pow(0.25, i);
         if (i > 0 && rng.Rndm() < 0.2)
            volumes[i] = MakeXtru(name, 3 + rng.Integer(6), sizes[i], medium);
         else
            volumes[i] = new TGeoVolume(name, new TGeoBBox(sizes[i], sizes[i], sizes[i]), medium);
      }
      // a volume only contains volumes with a higher index, so the volume graph is acyclic
      for (int i = 0; i < nvol - 1; ++i) {
         const int ndaughters = rng.Integer(4);
         for (int k = 0; k < ndaughters; ++k) {
            const int j = i + 1 + rng.Integer(nvol - 1 - i);
            volumes[i]->AddNode(volumes[j], k, new TGeoTranslation((k - 1) * 0.6 * sizes[i], 0., 0.));
         }
      }
      gGeoManager->SetTopVolume(volumes[0]);
      gGeoManager->CloseGeometry();

      const Counts expected = CountByIteration(volumes[0]);
      const Counts counts = CountByManager();
      EXPECT_EQ(counts.maxLevel, expected.maxLevel) << "iteration " << iteration;
      EXPECT_EQ(counts.maxDaughters, expected.maxDaughters) << "iteration " << iteration;
      EXPECT_EQ(counts.maxXtruVert, expected.maxXtruVert) << "iteration " << iteration;
   }
   delete gGeoManager;
}

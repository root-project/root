#include <gtest/gtest.h>

#include <TGeoBBox.h>
#include <TGeoCompositeShape.h>
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TGeoMaterial.h>
#include <TGeoMedium.h>
#include <TGeoNavigator.h>
#include <TGeoNode.h>
#include <TGeoPatternFinder.h>
#include <TGeoPgon.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>
#include <TGeoXtru.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <mutex>
#include <thread>
#include <vector>

/**
   Navigating the same geometry from several threads must give exactly the same answer as
   navigating it from one. This exercises every class that keeps per-thread scratch state:
   TGeoXtru, TGeoPgon, TGeoVolumeAssembly, TGeoBoolNode (via a composite shape) and
   TGeoPatternFinder (via a divided volume).

   Worth running under ThreadSanitizer: the threads book their navigators lazily, so this
   also covers concurrent TGeoManager::AddNavigator() against navigator-map readers.
*/

namespace {

/// Geometry containing one instance of each shape family that owns per-thread data.
TGeoManager *MakeGeometry()
{
   auto *geom = new TGeoManager("mt_nav_geom", "geometry for MT navigation test");

   auto *matVac = new TGeoMaterial("Vacuum", 0, 0, 0);
   auto *matAl = new TGeoMaterial("Al", 26.98, 13, 2.7);
   auto *vac = new TGeoMedium("Vacuum", 1, matVac);
   auto *alu = new TGeoMedium("Aluminium", 2, matAl);

   TGeoVolume *top = geom->MakeBox("TOP", vac, 100., 100., 100.);
   geom->SetTopVolume(top);

   // --- TGeoXtru: convex, simple polygon extruded over two sections
   Double_t xv[5] = {-10., -5., 5., 10., 0.};
   Double_t yv[5] = {-6., -10., -10., -6., 10.};
   auto *xtru = new TGeoXtru(2);
   xtru->DefinePolygon(5, xv, yv);
   xtru->DefineSection(0, -20., 0., 0., 1.);
   xtru->DefineSection(1, 20., 0., 0., 1.);
   top->AddNode(new TGeoVolume("XTRU", xtru, alu), 1, new TGeoTranslation(-45., 0., 0.));

   // --- TGeoPgon
   auto *pgon = new TGeoPgon("pgon", 0., 360., 8, 2);
   pgon->DefineSection(0, -20., 5., 12.);
   pgon->DefineSection(1, 20., 5., 12.);
   top->AddNode(new TGeoVolume("PGON", pgon, alu), 1, new TGeoTranslation(45., 0., 0.));

   // --- TGeoBoolNode, through a composite shape
   new TGeoBBox("cbox", 12., 12., 12.);
   new TGeoTube("ctub", 0., 6., 20.);
   auto *comp = new TGeoCompositeShape("comp", "cbox - ctub");
   top->AddNode(new TGeoVolume("COMP", comp, alu), 1, new TGeoTranslation(0., 45., 0.));

   // --- TGeoPatternFinder, through a divided volume
   TGeoVolume *slab = geom->MakeBox("SLAB", alu, 20., 5., 5.);
   slab->Divide("SLABDIV", 1, 10, -20., 4.);
   top->AddNode(slab, 1, new TGeoTranslation(0., -45., 0.));

   // --- TGeoVolumeAssembly
   auto *assembly = new TGeoVolumeAssembly("ASSEMBLY");
   TGeoVolume *brick = geom->MakeBox("BRICK", alu, 3., 3., 3.);
   for (int i = 0; i < 6; ++i)
      assembly->AddNode(brick, i + 1, new TGeoTranslation(0., 0., -25. + 10. * i));
   top->AddNode(assembly, 1, new TGeoTranslation(0., 0., 0.));

   geom->CloseGeometry();
   return geom;
}

struct GeometryWithFinders {
   TGeoManager *manager;
   TGeoPatternFinder *linearFinder;
   TGeoPatternFinder *radialFinder;
};

/// Build two divided volumes without touching their lazily-created matrices.
GeometryWithFinders MakeDividedGeometry(const char *name)
{
   auto *manager = new TGeoManager(name, name);
   auto *material = new TGeoMaterial("vacuum", 0., 0., 0.);
   auto *medium = new TGeoMedium("vacuum", 1, material);
   auto *top = manager->MakeBox("top", medium, 100., 100., 100.);
   manager->SetTopVolume(top);

   auto *slab = manager->MakeBox("slab", medium, 20., 5., 5.);
   slab->Divide("linear_slice", 1, 10, -20., 4.);
   top->AddNode(slab, 1);

   auto *tube = manager->MakeTube("tube", medium, 1., 20., 10.);
   tube->Divide("radial_slice", 1, 4, 1., 19. / 4.);
   top->AddNode(tube, 1, new TGeoTranslation(0., 40., 0.));

   manager->CloseGeometry();
   return {manager, slab->GetFinder(), tube->GetFinder()};
}

void MakeCurrent(TGeoManager *manager)
{
   gGeoManager = manager;
   gGeoIdentity = static_cast<TGeoIdentity *>(manager->GetListOfMatrices()->At(0));
}

class InspectablePgon : public TGeoPgon {
public:
   using TGeoPgon::TGeoPgon;

   std::size_t GetOwnedThreadDataCount() const
   {
      std::lock_guard<std::mutex> guard(fOwnedDataMutex);
      return fOwnedData.size();
   }
};

class InspectableXtru : public TGeoXtru {
public:
   using TGeoXtru::TGeoXtru;

   std::size_t GetOwnedThreadDataCount() const
   {
      std::lock_guard<std::mutex> guard(fOwnedDataMutex);
      return fOwnedData.size();
   }
};

struct Ray {
   Double_t point[3];
   Double_t dir[3];
};

/// Deterministic fan of rays aimed at each shape, so every class with per-thread state is
/// actually traversed. Start points sit 40 cm from the target, well inside the world volume.
std::vector<Ray> MakeRays()
{
   const Double_t targets[][3] = {
      {-45., 0., 0.}, {-45., 0., 10.}, {-45., 3., -10.}, // XTRU
      {45., 0., 0.},  {45., 0., 10.},  {45., 8., -10.},  // PGON
      {0., 45., 0.},  {0., 45., 8.},   {8., 45., 0.},    // composite (box minus tube)
      {0., -45., 0.}, {5., -45., 0.},  {-5., -45., 2.},  // divided slab
      {0., 0., -25.}, {0., 0., -5.},   {0., 0., 15.},    // assembly bricks
   };
   std::vector<Ray> rays;
   for (const auto &t : targets) {
      for (int i = 0; i < 12; ++i) {
         const double phi = 2. * M_PI * i / 12.;
         const double theta = 0.4 + 0.15 * (i % 5);
         const double d[3] = {std::sin(theta) * std::cos(phi), std::sin(theta) * std::sin(phi), std::cos(theta)};
         Ray ray;
         for (int k = 0; k < 3; ++k) {
            ray.point[k] = t[k] + 40. * d[k];
            ray.dir[k] = -d[k];
         }
         rays.push_back(ray);
      }
   }
   return rays;
}

struct RayResult {
   Int_t nsteps{0};
   Double_t pathlen{0.};
   Long64_t checksum{0}; // sequence of volumes traversed

   bool operator==(const RayResult &o) const
   {
      return nsteps == o.nsteps && checksum == o.checksum && std::abs(pathlen - o.pathlen) < 1e-9;
   }
};

RayResult ShootRay(TGeoNavigator *nav, const Ray &ray)
{
   RayResult res;
   nav->InitTrack(ray.point, ray.dir);
   while (!nav->IsOutside() && res.nsteps < 500) {
      TGeoNode *node = nav->GetCurrentNode();
      res.checksum = res.checksum * 31 + (node ? node->GetVolume()->GetNumber() : -1);
      nav->FindNextBoundaryAndStep(1.e6, kFALSE);
      res.pathlen += nav->GetStep();
      ++res.nsteps;
      if (!nav->IsOnBoundary())
         break;
   }
   return res;
}

} // namespace

TEST(Geometry, MultiThreadedNavigationMatchesSerial)
{
   TGeoManager *geom = MakeGeometry();
   ASSERT_NE(geom, nullptr);
   const std::vector<Ray> rays = MakeRays();

   // Reference: single-threaded, default navigator.
   std::vector<RayResult> reference;
   reference.reserve(rays.size());
   for (const Ray &ray : rays)
      reference.push_back(ShootRay(geom->GetCurrentNavigator(), ray));

   // Rays that miss every object legitimately cross a single boundary (the world). Guard against
   // a vacuous comparison by requiring that a good share of them actually traverse the shapes.
   const size_t nTraversing =
      std::count_if(reference.begin(), reference.end(), [](const RayResult &r) { return r.nsteps > 2; });
   ASSERT_GT(nTraversing, reference.size() / 2);

   constexpr int kNThreads = 8;
   geom->SetMaxThreads(kNThreads);

   std::vector<std::vector<RayResult>> perThread(kNThreads);
   std::vector<std::thread> threads;
   threads.reserve(kNThreads);
   for (int t = 0; t < kNThreads; ++t) {
      threads.emplace_back([&, t] {
         // Booked lazily and concurrently, exactly as a task-parallel workload would.
         TGeoNavigator *nav = geom->AddNavigator();
         perThread[t].reserve(rays.size());
         for (const Ray &ray : rays)
            perThread[t].push_back(ShootRay(nav, ray));
      });
   }
   for (std::thread &th : threads)
      th.join();

   for (int t = 0; t < kNThreads; ++t) {
      ASSERT_EQ(perThread[t].size(), reference.size()) << "thread " << t;
      for (size_t i = 0; i < reference.size(); ++i)
         EXPECT_TRUE(perThread[t][i] == reference[i]) << "thread " << t << ", ray " << i;
   }

   delete geom;
}

TEST(Geometry, PatternMatricesBelongToOwningManager)
{
   auto geometryA = MakeDividedGeometry("pattern_owner_A");

   // Keep A alive while constructing B. This is how applications that cache several
   // managers switch away from the current geometry before creating another one.
   gGeoManager = nullptr;
   gGeoIdentity = nullptr;
   auto geometryB = MakeDividedGeometry("pattern_owner_B");

   // First-touch A while B is current. Matrix ownership must follow the divided volume,
   // not the ambient globals.
   TGeoMatrix *matrix = geometryA.linearFinder->GetMatrix();
   ASSERT_NE(matrix, nullptr);
   EXPECT_GE(geometryA.manager->GetListOfMatrices()->IndexOf(matrix), 0);
   EXPECT_LT(geometryB.manager->GetListOfMatrices()->IndexOf(matrix), 0);

   TGeoMatrix *identity = geometryA.radialFinder->GetMatrix();
   EXPECT_EQ(identity, geometryA.manager->GetListOfMatrices()->At(0));
   EXPECT_NE(identity, geometryB.manager->GetListOfMatrices()->At(0));

   delete geometryB.manager;
   MakeCurrent(geometryA.manager);

   // Under ASan this dereference also catches a matrix that was wrongly owned and deleted by B.
   geometryA.linearFinder->cd(0);
   EXPECT_EQ(geometryA.linearFinder->GetMatrix(), matrix);
   EXPECT_TRUE(geometryA.radialFinder->GetMatrix()->IsIdentity());

   delete geometryA.manager;
}

TEST(Geometry, ShapeScratchDataReleasedOnClear)
{
   InspectablePgon pgon(0., 360., 64, 2);
   pgon.DefineSection(0, -10., 1., 5.);
   pgon.DefineSection(1, 10., 1., 5.);

   InspectableXtru xtru(2);
   Double_t x[] = {-5., 5., 5., -5.};
   Double_t y[] = {-5., -5., 5., 5.};
   xtru.DefinePolygon(4, x, y);
   xtru.DefineSection(0, -10.);
   xtru.DefineSection(1, 10.);

   // DefineSection computes the Xtru bounding box using the main-thread slot.
   pgon.ClearThreadData();
   xtru.ClearThreadData();

   constexpr int kNThreads = 8;
   std::atomic<bool> valid{true};
   std::vector<std::thread> threads;
   threads.reserve(kNThreads);
   for (int i = 0; i < kNThreads; ++i) {
      threads.emplace_back([&] {
         auto &pgonData = pgon.GetThreadData();
         auto &xtruData = xtru.GetThreadData();
         if (!pgonData.fIntBuffer || !pgonData.fDblBuffer || !xtruData.fXc || !xtruData.fYc || !xtruData.fPoly)
            valid.store(false, std::memory_order_relaxed);
      });
   }
   for (auto &thread : threads)
      thread.join();

   ASSERT_TRUE(valid.load(std::memory_order_relaxed));
   EXPECT_EQ(pgon.GetOwnedThreadDataCount(), kNThreads);
   EXPECT_EQ(xtru.GetOwnedThreadDataCount(), kNThreads);

   pgon.ClearThreadData();
   xtru.ClearThreadData();
   EXPECT_EQ(pgon.GetOwnedThreadDataCount(), 0u);
   EXPECT_EQ(xtru.GetOwnedThreadDataCount(), 0u);

   // The main thread's stale non-owning slots must rebuild on the next access.
   EXPECT_NE(pgon.GetThreadData().fIntBuffer, nullptr);
   EXPECT_NE(xtru.GetThreadData().fPoly, nullptr);
   EXPECT_EQ(pgon.GetOwnedThreadDataCount(), 1u);
   EXPECT_EQ(xtru.GetOwnedThreadDataCount(), 1u);
}

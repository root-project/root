#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <ROOT/TestSupport.hxx>
#include <TGeoManager.h>
#include <TGeoMaterial.h>
#include <TGeoMedium.h>
#include <TGeoNavigator.h>
#include <TGeoVolume.h>
#include <TROOT.h>

namespace {

TGeoManager *MakeGeometry(int iteration)
{
   TGeoManager *geom = nullptr;
   // TGeoManager::Init deletes the existing gGeoManager, as happens when the
   // interpreted macro is executed again from the main thread.
   if (gGeoManager) {
      ROOT_EXPECT_WARNING_PARTIAL(geom =
                                     new TGeoManager(TString::Format("world_%d", iteration), "navigator lifetime test"),
                                  "TGeoManager::Init", "Deleting previous geometry:");
   } else {
      geom = new TGeoManager(TString::Format("world_%d", iteration), "navigator lifetime test");
   }
   auto *material = new TGeoMaterial("Vacuum", 0., 0., 0.);
   auto *medium = new TGeoMedium("Vacuum", 1, material);
   auto *top = geom->MakeBox("top", medium, 10., 10., 10.);
   auto *sphere = geom->MakeSphere("sphere", medium, 0., 5.);
   top->AddNode(sphere, 1);
   geom->SetTopVolume(top);
   geom->CloseGeometry();
   return geom;
}

class ImplicitMTGuard {
public:
   explicit ImplicitMTGuard(unsigned int numThreads) { ROOT::EnableImplicitMT(numThreads); }
   ~ImplicitMTGuard() { ROOT::DisableImplicitMT(); }
};

} // namespace

TEST(TGeoManager, ConcurrentNavigatorsStayThreadLocal)
{
   constexpr unsigned int numThreads = 4;
   constexpr int numNavigators = 16;

   auto *geom = MakeGeometry(0);
   geom->SetMaxThreads(numThreads);

   // Start all workers together so navigator creation and lookup overlap.
   std::atomic<unsigned int> numReady{0};
   std::atomic<bool> start{false};
   std::atomic<bool> failed{false};
   std::vector<std::thread> workers;
   workers.reserve(numThreads);

   for (unsigned int i = 0; i < numThreads; ++i) {
      workers.emplace_back([&] {
         numReady.fetch_add(1, std::memory_order_release);
         while (!start.load(std::memory_order_acquire))
            std::this_thread::yield();

         // Every worker owns an independent navigator array. Adding a navigator
         // here must update only this worker's current-navigator cache, even as
         // the other workers add their navigators concurrently.
         TGeoNavigator *first = nullptr;
         TGeoNavigator *current = nullptr;
         for (int j = 0; j < numNavigators; ++j) {
            current = geom->AddNavigator();
            if (!first)
               first = current;
            for (int check = 0; check < 32; ++check) {
               if (geom->GetCurrentNavigator() != current)
                  failed.store(true, std::memory_order_relaxed);
            }
         }

         // Selecting another navigator must likewise affect only this worker.
         if (!geom->SetCurrentNavigator(0) || geom->GetCurrentNavigator() != first)
            failed.store(true, std::memory_order_relaxed);
      });
   }

   while (numReady.load(std::memory_order_acquire) != numThreads)
      std::this_thread::yield();
   start.store(true, std::memory_order_release);

   for (auto &worker : workers)
      worker.join();

   EXPECT_FALSE(failed.load(std::memory_order_relaxed));
   delete gGeoManager;
}

TEST(TGeoManager, RecreateAfterParallelOverlapCheck)
{
   ImplicitMTGuard imtGuard(2);

   // CheckOverlaps populates navigator caches in persistent IMT worker threads.
   // On the next iteration, MakeGeometry deletes the old manager on the main
   // thread and creates a new one, while the worker threads remain alive.
   for (int iteration = 0; iteration < 10; ++iteration) {
      auto *geom = MakeGeometry(iteration);

      // Foreach waits for all workers before returning. Reused workers must
      // reject a navigator cached for the previous manager generation and book
      // a navigator belonging to this manager.
      geom->CheckOverlaps(0.001);

      // The calling thread must also resolve the current manager's navigator.
      ASSERT_TRUE(geom->IsMultiThread());
      ASSERT_NE(geom->GetListOfNavigators(), nullptr);
      ASSERT_EQ(geom->GetCurrentNavigator(), geom->GetListOfNavigators()->GetCurrentNavigator());
      EXPECT_EQ(geom->GetStackLevel(), 0);
   }

   delete gGeoManager;
}

#include "ROOT/TPoolManager.hxx"
#include "TError.h"
#include "TROOT.h"
#include <algorithm>
#include "tbb/task_scheduler_init.h"

namespace ROOT {

   namespace Internal {
      //Returns the weak_ptr reflecting a shared_ptr to the only instance of the Pool Manager.
      //This will allow to check if the shared_ptr is still alive, solving the dangling pointer problem.
      std::weak_ptr<TPoolManager> &GetWP()
      {
         static std::weak_ptr<TPoolManager> weak_sched;
         return weak_sched;
      }

      UInt_t TPoolManager::fgPoolSize = 0;

      TPoolManager::TPoolManager(UInt_t nThreads): fSched(new tbb::task_scheduler_init(tbb::task_scheduler_init::deferred))
      {
         //Is it there another instance of the tbb scheduler running?
         if (fSched->is_active()) {
            mustDelete = false;
         }

         nThreads = nThreads != 0 ? nThreads : tbb::task_scheduler_init::default_num_threads();
         fSched ->initialize(nThreads);
         fgPoolSize = nThreads;
      };

      TPoolManager::~TPoolManager()
      {
         //Only terminate the tbb scheduler if there was not another instance already
         // running when the constructor was called.
         if (mustDelete) {
            fSched->terminate();
            fgPoolSize = 0;
         }
      }

      //Number of threads the PoolManager has been initialized with.
      UInt_t TPoolManager::GetPoolSize()
      {
         return fgPoolSize;
      }

      //Factory function returning a shared pointer to the only instance of the PoolManager.
      std::shared_ptr<TPoolManager> GetPoolManager(UInt_t nThreads)
      {
         if (GetWP().expired()) {
            std::shared_ptr<TPoolManager> shared(new TPoolManager(nThreads));
            GetWP() = shared;
            return GetWP().lock();
         }
         return GetWP().lock();
      }
   }
}

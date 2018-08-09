#include "TPoolManager.hxx"
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

      TPoolManager::TPoolManager(): fSched(new tbb::task_scheduler_init(tbb::task_scheduler_init::deferred))
      {
         //Is it there another instance of the tbb scheduler running?
         if (fSched->is_active()) {
            mustDelete = false;
         }

         fSched ->initialize();
      };

      TPoolManager::~TPoolManager()
      {
         //Only terminate the tbb scheduler if there was not another instance already
         // running when the constructor was called.
         if (mustDelete) {
            fSched->terminate();
         }
      }

      //Factory function returning a shared pointer to the only instance of the PoolManager.
      std::shared_ptr<TPoolManager> GetPoolManager()
      {
         if (GetWP().expired()) {
            // why not make_shared: https://stackoverflow.com/a/20895705/
            //(under "Why do instances of weak_ptrs keep the control block alive?"")
            std::shared_ptr<TPoolManager> shared(new TPoolManager());
            GetWP() = shared;
            return GetWP().lock();
         }
         return GetWP().lock();
      }
   }
}

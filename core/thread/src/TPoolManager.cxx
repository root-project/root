#include "TError.h"
#include "ROOT/TPoolManager.hxx"
#include "TROOT.h"
#include <algorithm>
#include "tbb/task_scheduler_init.h"


namespace ROOT {


   UInt_t TPoolManager::fgPoolSize = 0;

   TPoolManager::TPoolManager(UInt_t nThreads): fSched(new tbb::task_scheduler_init(tbb::task_scheduler_init::deferred))
   {
      if (fSched->is_active()) {
         mustDelete = false;
      }

      nThreads = nThreads != 0 ? nThreads : tbb::task_scheduler_init::default_num_threads();
      fSched ->initialize(nThreads);
      fgPoolSize = nThreads;
   };

   TPoolManager::~TPoolManager()
   {
      if (mustDelete) {
         fSched->terminate();
         fgPoolSize = 0;
      }
      GetWP().reset();
   }

   //Size of the task pool the PoolManager has been initialized with. Can be greater than number of threads.
   UInt_t TPoolManager::GetNThreads()
   {
      return fgPoolSize;
   }

}

std::weak_ptr<ROOT::TPoolManager> &GetWP()
{
   static std::weak_ptr<ROOT::TPoolManager> weak_sched;
   return weak_sched;
}

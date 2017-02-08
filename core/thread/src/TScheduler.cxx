#include "TError.h"
#include "TROOT.h"
#include "ROOT/TScheduler.hxx"
#include <algorithm>
#include "tbb/task_scheduler_init.h"

namespace ROOT {
   namespace Internal {

      static tbb::task_scheduler_init &GetScheduler()
      {
         static tbb::task_scheduler_init scheduler(tbb::task_scheduler_init::deferred);
         return scheduler;
      }

      UInt_t TScheduler::fgPoolSize = 0;
      UInt_t TScheduler::fgSubscriptionsCount = 0;

      TScheduler::TScheduler() {};

      TScheduler::~TScheduler()
      {
         if (fSubscribed)
            Unsubscribe();
      }

      void TScheduler::Subscribe()
      {
         Subscribe(tbb::task_scheduler_init::default_num_threads());
      }

      void TScheduler::Subscribe(UInt_t nThreads)
      {
         if (!fSubscribed) {
            fSubscribed = true;
            fgSubscriptionsCount++;
            if (fgPoolSize != 0 && fgPoolSize != nThreads) {
               Warning("TScheduler::Subscribe", "Can't change the number of threads specified by a previous instantiation of TScheduler. Proceeding with %d threads.", fgPoolSize);
            } else {
               fgPoolSize = nThreads;
               GetScheduler().initialize(nThreads);
            }
         } else {
            Warning("TScheduler::Subscribe", "Unsubscribe before subscribing again");
         }
      }

      UInt_t TScheduler::GetNThreads()
      {
         //Avoiding the warning: default_num_threads is an int (and it doesn't make sense)
         return std::min(fgPoolSize, static_cast<UInt_t>(tbb::task_scheduler_init::default_num_threads()));
      }

      //Size of the task pool the scheduler has been initialized with. Can be greater than number of threads.
      UInt_t TScheduler::GetPoolSize()
      {
         return fgPoolSize;
      }

      UInt_t TScheduler::GetNSubscribers()
      {
         return fgSubscriptionsCount;
      }

      void TScheduler::Unsubscribe()
      {
         if (fSubscribed) {
            fSubscribed = false;
            if (--fgSubscriptionsCount == 0) {
               if (GetScheduler().is_active())
                  GetScheduler().terminate();
               fgPoolSize = 0;
            }
         } else {
            Warning("TScheduler::Unsubscribe", "Scheduler still not in use by this instance. Call the constructor first.");
         }
      }
   }
}

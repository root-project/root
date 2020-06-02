#include "ROOT/TPoolManager.hxx"
#include "TError.h"
#include <cmath>
#include <algorithm>
#include <fstream>
#ifdef R__LINUX
#include <unistd.h>
#include <sys/stat.h>
#endif
#include "tbb/task_scheduler_init.h"


////////////////////////////////////////////////////////////////////////////////
/// Returns the available number of logical cores.
///
///  - Checks if there is CFS bandwith control in place (linux, via cgroups,
///    assuming standard paths)
///  - Otherwise, returns the number of logical cores provided by tbb by default.
///    This is processor affinity aware, at least in Linux.
////////////////////////////////////////////////////////////////////////////////


namespace ROOT {

   namespace Internal {

      //Returns the available number of logical cores.
      // - Checks if there is CFS bandwith control in place (linux, via cgroups,
      //   assuming standard paths)
      // - Otherwise, returns the number of logical cores provided by tbb by default.
      //   This is processor affinity aware, at least in Linux.
      Int_t NLogicalCores()
      {
      #ifdef R__LINUX
         // Check for CFS bandwith control
         std::ifstream f;
         std::string quotaFile("/sys/fs/cgroup/cpuacct/cpu.cfs_quota_us");
         struct stat buffer;
         // Does the file exist?
         if(stat(quotaFile.c_str(), &buffer) == 0) {
            f.open(quotaFile);
            float cfs_quota;
            f>>cfs_quota;
            f.close();
            if(cfs_quota > 0) {
               std::string periodFile("/sys/fs/cgroup/cpuacct/cpu.cfs_period_us");
               f.open(periodFile);
               float cfs_period;
               f>>cfs_period;
               f.close();
               return static_cast<int>(std::ceil(cfs_quota/cfs_period));
            }
         }
      #endif
         return tbb::task_scheduler_init::default_num_threads();
      }

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

         nThreads = nThreads != 0 ? nThreads : NLogicalCores();
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

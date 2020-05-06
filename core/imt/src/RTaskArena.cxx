#include "ROOT/RTaskArena.hxx"
#include "TError.h"
#include "TROOT.h"
#include "TThread.h"
#include <fstream>
#include <sys/stat.h>
#include <thread>
#include "tbb/task_arena.h"


//////////////////////////////////////////////////////////////////////////
///
/// \class ROOT::Internal::RTaskArenaWrapper
/// \ingroup Parallelism
/// \brief Wrapper over tbb::task_arena
///
/// This class is a wrapper over tbb::task_arena, in order to keep
/// TBB away from ROOT's headers. We keep a single global instance,
/// obtained with `ROOT::Internal::GetGlobalTaskArena()`, to be used by any
/// parallel ROOT class with TBB as a backend. This has several advantages:
///
///   - Provides a unique interface to the TBB scheduler: TThreadExecutor,
///     IMT and any class relying on TBB will get a pointer to the scheduler
///     through `ROOT::Internal::GetGlobalTaskArena()`, which will return a
///     reference to the only pointer to the TBB scheduler that will be
///     active in any ROOT Process
///   - Solves multiple undefined behaviors. Guaranteeing that all classes
///    use the same task arena avoids interferences and undefined behavior
///    by providing a single instance of the tbb::task_arena and automated
///    bookkeeping, instantiation and destruction.
///
/// #### Examples:
/// ~~~{.cpp}
/// root[] auto gTA = ROOT::Internal::GetGlobalTaskArena() //get a shared_ptr to the global arena
/// root[] gTA->InitGlobalTaskArena(nWorkers) // Initialize the global arena and enable Thread Safety in ROOT
/// root[] gTA->TaskArenaSize() // Get the current size of the arena (number of worker threads)
/// root[] gTA->Access() //std::unique_ptr to the internal tbb::task_arena for interacting directly with it (needed to call operations such as execute)
/// root[] root[] gTA->Access()->max_concurrency() // call to tbb::task_arena::max_concurrency()
/// ~~~
///
//////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
/// Returns the available number of logical cores.
///
///  - Checks if there is CFS bandwidth control in place (linux, via cgroups,
///    assuming standard paths)
///  - Otherwise, returns the number of logical cores provided by
///    std::thread::hardware_concurrency()
////////////////////////////////////////////////////////////////////////////////
static Int_t LogicalCPUBandwithControl()
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
   return std::thread::hardware_concurrency();
}

namespace ROOT{
namespace Internal {

RTaskArenaWrapper::RTaskArenaWrapper(): fTBBArena(new tbb::task_arena{}){}

unsigned RTaskArenaWrapper::TaskArenaSize()
{
   return fTBBArena->is_active()? static_cast<unsigned>(fTBBArena->max_concurrency()) : 0u;
}

std::unique_ptr<tbb::task_arena> &RTaskArenaWrapper::Access()
{
   return fTBBArena;
}


std::shared_ptr<ROOT::Internal::RTaskArenaWrapper> GetGlobalTaskArena()
{
   static std::weak_ptr<ROOT::Internal::RTaskArenaWrapper> weak_GTAWrapper;
   if (weak_GTAWrapper.expired()) {
      std::shared_ptr<ROOT::Internal::RTaskArenaWrapper> shared_GTAWrapper(new ROOT::Internal::RTaskArenaWrapper());
      weak_GTAWrapper = shared_GTAWrapper;
      return weak_GTAWrapper.lock();
   }

   return weak_GTAWrapper.lock();
}

std::shared_ptr<ROOT::Internal::RTaskArenaWrapper> InitGlobalTaskArena(unsigned maxConcurrency)
{
   auto globalTBBTaskArena = GetGlobalTaskArena();
   if (!globalTBBTaskArena->Access()->is_active()) {
      unsigned tbbDefaultNumberThreads = globalTBBTaskArena->Access()->max_concurrency(); // not initialized, automatic state
      maxConcurrency = maxConcurrency > 1 ? std::min(maxConcurrency, tbbDefaultNumberThreads) : tbbDefaultNumberThreads;
      unsigned bcCpus = LogicalCPUBandwithControl();
      auto taskArenaSize = std::min({maxConcurrency, bcCpus});
      globalTBBTaskArena->Access()->initialize(taskArenaSize);
      ROOT::EnableThreadSafety();
   } else {
      unsigned current = globalTBBTaskArena->Access()->max_concurrency();
      if (maxConcurrency && (current != maxConcurrency)) {
         Warning("InitGlobalTaskArena", "There's already an active task arena. Proceeding with the current %d threads",
            current);
      }
   }

   return globalTBBTaskArena;
}


} // namespace Internal
} // namespace ROOT

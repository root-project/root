#include "ROOT/RTaskArena.hxx"
#include "TError.h"
#include "TROOT.h"
#include "TThread.h"
#include <fstream>
#include <mutex>
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
/// root[] auto gTA = ROOT::Internal::GetGlobalTaskArena(nWorkers) //get a shared_ptr to the global arena and initialize
///                                                                //it with nWorkers. Enable thread safety in ROOT
/// root[] gTA->TaskArenaSize() // Get the current size of the arena (number of worker threads)
/// root[] gTA->Access() //std::unique_ptr to the internal tbb::task_arena for interacting directly with it (needed to
///                      //call operations such as execute)
/// root[] gTA->Access().max_concurrency() // call to tbb::task_arena::max_concurrency()
/// ~~~
///
//////////////////////////////////////////////////////////////////////////

namespace ROOT{
namespace Internal {

int LogicalCPUBandwithControl()
{
#ifdef R__LINUX
   // Check for CFS bandwith control
   std::ifstream f("/sys/fs/cgroup/cpuacct/cpu.cfs_quota_us"); // quota file
   if(f) {
      float cfs_quota;
      f>>cfs_quota;
      f.close();
      if(cfs_quota > 0) {
         f.open("/sys/fs/cgroup/cpuacct/cpu.cfs_period_us"); // period file
         float cfs_period;
         f>>cfs_period;
         f.close();
         return static_cast<int>(std::ceil(cfs_quota/cfs_period));
      }
   }
#endif
   return std::thread::hardware_concurrency();
}

////////////////////////////////////////////////////////////////////////////////
/// Initializes the tbb::task_arena within RTaskArenaWrapper.
///
/// * Can't be reinitialized
/// * Checks for CPU bandwidth control and avoids oversubscribing
/// * If no BC in place and maxConcurrency<1, defaults to the default tbb number of threads,
/// which is CPU affinity aware
////////////////////////////////////////////////////////////////////////////////
RTaskArenaWrapper::RTaskArenaWrapper(unsigned maxConcurrency): fTBBArena(new tbb::task_arena{})
{
   const unsigned tbbDefaultNumberThreads = fTBBArena->max_concurrency(); // not initialized, automatic state
   maxConcurrency = maxConcurrency > 0 ? std::min(maxConcurrency, tbbDefaultNumberThreads) : tbbDefaultNumberThreads;
   const unsigned bcCpus = LogicalCPUBandwithControl();
   if (maxConcurrency>bcCpus) {
      Warning("RTaskArenaWrapper", "CPU Bandwith Control Active. Proceeding with %d threads accordingly",
         bcCpus);
      maxConcurrency = bcCpus;
   }
   fTBBArena->initialize(maxConcurrency);
   fNWorkers = maxConcurrency;
   ROOT::EnableThreadSafety();
}

RTaskArenaWrapper::~RTaskArenaWrapper()
{
   fNWorkers = 0u;
}

unsigned RTaskArenaWrapper::fNWorkers = 0u;

unsigned RTaskArenaWrapper::TaskArenaSize()
{
   return fNWorkers;
}
////////////////////////////////////////////////////////////////////////////////
/// Provides access to the wrapped tbb::task_arena.
////////////////////////////////////////////////////////////////////////////////
tbb::task_arena &RTaskArenaWrapper::Access()
{
   return *fTBBArena;
}

std::shared_ptr<ROOT::Internal::RTaskArenaWrapper> GetGlobalTaskArena(unsigned maxConcurrency)
{
   static std::weak_ptr<ROOT::Internal::RTaskArenaWrapper> weak_GTAWrapper;

   static std::mutex m;
   const std::lock_guard<std::mutex> lock{m};
   if (auto sp = weak_GTAWrapper.lock()) {
      if (maxConcurrency && (sp->TaskArenaSize() != maxConcurrency)) {
         Warning("RTaskArenaWrapper", "There's already an active task arena. Proceeding with the current %d threads",
            sp->TaskArenaSize());
      }
      return sp;
   }
   auto sp = std::make_shared<ROOT::Internal::RTaskArenaWrapper>(maxConcurrency);
   weak_GTAWrapper = sp;
   return sp;
}

} // namespace Internal
} // namespace ROOT

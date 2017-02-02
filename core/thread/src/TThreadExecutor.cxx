#include "ROOT/TThreadExecutor.hxx"
#include "TError.h"
#include "TROOT.h"
#include "tbb/tbb.h"
#include <iostream>

namespace ROOT{

  unsigned TThreadExecutor::fgPoolSize = 0;

  TThreadExecutor::TThreadExecutor():TThreadExecutor::TThreadExecutor(tbb::task_scheduler_init::default_num_threads()){}

  TThreadExecutor::TThreadExecutor(size_t nThreads):fInitTBB(new tbb::task_scheduler_init(nThreads)){
    //ImplicitMT and TThreadExecutor share the same pool. If EnableImplicitMT has been called we need 
    // to get the size of the already initialized pool of threads
    fgPoolSize += fgPoolSize == 0? ROOT::GetImplicitMTPoolSize(): 0;

    if(fgPoolSize != 0){
      Warning("TThreadExecutor::TThreadExecutor", "Can't change the number of threads specified by a previous instantiation of TThreadExecutor or EnableImplicitMT. Proceeding with %d threads", fgPoolSize);
    } else {
      fgPoolSize = nThreads;
    }
  }

  TThreadExecutor::~TThreadExecutor() {
    if(!ROOT::IsImplicitMTEnabled())
      fInitTBB->terminate();
  }

  void TThreadExecutor::ParallelFor(unsigned int start, unsigned int end, unsigned step, const std::function<void(unsigned int i)> &f){
    tbb::parallel_for(start, end, step, f);
  }

  double TThreadExecutor::ParallelReduce(const std::vector<double> &objs, const std::function<double(double a, double b)> &redfunc){
   return tbb::parallel_reduce(tbb::blocked_range<decltype(objs.begin())>(objs.begin(), objs.end()), double{},
                              [redfunc](tbb::blocked_range<decltype(objs.begin())> const & range, double init) {
                              return std::accumulate(range.begin(), range.end(), init, redfunc);
                              }, redfunc);
  }

  float TThreadExecutor::ParallelReduce(const std::vector<float> &objs, const std::function<float(float a, float b)> &redfunc){
   return tbb::parallel_reduce(tbb::blocked_range<decltype(objs.begin())>(objs.begin(), objs.end()), float{},
                              [redfunc](tbb::blocked_range<decltype(objs.begin())> const & range, float init) {
                              return std::accumulate(range.begin(), range.end(), init, redfunc);
                              }, redfunc);
  }
}
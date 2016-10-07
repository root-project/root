#include "ROOT/TThreadExecutor.hxx"
#include "tbb/tbb.h"

namespace ROOT{
  TThreadExecutor::TThreadExecutor(){
    fInitTBB = new tbb::task_scheduler_init();
  }

  TThreadExecutor::TThreadExecutor(size_t nThreads){
    fInitTBB = new tbb::task_scheduler_init(nThreads);
  }

  TThreadExecutor::~TThreadExecutor() {
    fInitTBB->terminate();
  }

  void TThreadExecutor::_parallelFor(unsigned int start, unsigned int end, const std::function<void(unsigned int i)> &f){
    tbb::parallel_for(start, end, f);
  }


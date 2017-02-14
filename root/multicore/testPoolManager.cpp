#include "ROOT/TPoolManager.hxx"
#include "tbb/tbb.h"
#include "TROOT.h"
#include "ROOT/TThreadExecutor.hxx"

unsigned testTPoolManager(){

    {
    auto sched = ROOT::GetPoolManager(tbb::task_scheduler_init::default_num_threads());
    if(ROOT::TPoolManager::GetPoolSize()!= tbb::task_scheduler_init::default_num_threads())
       return 1;

    auto sched1 = ROOT::GetPoolManager(5);
    if(ROOT::TPoolManager::GetPoolSize()!= tbb::task_scheduler_init::default_num_threads())
       return 2;
    }

    if(ROOT::TPoolManager::GetPoolSize()!=0)
       return 3;

    ///////ImplicitMT and TThreadExecutor//////
    
    ROOT::EnableImplicitMT(8);
    ROOT::TThreadExecutor pool(5);
    if(ROOT::TPoolManager::GetPoolSize()!=8)
       return 4;

    ROOT::DisableImplicitMT();
    if(ROOT::TPoolManager::GetPoolSize()!=8)
       return 5;
    
    return 0;

}

int main(){
    return testTPoolManager();
}
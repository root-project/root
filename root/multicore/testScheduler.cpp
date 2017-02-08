#include "TScheduler.hxx"
#include "tbb/tbb.h"
#include "TROOT.h"
#include "ROOT/TThreadExecutor.hxx"

unsigned testScheduler(){
    
    ROOT::Internal::TScheduler sched;
    sched.Subscribe(tbb::task_scheduler_init::default_num_threads());
    if(ROOT::Internal::TScheduler::GetPoolSize()!= tbb::task_scheduler_init::default_num_threads())
       return 1;

    ROOT::Internal::TScheduler sched1;
    sched1.Subscribe(5);
    if(ROOT::Internal::TScheduler::GetPoolSize()!= tbb::task_scheduler_init::default_num_threads())
       return 2;
    
    sched.Unsubscribe();
    if(ROOT::Internal::TScheduler::GetNSubscribers()!=1)
       return 3;
    
    sched1.Unsubscribe();
    if(ROOT::Internal::TScheduler::GetPoolSize()!=0)
       return 4;
    
    sched.Subscribe(25);
    if(ROOT::Internal::TScheduler::GetPoolSize()<ROOT::Internal::TScheduler::GetNThreads())
       return 5;

    sched.Unsubscribe();

    ///////ImplicitMT and TThreadExecutor//////
    
    ROOT::EnableImplicitMT(8);
    ROOT::TThreadExecutor pool(5);
    if(ROOT::Internal::TScheduler::GetPoolSize()!=8)
       return 4;

    ROOT::DisableImplicitMT();
    if(ROOT::Internal::TScheduler::GetPoolSize()!=8)
       return 5;
    
    return 0;

}

int main(){
    return testScheduler();
}
#include<Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;


void env(Int_t level = THREAD_SINGLE)
{
   TEnvironment env(level);          //environment to start communication system

   assert(env.IsFinalized() == kFALSE);
   assert(env.IsInitialized() == kTRUE);
   assert(env.GetThreadLevel() == level);

   assert(env.IsMainThread() == kTRUE);

   env.Finalize();
   assert(env.IsFinalized() == kTRUE);

}

Int_t main()
{
   env();
   return 0;
}

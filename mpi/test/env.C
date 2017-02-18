#include<Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;


void env()
{
   TEnvironment env;          //environment to start communication system

   assert(env.IsFinalized() == kFALSE);
   assert(env.IsInitialized() == kTRUE);
   assert(env.GetThreadLevel() == THREAD_SINGLE);

   assert(env.IsMainThread() == kTRUE);

   env.Finalize();
   assert(env.IsFinalized() == kTRUE);

}

Int_t main()
{
   env();
   return 0;
}

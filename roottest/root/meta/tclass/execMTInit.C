#include "TSystem.h"
#include "TROOT.h"
#include "TClass.h"
#include <thread>
#include <vector>

#include "TError.h"

constexpr size_t gMaxThread = 128;

TClass *fgPtr[gMaxThread];

std::atomic<bool> gWaitToStart{true};

class UserClass {};

void init(int i) {

   while(gWaitToStart) {}

   TClass *cl = TClass::GetClass("UserClass");

   if ( !cl )
      Fatal("execMTInit","Thread %d could not find the TClass for UserClass\n",i);
   if ( cl->GetState() != TClass::kInterpreted )
      Fatal("execMTInit","Thread %d has the wrong State for UserClass: %d\n",i, cl->GetState());
   fgPtr[i] = cl;
}


int execMTInit(int nthreads = 4)
{
   //gSystem->Load("libHist");
   ROOT::EnableThreadSafety();

   std::vector<std::thread> threads;

   //auto hint = ROOT::gCoreMutex->ReadLock();
   auto hint = ROOT::gCoreMutex->WriteLock();

   for(int i = 0; i < nthreads; ++i)
   {
      threads.emplace_back(std::thread([i]() {
         init(i);
      }));
   }

   gWaitToStart = false;
   gSystem->Sleep(2*1000);
   //ROOT::gCoreMutex->ReadUnLock(hint);
   // Now all the thread should be stuck in GetListOfClasses()->FindObject
   // let's release and there is a high likelyhood that two or more thread
   // will get the 'answer' (no TClass yet) and then be waiting on the
   // write lock and we will then
   ROOT::gCoreMutex->WriteUnLock(hint);
   for(auto&& t : threads) {
      t.join();
   }

   TClass *cl = TClass::GetClass("UserClass");

   if ( !cl )
      Fatal("execMTInit","Final check could not find the TClass for UserClass\n");
   if ( cl->GetState() != TClass::kInterpreted )
      Fatal("execMTInit","Final check has the wrong State for UserClass: %d\n",cl->GetState());

   for(int i = 0; i < nthreads; ++i)
   {
      if (fgPtr[i] != cl)
         Fatal("execMTInit","Final check has a different address for UserClass than thread %d\n",i);
   }

   return 0;
}


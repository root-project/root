#include "executorTests.hxx"
#include "ROOT/EExecutionPolicy.hxx"
#include "ROOT/TExecutor.hxx"
#include "TROOT.h"

int main()
{
   ROOT::Internal::TExecutor ex{ROOT::EExecutionPolicy::kMultiProcess, 4u};
   if (ex.Policy() != ROOT::EExecutionPolicy::kMultiProcess) {
      return 1;
   }

   // On MacOS, Cocoa spawns threads. That's very bad for TProcessExecutor's `fork`:
   // forking a multi-thread program can easily break, see e.g. `man 2 fork`.
   gROOT->SetBatch(kTRUE);
   return ExecutorTest(ex);
}

#include "executorTests.hxx"
#include "ROOT/TProcessExecutor.hxx"
#include "TROOT.h"

int PoolTest() {
  ROOT::TProcessExecutor pool;
  return ExecutorTest(pool);
}

int main() {
   // On MacOS, Cocoa spawns threads. That's very bad for TProcessExecutor's `fork`:
   // forking a multi-thread program can easily break, see e.g. `man 2 fork`.
   gROOT->SetBatch(kTRUE);
   return PoolTest();
}

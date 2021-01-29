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

   return ExecutorTest(ex);
}

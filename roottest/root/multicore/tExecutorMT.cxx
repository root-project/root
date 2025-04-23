#include "executorTests.hxx"
#include "ROOT/EExecutionPolicy.hxx"
#include "ROOT/TExecutor.hxx"
#include "TROOT.h"

int main()
{

   // Build tests
   ROOT::Internal::TExecutor ex{4u};
   if (!ROOT::IsImplicitMTEnabled()) {
      if (ex.Policy() != ROOT::EExecutionPolicy::kSequential) {
         return 1;
      }
   }
#ifdef R__USE_IMT
   else {
      if (ex.Policy() != ROOT::EExecutionPolicy::kSequential) {
         return 2;
      }
   }

   ROOT::Internal::TExecutor ex1{ROOT::EExecutionPolicy::kMultiThread, 4u};
   if (ex1.Policy() != ROOT::EExecutionPolicy::kMultiThread) {
      return 3;
   }
#endif

   ROOT::Internal::TExecutor ex2{ROOT::EExecutionPolicy::kSequential};
   if (ex2.Policy() != ROOT::EExecutionPolicy::kSequential) {
      return 4;
   }

   auto offset = 14; // number of tests in TExecutorTest
   auto res = 0;
   res = ExecutorTest(ex);
   if (res)
      return res + 4;
#ifdef R__USE_IMT
   res = ExecutorTest(ex1);
   if (res)
      return res + offset + 4;
#endif
   res = ExecutorTest(ex2);
   if (res)
      return res + 2 * offset + 4;

   return 0;
}

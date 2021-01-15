#include "executorTests.hxx"
#include "ROOT/EExecutionPolicy.hxx"
#include "ROOT/TExecutor.hxx"
#include "TROOT.h"

int main() {

   // Build tests
   ROOT::Internal::TExecutor ex{4u};
   if (!ROOT::IsImplicitMTEnabled()) {
      if(ex.Policy()!= ROOT::EExecutionPolicy::kSequential) {
         return 1;
      }
   }
#ifdef R__USE_IMT
   else {
      if(ex.Policy()!= ROOT::EExecutionPolicy::kSequential) {
         return 2;
      }
   }

   ROOT::Internal::TExecutor ex1{ROOT::EExecutionPolicy::kMultiThread, 4u};
   if(ex1.Policy()!= ROOT::EExecutionPolicy::kMultiThread) {
         return 3;
   }
#endif

#ifndef _MSC_VER
   ROOT::Internal::TExecutor ex2{ROOT::EExecutionPolicy::kMultiProcess, 4u};
   if(ex2.Policy()!= ROOT::EExecutionPolicy::kMultiProcess) {
         return 4;
   }
#endif

   ROOT::Internal::TExecutor ex3{ROOT::EExecutionPolicy::kSequential};
   if(ex3.Policy()!= ROOT::EExecutionPolicy::kSequential) {
         return 5;
   }

   auto offset = 14; //number of tests in TExecutorTest
   auto res = 0;
   res = ExecutorTest(ex);
   if(res) return res+5;
#ifdef R__USE_IMT
   res = ExecutorTest(ex1);
   if(res) return res+offset+5;
#endif
#ifndef _MSC_VER
   res = ExecutorTest(ex2);
   if(res) return res+2*offset+5;
#endif
   res = ExecutorTest(ex3);
   if(res) return res+3*offset+5;
   return res? res+4*offset+5: 0;
}

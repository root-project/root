#include "executorTests.hxx"
#include "TROOT.h"
#include "ROOT/TExecutor.hxx"


int main() {

   // Build tests
   ROOT::Internal::TExecutor ex{};
   if (!ROOT::IsImplicitMTEnabled()) {
      if(ex.Policy()!= ROOT::EExecutionPolicy::kSequential) {
         return 0;
      }
   }
#ifdef R__USE_IMT
   else {
      if(ex.Policy()!= ROOT::EExecutionPolicy::kSequential) {
         return 1;
      }
   }

   ROOT::Internal::TExecutor ex1{ROOT::EExecutionPolicy::kMultiThread};
   if(ex1.Policy()!= ROOT::EExecutionPolicy::kMultiThread) {
         return 2;
   }
#endif

   ROOT::Internal::TExecutor ex2{ROOT::EExecutionPolicy::kMultiProcess};
   if(ex2.Policy()!= ROOT::EExecutionPolicy::kMultiProcess) {
         return 3;
   }

   ROOT::Internal::TExecutor ex3{ROOT::EExecutionPolicy::kSequential};
   if(ex3.Policy()!= ROOT::EExecutionPolicy::kSequential) {
         return 4;
   }

   auto offset = 14; //number of tests in TExecutorTest
   auto res = 0;
   res = ExecutorTest(ex);
   if(res) return res;
#ifdef R__USE_IMT
   res = ExecutorTest(ex1);
   if(res) return res+offset+4;
#endif
   res = ExecutorTest(ex2);
   if(res) return res+2*offset+4;
   res = ExecutorTest(ex3);
   if(res) return res+3*offset+4;
   return res? res+4*offset: 0;
}

#include "executorTests.hxx"
#include "TROOT.h"
#include "ROOT/TExecutor.hxx"


int main() {

   // Build tests
   ROOT::Internal::TExecutor ex{};
   if (!ROOT::IsImplicitMTEnabled()) {
      if(ex.Policy()!= ROOT::Internal::ExecutionPolicy::kSerial) {
         return 0;
      }
   }
#ifdef R__USE_IMT
   else {
      if(ex.Policy()!= ROOT::Internal::ExecutionPolicy::kSerial) {
         return 1;
      }
   }
#endif

   ROOT::Internal::TExecutor ex1{ROOT::Internal::ExecutionPolicy::kMultithread};
   if(ex1.Policy()!= ROOT::Internal::ExecutionPolicy::kMultithread) {
         return 2;
   }

   ROOT::Internal::TExecutor ex2{ROOT::Internal::ExecutionPolicy::kMultiprocess};
   if(ex2.Policy()!= ROOT::Internal::ExecutionPolicy::kMultiprocess) {
         return 3;
   }

   ROOT::Internal::TExecutor ex3{ROOT::Internal::ExecutionPolicy::kSerial};
   if(ex3.Policy()!= ROOT::Internal::ExecutionPolicy::kSerial) {
         return 4;
   }

   auto offset = 13; //number of tests in TExecutorTest
   auto res = 0;
   res = ExecutorTest(ex);
   if(res) return res;
   res = ExecutorTest(ex1);
   if(res) return res+offset+4;
   res = ExecutorTest(ex2);
   if(res) return res+2*offset+4;
   res = ExecutorTest(ex3);
   if(res) return res+3*offset+4;
   return res? res+4*offset: 0;
}

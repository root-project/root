#include "tExecutor.h"
#include "TROOT.h"
#include "ROOT/TSequentialExecutor.hxx"

int TSequentialExecutorTest() {

   //////////////Tests 1-11///////////////
   ROOT::TSequentialExecutor pool;
   auto res = TExecutorPoolTest(pool);
   if (res!=0)
      return res;

   unsigned j{};
   auto lambdaNTimes = [&](){j++;};
   
   pool.Foreach(lambdaNTimes, 4);
   if(j!=4)
      return 12;

   std::vector<int> vec{1,1,1,1};
   auto redfunc = [](std::vector<int> a) -> int { return std::accumulate(a.begin(), a.end(), 0); };
   pool.Foreach([&](int &i){i=2;}, vec);
   if(redfunc(vec) != 8)
      return 13;
   return 0;
}

int main(){
   return TSequentialExecutorTest();
}
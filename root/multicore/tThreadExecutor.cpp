#include "tExecutor.h"
#include "TROOT.h"
#include "ROOT/TThreadExecutor.hxx"

int PoolTest() {

   if(ROOT::TThreadExecutor::GetPoolSize()==0){
       ROOT::TThreadExecutor pool;
       if(ROOT::TThreadExecutor::GetPoolSize()==0)
        return 12;
       if(ROOT::GetImplicitMTPoolSize()!=0 && ROOT::TThreadExecutor::GetPoolSize()!=ROOT::GetImplicitMTPoolSize())
        return 13;

       auto tmp = ROOT::TThreadExecutor::GetPoolSize();
       ROOT::TThreadExecutor pool2(5);
       if(tmp!=ROOT::TThreadExecutor::GetPoolSize())
        return 14;
   } else{
       auto tmp = ROOT::TThreadExecutor::GetPoolSize();
       ROOT::TThreadExecutor pool2(5);
       if(tmp != ROOT::TThreadExecutor::GetPoolSize())
       return 15;
   }


   //////////////Tests 1-11///////////////
   ROOT::TThreadExecutor pool;
   auto res = TExecutorPoolTest(pool);
   if (res!=0)
      return res;
  ///////////////////////////////////////

   /***** chunking tests *****/

  auto redfunc = [](std::vector<int> a) -> int { return std::accumulate(a.begin(), a.end(), 0); };

  // init list and C++ function
   auto chunkedredres1 = pool.MapReduce(f, {0,1,0,1}, redfunc, 2);
   if(chunkedredres1 != 6)
      return 16;

  //nTimes + lambda signature
   auto chunkedredres2 = pool.MapReduce([](){return 1;}, 6, redfunc, 3);
   if(chunkedredres2 != 6)
      return 17;

  //TSeq signature, uneven chunks (8 elements in 3 chunks)
   auto chunkedredres3 = pool.MapReduce(f, ROOT::TSeq<int>(0,6), redfunc, 2);
   if(chunkedredres3 != 21)
      return 18;

   return 0;
}

int main() {
	return PoolTest();
}

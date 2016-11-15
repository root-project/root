#include "tExecutor.h"
#include "ROOT/TThreadExecutor.hxx"

int PoolTest() {
   ROOT::TThreadExecutor pool;
   auto res = TExecutorPoolTest(pool);
   if (res!=0)
      return res;

   /***** chunking tests *****/

  auto redfunc = [](std::vector<int> a) -> int { return std::accumulate(a.begin(), a.end(), 0); };

  // init list and C++ function
   auto chunkedredres1 = pool.MapReduce(f, {0,1,0,1}, redfunc, 2);
   if(chunkedredres1 != 6)
      return 12;

  //nTimes + lambda signature
   auto chunkedredres2 = pool.MapReduce([](){return 1;}, 6, redfunc, 3);
   if(chunkedredres2 != 6)
      return 13;

  //TSeq signature, uneven chunks (8 elements in 3 chunks)
   auto chunkedredres3 = pool.MapReduce(f, ROOT::TSeq<int>(0,6), redfunc, 2);
   if(chunkedredres3 != 21)
      return 14;

   return 0;
}

int main() {
	return PoolTest();
}

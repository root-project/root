#include "TH1F.h"
#include "ROOT/TThreadExecutor.hxx"
#include <vector>
#include <numeric> //accumulate

int f(int a)
{
   return a+1;
}

class fClass {
   public:
   int operator()(int a)
   {
      return a+1;
   }
};

TObject *rootF(TObject *o)
{
   TH1F *h = (TH1F*) o;
   h->FillRandom("gaus", 1);
   return h;
}

int PoolTest() {
   ROOT::TThreadExecutor pool;
   fClass c;
   auto boundF = std::bind(f, 1);

   /**** TThreadExecutor::Map ****/
   std::vector<int> truth = {1,1,1,1};

   // init list and lambda
   auto res = pool.Map([](int a) -> int { return a+1; }, {0,0,0,0});
   if( res != truth)
      return 1;

   // vector and C++ function
   std::vector<int> vargs = {0,0,0,0};
   auto res2 = pool.Map(f, vargs);
   if(res2 != truth)
      return 2;

   // vector and functor class
   auto res3 = pool.Map(c, vargs);
   if(res3 != truth)
      return 3;

   //nTimes signature and bound function
   auto res6 = pool.Map(boundF, 100);
   if(res6 != std::vector<int>(100,2))
      return 4;

   /**** TThreadExecutor::MapReduce ****/
   int redtruth = 4;
   auto redfunc = [](std::vector<int> a) -> int { return std::accumulate(a.begin(), a.end(), 0); };

   // init list and lambda
   auto redres = pool.MapReduce([](int a) { return a+1; }, {0,0,0,0}, redfunc);
   if(redres != redtruth)
      return 5;

   // vector and C++ function
   std::vector<int> vargs2 = {0,0,0,0};
   auto redres2 = pool.MapReduce(f, vargs2, redfunc);
   if(redres2 != redtruth)
      return 6;

   // vector and functor class
   auto redres3 = pool.MapReduce(c, vargs2, redfunc);
   if(redres3 != redtruth)
      return 7;

   //nTimes signature and bound function
   auto redres6 = pool.MapReduce(boundF, 100, redfunc);
   if(redres6 != 200)
      return 8;

   /***** chunking tests *****/

  // init list and C++ function
   std::vector<int> partialRedTruth = {3, 3};
   auto partialredres1 = pool.Map(f, {0,1,0,1}, redfunc, 2);
   auto chunkedredres1 = pool.MapReduce(f, {0,1,0,1}, redfunc, 2);
   if(partialredres1 != partialRedTruth || chunkedredres1 != 6)
      return 9;
  
  //nTimes + lambda signature
   auto partialredres2 = pool.Map([](){return 1;}, 6, redfunc, 2);
   unsigned xo=6;
   auto chunkedredres2 = pool.MapReduce([](){return 1;}, xo, redfunc);
   if(partialredres2 != partialRedTruth || chunkedredres2 != 6)
      return 10;

  //TSeq signature, uneven chunks (8 elements in 3 chunks)
   partialRedTruth = {6, 15, 15}; //1+2+3, 4+5+6, 7+8
   auto partialredres3 = pool.Map(f, ROOT::TSeq<int>(0,8), redfunc, 3);
   auto chunkedredres3 = pool.MapReduce(f, ROOT::TSeq<int>(0,6), redfunc, 2);
   if(partialredres3 != partialRedTruth  || chunkedredres3 != 21)
      return 11;

   /***** other tests *****/

   //returning a c-string
    auto extrares1 = pool.Map([]() { return "42"; }, 25);
    for(auto c_str : extrares1)
       if(strcmp(c_str, "42") != 0)
          return 12;

   //returning a string
   auto extrares2 = pool.Map([]() { return std::string("fortytwo"); }, 25);
   for(auto str : extrares2)
      if(str != "fortytwo")
         return 13;

   return 0;
}

int main() {
	return PoolTest();
}

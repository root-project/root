#include "TApplication.h"
#include "TH1F.h"
#include "TPool.h"
#include "TList.h"
#include <functional>
#include <list>
#include <vector>
#include <numeric> //accumulate
#include <iostream>

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
   TH1F *h = (TH1F*)o;
   h->FillRandom("gaus", 1);
   return h;
}

int PoolTest() {
   TPool pool; //To be changed back to TProcPool
   fClass c;
   auto boundF = std::bind(f, 1);

   /**** TProcPool::Map ****/
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
   // std::list and functor class
   std::list<int> largs = {0,0,0,0};
   auto res3 = pool.Map(c, largs);
   if(res3 != truth)
      return 3;
   // TList
   TList tlargs;
   tlargs.Add(new TH1F("h1","h",100,-3,3));
   tlargs.Add(new TH1F("h2","h",100,-3,3));
   auto res4 = pool.Map(rootF, tlargs);
   if(res4.GetEntries() != 2 || ((TH1F*)res4[0])->GetEntries() != 1 || ((TH1F*)res4[1])->GetEntries() != 1)
      return 4;
   res4.Delete();
   tlargs.Delete();
   //nTimes signature and bound function
   auto res6 = pool.Map(boundF, 100);
   if(res6 != std::vector<int>(100,2))
      return 6;

   /**** TProcPool::MapReduce ****/
   int redtruth = 4;
   auto redfunc = [](std::vector<int> a) -> int { return std::accumulate(a.begin(), a.end(), 0); };
   // init list and lambda
   auto redres = pool.MapReduce([](int a) { return a+1; }, {0,0,0,0}, redfunc);
   if(redres != redtruth)
      return 8;
   // vector and C++ function
   std::vector<int> vargs2 = {0,0,0,0};
   auto redres2 = pool.MapReduce(f, vargs2, redfunc);
   if(redres2 != redtruth)
      return 9;
   // std::list and functor class
   std::list<int> largs2 = {0,0,0,0};
   auto redres3 = pool.MapReduce(c, largs2, redfunc);
   if(redres3 != redtruth)
      return 10;
   // TList
   TList tlargs2;
   tlargs2.Add(new TH1F("h1","h",100,-3,3));
   tlargs2.Add(new TH1F("h2","h",100,-3,3));
   TH1F* redres4 = static_cast<TH1F*>(pool.MapReduce(rootF, tlargs2, PoolUtils::ReduceObjects));
   if(redres4->GetEntries() != 2)
      return 11;
   tlargs2.Delete();
   delete redres4;
   //nTimes signature and bound function
   auto redres6 = pool.MapReduce(boundF, 100, redfunc);
   if(redres6 != 200)
      return 12;

   /***** other tests *****/

   //returning a c-string
   auto extrares1 = pool.Map([]() { return "42"; }, 25);
   for(auto c_str : extrares1)
      if(strcmp(c_str, "42") != 0)
         return 13;
   for(auto c_str : extrares1)
      delete [] c_str;

   //returning a string
   auto extrares2 = pool.Map([]() { return std::string("fortytwo"); }, 25);
   for(auto str : extrares2)
      if(str != "fortytwo")
         return 14;

   return 0;
}

int main() {
	TApplication app("app",nullptr,nullptr);
	return PoolTest();
}

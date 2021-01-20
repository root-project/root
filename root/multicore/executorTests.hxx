#include "ROOT/TSeq.hxx"
#include "TH1F.h"
#include "TRandom.h"
#include <functional>
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

template<class T>
int ExecutorTest(T &executor) {
   fClass c;
   auto boundF = std::bind(f, 1);

   /**** TProcessExecutor::Map ****/
   std::vector<int> truth = {1,1,1,1};

   //nTimes signature and bound function
   auto res1 = executor.Map(boundF, 100);
   if(res1 != std::vector<int>(100,2))
      return 1;

   // TSeq and lambda
   auto res2 = executor.Map([](int) -> int { return 1; } , ROOT::TSeq<int>(0, 4));
   if(res2 != truth)
      return 2;

   // init list and lambda
   auto res3 = executor.Map([](int a) -> int { return a+1; }, {0,0,0,0});
   if( res3 != truth)
      return 3;

   // vector and C++ function
   std::vector<int> vargs = {0,0,0,0};
   auto res4 = executor.Map(f, vargs);
   if(res4 != truth)
      return 4;

   // vector and functor class
   auto res5 = executor.Map(c, vargs);
   if(res5 != truth)
      return 5;


   /**** TProcessExecutor::MapReduce ****/
   int redtruth = 4;
   auto redfunc = [](std::vector<int> a) -> int { return std::accumulate(a.begin(), a.end(), 0); };

   //nTimes signature and bound function
   auto redres1 = executor.MapReduce(boundF, 100, redfunc);
   if(redres1 != 200)
      return 6;

   // TSeq and lambda
   auto redes2 = executor.MapReduce([](int) -> int { return 1; } , ROOT::TSeq<int>(0, 4), redfunc);
   if(redes2 != redtruth)
      return 7;

   // init list and lambda
   auto redres3 = executor.MapReduce([](int a) { return a+1; }, {0,0,0,0}, redfunc);
   if(redres3 != redtruth)
      return 8;

   // vector and C++ function
   std::vector<int> vargs2 = {0,0,0,0};
   auto redres4 = executor.MapReduce(f, vargs2, redfunc);
   if(redres4 != redtruth)
      return 9;

   // vector and functor class
   auto redres5 = executor.MapReduce(c, vargs2, redfunc);
   if(redres5 != redtruth)
      return 10;

   // const vector and functor class
   std::vector<int> vargs3 = {0,0,0,0};
   auto redres6 = executor.MapReduce(c, vargs3, redfunc);
   if(redres6 != redtruth)
      return 11;

   //TObject::Merge() reduction signature.
   TH1F *htot = new TH1F("htot", "htot", 10, 0, 1);
   std::vector<TH1 *> vhist(5);
   vhist[0] = new TH1F("h0", "h0", 10, 0, 1);
   vhist[1] = new TH1F("h1", "h1", 10, 0, 1);
   vhist[2] = new TH1F("h2", "h2", 10, 0, 1);
   vhist[3] = new TH1F("h3", "h3", 10, 0, 1);
   vhist[4] = new TH1F("h4", "h4", 10, 0, 1);

   for(auto i=0; i<50; i++){
      auto x = gRandom->Gaus(-3,2);
      vhist[i/10]->Fill(x);
      htot->Fill(x);
   }
   auto hred = executor.Reduce(vhist);

   for (auto i = 0; i < 12; i++) {
      if(htot->GetBinContent(i) != hred->GetBinContent(i))
         return 12;
   }

   delete htot;
   delete hred;
   for(auto el: vhist){
      delete el;
   }

    /***** other tests *****/

   //returning a c-string
   auto extrares1 = executor.Map([]() { return "42"; }, 25);
   for(auto c_str : extrares1)
      if(strcmp(c_str, "42") != 0)
         return 13;

   //returning a string
   auto extrares2 = executor.Map([]() { return std::string("fortytwo"); }, 25);
   for(auto str : extrares2)
      if(str != "fortytwo")
         return 14;

   return 0;
}

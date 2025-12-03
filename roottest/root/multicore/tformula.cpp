#include "TFormula.h"
#include "TROOT.h"
#include "TObject.h"
#include <thread>
#include <memory>
#include <atomic>
#include <cassert>

int main()
{
  constexpr int kNThreads = 20;
 std::atomic<int> canStart{kNThreads};
 std::vector<std::thread> threads;

 //Tell Root we want to be multi-threaded
 ROOT::EnableThreadSafety();
 //When threading, also have to keep ROOT from logging all TObjects into a list
 TObject::SetObjectStat(false);

 for(unsigned int i=0; i<kNThreads; ++i) {
   threads.emplace_back([i,&canStart]() {
       --canStart;
       while( canStart > 0 ) {}

       TFormula f("testFormula","1./(1.+(4.61587e+06*(((1./(0.5*TMath::Max(1.e-6,x+1.)))-1.)/1.16042e+07)))");

       for(int i=0; i<1000;++i) {
         double x = double(i)/100.;
         f.Eval(x);
       }
     });
 }
 canStart = true;

 for(auto& thread: threads) {
   thread.join();
 }

 return 0;
}



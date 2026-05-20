#include "commonutils.h"

int exectsautoparse(){

   gEnv->SetValue("RooFit.Banner", 0);

   ROOT::EnableThreadSafety();
   
   std::atomic<bool> fire(false);
   vector<thread> threads;
   for (auto const & key : keys){
      auto f = [&](){
         while(true){
            if (fire.load()){
               gInterpreter->AutoParse(key);
               break;
            }
         }
      };
      threads.emplace_back(f);
   }
   fire.store(true);
   for (auto&& t : threads) t.join();
   return 0;
}

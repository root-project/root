void execautoloadts(){
   auto keys {"TH1F",
              "TXMLFile",
              "TGraph",
              "RooRealVar",
              "RooVoigtian",
              "RooStats::LikelihoodInterval",
              "TMultiLayerPerceptron",
              "TBrowser",
              "THtml",
              "ROOT::Math::GeneticMinimizer"};

   TThread::Initialize();
//    gSystem->ListLibraries();
   std::atomic<bool> fire(false);
   vector<thread> threads;
   for (auto const & key : keys){
      auto f = [&](){
         while(true){
            if (fire.load()){
               TClass::GetClass(key);
//                printf("Autoloaded for key %s\n",key);
               break;
            }
         }
      };
      threads.emplace_back(f);
   }
   fire.store(true);
   for (auto&& t : threads) t.join();
//    gSystem->ListLibraries();
}

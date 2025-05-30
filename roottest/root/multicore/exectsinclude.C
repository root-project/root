class tsStringlist {
public:
   void addString(const std::string& str){
   std::lock_guard<std::mutex> lg(fMutex);
      fNames.push_back(str);
   }
   const std::list<std::string>& getStrings() const {return fNames;}
private:
   std::list<std::string> fNames;
   std::mutex fMutex;

};

void exectsinclude(){

   auto headers = {"TH1F.h",
                   "TGraph.h",
#if __has_include("RooRealVar.h")
                   "RooRealVar.h",
		   "RooGaussian.h",
#endif
                   "TBrowser.h",
                   "TMatrixD.h",
                   "TF1.h",
#if __has_include("TGeoSphere.h")
                   "TGeoSphere.h",
#endif
#if __has_include("TGLClip.h")
                   "TGLClip.h",
#endif
#if __has_include("TEveBox.h")
                   "TEveBox.h",
#endif
                   "TBufferXML.h",
                   "TQClass.h"};

   tsStringlist inclusions;
   gInterpreter->SetClassAutoloading(false);
   ROOT::EnableThreadSafety();

   std::atomic<bool> fire(false);
   vector<thread> threads;
   for (auto const & header : headers){
      auto f = [&](){
         while(true){
            string s = "#include \"";
            s+=header;
            s+="\"";
            if (fire.load()){
               gInterpreter->ProcessLine(s.c_str());
               inclusions.addString(s);
               break;
            }
         }
      };
      threads.emplace_back(f);
   }
   fire.store(true);
   for (auto&& t : threads) t.join();
   std::list<std::string> inclusionsList (inclusions.getStrings());
   inclusionsList.sort();
   for (auto&& inc:inclusionsList) printf("Line processed \"%s\"\n",inc.c_str());
}


#include "TProcPool.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TFileCollection.h"
#include "TH1F.h"
#include "TChain.h"
#include "TROOT.h"
#include <memory>

TH1F* myMacro(TTreeReader& reader) {

   TTreeReaderValue<Float_t> px(reader, "px");

   TH1F *myhisto = new TH1F("h","h",100,-3,3);
   while(reader.Next())
      myhisto->Fill(*px);

   return myhisto;
}

int main() {

   TString hsimpleLocation = gROOT->GetTutorialsDir();
   hsimpleLocation+="/hsimple.root";

   std::vector<std::unique_ptr<TH1F>> res(6);
   std::vector<std::string> files(4,hsimpleLocation.Data());
   TProcPool pool(3);

   //TProcPool::Process with single file name and tree name
   //Note: we have less files than workers here
   res[0].reset(pool.Process(hsimpleLocation.Data(), myMacro, "ntuple"));

   //TProcPool::Process with vector of files and tree name
   //Note: we have more files than workers here (different behaviour)
   res[1].reset(pool.Process(files, myMacro, "ntuple"));

   //TProcPool::Process with single file name, tree name and entries limit
   res[2].reset(pool.Process(hsimpleLocation.Data(), myMacro, "ntuple", 42));

   //TProcPool::Process with vector of files, no tree name and entries limit
   res[3].reset(pool.Process(files, myMacro, "ntuple", 10000));

   //TProcPool::Process with TFileCollection, no tree name and entries limit
   TFileCollection fc;
   fc.Add(hsimpleLocation.Data());
   fc.Add(hsimpleLocation.Data());
   res[4].reset(pool.Process(fc, myMacro, "", 42));

   //TProcPool::Process with TChain, no tree name and entries limit
   TChain c;
   c.Add(hsimpleLocation.Data());
   c.Add(hsimpleLocation.Data());
   res[5].reset(pool.Process(c, myMacro));

   for(auto&& r : res)
      std::cout << r->GetEntries() << "\n";
   std::cout << std::flush;

   return 0;
}


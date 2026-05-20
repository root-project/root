#include "TString.h"
#include "TROOT.h"
#include "TTree.h"
#include "TH1F.h"
#include "TTreeReader.h"
#include "ROOT/TTreeProcessorMP.hxx"

TH1F* myMacro(TTreeReader& reader) {

   TTreeReaderValue<Float_t> px(reader, "px");

   TH1F *myhisto = new TH1F("h","h",100,-3,3);
   while(reader.Next())
      myhisto->Fill(*px);

   return myhisto;
}

int main() {
   // MacOSX may generate connection to WindowServer errors
   gROOT->SetBatch(kTRUE);
   
   TString hsimpleLocation = gROOT->GetTutorialsDir();
   hsimpleLocation+="/hsimple.root";

   std::unique_ptr<TFile> fp(TFile::Open(hsimpleLocation));
   TTree* tree;
   fp->GetObject("ntuple",tree);

   ROOT::TTreeProcessorMP pool(2);
   auto res = pool.Process(*tree, myMacro);

   std::cout << res->GetEntries() << std::endl;

   return 0;
}

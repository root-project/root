#include "TString.h"
#include "TROOT.h"
#include "TTree.h"
#include "TH1F.h"
#include "TTreeReader.h"
#include "TProcPool.h"

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

   std::unique_ptr<TFile> fp(TFile::Open(hsimpleLocation));
   TTree *tree = static_cast<TTree*>(fp->Get("ntuple"));

   TProcPool pool(2);
   auto res = static_cast<TH1F*>(pool.Process(*tree, myMacro));

   std::cout << res->GetEntries() << std::endl;

   return 0;
}

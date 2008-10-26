#include "TFile.h"
#include "TChain.h"
#include "Riostream.h"

class Bottom {
public:
   int i;
   int j;
};

class Top {
   int a;
   Bottom b;
};

void write(const char *stem, Int_t splitlevel) {
   TFile file(TString::Format("%s-%d.root",stem,splitlevel),"RECREATE");
   TTree tree("T","T title");
   Top obj;
   tree.Branch("obj.",&obj,32000,splitlevel);
   tree.Fill();
   file.Write();
}

void runSplitMismatch() {
   write("missplit",1);
   write("missplit",99);
   TChain c("T");
   c.Add("missplit-*.root");
   TFile f("missplit.fail.root","RECREATE");
   c.CloneTree(-1,"fast");
   f.Write();
   TTree *output; f.GetObject("T",output);
   if (output->GetEntries() > 1) {
      cout << "Error: the split 99 TTree was used eventhough it is not compatible with the first (split 1) TTree)\n";
   }
}


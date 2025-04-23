#include "TFile.h"
#include "TChain.h"
#include "Riostream.h"

class Bottom {
public:
   int i;
   int j;
};

class Top {
public:
   int a;
   Bottom b;
};

void write(const char *stem, Int_t splitlevel) {
   TFile file(TString::Format("%s-%d.root",stem,splitlevel),"RECREATE");
   TTree tree("T","T title");
   Top obj;
   obj.a = 100+splitlevel;
   obj.b.i = 200+splitlevel;
   obj.b.j = 300+splitlevel;
   tree.Branch("obj.",&obj,32000,splitlevel);
   tree.Fill();
   file.Write();
   tree.Scan("obj.a:obj.b.i:obj.b.j");
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
   output->Scan("obj.a:obj.b.i:obj.b.j");
}


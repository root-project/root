#include "TH1F.h"
#include "parent.h"
#include <stdio.h>
#include "TStopwatch.h"


class child :public parent {
  public:
    child(TTree* t):parent(t) {};
    virtual void Loop() {
      TH1F* h = new TH1F("h","h",1000,0,10000);
      if (fChain == 0) return;

      Long64_t nentries = fChain->GetEntriesFast();

      Long64_t nbytes = 0, nb = 0;
      for (Long64_t jentry=0; jentry<nentries;jentry++) {
        Long64_t ientry = LoadTree(jentry);
        if (ientry < 0) break;
        nb = fChain->GetEntry(jentry);   nbytes += nb;
        // if (Cut(ientry) < 0) continue;
        //    }
        //

        h->Fill(B0_MM);
      }
      delete h;
      return;
    }
};

int main(int argc,char** argv) {
  TStopwatch watch;
  TFile* inf = new TFile(Form("size.%d.%d.root",atoi(argv[1]),atoi(argv[2])),"read");
  TTree* intree;
  inf->GetObject("B02DD",intree);
  child p(intree);
  p.Loop();
  watch.Start();
  p.Loop();
  watch.Stop();
  printf("wall clock time: %f\n",watch.RealTime());
  return 0;
}

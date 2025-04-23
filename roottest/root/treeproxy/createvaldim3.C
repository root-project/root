#include <TFile.h>
#include <TTree.h>
#include <TClonesArray.h>

#if defined(R__MACOSX) && !defined(MAC_OS_X_VERSION_10_5)
# include "Track.C"
#else
# if defined(__CINT__) && !defined(__MAKECINT__)
#  include "Track.C+"
# else
#  include "Track.h"
# endif
#endif

void createvaldim3(bool process = false) 
{
  Double_t a[2][3][4];
  Double_t bb[2][3];
  Int_t c[2];
  TClonesArray *clones = new TClonesArray("Track");
  Track *tr = new Track;

  TFile file("forproxy.root","RECREATE");
  TTree *t = new TTree("t","t");
  t->Branch("a",a,"a[2][3][4]/D");
  t->Branch("bb",bb,"bb[2][3]/D");
  t->Branch("c",c,"c[2]/I");
  t->Branch("tr.",&tr);
  t->Branch("trs", &clones);

  tr->Set(3);
  for(int i=0;i<2;++i) {
     c[i] = i;
     for(int j=0;j<3;++j) {
        bb[i][j] = i*100+j*10;
        for(int k=0;k<4;++k) {
           a[i][j][k] = i*100+j*10+k;
        }
     }
     Track *otrack = new ( (*clones)[i] ) Track();
     otrack->Set(i);
  }
  t->Fill();
  file.Write();

  //t->Print();

  t->MakeProxy("val3dimSel","val3dim.C");
  if (process) t->Process("val3dimSel.h+");
  
}

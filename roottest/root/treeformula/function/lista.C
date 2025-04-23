#include "TFile.h"
#include "TObject.h"
#include "TH1F.h"
#include "TTree.h"
#include "TClonesArray.h"
#include <list>

//_____________________________________________________________________________

class TStrip: public TObject {
  
public:
  TStrip(const char * /* name */ =0){}; 
  ~TStrip() override { // delete all hists here - well, not in your example case where you have 5 global ones 
  }
  void pushback(TH1F *histog){fHists.push_back(histog);};
private:
  std::list<TH1F*> fHists;
  ClassDefOverride(TStrip,1); // a strip
};

class TPlate: public TObject {
  
public:
  TPlate(const char * /* name */ =0){};
  ~TPlate() override { // delete all entries in fStrips here 
  }
  void pushback(TStrip *strip){fStrips.push_back(strip);};
  
private:
  std::list<TStrip*> fStrips;
  ClassDefOverride(TPlate,1); // a plate
};

#ifdef __MAKECINT__
#pragma link C++ class TStrip+;
#pragma link C++ class TPlate+;
#endif


void lista(){
  
  
  TH1F *hist[5];
  hist[0] = new TH1F("hist0", "hist0", 100,-1.,1.);
  hist[1] = new TH1F("hist1", "hist1", 100,-1.,1.);
  hist[2] = new TH1F("hist2", "hist2", 100,-1.,1.);
  hist[3] = new TH1F("hist3", "hist3", 100,-1.,1.);
  hist[4] = new TH1F("hist4", "hist4", 100,-1.,1.);
  
  char stripname[10];
  char platename[10];

  TFile *fileout = new TFile("prova.root","recreate");

  TClonesArray *ca=new TClonesArray("TPlate");
  TClonesArray &aca=*ca;

  TTree *fTreeEq = new TTree("TreeEq","Tree of Plates and Sectors"); 
  fTreeEq->Branch("sectors", &ca);

  // for (Int_t entry=0; entry<10; entry++) {
  for (Int_t k=0; k<2; k++){
    sprintf(platename,"plate%i",k);
    TPlate *Plate = new(aca[k]) TPlate (platename);
    for (Int_t i=0; i<1; i++){
      sprintf(stripname,"strip%i",i);
      TStrip *Strip = new TStrip(stripname);
      for (Int_t j=0; j<5; j++){
	Strip->pushback(hist[j]);
      }
      Plate->pushback(Strip);
    }
  }
  fTreeEq->Fill();
  // for (Int_t k=0; k<2; k++){
  //   TPlate *Plate = (TPlate*)aca[k];
  //   Plate->Clear() // deletes all entries in the list
  // }
  // ca->Clear();
  
  fTreeEq->Write();
  delete fileout;

}


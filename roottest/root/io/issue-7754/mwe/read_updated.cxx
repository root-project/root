#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <iostream>
#include <TSystem.h>

#include "include/TSpectrometerEvent.hh"
#include "include/TSpectrometerHit.hh"

void read_updated(bool newpers) {

  TFile *fd = nullptr;
  if(newpers)
    fd = TFile::Open("pluto_newpers.root", "OPEN");
  else
    fd = TFile::Open("pluto_oldpers.root", "OPEN");
  std::cout << "File opened: " << fd->GetName() << std::endl;
  TTree *SlimMC = static_cast<TTree*>(fd->Get("SlimMC"));

  //fd->ShowStreamerInfo();

  TSpectrometerEvent *evt = 0;
  TBranch *br = nullptr;
  SlimMC->SetBranchAddress("Spectrometer", &evt, &br);

  std::cout << "Number of entries " << SlimMC->GetEntries() << std::endl;
  //SlimMC->GetEntry(0);
  br->GetEntry(0);
  // br->Print("debugInfo");
 
  std::cout << evt->GetNHits() << std::endl;
  for(int iHit=0; iHit<evt->GetNHits() /* && iHit < 10 */; ++iHit){
    TSpectrometerHit *hit = static_cast<TSpectrometerHit*>(evt->GetHit(iHit));
    std::cout << hit->GetMCTrackID() << " " << hit->GetKinePartIndex() << std::endl;
  }
}


int main(int argc, char**argv) {
  bool newpers = false;
  if(argc>1)
    newpers = argv[1][0]=='1';
  read_updated(newpers);
  return 0;
}

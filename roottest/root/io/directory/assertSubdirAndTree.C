#include "TTree.h"
#include "TFile.h"

void assertSubdirAndTree() 
{
  TFile *file = TFile::Open("Collision12-ANNPID.root");
  file->cd("Tuple_BDKstar");
  TTree *BDKstarTuple; gDirectory->GetObject("BDKstarTuple",BDKstarTuple);
  BDKstarTuple->Draw("KsPip_Hlt2TopoOSTF2BodyDecision_TOS");
  delete file;
}


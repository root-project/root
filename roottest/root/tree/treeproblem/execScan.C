#include "TFile.h"
#include "TNtuple.h"
#include <iostream>

void execScan()
{
  TFile *f = TFile::Open("trial.root", "RECREATE");
  TNtuple *n = new TNtuple("n", "n", "x");
  for (Int_t i = 0; i < 10; i++) n->Fill(-i);
  Double_t xmin = 0, xmax = 0;
  xmin = n->GetMinimum("x"); // after "f->Write();", breaks "n->Scan();" below
  xmax = n->GetMaximum("x"); // ... ditto ...
  n->Scan();
  n->GetEntry(3);
  std::cout << xmin << " ... " << xmax << std::endl;
  f->Write(); // breaks "n->Scan();" below if "xmin" or "xmax" were calculated
  n->Scan();
}

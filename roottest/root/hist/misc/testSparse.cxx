// This script is mostly based on Wile's one from https://root-forum.cern.ch/t/reading-and-writing-thnsparse-to-file-with-variable-bin-size/15365

#include "TCanvas.h"
#include "TFile.h"
#include "THnSparse.h"
#include "TH1.h"

#include <iostream>

using std::cout, std::endl;

void writeSparse(bool useVarBins=false, bool draw=false)
{
  std::cout << "... writeSparse ..." << std::endl;
  // Define the fixed bins for initialization
  const int nDim = 3;
  // Int_t nBins[nDim] = {10, 4, 4};
  Int_t nBins[nDim] = {5, 2, 2};
  Double_t min[nDim] = {0., 0., 0.};
  Double_t max[nDim] = {5., 2., 2.};

  TFile* f = new TFile("testSparse.root", "recreate");
  THnSparseF* h = new THnSparseF("sparseHist", "sparseHist", nDim, nBins, min, max);

  // Set variable bins
  if(useVarBins){
    const int nVarBins0 = 5; // 5
    Double_t varBins0[nVarBins0+1] = {0., 1., 2., 3., 4., 5.};//, 6, 7, 8, 9, 10};
    h->GetAxis(0)->Set(nVarBins0, varBins0);

    const int nVarBins1 = 2; // 2
    Double_t varBins1[nVarBins1+1] = {0., 0.5, 2.};//, 3, 4};
    h->GetAxis(1)->Set(nVarBins1, varBins1);

    const int nVarBins2 = 2; // 2
    Double_t varBins2[nVarBins2+1] = {0., 1.5, 2.};//, 3, 4};
    h->GetAxis(2)->Set(nVarBins2, varBins2);
  }

  // Fill 2 bins
  Double_t val1[nDim] = {0, 1, 0};
  Double_t val2[nDim] = {3, 0, 1};
  h->Fill(val2);
  h->Fill(val2);
  h->Fill(val1);

  cout << "printing entries with no options:" << endl;
  h->PrintEntries();
  cout << "printing entries with '0':" << endl;
  h->PrintEntries(0, -1, "0");
  cout << "printing entries with 'x':" << endl;
  h->PrintEntries(0, -1, "x");
  cout << "printing entries with 'x0':" << endl;
  h->PrintEntries(0, -1, "0x");

  // Draw projections
  if(draw){
    TCanvas* c0 = new TCanvas("c0", "c0", 500, 500);
    h->Projection(0)->Draw();

    TCanvas* c1 = new TCanvas("c1", "c1", 500, 500);
    h->Projection(1)->Draw();

    TCanvas* c2 = new TCanvas("c2", "c2", 500, 500);
    h->Projection(2)->Draw();
  }

  h->Write();
  if (!draw) delete f;
}

void readSparse(bool draw=false)
{
  std::cout << "... readSparse ..." << std::endl;
  TFile* f = new TFile("testSparse.root");
  THnSparseF* h = (THnSparseF*) f->Get("sparseHist");

  cout << "printing entries with no options:" << endl;
  h->PrintEntries();
  cout << "printing entries with '0':" << endl;
  h->PrintEntries(0, -1, "0");
  cout << "printing entries with 'x':" << endl;
  h->PrintEntries(0, -1, "x");
  cout << "printing entries with 'x0':" << endl;
  h->PrintEntries(0, -1, "x0");

  // Draw projections
  if(draw){
    TCanvas* c0 = new TCanvas("c0", "c0", 500, 500);
    h->Projection(0)->Draw();

    TCanvas* c1 = new TCanvas("c1", "c1", 500, 500);
    h->Projection(1)->Draw();

    TCanvas* c2 = new TCanvas("c2", "c2", 500, 500);
    h->Projection(2)->Draw();
  }

  if (!draw) delete f;
}

void testSparse() {
  writeSparse();
  readSparse();
}

#include <cstdlib>
#include <iostream>
#include <string>

#include "TROOT.h"
#include "TFile.h"
#include "TClonesArray.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1.h"
#include "TCanvas.h"

#include "bar.h"

using namespace std;

bool runHisto(TTree *tree, const char *what, const char* where, double mean) {
  string cmd = what;
  cmd += (">>"); cmd += where;
  
  tree->Draw(cmd.c_str());
  
  TH1F* hist = (TH1F*)gROOT->FindObject(where);

  if (hist==0) {
    cerr << "Histograms for " << what << " not created" << endl;
    return false;
  } else if (hist->GetMean()!=mean) {
    cerr << "Histograms for " << what << " improperly created mean is " 
         << hist->GetMean() << " instead of " << mean << endl;
    return false;
  }
  return true;

}


int run() {
  bool result = true;

  TFile *h = new TFile("Event.root", "RECREATE", "ROOT file");

  bar *b = new bar();
  for (int k = 0; k < 2; k++) {
    b->v[k] = k;
  }
  // b->f = new TClonesArray("foo", 10);
  for (int j = 0; j < 2; j++) {
    b->fop[j]->i = j+1;
    b->fop[j]->f = 2*b->fop[j]->i;

    b->fo[j].i = j+1;
    b->fo[j].f = 2*b->fo[j].i;

    b->fp[j]->i = j+1;
    b->fp[j]->f = 2*b->fp[j]->i;

    b->f[j].i = j+1;
    b->f[j].f = 2*b->f[j].i;


    b->fov[j].i = j+1;
    b->fov[j].f = 2*b->fop[j]->i;

    b->fvop[j].i = j+1;
    b->fvop[j].f = 2*b->fo[j].i;

    b->fv[j].i = j+1;
    b->fv[j].f = 2*b->fp[j]->i;

    b->fvp[j].i = j+1;
    b->fvp[j].f = 2*b->f[j].i;

    
  }

  TTree *tree = new TTree("T","An example of a ROOT tree");
  // T->Draw("f[].i"); does not work with a .!
  TBranch *br = tree->Branch("a/b", "bar", &b,32000,0);
  
  for (int i = 0; i < 10; i++) {
    tree->Fill();
  }
  
  h->Write();

  // Now do the actual test ... quickly
  gROOT->SetBatch(kTRUE);
  new TCanvas("c1");

  delete b;
  b = new bar();
  tree->SetBranchAddress("a/b",&b);
  b->print();
  tree->GetEntry(3);
  b->print();

  b = new bar();
  tree->SetBranchAddress("a/b",&b);
  b->print();
  tree->GetEntry(2);
  b->print();

  return 1;

  result &= runHisto(tree, "fo.i","hist0",1.5);
  result &= runHisto(tree, "fo[].i","hist1",1.5);
  result &= runHisto(tree, "fo[].f","hist1",3);

  result &= runHisto(tree, "fop.i","hist0",1.5);
  result &= runHisto(tree, "fop[].i","hist1",1.5);
  result &= runHisto(tree, "fop[].f","hist1",3);

  result &= runHisto(tree, "fp.i","hist0",1.5);
  result &= runHisto(tree, "fp[].i","hist1",1.5);
  result &= runHisto(tree, "fp[].f","hist1",3);

  result &= runHisto(tree, "f.i","hist0",1.5);
  result &= runHisto(tree, "f[].i","hist1",1.5);
  result &= runHisto(tree, "f[].f","hist1",3);


  // Note the same fail without the original library.
  // Should also test arrays of TString, TNamed and TObject.

  h->Close();

  return result;

}

#ifndef __CINT__
int
main()
{
   return !run();
}
#endif

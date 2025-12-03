#include <cstdlib>
#include <iostream>

#include "TROOT.h"
#include "TFile.h"
#include "TClonesArray.h"
#include "TTree.h"
#include "TBranch.h"
#include "TH1.h"
#include "TCanvas.h"

#include "bar.h"
#include "foo.h"

#ifdef __ROOTCLING__
#pragma link C++ class bar+;
#pragma link C++ class foo+;
#endif

using namespace std;

int run()
{
   int result = 0;

   TROOT s("simple", "Example of creation of a tree");
   TFile *h = new TFile("Event.root", "RECREATE", "ROOT file");

   bar *b = new bar();
   for (int k = 0; k < 2; k++) {
      b->v[k] = k;
   }
   b->f = new TClonesArray("foo", 10);
   for (int j = 0; j < 10; j++) {
      new((*(b->f))[j]) foo(j);
   }

   TTree *tree = new TTree("T","An example of a ROOT tree");
   // T->Draw("f[].i"); does not work with a .!
   // TBranch *br =
   tree->Branch("a/b", "bar", &b);

   for (int i = 0; i < 100; i++) {
      tree->Fill();
   }

   h->Write();

   // Now do the actual test ... quickly
   gROOT->SetBatch(kTRUE);
   new TCanvas("c1");

   tree->Draw("f.i>>hist0");
   TH1F* hist0 = (TH1F*)gROOT->FindObject("hist0");
   if (hist0==0) {
      cerr << "Histograms for f[].i not created" << endl;
      result = 1;
   } else if (hist0->GetMean()!=4.5) {
      cerr << "Histograms for f.i improperly created mean is "
           << hist0->GetMean() << " instead of 4.5" << endl;
      result = 2;
   }

   tree->Draw("f[].i>>hist1");
   TH1F* hist1 = (TH1F*)gROOT->FindObject("hist1");
   if (hist1==0) {
      cerr << "Histograms for f[].i not created" << endl;
      result = 1;
   } else if (hist1->GetMean()!=4.5) {
      cerr << "Histograms for f[].i improperly created mean is "
           << hist1->GetMean() << " instead of 4.5" << endl;
      result = 2;
   }

   tree->Draw("f>>hist2");
   TH1F* hist2 = (TH1F*)gROOT->FindObject("hist2");
   if (hist2==0) {
      cerr << "Histograms for f not created" << endl;
      result = 1;
   } else if (hist2->GetMean()!=9) {
      cerr << "Histograms for f[].i improperly created mean is "
           << hist2->GetMean() << " instead of 9" << endl;
      result = 2;
   }

   h->Close();
   return result;
}

int
main()
{
   return run();
}


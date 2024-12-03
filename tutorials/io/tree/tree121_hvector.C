/// \file
/// \ingroup tutorial_tree
/// \notebook
/// Write and read STL vectors in a tree.
///
/// \macro_image
/// \macro_code
///
/// \author The ROOT Team

#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TCanvas.h"
#include "TFrame.h"
#include "TH1F.h"
#include "TBenchmark.h"
#include "TRandom.h"
#include "TSystem.h"

void write_vector()
{
   auto f = TFile::Open("hvector.root","RECREATE");

   if (!f)
      return;

   // Create one histograms
   auto hpx = new TH1F("hpx","This is the px distribution", 100, -4, 4);
   hpx->SetFillColor(48);

   std::vector<float> vpx;
   std::vector<float> vpy;
   std::vector<float> vpz;
   std::vector<float> vrand;

   // Create a TTree
   TTree *t = new TTree("tvec", "Tree with vectors");
   t->Branch("vpx", &vpx);
   t->Branch("vpy", &vpy);
   t->Branch("vpz", &vpz);
   t->Branch("vrand", &vrand);

   // Create a new canvas.
   auto c1 = new TCanvas("c1", "Dynamic Filling Example", 200, 10, 700, 500);

   gRandom->SetSeed();
   const Int_t kUPDATE = 1000;
   for (Int_t i = 0; i < 25000; i++) {
      Int_t npx = (Int_t)(gRandom->Rndm(1) * 15);

      vpx.clear();
      vpy.clear();
      vpz.clear();
      vrand.clear();

      for (Int_t j = 0; j < npx; ++j) {

         Float_t px,py,pz;
         gRandom->Rannor(px, py);
         pz = px * px + py * py;
         Float_t random = gRandom->Rndm(1);

         hpx->Fill(px);

         vpx.emplace_back(px);
         vpy.emplace_back(py);
         vpz.emplace_back(pz);
         vrand.emplace_back(random);

      }
      if (i && (i%kUPDATE) == 0) {
         if (i == kUPDATE)
            hpx->Draw();
         c1->Modified();
         c1->Update();
         if (gSystem->ProcessEvents())
            break;
      }
      t->Fill();
   }
   f->Write();

   delete f;
}


void read_vector()
{
   auto f = TFile::Open("hvector.root", "READ");

   if (!f)
      return;

   auto t = f->Get<TTree>("tvec");

   std::vector<float> *vpx = nullptr;

   // Create a new canvas.
   auto c1 = new TCanvas("c1", "Dynamic Filling Example", 200, 10, 700, 500);

   const Int_t kUPDATE = 1000;

   TBranch *bvpx = nullptr;
   t->SetBranchAddress("vpx", &vpx, &bvpx);


   // Create one histograms
   auto h = new TH1F("h", "This is the px distribution", 100, -4, 4);
   h->SetFillColor(48);

   for (Int_t i = 0; i < 25000; i++) {

      Long64_t tentry = t->LoadTree(i);
      bvpx->GetEntry(tentry);

      for (UInt_t j = 0; j < vpx->size(); ++j) {

         h->Fill(vpx->at(j));

      }
      if (i && (i%kUPDATE) == 0) {
         if (i == kUPDATE)
            h->Draw();
         c1->Modified();
         c1->Update();
         if (gSystem->ProcessEvents())
            break;
      }
   }

   // Since we passed the address of a local variable we need
   // to remove it.
   t->ResetBranchAddresses();
}


void tree121_hvector()
{
   gBenchmark->Start("hvector");
   write_vector();
   read_vector();
   gBenchmark->Show("hvector");
}

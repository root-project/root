#include "TH3.h"
#include "TCanvas.h"
#include "TFile.h"
#include "THnSparse.h"
#include "TRandom.h"
#include <iostream>

using namespace std;

// Compare TH3 operations with THnSparse operations

void Test(TH1* h, TH1* s, const char* test)
{
   // Check that hist and sparse are equal, print the test result
   cout << test << ": ";
   
   // What exactly is "equal"?
   // Define it as  the max of 1/1000 of the "amplitude" of the 
   // original hist, or 1E-4, whatever is larger.
   Double_t epsilon = 1E-4;
   Double_t diffH = h->GetMaximum() - h->GetMinimum();
   if (diffH < 0.) diffH = -diffH;
   if (diffH / 1000. > epsilon)
      epsilon = diffH / 1000.;

   TH1* diff = (TH1*)s->Clone("diff");
   diff->Add(h, -1);
   Double_t max = diff->GetMaximum();
   Double_t min = diff->GetMinimum();
   if (max < -min) max = -min;
   if (max < epsilon) cout << "SUCCESS";
   else {
      cout << "FAIL: delta=" << max;
      TCanvas* c = new TCanvas(test, test);
      c->Divide(1,3);
      c->cd(1); h->Draw();
      c->cd(2); s->Draw();
      c->cd(3); diff->Draw();
      TFile f("runsparse.root", "UPDATE");
      c->Write();
      delete c;
   }
   cout <<endl;
   delete diff;
}


void CheckFillProjection(TH3* h, THnSparse* sparse) 
{
   // Check filling of THnSparse, projection of TH3:
   TH3D* proj = sparse->Projection(0, 1, 2);
   Test(h, proj, "Projecting a THnSparse");
   delete proj;
}

void CheckProjection1(TH3* h, THnSparse* sparse) 
{
   // Check projection of THnSparse and TH3 to a TH1
   TH1* proj = sparse->Projection(1);
   TH1* hproj = h->Project3D("y");
   Test(hproj, proj, "Projecting a THnSparse to a TH1 (TH3::Project3D)");
   delete hproj;
   delete proj;

   proj = sparse->Projection(2);
   hproj = h->ProjectionZ();
   Test(hproj, proj, "Projecting a THnSparse to a TH1 (TH3::ProjectZ)");
   delete hproj;
   delete proj;
}

void CheckProjection2(TH3* h, THnSparse* sparse) 
{
   // Check projection of THnSparse and TH3 to a TH2
   TH2D* proj = sparse->Projection(1, 2);
   TH1* hproj = h->Project3D("yz");
   Test(hproj, proj, "Projecting a THnSparse to a TH2");
   delete proj;
   delete hproj;
}

void CheckScale(TH3* h, THnSparse* sparse)
{
   // Check scaling
   h->Scale(0.42);
   sparse->Scale(0.42);
   TH3* proj = sparse->Projection(0, 1, 2);
   Test(h, proj, "Scaling of THnSparse");
   delete proj;

}

void CheckClone(TH3* h, THnSparse* sparse)
{
   // Check cloning of a THnSparse
   if (h) {}
   THnSparse* clone = (THnSparse*)sparse->Clone("clone");
   TH3* proj = sparse->Projection(0, 1, 2);
   TH3* projclone = clone->Projection(0, 1, 2);
   Test(proj, projclone, "Cloning of THnSparse");
   delete clone;
   delete proj;
   delete projclone;
}

void CheckSubtract(TH3* h, THnSparse* sparse)
{
   // Check subtracting THnSparse from THnSparse
   TH3* h42 = (TH3*)h->Clone("h42");
   h42->Scale(.42);
   THnSparse* clone = (THnSparse*)sparse->Clone("clone");
   clone->Scale(1.42);
   clone->Add(sparse, -1.);
   TH3* proj = clone->Projection(0, 1, 2);
   Test(h42, proj, "Subtracting two THnSparse");
   delete proj;
   delete clone;
   delete h42;
}

void CheckDivide(TH3* h, THnSparse* sparse)
{
   // check division of two THnSparses
   TH3* h31 = (TH3*)h->Clone();
   h31->Scale(3.1);
   h31->Divide(h);

   THnSparse* s31 = (THnSparse*)sparse->Clone("s31");
   s31->Scale(3.1);
   s31->Divide(sparse);
   TH3* hs31 = s31->Projection(0,1,2);

   Test(h31, hs31, "Dividing two THnSparse");
   delete h31;
   delete s31;
   delete hs31;
}

TH3* GetErrors(TH3* h)
{
   // return a histogram fileld with h's errors
   TH3* herr = (TH3*)h->Clone("herr");
   herr->Reset();
   Int_t binsh = (h->GetNbinsX() + 2) * (h->GetNbinsY() + 2) * (h->GetNbinsZ() + 2);
   for (Long64_t i = 0; i < binsh; ++i) {
      Double_t eh = h->GetBinError(i);
      herr->SetBinContent(i, eh);
   }
   herr->SetEntries(h->GetEntries());
   return herr;
}

void CheckErrors(TH3* h, THnSparse* sparse)
{
   // compare the errors of the TH3 and the THnSparse
   TH3* hserr = (TH3*) h->Clone("hserr");
   hserr->Reset();
   Int_t coord[3];
   memset(coord, 0, sizeof(Int_t) * 3);
   for (Long64_t i = 0; i < sparse->GetNbins(); ++i) {
      // Get the content of the bin from the first histogram
      sparse->GetBinContent(i, coord);
      Double_t es = sparse->GetBinError(i);
      hserr->SetBinContent(hserr->GetBin(coord[0], coord[1], coord[2]), es);
   }
   hserr->SetEntries(sparse->GetEntries());
   TH3* herr = GetErrors(h);
   Test(hserr, herr, "Error calculation");
   delete hserr;
   delete herr;
}

void CheckBinomial(TH3* h, THnSparse* sparse)
{
   // check division of two THnSparses
   TH3* h31 = (TH3*)h->Clone();
   TH3* hb31 = (TH3*)h31->Clone("hb31");
   h31->Scale(3.1);
   hb31->Divide(h, h31, 1., 1., "b");

   THnSparse* s31 = (THnSparse*)sparse->Clone("s31");
   THnSparse* sb31 = (THnSparse*)s31->Clone("sb31");
   s31->Scale(3.1);
   sb31->Divide(sparse, s31, 1., 1., "b");
   TH3* hsb31 = sb31->Projection(0,1,2);

   Test(hb31, hsb31, "Dividing two THnSparse (binomial)");

   TH3* hb31err = GetErrors(hb31);
   TH3* hsb31err = GetErrors(hsb31);
   Test(hsb31err, hb31err, "Binomial errors of division result");

   delete h31;
   delete hb31;
   delete s31;
   delete sb31;
   delete hsb31;
   delete hb31err;
   delete hsb31err;
}


void CheckMerge(TH3* h, THnSparse* sparse)
{
   // check merging of three THnSparse
   TList h3_clones;
   h3_clones.SetOwner();
   for (int i = 0; i < 4; ++i) {
      h3_clones.AddLast(h->Clone());
      ((TH3*)h3_clones.Last())->Scale(i);
   }

   TList hs_clones;
   hs_clones.SetOwner();
   for (int i = 0; i < 4; ++i) {
      hs_clones.AddLast(sparse->Clone());
      ((THnSparse*)hs_clones.Last())->Scale(i);
   }

   TH3* h3Merge = (TH3*) h->Clone();
   THnSparse* hsMerge = (THnSparse*) sparse->Clone();

   h3Merge->Merge(&h3_clones);
   hsMerge->Merge(&hs_clones);

   TH3* proj = hsMerge->Projection(0, 1, 2);
   Test(h3Merge, proj, "Merging histograms");

   delete proj;
   delete h3Merge;
   delete hsMerge;
}

void doit(bool small) {
   Int_t blowup = 1;
   if (!small) blowup = 4;
   Int_t nbins[] = {10 * blowup, 20 * blowup, 14 * blowup};
   Double_t xmin[] = {0., -1., 0.};
   Double_t xmax[] = {1., 1., 10.};
         
   THnSparse* sparse = new THnSparseF("sparse" ,"sparse TH3", 3,
                                      nbins, xmin, xmax);
   sparse->Sumw2();

   TH3F* h = new TH3F("h", "nonsparse TH3", nbins[0], xmin[0], xmax[0],
                      nbins[1], xmin[1], xmax[1],
                      nbins[2], xmin[2], xmax[2]);
   h->Sumw2();

   for (Int_t entries = 0; entries < 10000; ++entries) {
      Double_t x[3];
      for (Int_t d = 0; d < 3; ++d)
         // 10% overshoot to tests overflows
         x[d] = gRandom->Rndm()*(xmax[d]*1.2 - xmin[d]*1.2) + xmin[d]*1.1;
      sparse->Fill(x);
      h->Fill(x[0], x[1], x[2]);
   }

   CheckFillProjection(h, sparse);
   CheckProjection1(h, sparse);
   CheckProjection2(h, sparse);
   CheckScale(h, sparse);
   CheckClone(h, sparse);
   CheckSubtract(h, sparse);
   CheckDivide(h, sparse);
   CheckErrors(h, sparse);
   CheckBinomial(h, sparse);
   CheckMerge(h, sparse);

   delete h;
   delete sparse;
}

void runsparse() 
{
   doit(true);
   doit(false);
}

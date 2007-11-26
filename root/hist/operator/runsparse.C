#include "TH3.h"
#include "THnSparse.h"
#include "TRandom.h"
#include <iostream>

using namespace std;

// Compare TH3 operations with THnSparse operations

void Test(TH1* h, TH1* s, const char* test)
{
   // Check that ihst and sparse are equal, print the test result
   cout << test << ": ";
   
   TH1* diff = (TH1*)s->Clone("diff");
   diff->Add(h, -1);
   Double_t max = diff->GetMaximum();
   Double_t min = diff->GetMinimum();
   if (max < -min) max = -min;
   if (max < 0.5) cout << "SUCCESS";
   else cout << "FAIL: delta=" << max;
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

void CheckScale(TH3* h, THnSparse* sparse)
{
   // Check scaling
   h->Scale(0.42);
   sparse->Scale(0.42);
   TH3* proj = sparse->Projection(0, 1, 2);
   Test(h, proj, "Scaling of THnSparse");
   delete proj;

}

void CheckClone(TH3*, THnSparse* sparse)
{
   // Check cloning of a THnSparse
   THnSparse* clone = (THnSparse*)sparse->Clone("clone");
   TH3* proj = sparse->Projection(0, 1, 2);
   TH3* projclone = clone->Projection(0, 1, 2);
   Test(proj, projclone, "Cloning of THnSparse");
   delete clone;
   delete proj;
   delete projclone;
}

void CheckSubstract(TH3* h, THnSparse* sparse)
{
   // Check substracting THnSparse from THnSparse
   TH3* h42 = (TH3*)h->Clone("h42");
   h42->Scale(.42);
   THnSparse* clone = (THnSparse*)sparse->Clone("clone");
   clone->Scale(1.42);
   clone->Add(sparse, -1.);
   TH3* proj = clone->Projection(0, 1, 2);
   Test(h42, proj, "Substracting two THnSparse");
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


void runsparse() 
{
   Int_t nbins[] = {10, 12, 14};
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
         x[d] = gRandom->Rndm()*(xmax[d]*1.1 - xmin[d]*1.1) + xmin[d]*1.1;
      sparse->Fill(x);
      h->Fill(x[0], x[1], x[2]);
   }

   CheckFillProjection(h, sparse);
   CheckScale(h, sparse);
   CheckClone(h, sparse);
   CheckSubstract(h, sparse);
   CheckDivide(h, sparse);
   CheckErrors(h, sparse);
   CheckBinomial(h, sparse);

   delete h;
   delete sparse;
}

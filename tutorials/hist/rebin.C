/// \file
/// \ingroup tutorial_hist
/// \notebook -js
/// Rebin a variable bin-width histogram.
///
/// This tutorial illustrates how to:
///   - create a variable bin-width histogram with a binning such
///     that the population per bin is about the same.
///   - rebin a variable bin-width histogram into another one.
///
/// \macro_image
/// \macro_code
///
/// \author Rene Brun

#include "TH1.h"
#include "TCanvas.h"
void rebin() {
   //create a fix bin histogram
   TH1F *h = new TH1F("h","test rebin",100,-3,3);
   Int_t nentries = 1000;
   h->FillRandom("gaus",nentries);
   Double_t xbins[1001];
   Int_t k=0;
   TAxis *axis = h->GetXaxis();
   for (Int_t i=1;i<=100;i++) {
      Int_t y = (Int_t)h->GetBinContent(i);
      if (y <=0) continue;
      Double_t dx = axis->GetBinWidth(i)/y;
      Double_t xmin = axis->GetBinLowEdge(i);
      for (Int_t j=0;j<y;j++) {
         xbins[k] = xmin +j*dx;
         k++;
      }
   }
   xbins[k] = axis->GetXmax();
   //create a variable bin-width histogram out of fix bin histogram
   //new rebinned histogram should have about 10 entries per bin
   TH1F *hnew = new TH1F("hnew","rebinned",k,xbins);
   hnew->FillRandom("gaus",10*nentries);

   //rebin hnew keeping only 50% of the bins
   Double_t xbins2[501];
   Int_t kk=0;
   for (Int_t j=0;j<k;j+=2) {
      xbins2[kk] = xbins[j];
      kk++;
   }
   xbins2[kk] = xbins[k];
   TH1F *hnew2 = (TH1F*)hnew->Rebin(kk,"hnew2",xbins2);

   //draw the 3 histograms
   TCanvas *c1 = new TCanvas("c1","c1",800,1000);
   c1->Divide(1,3);
   c1->cd(1);
   h->Draw();
   c1->cd(2);
   hnew->Draw();
   c1->cd(3);
   hnew2->Draw();
}

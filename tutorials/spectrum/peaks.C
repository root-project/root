/// \file
/// \ingroup tutorial_spectrum
/// \notebook
/// Illustrates how to find peaks in histograms.
///
/// This script generates a random number of gaussian peaks
/// on top of a linear background.
/// The position of the peaks is found via TSpectrum and injected
/// as initial values of parameters to make a global fit.
/// The background is computed and drawn on top of the original histogram.
///
/// This script can fit "peaks' heights" or "peaks' areas" (comment out
/// or uncomment the line which defines `__PEAKS_C_FIT_AREAS__`).
///
/// To execute this example, do (in ROOT 5 or ROOT 6):
///
/// ~~~{.cpp}
///  root > .x peaks.C  (generate 10 peaks by default)
///  root > .x peaks.C++ (use the compiler)
///  root > .x peaks.C++(30) (generates 30 peaks)
/// ~~~
///
/// To execute only the first part of the script (without fitting)
/// specify a negative value for the number of peaks, eg
///
/// ~~~{.cpp}
///  root > .x peaks.C(-20)
/// ~~~
///
/// \macro_output
/// \macro_image
/// \macro_code
///
/// \author Rene Brun

#include "TCanvas.h"
#include "TMath.h"
#include "TH1.h"
#include "TF1.h"
#include "TRandom.h"
#include "TSpectrum.h"
#include "TVirtualFitter.h"

//
// Comment out the line below, if you want "peaks' heights".
// Uncomment the line below, if you want "peaks' areas".
//
// #define __PEAKS_C_FIT_AREAS__ 1 /* fit peaks' areas */

Int_t npeaks = 30;
Double_t fpeaks(Double_t *x, Double_t *par) {
   Double_t result = par[0] + par[1]*x[0];
   for (Int_t p=0;p<npeaks;p++) {
      Double_t norm  = par[3*p+2]; // "height" or "area"
      Double_t mean  = par[3*p+3];
      Double_t sigma = par[3*p+4];
#if defined(__PEAKS_C_FIT_AREAS__)
      norm /= sigma * (TMath::Sqrt(TMath::TwoPi())); // "area"
#endif /* defined(__PEAKS_C_FIT_AREAS__) */
      result += norm*TMath::Gaus(x[0],mean,sigma);
   }
   return result;
}
void peaks(Int_t np=10) {
   npeaks = TMath::Abs(np);
   TH1F *h = new TH1F("h","test",500,0,1000);
   // Generate n peaks at random
   Double_t par[3000];
   par[0] = 0.8;
   par[1] = -0.6/1000;
   Int_t p;
   for (p=0;p<npeaks;p++) {
      par[3*p+2] = 1; // "height"
      par[3*p+3] = 10+gRandom->Rndm()*980; // "mean"
      par[3*p+4] = 3+2*gRandom->Rndm(); // "sigma"
#if defined(__PEAKS_C_FIT_AREAS__)
      par[3*p+2] *= par[3*p+4] * (TMath::Sqrt(TMath::TwoPi())); // "area"
#endif /* defined(__PEAKS_C_FIT_AREAS__) */
   }
   TF1 *f = new TF1("f",fpeaks,0,1000,2+3*npeaks);
   f->SetNpx(1000);
   f->SetParameters(par);
   TCanvas *c1 = new TCanvas("c1","c1",10,10,1000,900);
   c1->Divide(1,2);
   c1->cd(1);
   h->FillRandom("f",200000);
   h->Draw();
   TH1F *h2 = (TH1F*)h->Clone("h2");
   // Use TSpectrum to find the peak candidates
   TSpectrum *s = new TSpectrum(2*npeaks);
   Int_t nfound = s->Search(h,2,"",0.10);
   printf("Found %d candidate peaks to fit\n",nfound);
   // Estimate background using TSpectrum::Background
   TH1 *hb = s->Background(h,20,"same");
   if (hb) c1->Update();
   if (np <0) return;

   //estimate linear background using a fitting method
   c1->cd(2);
   TF1 *fline = new TF1("fline","pol1",0,1000);
   h->Fit("fline","qn");
   // Loop on all found peaks. Eliminate peaks at the background level
   par[0] = fline->GetParameter(0);
   par[1] = fline->GetParameter(1);
   npeaks = 0;
#if ROOT_VERSION_CODE >= ROOT_VERSION(6,00,00)
   Double_t *xpeaks; // ROOT 6
#else
   Float_t *xpeaks;  // ROOT 5
#endif
   xpeaks = s->GetPositionX();
   for (p=0;p<nfound;p++) {
      Double_t xp = xpeaks[p];
      Int_t bin = h->GetXaxis()->FindBin(xp);
      Double_t yp = h->GetBinContent(bin);
      if (yp-TMath::Sqrt(yp) < fline->Eval(xp)) continue;
      par[3*npeaks+2] = yp; // "height"
      par[3*npeaks+3] = xp; // "mean"
      par[3*npeaks+4] = 3; // "sigma"
#if defined(__PEAKS_C_FIT_AREAS__)
      par[3*npeaks+2] *= par[3*npeaks+4] * (TMath::Sqrt(TMath::TwoPi())); // "area"
#endif /* defined(__PEAKS_C_FIT_AREAS__) */
      npeaks++;
   }
   printf("Found %d useful peaks to fit\n",npeaks);
   printf("Now fitting: Be patient\n");
   TF1 *fit = new TF1("fit",fpeaks,0,1000,2+3*npeaks);
   // We may have more than the default 25 parameters
   TVirtualFitter::Fitter(h2,10+3*npeaks);
   fit->SetParameters(par);
   fit->SetNpx(1000);
   h2->Fit("fit");
}

#if !defined(__CINT__) && defined(__PEAKS_C_FIT_AREAS__)
#undef __PEAKS_C_FIT_AREAS__
#endif /* !defined(__CINT__) && defined(__PEAKS_C_FIT_AREAS__) */

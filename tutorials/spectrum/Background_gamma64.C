/// \file
/// \ingroup tutorial_spectrum
/// \notebook
/// Example to illustrate the background estimator (class TSpectrum2).
///
/// \macro_image
/// \macro_code
///
/// \authors Miroslav Morhac, Olivier Couet

#include <TSpectrum2.h>

#include <TSpectrum2.h>
void Background_gamma64() {
   Int_t i, j;
   const Int_t nbinsx = 64;
   const Int_t nbinsy = 64;
   Double_t xmin = 0;
   Double_t xmax = (Double_t)nbinsx;
   Double_t ymin = 0;
   Double_t ymax = (Double_t)nbinsy;
   Double_t ** source = new Double_t*[nbinsx];
   for (i=0;i<nbinsx;i++)
      source[i]=new Double_t[nbinsy];
   TString dir  = gROOT->GetTutorialDir();
   TString file = dir+"/spectrum/TSpectrum2.root";
   TFile *f     = new TFile(file.Data());
   auto back = (TH2F*) f->Get("back1");
   gStyle->SetOptStat(0);
   auto s = new TSpectrum2();
   for (i = 0; i < nbinsx; i++){
      for (j = 0; j < nbinsy; j++){
         source[i][j] = back->GetBinContent(i + 1,j + 1);
      }
   }
   s->Background(source,nbinsx,nbinsy,4,4,TSpectrum2::kBackDecreasingWindow,TSpectrum2::kBackSuccessiveFiltering);
   for (i = 0; i < nbinsx; i++){
      for (j = 0; j < nbinsy; j++)
         back->SetBinContent(i + 1,j + 1, source[i][j]);
   }
   back->Draw("SURF1");
}

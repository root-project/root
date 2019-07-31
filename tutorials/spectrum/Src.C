/// \file
/// \ingroup tutorial_spectrum
/// \notebook
/// Example to illustrate high resolution peak searching function (class TSpectrum2).
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \authors Miroslav Morhac, Olivier Couet

#include <TSpectrum2.h>

void Src() {
   Int_t i, j, nfound;
   Double_t nbinsx = 64;
   Double_t nbinsy = 64;
   Double_t xmin = 0;
   Double_t xmax = (Double_t)nbinsx;
   Double_t ymin = 0;
   Double_t ymax = (Double_t)nbinsy;
   Double_t** source = new Double_t*[nbinsx];
   for (i=0;i<nbinsx;i++)
      source[i]=new Double_t[nbinsy];
   Double_t** dest = new Double_t*[nbinsx];
   for (i=0;i<nbinsx;i++)
      dest[i]=new Double_t[nbinsy];
   TString dir  = gROOT->GetTutorialDir();
   TString file = dir+"/spectrum/TSpectrum2.root";
   TFile *f     = new TFile(file.Data());
   gStyle->SetOptStat(0);
   auto search = (TH2F*) f->Get("search4");
   auto *s = new TSpectrum2();
   for (i = 0; i < nbinsx; i++){
      for (j = 0; j < nbinsy; j++){
         source[i][j] = search->GetBinContent(i + 1,j + 1);
      }
   }
   nfound = s->SearchHighRes(source, dest, nbinsx, nbinsy, 2, 5, kTRUE, 3, kFALSE, 3);
   printf("Found %d candidate peaks\n",nfound);
   Double_t *PositionX = s->GetPositionX();
   Double_t *PositionY = s->GetPositionY();
   search->Draw("COL");
   auto m = new TMarker();
   m->SetMarkerStyle(23);
   m->SetMarkerColor(kRed);
   for (i=0;i<nfound;i++) {
      printf("posx= %d, posy= %d, value=%d\n",(Int_t)(PositionX[i]+0.5), (Int_t)(PositionY[i]+0.5),
      (Int_t)source[(Int_t)(PositionX[i]+0.5)][(Int_t)(PositionY[i]+0.5)]);
      m->DrawMarker(PositionX[i],PositionY[i]);
   }
}
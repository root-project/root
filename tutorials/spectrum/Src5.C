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

void Src5()
{
   const Int_t nbinsx = 64;
   const Int_t nbinsy = 64;
   std::vector<Double_t *> source(nbinsx), dest(nbinsx);
   for (Int_t i = 0; i < nbinsx; i++) {
      source[i] = new Double_t[nbinsy];
      dest[i] = new Double_t[nbinsy];
   }
   TString dir = gROOT->GetTutorialDir();
   TString file = dir + "/spectrum/TSpectrum2.root";
   TFile *f = TFile::Open(file.Data());
   gStyle->SetOptStat(0);
   auto search = (TH2F *)f->Get("search3;1");
   TSpectrum2 s;
   for (Int_t i = 0; i < nbinsx; i++) {
      for (Int_t j = 0; j < nbinsy; j++) {
         source[i][j] = search->GetBinContent(i + 1, j + 1);
      }
   }
   Int_t nfound = s.SearchHighRes(source.data(), dest.data(), nbinsx, nbinsy, 2, 5, kFALSE, 10, kFALSE, 1);
   printf("Found %d candidate peaks\n", nfound);
   Double_t *PositionX = s.GetPositionX();
   Double_t *PositionY = s.GetPositionY();
   search->Draw("COL");
   TMarker m;
   m.SetMarkerStyle(23);
   m.SetMarkerColor(kRed);
   for (Int_t i = 0; i < nfound; i++) {
      printf("posx= %d, posy= %d, value=%d\n", (Int_t)(PositionX[i] + 0.5), (Int_t)(PositionY[i] + 0.5),
             (Int_t)source[(Int_t)(PositionX[i] + 0.5)][(Int_t)(PositionY[i] + 0.5)]);
      m.DrawMarker(PositionX[i], PositionY[i]);
   }

   for (Int_t i = 0; i < nbinsx; i++) {
      delete[] source[i];
      delete[] dest[i];
   }
}


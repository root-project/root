/// \file
/// \ingroup tutorial_spectrum
/// \notebook
/// Example to illustrate high resolution peak searching function (class TSpectrum).
///
/// \macro_output
/// \macro_image
/// \macro_code
///
/// \authors Miroslav Morhac, Olivier Couet

void SearchHR1() {
   Double_t fPositionX[100];
   Double_t fPositionY[100];
   Int_t fNPeaks = 0;
   Int_t i,nfound,bin;
   const Int_t nbins = 1024;
   Double_t xmin     = 0;
   Double_t xmax     = nbins;
   Double_t a;
   Double_t source[nbins], dest[nbins];
   gROOT->ForceStyle();

   TString dir  = gROOT->GetTutorialDir();
   TString file = dir+"/spectrum/TSpectrum.root";
   TFile *f     = new TFile(file.Data());
   TH1F *h = (TH1F*) f->Get("back2");
   h->SetTitle("High resolution peak searching, number of iterations = 3");
   h->GetXaxis()->SetRange(1,nbins);
   TH1F *d = new TH1F("d","",nbins,xmin,xmax);
   h->Draw("L");

   for (i = 0; i < nbins; i++) source[i]=h->GetBinContent(i + 1);

   h->Draw("L");

   TSpectrum *s = new TSpectrum();

   nfound = s->SearchHighRes(source, dest, nbins, 8, 2, kTRUE, 3, kTRUE, 3);
   Double_t *xpeaks = s->GetPositionX();
   for (i = 0; i < nfound; i++) {
      a=xpeaks[i];
      bin = 1 + Int_t(a + 0.5);
      fPositionX[i] = h->GetBinCenter(bin);
      fPositionY[i] = h->GetBinContent(bin);
   }

   TPolyMarker * pm = (TPolyMarker*)h->GetListOfFunctions()->FindObject("TPolyMarker");
   if (pm) {
      h->GetListOfFunctions()->Remove(pm);
      delete pm;
   }
   pm = new TPolyMarker(nfound, fPositionX, fPositionY);
   h->GetListOfFunctions()->Add(pm);
   pm->SetMarkerStyle(23);
   pm->SetMarkerColor(kRed);
   pm->SetMarkerSize(1.3);

   for (i = 0; i < nbins; i++) d->SetBinContent(i + 1,dest[i]);
   d->SetLineColor(kRed);
   d->Draw("SAME");

   printf("Found %d candidate peaks\n",nfound);
   for( i=0;i<nfound;i++) printf("posx= %f, posy= %f\n",fPositionX[i], fPositionY[i]);
}

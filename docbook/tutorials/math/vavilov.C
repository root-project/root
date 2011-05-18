#include "TMath.h"
#include "TCanvas.h"
#include "TRandom.h"
#include "TGraph.h"
   
void vavilov()
{
   //test of the TMath::Vavilov distribution
   //Author: Anna Kreshuk
   
   Int_t n = 1000;
   Double_t *x  = new Double_t[n];
   Double_t *y1 = new Double_t[n];
   Double_t *y2 = new Double_t[n];
   Double_t *y3 = new Double_t[n];
   Double_t *y4 = new Double_t[n];

   TRandom r;
   for (Int_t i=0; i<n; i++) {
      x[i]  = r.Uniform(-2, 10);
      y1[i] = TMath::Vavilov(x[i], 0.3, 0.5);
      y2[i] = TMath::Vavilov(x[i], 0.15, 0.5);
      y3[i] = TMath::Vavilov(x[i], 0.25, 0.5);
      y4[i] = TMath::Vavilov(x[i], 0.05, 0.5);
   }

   TCanvas *c1 = new TCanvas("c1", "Vavilov density");
   c1->SetGrid();
   c1->SetHighLightColor(19);
   TGraph *gr1 = new TGraph(n, x, y1);
   TGraph *gr2 = new TGraph(n, x, y2);
   TGraph *gr3 = new TGraph(n, x, y3);
   TGraph *gr4 = new TGraph(n, x, y4);
   gr1->SetTitle("TMath::Vavilov density");
   gr1->Draw("ap");
   gr2->Draw("psame");
   gr2->SetMarkerColor(kRed);
   gr3->Draw("psame");
   gr3->SetMarkerColor(kBlue);
   gr4->Draw("psame");
   gr4->SetMarkerColor(kGreen);

   TF1 *f1 = new TF1("f1", "TMath::Vavilov(x, 0.3, 0.5)", -2, 10);

   TH1F *hist = new TH1F("vavilov", "vavilov", 100, -2, 10);
   for (int i=0; i<10000; i++) {
      hist->Fill(f1->GetRandom());
   }
   hist->Scale(1/1200.);
   hist->Draw("same");
}

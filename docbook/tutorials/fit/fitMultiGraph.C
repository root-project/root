#include "TMultiGraph.h"
#include "TRandom.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "TCanvas.h"
#include "TMath.h"

void fitMultiGraph()
{
   //fitting a parabola to a multigraph of 3 partly overlapping graphs
   //with different errors
   //Author: Anna Kreshuk
      
   Int_t n = 30;
   Double_t *x1 = new Double_t[n];
   Double_t *x2 = new Double_t[n];
   Double_t *x3 = new Double_t[n];
   Double_t *y1 = new Double_t[n];
   Double_t *y2 = new Double_t[n];
   Double_t *y3 = new Double_t[n];
   Double_t *e1 = new Double_t[n];
   Double_t *e2 = new Double_t[n];
   Double_t *e3 = new Double_t[n];
   
   //generate the data for the graphs
   TRandom r;
   Int_t i;
   for (i=0; i<n; i++) {
      x1[i] = r.Uniform(0.1, 5);
      x2[i] = r.Uniform(3, 8);
      x3[i] = r.Uniform(9, 15);
      y1[i] = 3 + 2*x1[i] + x1[i]*x1[i] + r.Gaus();
      y2[i] = 3 + 2*x2[i] + x2[i]*x2[i] + r.Gaus()*10;
      e1[i] = 1;
      e2[i] = 10;
      e3[i] = 20;
      y3[i] = 3 + 2*x3[i] + x3[i]*x3[i] + r.Gaus()*20;
   }
   
   //create the graphs and set their drawing options
   TGraphErrors *gr1 = new TGraphErrors(n, x1, y1, 0, e1);
   TGraphErrors *gr2 = new TGraphErrors(n, x2, y2, 0, e2);
   TGraphErrors *gr3 = new TGraphErrors(n, x3, y3, 0, e3);
   gr1->SetLineColor(kRed);
   gr2->SetLineColor(kBlue);
   gr2->SetMarkerStyle(24);
   gr2->SetMarkerSize(0.3);
   gr3->SetLineColor(kGreen);
   gr3->SetMarkerStyle(24);
   gr3->SetMarkerSize(0.3);

   //add the graphs to the multigraph
   TMultiGraph *mg=new TMultiGraph("mg", 
      "TMultiGraph of 3 TGraphErrors");
   mg->Add(gr1);
   mg->Add(gr2);
   mg->Add(gr3);

   TCanvas *myc = new TCanvas("myc", 
      "Fitting a MultiGraph of 3 TGraphErrors");
   myc->SetFillColor(42);
   myc->SetGrid();
   
   mg->Draw("ap");
   
   //fit
   mg->Fit("pol2", "F");

   //access to the fit function
   TF1 *fpol = mg->GetFunction("pol2");
   fpol->SetLineWidth(1);

}

void fitminuit()
{
   Int_t n = 30;
   Double_t *x1 = new Double_t[n];
   Double_t *x2 = new Double_t[n];
   Double_t *x3 = new Double_t[n];
   Double_t *y1 = new Double_t[n];
   Double_t *y2 = new Double_t[n];
   Double_t *y3 = new Double_t[n];
   Double_t *e1 = new Double_t[n];
   Double_t *e2 = new Double_t[n];
   Double_t *e3 = new Double_t[n];
   Double_t *xtotal = new Double_t[n*3];
   Double_t *ytotal = new Double_t[n*3];
   Double_t *etotal = new Double_t[n*3];
   
   TRandom r;
   Int_t i;
   for (i=0; i<n; i++) {
      x1[i] = r.Uniform(-3, -1);
      x2[i] = r.Uniform(-1, 1);
      x3[i] = r.Uniform(1, 3);
      y1[i] = TMath::Gaus(x1[i], 0, 1);
      y2[i] = TMath::Gaus(x2[i], 0, 1);
      e1[i] = 0.00001;
      e2[i] = 0.00001;
      e3[i] = 0.00001;
      y3[i] = TMath::Gaus(x3[i], 0, 1);
   }
   for (i=0; i<n; i++) 
      {xtotal[i]=x1[i]; ytotal[i]=y1[i]; etotal[i]=0.00001;}
   for (i=n; i<2*n; i++) 
      {xtotal[i] = x2[i-n]; ytotal[i]=y2[i-n]; etotal[i]=0.00001;}
   for (i=2*n; i<3*n; i++) 
      {xtotal[i] = x3[i-2*n]; ytotal[i]=y3[i-2*n]; etotal[i]=0.00001;}

   //create the graphs and set their drawing options
   TGraphErrors *gr1 = new TGraphErrors(n, x1, y1, 0, e1);
   TGraphErrors *gr2 = new TGraphErrors(n, x2, y2, 0, e2);
   TGraphErrors *gr3 = new TGraphErrors(n, x3, y3, 0, e3);
   TGraphErrors *grtotal = new TGraphErrors(n*3, xtotal, ytotal, 0, etotal);
   TMultiGraph *mg=new TMultiGraph("mg", "TMultiGraph of 3 TGraphErrors");
   mg->Add(gr1);
   mg->Add(gr2);
   mg->Add(gr3);
   //mg->Draw("ap");
   //TF1 *ffit = new TF1("ffit", "TMath::Gaus(x, [0], [1], [2])", -3, 3);
   //ffit->SetParameters(0, 1, 0);
   //mg->Fit(ffit);

   grtotal->Fit("gaus");
   mg->Fit("gaus");
}

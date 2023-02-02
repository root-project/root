/// \file
/// \ingroup tutorial_fit
/// \notebook -js
/// fitting a parabola to a multigraph of 3 partly overlapping graphs
/// with different errors
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \author Anna Kreshuk

#include "TMultiGraph.h"
#include "TRandom.h"
#include "TF1.h"
#include "TGraphErrors.h"
#include "TCanvas.h"
#include "TMath.h"

void fitMultiGraph()
{
   int n = 30;
   double *xvalues1 = new double[n];
   double *xvalues2 = new double[n];
   double *xvalues3 = new double[n];
   double *yvalues1 = new double[n];
   double *yvalues2 = new double[n];
   double *yvalues3 = new double[n];
   double *evalues1 = new double[n];
   double *evalues2 = new double[n];
   double *evalues3 = new double[n];

   //generate the data for the graphs
   TRandom r;
   int i;
   for (i=0; i<n; i++) {
      xvalues1[i] = r.Uniform(0.1, 5);
      xvalues2[i] = r.Uniform(3, 8);
      xvalues3[i] = r.Uniform(9, 15);
      yvalues1[i] = 3 + 2*xvalues1[i] + xvalues1[i]*xvalues1[i] + r.Gaus();
      yvalues2[i] = 3 + 2*xvalues2[i] + xvalues2[i]*xvalues2[i] + r.Gaus()*10;
      evalues1[i] = 1;
      evalues2[i] = 10;
      evalues3[i] = 20;
      yvalues3[i] = 3 + 2*xvalues3[i] + xvalues3[i]*xvalues3[i] + r.Gaus()*20;
   }

   //create the graphs and set their drawing options
   TGraphErrors *gr1 = new TGraphErrors(n, xvalues1, yvalues1, 0, evalues1);
   TGraphErrors *gr2 = new TGraphErrors(n, xvalues2, yvalues2, 0, evalues2);
   TGraphErrors *gr3 = new TGraphErrors(n, xvalues3, yvalues3, 0, evalues3);
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
   int n = 30;
   double *xvalues1 = new double[n];
   double *xvalues2 = new double[n];
   double *xvalues3 = new double[n];
   double *yvalues1 = new double[n];
   double *yvalues2 = new double[n];
   double *yvalues3 = new double[n];
   double *evalues1 = new double[n];
   double *evalues2 = new double[n];
   double *evalues3 = new double[n];
   double *xtotal = new double[n*3];
   double *ytotal = new double[n*3];
   double *etotal = new double[n*3];

   TRandom r;
   int i;
   for (i=0; i<n; i++) {
      xvalues1[i] = r.Uniform(-3, -1);
      xvalues2[i] = r.Uniform(-1, 1);
      xvalues3[i] = r.Uniform(1, 3);
      yvalues1[i] = TMath::Gaus(xvalues1[i], 0, 1);
      yvalues2[i] = TMath::Gaus(xvalues2[i], 0, 1);
      evalues1[i] = 0.00001;
      evalues2[i] = 0.00001;
      evalues3[i] = 0.00001;
      yvalues3[i] = TMath::Gaus(xvalues3[i], 0, 1);
   }
   for (i=0; i<n; i++)
      {xtotal[i]=xvalues1[i]; ytotal[i]=yvalues1[i]; etotal[i]=0.00001;}
   for (i=n; i<2*n; i++)
      {xtotal[i] = xvalues2[i-n]; ytotal[i]=yvalues2[i-n]; etotal[i]=0.00001;}
   for (i=2*n; i<3*n; i++)
      {xtotal[i] = xvalues3[i-2*n]; ytotal[i]=yvalues3[i-2*n]; etotal[i]=0.00001;}

   //create the graphs and set their drawing options
   TGraphErrors *gr1 = new TGraphErrors(n, xvalues1, yvalues1, 0, evalues1);
   TGraphErrors *gr2 = new TGraphErrors(n, xvalues2, yvalues2, 0, evalues2);
   TGraphErrors *gr3 = new TGraphErrors(n, xvalues3, yvalues3, 0, evalues3);
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

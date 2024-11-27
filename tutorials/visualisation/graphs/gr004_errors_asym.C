/// \file
/// \ingroup tutorial_graphs
/// \notebook
///
/// This tutorial demonstrates the use of TGraphAsymmErrors to plot a graph with asymmetrical errors on both the x and y axes.
/// The errors for the x values are divided into low (left side of the marker) and high (right side of the marker) errors.
/// Similarly, for the y values, there are low (lower side of the marker) and high (upper side of the marker) errors.
///
/// \macro_image
/// \macro_code
/// 
/// \author Miro Helbich

void gr004_errors_asym() {
   TCanvas *c2 = new TCanvas("c2","", 700, 500);

   c2->SetGrid();
   const Int_t npoints=3;
   Double_t xaxis[npoints] = {1.,2.,3.};
   Double_t yaxis[npoints] = {10.,20.,30.};

   Double_t exl[npoints] = {0.5,0.2,0.1}; //Lower x errors
   Double_t exh[npoints] = {0.5,0.3,0.4}; //Higher x errors
   Double_t eyl[npoints] = {3.,5.,4.}; //Lower y errors
   Double_t eyh[npoints] = {3.,5.,4.}; //Higher y errors

   TGraphAsymmErrors *gr = new TGraphAsymmErrors(npoints,xaxis,yaxis,exl,exh,eyl,eyh); //Create the TGraphAsymmErrors object with data and asymmetrical errors

   gr->SetTitle("A simple graph with asymmetrical errors");
   gr->Draw("A*"); //"A" = draw axes and "*" = draw markers at the points with error bars
}

/// \file
/// \ingroup tutorial_graphs
/// \notebook
///  
/// This tutorial demonstrates how to set a logarithmic scale for the axes of a graph using the `SetLogScale()` method.
/// The logarithmic scale can be applied to either the x-axis, the y-axis, or both axes at the same time.
/// When using a logarithmic scale, the data must be positive since the logarithm is undefined for non-positive values and zero.
///
/// \macro_image
/// \macro_code
/// \date 25/11/2024
/// \author Emanuele Chiamulera

void gr110_logscale() {

   auto c = new TCanvas("c","Reversed graphs",0,0,900,400);
   c->Divide(2,1);
   c->cd(1);

   const Int_t n = 6; //Fill the arrays x and y with the data points
   Double_t x[n], y[n];
   for (Int_t i=0;i<n;i++) {
     x[i] = i+1;
     y[i] = exp(i+1);
   }

   TGraph *gr1 = new TGraph(n,x,y);

   gr1->SetLineColor(2);
   gr1->SetLineWidth(4);
   gr1->SetMarkerColor(4);
   gr1->SetMarkerStyle(21);
   gr1->SetTitle("Graph without log scale");
   gr1->DrawClone("ACP");

   //The logarithmic scale setting can also be done in a ROOT interactive session.
   c->cd(2);
   gPad->SetLogy();
   //gPad->SetLogx(); Uncomment this line if log scale is needed on both axes
   gr1->SetTitle("Graph with y log scale");
   gr1->Draw("ACP");

}
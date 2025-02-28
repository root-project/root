/// \file
/// \ingroup tutorial_graphs
/// \notebook -js
/// \preview Create and draw a graph with error bars. If more graphs are needed, see the
/// [gr03_err2gr.C](https://root.cern/doc/master/gerrors2_8C.html) tutorial
///
/// See the [TGraphErrors documentation](https://root.cern/doc/master/classTGraphErrors.html)
///
/// \macro_image
/// \macro_code
/// \author Rene Brun

void gr002_errors() {
   TCanvas *c1 = new TCanvas("c1","A Simple Graph with error bars",200,10,700,500);

   c1->SetGrid();
   c1->GetFrame()->SetBorderSize(12);

   // We will use the constructor requiring: the number of points, arrays containing the x-and y-axis values, and arrays with the x- andy-axis errors
   const Int_t n = 10;
   Float_t x[n]  = {-0.22, 0.05, 0.25, 0.35, 0.5, 0.61,0.7,0.85,0.89,0.95};
   Float_t y[n]  = {1,2.9,5.6,7.4,9,9.6,8.7,6.3,4.5,1};
   Float_t ex[n] = {.05,.1,.07,.07,.04,.05,.06,.07,.08,.05};
   Float_t ey[n] = {.8,.7,.6,.5,.4,.4,.5,.6,.7,.8};

   // If all x-axis errors should zero, just provide a single 0 in place of ex
   TGraphErrors *gr = new TGraphErrors(n,x,y,ex,ey);

   gr->SetTitle("TGraphErrors Example");
   gr->SetMarkerColor(4);
   gr->SetMarkerStyle(21);

   // To draw in a new/empty canvas or pad, include the option "A" so that the axes are drawn (leave it out if the graph is to be drawn on top of an existing plot
   gr->Draw("ALP");

   c1->Update();
}

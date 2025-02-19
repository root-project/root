/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// \preview Show how to shade an area between two graphs.
///
/// \macro_image
/// \macro_code
///
/// \author Rene Brun

void gr101_shade_area() {
   TCanvas *c1 = new TCanvas("c1","A Simple Graph Example",200,10,700,500);

   c1->SetGrid();
   c1->DrawFrame(0,0,2.2,12);

   const Int_t n = 20;
   Double_t x[n], y[n],ymin[n], ymax[n];
   Int_t i;
   for (i=0;i<n;i++) {
     x[i] = 0.1+i*0.1;
     ymax[i] = 10*sin(x[i]+0.2); //Y points for the upper graph
     ymin[i] = 8*sin(x[i]+0.1); //Y point for the lower graph
     y[i] = 9*sin(x[i]+0.15);
   }
   TGraph *grmin = new TGraph(n,x,ymin); //Bottom graph
   TGraph *grmax = new TGraph(n,x,ymax); //Upper graph
   TGraph *gr    = new TGraph(n,x,y); //Middle graph
   TGraph *grshade = new TGraph(2*n); //Create a graph to represent the shaded region between the upper and lower graphs
   for (i=0;i<n;i++) {    //Populate points in the shaded graph
      grshade->SetPoint(i,x[i],ymax[i]); 
      grshade->SetPoint(n+i,x[n-i-1],ymin[n-i-1]);
   }
   grshade->SetFillStyle(3013);
   grshade->SetFillColor(16);
   grshade->Draw("f"); //Draw the shaded area with "f" option (filled graph)
   grmin->Draw("l");
   grmax->Draw("l");
   gr->SetLineWidth(4);
   gr->SetMarkerColor(4);
   gr->SetMarkerStyle(21);
   gr->Draw("CP");
}

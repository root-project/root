/// \file
/// \ingroup tutorial_graphs
/// \notebook  
/// \preview Simple graph with legend. For more details on TLegend [see documentation](https://root.cern/doc/master/classTLegend.html)
///
/// \macro_image
/// \macro_code
/// \date 25/11/2024
/// \author Emanuele Chiamulera

void gr111_legend() {

   auto c = new TCanvas("c","", 900, 800);
   c->SetGrid();

   const Int_t n = 5;
   Double_t x[n] = {-3.00, -2.68, -2.36, -2.05, -1.73};
   Double_t y1[n] = {-0.98, -0.89, -0.71, -0.46, -0.16};
   Double_t y2[n] = {-2.98, -2.89, -2.71, -2.46, -2.16};


   TGraph *gr1 = new TGraph(n, x, y1); //Create the graph
   gr1->SetLineColor(2);
   gr1->SetLineWidth(4);
   gr1->SetMarkerColor(4);
   gr1->SetMarkerStyle(21);
   gr1->SetTitle("Graph with legend");
   
   TLegend *legend = new TLegend(0.1,0.7,0.38,0.9); //Create the TLegend object and define it's position 
   legend->SetHeader("Legend title", "C"); //"C" Center alignment for the header ("L" Left and "R" Right)
   legend->SetFillColor(kWhite);
   legend->SetBorderSize(1);
   legend->SetTextSize(0.04);
   legend->AddEntry(gr1, "Data points", "lp");  // "p" for point marker, "l" for line, "e" for error bars if TGraphError is used.

   gr1->Draw("");

   // Draw the legend
   legend->Draw();
}

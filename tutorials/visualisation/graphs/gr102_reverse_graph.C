/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// \preview This example tests the reverse graphs obtained with Draw("a  pl rx ry ") on a TGraph, where rx and ry refere to the reversing of x and y axis.
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

void gr102_reverse_graph() {
   auto c = new TCanvas("c","Reversed graphs",0,0,900,400);
   c->Divide(2,1); //Create a canvas with a 2x1 grid layout

   auto gr = new TGraph();
   gr->GetXaxis()->SetNdivisions(514);
   gr->GetYaxis()->SetNdivisions(514);
   gr->SetMarkerStyle(kCircle);
   gr->SetMarkerColor(kBlue);
   gr->SetLineColor(kRed);
   gr->SetPoint(0,5,5);
   gr->SetPoint(1,9,9);
   gr->SetPoint(2,14,14);


   c->cd(1);
   gPad->SetGrid();
   gr->Draw("a  pl "); //Plot with axis ("a") and line+points ("pl")

   c->cd(2);
   gPad->SetGrid();
   gr->Draw("a  pl rx ry "); //Plot with axis ("a") and line+points ("pl") with reverse X and Y axes

}

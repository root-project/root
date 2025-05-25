/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// \preview This example tests all the various cases of reverse graphs obtained with Draw("a  pl rx ry "), where rx and ry refer to the reversing of x and y axis.
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

void gr112_reverse_graph_and_errors() {
   auto c = new TCanvas("c","Reversed graphs",0,0,900,900);
   c->Divide(3,3);

   //TGraphErrors (first row example)
   auto graphe = new TGraphErrors();
   graphe->GetXaxis()->SetNdivisions(514);
   graphe->GetYaxis()->SetNdivisions(514);
   graphe->SetMarkerStyle(kCircle);
   graphe->SetPoint(0,5,5);
   graphe->SetPointError(0,1,3);
   graphe->SetPoint(1,9,9);
   graphe->SetPointError(1,1,3);

   c->cd(1);
   gPad->SetGrid();
   graphe->Draw("a  pl "); //Plot with axis ("a") and line+points ("pl")

   c->cd(2);
   gPad->SetGrid();
   graphe->Draw("a  pl rx ry "); //Plot with axis ("a") and line+points ("pl") with reverse X and Y axes

   c->cd(3);
   gPad->SetGrid();
   gPad->SetLogx();
   gPad->SetLogy();
   graphe->GetXaxis()->SetMoreLogLabels();
   graphe->GetYaxis()->SetMoreLogLabels();
   graphe->Draw("a  pl rx ry"); //Plot with reverse axis and log scale

   //TGraphAsymmErrors (second row example)
   auto graphae = new TGraphAsymmErrors();
   graphae->GetXaxis()->SetNdivisions(514);
   graphae->GetYaxis()->SetNdivisions(514);
   graphae->SetMarkerStyle(kCircle);
   graphae->SetPoint(0,5,5);
   graphae->SetPointError(0,1,3,3,1);
   graphae->SetPoint(1,9,9);
   graphae->SetPointError(1,1,3,1,3);

   c->cd(4);
   gPad->SetGrid();
   graphae->Draw("a  pl ");

   c->cd(5);
   gPad->SetGrid();
   graphae->Draw("a  pl rx ry ");

   c->cd(6);
   gPad->SetGrid();
   gPad->SetLogx();
   gPad->SetLogy();
   graphae->GetXaxis()->SetMoreLogLabels();
   graphae->GetYaxis()->SetMoreLogLabels();
   graphae->Draw("a  pl rx ry");

   //TGraphBentErrors (third row example)
   auto graphbe = new TGraphBentErrors();
   graphbe->GetXaxis()->SetNdivisions(514);
   graphbe->GetYaxis()->SetNdivisions(514);
   graphbe->SetMarkerStyle(kCircle);
   graphbe->SetPoint(0,5,5);
   graphbe->SetPointError(0,1,3,3,1,.5,.2,.5,.2);
   graphbe->SetPoint(1,9,9);
   graphbe->SetPointError(1,1,3,1,3,-.5,-.2,-.5,-.2);

   c->cd(7);
   gPad->SetGrid();
   graphbe->Draw("a  pl ");

   c->cd(8);
   gPad->SetGrid();
   graphbe->Draw("a  pl rx ry ");

   c->cd(9);
   gPad->SetGrid();
   gPad->SetLogx();
   gPad->SetLogy();
   graphbe->GetXaxis()->SetMoreLogLabels();
   graphbe->GetYaxis()->SetMoreLogLabels();
   graphbe->Draw("a  pl rx ry");
}
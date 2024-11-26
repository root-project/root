/// \file
/// \ingroup tutorial_graphs
/// \notebook
/// This example test all the various case of reverse graphs
/// combined with logarithmic scale.
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

void graphreverse()
{
   auto c = new TCanvas("c", "Reversed graphs", 0, 0, 900, 900);
   c->Divide(3, 3);

   // TGraphErrors
   auto graphe = new TGraphErrors();
   graphe->GetXaxis()->SetNdivisions(514);
   graphe->GetYaxis()->SetNdivisions(514);
   graphe->SetMarkerStyle(kCircle);
   graphe->SetPoint(0, 5, 5);
   graphe->SetPointError(0, 1, 3);
   graphe->SetPoint(1, 9, 9);
   graphe->SetPointError(1, 1, 3);

   c->cd(1);
   gPad->SetGrid();
   graphe->Draw("a  pl ");

   c->cd(2);
   gPad->SetGrid();
   graphe->Draw("a  pl rx ry ");

   c->cd(3);
   gPad->SetGrid();
   gPad->SetLogx();
   gPad->SetLogy();
   graphe->GetXaxis()->SetMoreLogLabels();
   graphe->GetYaxis()->SetMoreLogLabels();
   graphe->Draw("a  pl rx ry");

   // TGraphAsymmErrors
   auto graphae = new TGraphAsymmErrors();
   graphae->GetXaxis()->SetNdivisions(514);
   graphae->GetYaxis()->SetNdivisions(514);
   graphae->SetMarkerStyle(kCircle);
   graphae->SetPoint(0, 5, 5);
   graphae->SetPointError(0, 1, 3, 3, 1);
   graphae->SetPoint(1, 9, 9);
   graphae->SetPointError(1, 1, 3, 1, 3);

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

   // TGraphBentErrors
   auto graphbe = new TGraphBentErrors();
   graphbe->GetXaxis()->SetNdivisions(514);
   graphbe->GetYaxis()->SetNdivisions(514);
   graphbe->SetMarkerStyle(kCircle);
   graphbe->SetPoint(0, 5, 5);
   graphbe->SetPointError(0, 1, 3, 3, 1, .5, .2, .5, .2);
   graphbe->SetPoint(1, 9, 9);
   graphbe->SetPointError(1, 1, 3, 1, 3, -.5, -.2, -.5, -.2);

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

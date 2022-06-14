/// \file
/// \ingroup tutorial_hist
/// \notebook
/// The legend can be placed automatically in the current pad in an empty space
/// found at painting time.
///
/// The following example illustrate this facility. Only the width and height of the
/// legend is specified in percentage of the pad size.
///
/// \macro_image
/// \macro_code
///
/// \author Olivier Couet

void legendautoplaced()
{
   auto c4 = new TCanvas("c", "c", 600,500);
   auto hpx = new TH1D("hpx","This is the hpx distribution",100,-4.,4.);
   hpx->FillRandom("gaus", 50000);
   hpx->Draw("E");
   hpx->GetYaxis()->SetTitle("Y Axis title");
   hpx->GetYaxis()->SetTitleOffset(1.3); hpx->GetYaxis()->CenterTitle(true);
   hpx->GetXaxis()->SetTitle("X Axis title");
   hpx->GetXaxis()->CenterTitle(true);

   auto h1 = new TH1D("h1","A green histogram",100,-2.,2.);
   h1->FillRandom("gaus", 10000);
   h1->SetLineColor(kGreen);
   h1->Draw("same");

   auto g = new TGraph();
   g->SetPoint(0, -3.5, 100 );
   g->SetPoint(1, -3.0, 300 );
   g->SetPoint(2, -2.0, 1000 );
   g->SetPoint(3,  1.0, 800 );
   g->SetPoint(4,  0.0, 200 );
   g->SetPoint(5,  3.0, 200 );
   g->SetPoint(6,  3.0, 700 );
   g->Draw("L");
   g->SetTitle("This is a TGraph");
   g->SetLineColor(kRed);
   g->SetFillColor(0);

   // TPad::BuildLegend() default placement values are such that they trigger
   // the automatic placement.
   c4->BuildLegend();
}

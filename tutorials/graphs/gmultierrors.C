/// \file
/// \ingroup tutorial_graphs
/// \notebook -js
/// Draw a graph with multiple y errors
///
/// \macro_image
/// \macro_code
///
/// \author Simon Spies

void gmultierrors() {
   TCanvas *c1 = new TCanvas("c1","A Simple Graph with multiple errors",200,10,700,500);

   c1->SetGrid();
   c1->GetFrame()->SetBorderSize(12);

   const Int_t np = 5, ne = 2;
   Double_t x[np]    = {0, 1, 2, 3, 4};
   Double_t y[np]    = {0, 2, 4, 1, 3};
   Double_t exl[np]  = {0.3, 0.3, 0.3, 0.3, 0.3};
   Double_t exh[np]  = {0.3, 0.3, 0.3, 0.3, 0.3};
   Double_t* eylstat = new double[np]  {1, 0.5, 1, 0.5, 1};
   Double_t* eyhstat = new double[np]  {0.5, 1, 0.5, 1, 0.5};
   Double_t* eylsys  = new double[np]  {0.5, 0.4, 0.8, 0.3, 1.2};
   Double_t* eyhsys  = new double[np]  {0.6, 0.7, 0.6, 0.4, 0.8};
   Double_t** eyl    = new double*[ne] {eylstat, eylsys};
   Double_t** eyh    = new double*[ne] {eyhstat, eyhsys};

   TGraphMultiErrors* gme = new TGraphMultiErrors(5, 2, x, y, exl, exh, eyl, eyh);
   gme->SetMarkerStyle(20);
   gme->SetLineColor(kRed);
   gme->GetAttLine(0)->SetLineColor(kRed);
   gme->GetAttLine(1)->SetLineColor(kBlue);
   gme->GetAttFill(1)->SetFillStyle(0);

   // Graph and x erros drawn with "APS"
   // Stat Errors drawn with "Z"
   // Sys Errors drawn with "5 s=0.5"
   gme->Draw("APS ; Z ; 5 s=0.5");

   c1->Update();
}

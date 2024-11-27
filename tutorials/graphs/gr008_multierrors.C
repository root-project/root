/// \file
/// \ingroup tutorial_graphs
/// \notebook -js
/// Draw a graph with multiple errors. Multi errors can be usefull for distinguishing different type of errors,
/// for example statistical and systematic errors.
///
/// \macro_image
/// \macro_code
/// 
/// \author Simon Spies

void gr008_multierrors() {
   TCanvas *c1 = new TCanvas("c1", "A Simple Graph with multiple y-errors", 200, 10, 700, 500);
   c1->SetGrid();
   c1->GetFrame()->SetBorderSize(12);

   const Int_t np = 5;
   Double_t x[np]       = {0, 1, 2, 3, 4};
   Double_t y[np]       = {0, 2, 4, 1, 3};
   Double_t exl[np]     = {0.3, 0.3, 0.3, 0.3, 0.3}; //Lower x errors
   Double_t exh[np]     = {0.3, 0.3, 0.3, 0.3, 0.3}; //Higher x errors
   Double_t eylstat[np] = {1, 0.5, 1, 0.5, 1}; //Lower y statistical errors
   Double_t eyhstat[np] = {0.5, 1, 0.5, 1, 0.5}; //Higher y statistical errors
   Double_t eylsys[np]  = {0.5, 0.4, 0.8, 0.3, 1.2}; //Lower y systematic errors
   Double_t eyhsys[np]  = {0.6, 0.7, 0.6, 0.4, 0.8}; //Higher y systematic errors

   TGraphMultiErrors *gme = new TGraphMultiErrors("gme", "TGraphMultiErrors Example", np, x, y, exl, exh, eylstat, eyhstat); //Create the TGraphMultiErrors object
   gme->AddYError(np, eylsys, eyhsys); //Add the systematic y-errors to the graph
   gme->SetMarkerStyle(20);
   gme->SetLineColor(kRed);
   gme->GetAttLine(0)->SetLineColor(kRed); //Color for statistical error bars
   gme->GetAttLine(1)->SetLineColor(kBlue); //Color for systematic error bars
   gme->GetAttFill(1)->SetFillStyle(0);

   //Graph is drawn with the option "APS": "A" draw axes, "P" draw points and "S" draw symmetric horizontal error bars (x-errors)
   //Statistical y-errors are drawn with the option "Z" vertical error bars.
   //Systematic y-errors are drawn with the option "5 s=0.5":
   //"5" draw rectangles to represent the systematic y-error bars.
   //"s=0.5" scale the rectangles horizontally (along the x-axis) by a factor of 0.5.
   gme->Draw("APS ; Z ; 5 s=0.5");


   c1->Update();
}

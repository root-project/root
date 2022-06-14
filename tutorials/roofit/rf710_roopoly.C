/// \file
/// \ingroup tutorial_roofit
/// \notebook -js
/// Taylor expansion of RooFit functions using the taylorExpand function with RooPolyFunc
///
/// \macro_image
/// \macro_output
/// \macro_code
///
/// \date December 2021
/// \author Rahul Balasubramanian

#include "RooRealVar.h"
#include "RooPolyFunc.h"
#include "RooAbsCollection.h"
#include "RooPlot.h"
#include "TCanvas.h"

using namespace RooFit;

void rf710_roopoly()
{
   // C r e a t e   p o l y n o m i a l
   // f u n c t i o n  o f  f o u r t h  o r d e r
   // ---------------------------------------------
   // x^4 - 5x^3 + 5x^2 + 5x -6
   // ---------------------------------------------

   RooRealVar x("x", "x", 0, -3, 10);
   RooPolyFunc f("f", "f", RooArgSet(x));
   f.addTerm(+1, x, 4);
   f.addTerm(-5, x, 3);
   f.addTerm(+5, x, 2);
   f.addTerm(+5, x, 1);
   f.addTerm(-6, x, 0);

   // C r e a t e   t a y l o r  e x p a n s i o n
   // ---------------------------------------------
   double x0 = 2.0;
   auto taylor_o1 = RooPolyFunc::taylorExpand("taylor_o1", "taylor expansion order 1", f, RooArgSet(x), x0, 1);
   auto taylor_o2 = RooPolyFunc::taylorExpand("taylor_o2", "taylor expansion order 2", f, RooArgSet(x), x0, 2);

   // Plot polynomial with first and second order taylor expansion overlaid
   auto frame = x.frame(Title("x^{4} - 5x^{3} + 5x^{2} + 5x - 6"));
   auto c = new TCanvas("rf710_roopoly", "rf710_roopoly", 400, 400);
   c->cd();

   f.plotOn(frame, Name("f"));
   taylor_o1->plotOn(frame, LineColor(kRed), LineStyle(kDashed), Name("taylor_o1"));
   taylor_o2->plotOn(frame, LineColor(kRed - 9), LineStyle(kDotted), Name("taylor_o2"));

   frame->SetMinimum(-8.0);
   frame->SetMaximum(+8.0);
   frame->SetYTitle("function value");
   frame->Draw();

   auto legend = new TLegend(0.53, 0.7, 0.86, 0.87);
   legend->SetFillColor(kWhite);
   legend->SetLineColor(kWhite);
   legend->SetTextSize(0.02);
   legend->AddEntry("taylor_o1", "Taylor expansion upto first order", "L");
   legend->AddEntry("taylor_o2", "Taylor expansion upto second order", "L");
   legend->AddEntry("f", "Polynomial of fourth order", "L");
   legend->Draw();
   c->Draw();
   c->SaveAs("rf710_roopoly.png");
}

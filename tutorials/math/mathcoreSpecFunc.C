// Example macro describing how to use the special mathematical functions
// taking full advantage of the precision and speed of the C99 compliant
// environments. To execute the macro type in:
//
// root[0]: .x mathcoreSpecFunc.C 
//
// It will create two canvases: 
//
//   a) one with the representation of the tgamma, lgamma, erf and erfc functions
//   b) one with the relative difference between the old ROOT versions and the
//      C99 implementation (on obsolete platform+compiler combinations which are
//      not C99 compliant it will call the original ROOT implementations, hence
//      the difference will be 0)
//
// The naming and numbering of the functions is taken from
// <A HREF="http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2004/n1687.pdf">
// Matt Austern,
// (Draft) Technical Report on Standard Library Extensions,
// N1687=04-0127, September 10, 2004</A>
//
//  Author: Andras Zsenei


#include "TF1.h"
#include "TSystem.h"
#include "TCanvas.h"


void mathcoreSpecFunc() {

gSystem->Load("libMathCore");
TF1 *f1a = new TF1("f1a","ROOT::Math::tgamma(x)",0,20);
TF1 *f1b = new TF1("f1b","abs((ROOT::Math::tgamma(x)-TMath::Gamma(x))/ROOT::Math::tgamma(x))",0,20);

TF1 *f2a = new TF1("f2a","ROOT::Math::lgamma(x)",0,100);
TF1 *f2b = new TF1("f2b","abs((ROOT::Math::lgamma(x)-TMath::LnGamma(x))/ROOT::Math::lgamma(x))",0,100);

TF1 *f3a = new TF1("f3a","ROOT::Math::erf(x)",0,5);
TF1 *f3b = new TF1("f3b","abs((ROOT::Math::erf(x)-TMath::Erf(x))/ROOT::Math::erf(x))",0,5);

TF1 *f4a = new TF1("f4a","ROOT::Math::erfc(x)",0,5);
TF1 *f4b = new TF1("f4b","abs((ROOT::Math::erfc(x)-TMath::Erfc(x))/ROOT::Math::erfc(x))",0,5);


TCanvas *c1 = new TCanvas("c1","c1",1000,750);
c1->SetFillColor(kYellow-10);

f1a->SetLineColor(kBlue);
f1b->SetLineColor(kBlue);
f2a->SetLineColor(kBlue);
f2b->SetLineColor(kBlue);
f3a->SetLineColor(kBlue);
f3b->SetLineColor(kBlue);
f4a->SetLineColor(kBlue);
f4b->SetLineColor(kBlue);

c1->Divide(2,2);

c1->cd(1);
f1a->Draw();
c1->cd(2);
f2a->Draw();
c1->cd(3);
f3a->Draw();
c1->cd(4);
f4a->Draw();


TCanvas *c2 = new TCanvas("c2","c2",1000,750);

c2->SetFillColor(kYellow-10);

c2->Divide(2,2);

c2->cd(1);
f1b->Draw();
c2->cd(2);
f2b->Draw();
c2->cd(3);
f3b->Draw();
c2->cd(4);
f4b->Draw();

}

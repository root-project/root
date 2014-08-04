/////////////////////////////////////////////////////////////////////////
//
// Demonstraite Z_Bi = Z_Gamma
// author: Kyle Cranmer & Wouter Verkerke
// date May 2010
//
//
/////////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooRealVar.h"
#include "RooProdPdf.h"
#include "RooWorkspace.h"
#include "RooDataSet.h"
#include "TCanvas.h"
#include "TH1.h"

using namespace RooFit;
using namespace RooStats;

void Zbi_Zgamma() {

  // Make model for prototype on/off problem
  // Pois(x | s+b) * Pois(y | tau b )
  // for Z_Gamma, use uniform prior on b.
  RooWorkspace* w = new RooWorkspace("w",true);
  w->factory("Poisson::px(x[150,0,500],sum::splusb(s[0,0,100],b[100,0,300]))");
  w->factory("Poisson::py(y[100,0,500],prod::taub(tau[1.],b))");
  w->factory("Uniform::prior_b(b)");

  // construct the Bayesian-averaged model (eg. a projection pdf)
  // p'(x|s) = \int db p(x|s+b) * [ p(y|b) * prior(b) ]
  w->factory("PROJ::averagedModel(PROD::foo(px|b,py,prior_b),b)") ;

  // plot it, blue is averaged model, red is b known exactly
  RooPlot* frame = w->var("x")->frame() ;
  w->pdf("averagedModel")->plotOn(frame) ;
  w->pdf("px")->plotOn(frame,LineColor(kRed)) ;
  frame->Draw() ;

  // compare analytic calculation of Z_Bi
  // with the numerical RooFit implementation of Z_Gamma
  // for an example with x = 150, y = 100

  // numeric RooFit Z_Gamma
  w->var("y")->setVal(100);
  w->var("x")->setVal(150);
  RooAbsReal* cdf = w->pdf("averagedModel")->createCdf(*w->var("x"));
  cdf->getVal(); // get ugly print messages out of the way

  cout << "Hybrid p-value = " << cdf->getVal() << endl;
  cout << "Z_Gamma Significance  = " <<
    PValueToSignificance(1-cdf->getVal()) << endl;

  // analytic Z_Bi
  double Z_Bi = NumberCountingUtils::BinomialWithTauObsZ(150, 100, 1);
  std::cout << "Z_Bi significance estimation: " << Z_Bi << std::endl;

  // OUTPUT
  // Hybrid p-value = 0.999058
  // Z_Gamma Significance  = 3.10804
  // Z_Bi significance estimation: 3.10804


}

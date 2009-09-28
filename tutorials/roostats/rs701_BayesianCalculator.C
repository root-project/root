/////////////////////////////////////////////////////////////////////////
//
// 'Bayesian Calculator' RooStats tutorial macro #701
// author: Gregory Schott
// date Sep 2009
//
// This tutorial shows an example of using the BayesianCalculator class 
//
/////////////////////////////////////////////////////////////////////////


#include "RooRealVar.h"
#include "RooProdPdf.h"
#include "RooWorkspace.h"
#include "RooDataSet.h"


#include "RooStats/BayesianCalculator.h"
#include "RooStats/SimpleInterval.h"

using namespace RooFit;
using namespace RooStats;

void rs701_BayesianCalculator()
{

  RooWorkspace* w = new RooWorkspace("w",true);
  w->factory("SUM::pdf(s[0,15]*Uniform(x[0,1]),b[0.001,0,2]*Uniform(x))");
  w->factory("Gaussian::prior_b(b,0.001,1)");
  w->factory("PROD::model(pdf,prior_b)");
  RooAbsPdf* model = w->pdf("model");  // pdf*priorNuisance
  const RooArgSet nuisanceParameters(*(w->var("b")));

  w->factory("Uniform::priorPOI(s)");
  RooAbsRealLValue* POI = w->var("s");
  RooAbsPdf* priorPOI = w->pdf("priorPOI");

  w->factory("n[1]"); // observed number of events
  RooDataSet data("data","",RooArgSet(*(w->var("x")),*(w->var("n"))),"n");
  data.add(RooArgSet(*(w->var("x"))),w->var("n")->getVal());

  BayesianCalculator bcalc(data,*model,RooArgSet(*POI),*priorPOI,&nuisanceParameters);
  SimpleInterval* interval = bcalc.GetInterval();
  std::cout << "90% CL interval: [ " << interval->LowerLimit() << " - " << interval->UpperLimit() << " ] or 95% CL limits\n";
  bcalc.PlotPosterior();
  
  // observe one event while expecting one background event -> the 95% CL upper limit on s is 4.10
  // observe one event while expecting zero background event -> the 95% CL upper limit on s is 4.74
}

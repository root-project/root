

#include "RooRealVar.h"
#include "RooStats/SPlot.h"
#include "RooGaussian.h"
#include "RooExponential.h"
#include "RooLinearVar.h"
#include "RooConstVar.h"
#include "RooAddPdf.h"

#include "gtest/gtest.h"

TEST(SPlot, UseWithRooLinearVar) {
  RooRealVar x("x", "observable", 0, 0, 20);
  RooRealVar m("m", "mean", 5., -10, 10);
  RooRealVar s("s", "sigma", 2., 0.1, 10);
  RooGaussian gauss("gauss", "gauss", x, m, s);

  RooRealVar a("a", "exp", -0.2, -10., 0.);
  RooExponential ex("ex", "ex", x, a);

  RooRealVar common("common", "common scale", 3., 0, 10);
  RooRealVar r1("r1", "ratio 1", 0.3, 0, 10);
  RooRealVar r2("r2", "ratio 2", 0.5, 0, 10);
  RooLinearVar c1("c1", "c1", r1, common, RooFit::RooConst(0.));
  RooLinearVar c2("c2", "c2", r2, common, RooFit::RooConst(0.));
  RooAddPdf sum("sum", "sum", RooArgSet(gauss, ex), RooArgSet(c1, c2));

  std::unique_ptr<RooDataSet> data{sum.generate(x, 1000)};

  RooStats::SPlot splot("splot", "splot", *data, &sum, RooArgSet(c1, c2));
  EXPECT_EQ(splot.GetNumSWeightVars(), 2);
  EXPECT_NE(data->get(0)->find("c1_sw"), nullptr);
}

// Regression test for https://github.com/root-project/root/issues/11768:
// passing a discriminating variable of the fit in the yields list must
// produce an error, since the resulting sWeights would be invalid.
TEST(SPlot, ErrorOnDiscriminatingVariableAsYield)
{
   RooRealVar x("x", "observable", 0, 0, 20);
   RooRealVar m("m", "mean", 5., -10, 10);
   RooRealVar s("s", "sigma", 2., 0.1, 10);
   RooGaussian gauss("gauss", "gauss", x, m, s);

   RooRealVar a("a", "exp", -0.2, -10., 0.);
   RooExponential ex("ex", "ex", x, a);

   RooRealVar nsig("nsig", "nsig", 500, 0, 1000);
   RooRealVar nbkg("nbkg", "nbkg", 500, 0, 1000);
   RooAddPdf sum("sum", "sum", RooArgSet(gauss, ex), RooArgSet(nsig, nbkg));

   std::unique_ptr<RooDataSet> data{sum.generate(x, 1000)};

   // "x" is a discriminating variable of the fit, so using it in the yields
   // list is a user error that must throw.
   EXPECT_THROW(RooStats::SPlot("splot", "splot", *data, &sum, RooArgList(nsig, x)), std::invalid_argument);
}

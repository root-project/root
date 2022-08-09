// Tests for the RooWrapperPdf
// Author: Stephan Hageboeck, CERN  09/2019

#include "RooWrapperPdf.h"
#include "RooRealVar.h"
#include "RooProduct.h"
#include "RooRealSumFunc.h"
#include "RooConstVar.h"
#include "RooPlot.h"
#include "RooFitResult.h"
#include "RooDataSet.h"

#include "TCanvas.h"

#include "gtest/gtest.h"

TEST(RooWrapperPdf, Basics)
{
  RooRealVar x("x", "x", -5., 5.);

  // Implement a poor-man's polynomial. Value ranges are chosen to keep it positive.
  RooRealVar a0("a0", "a0", 0.1, 0.1, 10.);
  RooRealVar a1("a1", "a1", -0.01, -2.1, 0.);
  RooRealVar a2("a2", "a2", 0.01, 0.01, 5.);
  RooProduct xId("xId", "x", RooArgList(x));
  RooProduct xSq("xSq", "x^2", RooArgList(x, x));
  RooConstVar one("one", "one", 1.);
  RooRealSumFunc pol("pol", "pol", RooArgList(one, xId, xSq), RooArgList(a0, a1, a2));

  RooWrapperPdf polPdf("polPdf", "polynomial PDF", pol);

  EXPECT_GT(pol.getVal(x)*1.05, polPdf.getVal(x)) << "Wrapper pdf normalises.";

  RooArgSet intSet(x);
  RooArgSet numSet;

  EXPECT_NE(polPdf.getAnalyticalIntegralWN(intSet, numSet, &intSet, nullptr), 0)
        << "Test that PDF claims to have analytic integral with norm.";

//  auto frame = x.frame();
//  pol.plotOn(frame);
//  polPdf.plotOn(frame);
//  TCanvas canv;
//  frame->Draw();
//  canv.SaveAs("/tmp/testWrapperPdf.png");
}


TEST(RooWrapperPdf, GenerateAndFit) {
  RooRealVar x("x", "x", -5., 5.);

  // Implement a poor-man's polynomial. Value ranges are chosen to keep it positive.
  RooRealVar a0("a0", "a0", 0.1);
  RooRealVar a1("a1", "a1", -0.01);
  RooRealVar a2("a2", "a2", 0.01, 0.001, 5.);
  RooProduct xId("xId", "x", RooArgList(x));
  RooProduct xSq("xSq", "x^2", RooArgList(x, x));
  RooConstVar one("one", "one", 1.);
  RooRealSumFunc pol("pol", "pol", RooArgList(one, xId, xSq), RooArgList(a0, a1, a2));

  RooWrapperPdf polPdf("polPdf", "polynomial PDF", pol);

  auto data = polPdf.generate(x, 50000);
  a2.setVal(0.02);
  auto result = polPdf.fitTo(*data, RooFit::Save(), RooFit::PrintLevel(-1));

  EXPECT_EQ(result->status(), 0) << "Fit converged.";
  EXPECT_LT(fabs(a2.getVal()-0.01), a2.getError());

//  auto frame = x.frame();
//  data->plotOn(frame);
//  polPdf.plotOn(frame);
//  TCanvas canv;
//  frame->Draw();
//  canv.SaveAs("/tmp/testWrapperPdf2.png");
}


TEST(RooWrapperPdf, DISABLED_FullAnalyticInt) {
  RooRealVar x("x", "x", 4., 0., 10.);
  RooRealVar y("y", "y", -0.5, -5., 5.);

  RooProduct xy("xy", "x*y", RooArgList(x, y));
  RooWrapperPdf prodPdf("prodPdf", "PDF(x*y)", xy);

  RooArgSet intSet(x);
  RooArgSet numSet;

  EXPECT_NE(prodPdf.getAnalyticalIntegral(intSet, numSet, nullptr), 0)
      << "Test that PDF claims to have analytic integral.";

  EXPECT_FLOAT_EQ(xy.getVal(), -1.5);
  std::cout << "The following error is expected:\n----" << std::endl;
  EXPECT_FLOAT_EQ(prodPdf.getVal(), 0.);
  std::cout << "----" << std::endl;

  constexpr double newY = 2.;
  y.setVal(newY);
  EXPECT_NEAR(prodPdf.getVal(), 3.*newY, 0.001);
  EXPECT_NEAR(prodPdf.getVal(x), 0.3, 0.001);

//  auto cdf = prodPdf.createCdf(x);
//  for (unsigned int i=0; i<10; ++i) {
//    x.setVal(i);
//    std::cout << i << " " << std::setprecision(4)
//        << std::setw(7) << xy.getVal(x)
//        << std::setw(7) << prodPdf.getVal(x)
//        << std::setw(7) << cdf->getVal() << std::endl;
//    EXPECT_NEAR(xy.getVal(x), newY*i, 1.E-5);
//    EXPECT_NEAR(prodPdf.getVal(x), newY*i/10., 1.E-5);
//    EXPECT_NEAR(cdf->getVal(), newY*i, 1.E-5);
//  }

//  auto frame = x.frame();
//  prodPdf.plotOn(frame);
//  cdf->plotOn(frame, RooFit::LineColor(kRed));
//  TCanvas canv;
//  frame->Draw();
//  canv.SaveAs("/tmp/testWrapperPdf3.png");
}


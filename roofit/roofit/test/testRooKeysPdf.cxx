// Test the legacy implementation of the RooKeyPdf against the updated version.
// Author: Stephan Hageboeck, Sep 19

#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooConstVar.h"
#include "RooPolynomial.h"
#include "RooKeysPdf.h"
#include "RooPlot.h"

#include "TCanvas.h"
#include "TH1.h"
#include "TDirectory.h"

#include "legacyPDFs/RooKeysPdf.h"
#include "legacyPDFs/RooKeysPdf.cxx"

#include "gtest/gtest.h"

#include <memory>

using namespace RooFit;

class RooKeysPdfTest :
//public ::testing::Test {
		public ::testing::TestWithParam<Legacy::RooKeysPdf::Mirror> {
protected:
  RooKeysPdfTest() {
    x.reset(new RooRealVar("x", "x", 0, 20));
    RooPolynomial p("p", "p", *x, RooArgList(RooConst(0.01), RooConst(-0.01), RooConst(0.0004)));
    data.reset(p.generate(*x, 200));

    TDirectory::AddDirectory(false);
  }

  ~RooKeysPdfTest() { }

  std::unique_ptr<RooRealVar> x;
  std::unique_ptr<RooDataSet> data;
  double kernelWidth{1.};
};


TEST_P(RooKeysPdfTest, LegacyImplementation) {
  auto mirrorState = GetParam();
  std::string name = ::testing::UnitTest::GetInstance()->current_test_info()->test_case_name();
  if (name.substr(0,3) == std::string("Two")) {
    kernelWidth = 2.;
  }

  RooKeysPdf keys("kest1", "kest1", *x, *data, static_cast<RooKeysPdf::Mirror>(mirrorState), kernelWidth);
  Legacy::RooKeysPdf keys_leg("kest1_leg", "kest1_leg", *x, *data, static_cast<Legacy::RooKeysPdf::Mirror>(mirrorState), kernelWidth);

  auto hist1 = std::unique_ptr<TH1>(keys.createHistogram("x", 500));
  auto hist1_leg = std::unique_ptr<TH1>(keys_leg.createHistogram("x", 500));

  unsigned int nOff = 0;
  for (int i = 0; i <= hist1->GetNbinsX()+1; ++i) {
    if (fabs(hist1->GetBinContent(i) - hist1_leg->GetBinContent(i)) > fabs(0.03*hist1_leg->GetBinContent(i)))
      nOff++;
  }

  EXPECT_LE(nOff, 3u) << "More than 3 bins deviate from reference implementation.";


  auto frame = std::unique_ptr<RooPlot>(x->frame());
  keys.plotOn(frame.get());
  keys_leg.plotOn(frame.get(), LineColor(kRed), LineStyle(kDotted));

  TCanvas canv;
  frame->Draw();
  canv.Draw();
  std::replace(name.begin(), name.end(), '/', '_');
  std::stringstream testName;
  testName << "/tmp/" << name << "_" << GetParam() << ".png";
  canv.SaveAs(testName.str().c_str());
}


INSTANTIATE_TEST_CASE_P(One, RooKeysPdfTest,
		::testing::Range(Legacy::RooKeysPdf::NoMirror,
				static_cast<Legacy::RooKeysPdf::Mirror>(Legacy::RooKeysPdf::MirrorAsymBoth+1)));
INSTANTIATE_TEST_CASE_P(Two, RooKeysPdfTest,
		::testing::Range(Legacy::RooKeysPdf::NoMirror,
				static_cast<Legacy::RooKeysPdf::Mirror>(Legacy::RooKeysPdf::MirrorAsymBoth+1)));

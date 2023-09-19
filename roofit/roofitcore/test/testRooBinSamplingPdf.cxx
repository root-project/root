// Tests for the RooBinSamplingPdf
// Authors: Jonas Rembser, CERN  03/2022

#include <RooAddPdf.h>
#include <RooArgSet.h>
#include <RooBinSamplingPdf.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooGenericPdf.h>
#include <RooHelpers.h>
#include <RooRandom.h>
#include <RooRealVar.h>

#include "gtest_wrapper.h"

#include <memory>

class ParamTest : public testing::TestWithParam<std::tuple<std::string>> {
   void SetUp() override
   {
      RooRandom::randomGenerator()->SetSeed(1337ul);
      _batchMode = std::get<0>(GetParam());
      _changeMsgLvl = std::make_unique<RooHelpers::LocalChangeMsgLevel>(RooFit::WARNING);
   }

   void TearDown() override { _changeMsgLvl.reset(); }

protected:
   std::string _batchMode;

private:
   std::unique_ptr<RooHelpers::LocalChangeMsgLevel> _changeMsgLvl;
};

// For a linear pdf, doing the bin sampling should make no difference because
// the integral of a linear function is the same as the central point.
TEST_P(ParamTest, LinearPdfCrossCheck)
{
   using namespace RooFit;

   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooRandom::randomGenerator()->SetSeed(1337ul);

   RooRealVar x("x", "x", 0.1, 5.1);
   x.setBins(10);

   RooGenericPdf pdf("lin", "x", {x});
   std::unique_ptr<RooDataHist> dataH(pdf.generateBinned(x, 10000));
   RooDataSet data("data", "data", x, RooFit::Import(*dataH));

   std::unique_ptr<RooAbsReal> nll1(pdf.createNLL(data));
   std::unique_ptr<RooAbsReal> nll2(pdf.createNLL(data, IntegrateBins(1.E-3)));

   EXPECT_FLOAT_EQ(nll2->getVal(), nll1->getVal());
}

// For a linear pdf, doing the bin sampling should make no difference because
// the integral of a linear function is the same as the central point.
// Similar to "LinearPdfCrossCheck", but this time for a subrange fit.
TEST_P(ParamTest, LinearPdfSubRangeCrossCheck)
{
   using namespace RooFit;

   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooRandom::randomGenerator()->SetSeed(1337ul);

   RooRealVar x("x", "x", 0.1, 5.1);
   x.setBins(10);
   x.setRange("range", 0.1, 4.1);
   x.setBins(8, "range"); // consistent binning

   RooGenericPdf pdf("lin", "x", {x});
   std::unique_ptr<RooDataHist> dataH(pdf.generateBinned(x, 10000));
   RooDataSet data("data", "data", x, RooFit::Import(*dataH));

   std::unique_ptr<RooAbsReal> nll1(pdf.createNLL(data, Range("range")));
   std::unique_ptr<RooAbsReal> nll2(pdf.createNLL(data, Range("range"), IntegrateBins(1.E-3)));

   EXPECT_FLOAT_EQ(nll2->getVal(), nll1->getVal());
}

TEST(RooBinSamplingPdf, CheckConsistentNormalization)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooRealVar x("x", "x", 0, 10);
   RooRealVar mean1("mean1", "mean1", 4., 0, 10);
   RooRealVar mean2("mean2", "mean2", 6., 0, 10);
   RooRealVar width("width", "width", 3., 0.1, 10);
   RooRealVar f("f", "f", 0.5, 0.0, 1.0);

   RooArgSet normSet{x};

   RooGenericPdf gaus1("gauss1", "gauss1", "std::exp(-0.5*(x - mean1)^2/width^2)", {x, mean1, width});
   RooGenericPdf gaus2("gauss2", "gauss2", "std::exp(-0.5*(x - mean2)^2/width^2)", {x, mean2, width});
   RooAddPdf pdf{"pdf", "pdf", {gaus1, gaus2}, {f}};
   pdf.fixCoefNormalization(normSet);

   RooBinSamplingPdf binSamplingPdf{"binSamplingPdf", "binSamplingPdf", x, pdf};

   // An integral over the normalization set normalized by an integral over the
   // normalization set should be unity by definition.
   std::unique_ptr<RooAbsReal> int1{binSamplingPdf.createIntegral(normSet, &normSet)};
   EXPECT_FLOAT_EQ(int1->getVal(), 1.0);

   // Evaluating the pdf with a given normalization set should not unexpectedly
   // change the value of it's unnormalized integral.
   std::unique_ptr<RooAbsReal> int2{binSamplingPdf.createIntegral(normSet)};
   binSamplingPdf.getVal(normSet);
   std::unique_ptr<RooAbsReal> int3{binSamplingPdf.createIntegral(normSet)};
   EXPECT_FLOAT_EQ(int2->getVal(), int3->getVal());
}

INSTANTIATE_TEST_SUITE_P(RooBinSamplingPdf, ParamTest, testing::Values("Off", "Cpu"),
                         [](testing::TestParamInfo<ParamTest::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << "BatchMode" << std::get<0>(paramInfo.param);
                            return ss.str();
                         });

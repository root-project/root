// Tests for RooMCStudy
// Authors: Jonas Rembser, CERN 2026

#include <RooAbsPdf.h>
#include <RooArgSet.h>
#include <RooCurve.h>
#include <RooDataSet.h>
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooMCStudy.h>
#include <RooPlot.h>
#include <RooRandom.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include "gtest_wrapper.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

void fillModel(RooWorkspace &ws)
{
   ws.factory("x[-10, 10]");
   ws.factory("Gaussian::pdf1(x, m[-1, 1], s[5, 10])");
   ws.factory("Gaussian::pdf2(x, m, s2[1, 3])");
   ws.factory("SUM::pdf(N1[0, 100] * pdf1, N2[0, 100] * pdf2)");
}

std::vector<double> getColumn(RooDataSet const &data, const char *name)
{
   auto *var = static_cast<RooRealVar const *>(data.get()->find(name));
   if (var == nullptr) {
      throw std::runtime_error(std::string{"dataset has no column named \""} + name + "\"");
   }
   std::vector<double> out;
   out.reserve(data.numEntries());
   for (int i = 0; i < data.numEntries(); ++i) {
      data.get(i);
      out.push_back(var->getVal());
   }
   return out;
}

/// Returns the peak position and the width of the last curve on the frame,
/// estimated from the full width at half maximum. Used to validate the
/// Gaussian that FitGauss() fits to the plotted distribution.
std::pair<double, double> curvePeakAndWidth(RooPlot &frame)
{
   // Not frame.getCurve(), because that returns the last item on the frame,
   // which is the box with the fit parameters and not the fitted curve.
   auto *curve = static_cast<RooCurve *>(frame.findObject(nullptr, RooCurve::Class()));
   if (curve == nullptr) {
      throw std::runtime_error("frame has no curve");
   }
   const int nPoints = curve->GetN();
   double yMax = -std::numeric_limits<double>::infinity();
   double xPeak = 0.0;
   for (int i = 0; i < nPoints; ++i) {
      if (curve->GetPointY(i) > yMax) {
         yMax = curve->GetPointY(i);
         xPeak = curve->GetPointX(i);
      }
   }
   double xLo = std::numeric_limits<double>::infinity();
   double xHi = -std::numeric_limits<double>::infinity();
   for (int i = 0; i < nPoints; ++i) {
      if (curve->GetPointY(i) >= 0.5 * yMax) {
         xLo = std::min(xLo, curve->GetPointX(i));
         xHi = std::max(xHi, curve->GetPointX(i));
      }
   }
   // 2 * sqrt(2 * log(2)) converts the full width at half maximum to a sigma
   return {xPeak, (xHi - xLo) / 2.3548200450309493};
}

} // namespace

/// Covers GitHub issue #9490: the generated parameter values must be saved
/// also when no constraints are used.
TEST(RooMCStudy, GenParDataSetNoConstraints)
{
   RooRandom::randomGenerator()->SetSeed(4357);
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING};

   RooWorkspace ws;
   fillModel(ws);

   RooMCStudy mcstudy{*ws.pdf("pdf"), *ws.var("x"), RooFit::Silence()};
   mcstudy.generate(3, 100, true);

   RooDataSet const *genParData = mcstudy.genParDataSet();
   ASSERT_NE(genParData, nullptr);
   EXPECT_EQ(genParData->numEntries(), 3);
   // The columns keep the original parameter names
   EXPECT_NE(genParData->get()->find("s"), nullptr);
}

/// Covers GitHub issue #9490: parameters constrained by external constraint
/// p.d.f.s must be sampled from them for each toy, like for internal
/// constraints.
TEST(RooMCStudy, GenParDataSetExternalConstraints)
{
   RooRandom::randomGenerator()->SetSeed(4357);
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING};

   RooWorkspace ws;
   fillModel(ws);
   ws.factory("Gaussian::constraint(s, cm[7], cs[0.5])");
   RooArgSet extCons{*ws.pdf("constraint")};

   RooMCStudy mcstudy{*ws.pdf("pdf"), *ws.var("x"), RooFit::ExternalConstraints(extCons), RooFit::Silence()};
   const int nToys = 20;
   mcstudy.generate(nToys, 100, true);

   RooDataSet const *genParData = mcstudy.genParDataSet();
   ASSERT_NE(genParData, nullptr);
   ASSERT_EQ(genParData->numEntries(), nToys);

   std::vector<double> svals = getColumn(*genParData, "s");
   double smin = svals[0];
   double smax = svals[0];
   double ssum = 0.0;
   for (double v : svals) {
      smin = std::min(smin, v);
      smax = std::max(smax, v);
      ssum += v;
   }
   // The values of "s" are sampled from Gaussian(s | 7, 0.5) for each toy
   EXPECT_GT(smax, smin);
   EXPECT_NEAR(ssum / nToys, 7.0, 1.0);
}

/// The internal-constraints behavior that worked before GitHub issue #9490
/// must be unchanged: per-toy sampled parameters in genParDataSet, and
/// "<name>_gen" columns merged into fitParDataSet.
TEST(RooMCStudy, GenParDataSetInternalConstraints)
{
   RooRandom::randomGenerator()->SetSeed(4357);
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING};

   RooWorkspace ws;
   fillModel(ws);
   ws.factory("Gaussian::constraint(s, cm[7], cs[0.5])");
   ws.factory("PROD::prodpdf({pdf,constraint})");

   const int nToys = 3;
   RooMCStudy mcstudy{*ws.pdf("prodpdf"), *ws.var("x"), RooFit::Constrain(*ws.var("s")), RooFit::Silence(),
                      RooFit::FitOptions(RooFit::PrintLevel(-1))};
   mcstudy.generateAndFit(nToys, 200);

   RooDataSet const *genParData = mcstudy.genParDataSet();
   ASSERT_NE(genParData, nullptr);
   EXPECT_EQ(genParData->numEntries(), nToys);
   EXPECT_NE(genParData->get()->find("s"), nullptr);
   EXPECT_NE(mcstudy.fitParDataSet().get()->find("s_gen"), nullptr);
}

/// Covers GitHub issue #12387: the FitGauss() command argument should work
/// not only for plotPull(), but also for the functions that plot the fitted
/// values, their errors and the NLL distribution, because users need Gaussian
/// fits of the parameter distribution for linearity tests.
TEST(RooMCStudy, FitGauss)
{
   RooRandom::randomGenerator()->SetSeed(4357);
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING};

   RooWorkspace ws;
   fillModel(ws);

   RooMCStudy mcstudy{*ws.pdf("pdf"), *ws.var("x"), RooFit::Silence(), RooFit::FitOptions(RooFit::PrintLevel(-1))};
   mcstudy.generateAndFit(50, 200);

   // Without FitGauss(), the frame only contains the data histogram
   std::unique_ptr<RooPlot> frame1{mcstudy.plotParam(*ws.var("s"), RooFit::Bins(40))};
   ASSERT_NE(frame1, nullptr);
   EXPECT_EQ(frame1->numItems(), 1);

   // With FitGauss(), the fitted Gaussian curve and the box with the fitted
   // mean and sigma are added to the frame
   std::unique_ptr<RooPlot> frame2{mcstudy.plotParam(*ws.var("s"), RooFit::Bins(40), RooFit::FitGauss(true))};
   ASSERT_NE(frame2, nullptr);
   EXPECT_EQ(frame2->numItems(), 3);

   // FitGauss() is also supported by plotError() and plotNLL(), which are
   // implemented via plotParam()
   std::unique_ptr<RooPlot> frame3{mcstudy.plotError(*ws.var("s"), RooFit::Bins(40), RooFit::FitGauss(true))};
   ASSERT_NE(frame3, nullptr);
   EXPECT_EQ(frame3->numItems(), 3);

   std::unique_ptr<RooPlot> frame4{mcstudy.plotNLL(RooFit::Bins(40), RooFit::FitGauss(true))};
   ASSERT_NE(frame4, nullptr);
   EXPECT_EQ(frame4->numItems(), 3);

   // The existing FitGauss() support in plotPull() must be unchanged
   std::unique_ptr<RooPlot> frame5{mcstudy.plotPull(*ws.var("s"), RooFit::Bins(40), RooFit::FitGauss(true))};
   ASSERT_NE(frame5, nullptr);
   EXPECT_EQ(frame5->numItems(), 3);
}

/// Also covers GitHub issue #12387: the Gaussian that FitGauss() adds must
/// actually describe the plotted distribution. The parameters of that Gaussian
/// are seeded from the moments of the distribution, because seeding them from
/// the frame range instead fails in two ways: the fitted width runs into a
/// limit if the distribution is much narrower than the frame, and the fitted
/// mean is off if the frame range can't be represented precisely enough in the
/// RooWorkspace factory expression that builds the Gaussian.
TEST(RooMCStudy, FitGaussSeeding)
{
   // Both the width of the frame relative to the distribution and the absolute
   // scale of the fitted parameter are relevant here, so they are scanned.
   // The offset 1000000.5 is deliberately a value that a low-precision string
   // representation of the frame range would not resolve.
   for (double offset : {0.0, 1000000.5}) {
      for (bool wideFrame : {false, true}) {
         RooRandom::randomGenerator()->SetSeed(4357);
         RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING};

         RooWorkspace ws;
         ws.factory("x[" + std::to_string(offset - 10.0) + ", " + std::to_string(offset + 10.0) + "]");
         ws.factory("Gaussian::pdf(x, m[" + std::to_string(offset) + ", " + std::to_string(offset - 5.0) + ", " +
                    std::to_string(offset + 5.0) + "], s[1.0, 0.1, 5.0])");

         RooMCStudy mcstudy{*ws.pdf("pdf"), *ws.var("x"), RooFit::Silence(),
                            RooFit::FitOptions(RooFit::PrintLevel(-1))};
         // Many events per toy, so that the distribution of the fitted parameter is
         // much narrower than the wide frame range below.
         mcstudy.generateAndFit(100, 5000);

         auto const &fitParData = mcstudy.fitParDataSet();
         auto const &mFit = static_cast<RooRealVar const &>(*fitParData.get()->find("m"));
         const double dataMean = fitParData.mean(mFit);
         const double dataSigma = fitParData.sigma(mFit);

         // Without FrameRange(), the frame is auto-ranged around the
         // distribution. With it, the frame is much wider than the distribution.
         RooCmdArg frameRange = wideFrame ? RooFit::Range(offset - 5.0, offset + 5.0) : RooCmdArg::none();
         std::unique_ptr<RooPlot> frame{
            mcstudy.plotParam(*ws.var("m"), RooFit::Bins(40), RooFit::FitGauss(true), frameRange)};
         ASSERT_NE(frame, nullptr);
         ASSERT_EQ(frame->numItems(), 3);

         auto const [peak, width] = curvePeakAndWidth(*frame);
         const std::string ctx =
            "offset = " + std::to_string(offset) + ", wideFrame = " + std::to_string(wideFrame);
         EXPECT_NEAR(peak, dataMean, 0.5 * dataSigma) << ctx;
         EXPECT_GT(width, 0.5 * dataSigma) << ctx;
         EXPECT_LT(width, 2.0 * dataSigma) << ctx;
      }
   }
}

/// Covers the fitParData/genParData size mismatch from the discussion in
/// GitHub issue #9490: when some fits fail, the "<name>_gen" columns merged
/// into fitParDataSet must stay aligned with the successful fits, while
/// genParDataSet keeps the entries of all generated toys.
TEST(RooMCStudy, FailedFitsMergeConsistency)
{
   RooRandom::randomGenerator()->SetSeed(4357);
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING};

   RooWorkspace ws;
   ws.factory("x[-10, 10]");
   ws.factory("Gaussian::gauss(x, m[0, -1, 1], s[7, 5, 10])");
   // With only 0.7 expected events, some toys have zero events and their fits fail
   ws.factory("ExtendPdf::model(gauss, nev[0.7])");
   ws.factory("Gaussian::constraint(s, cm[7], cs[0.5])");
   RooArgSet extCons{*ws.pdf("constraint")};

   const int nToys = 30;
   RooMCStudy mcstudy{*ws.pdf("model"),
                      *ws.var("x"),
                      RooFit::ExternalConstraints(extCons),
                      RooFit::Extended(),
                      RooFit::Silence(),
                      RooFit::FitOptions(RooFit::PrintLevel(-1))};
   mcstudy.generateAndFit(nToys, 0);

   RooDataSet const *genParData = mcstudy.genParDataSet();
   ASSERT_NE(genParData, nullptr);
   EXPECT_EQ(genParData->numEntries(), nToys);

   RooDataSet const &fitParData = mcstudy.fitParDataSet();
   const int nFit = fitParData.numEntries();
   // Make sure the test setup is meaningful: some toys must have failed
   ASSERT_LT(nFit, nToys);
   ASSERT_GT(nFit, 0);
   ASSERT_NE(fitParData.get()->find("s_gen"), nullptr);

   // The merged values must be an ordered subsequence of the generated ones
   std::vector<double> fitGenVals = getColumn(fitParData, "s_gen");
   std::vector<double> allGenVals = getColumn(*genParData, "s");
   std::size_t iAll = 0;
   for (double val : fitGenVals) {
      while (iAll < allGenVals.size() && allGenVals[iAll] != val) {
         ++iAll;
      }
      EXPECT_LT(iAll, allGenVals.size()) << "merged s_gen value " << val << " misaligned with generated toys";
      ++iAll;
   }
}

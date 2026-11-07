// Tests for RooMomentMorphFuncND.
// Authors: Jonas Rembser, CERN 05/2026

#include <RooBinning.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooFit/Detail/NormalizationHelpers.h>
#include <RooFit/Evaluator.h>
#include <RooGaussian.h>
#include <RooHelpers.h>
#include <RooHistPdf.h>
#include <RooMomentMorphFuncND.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>
#include <RooWrapperPdf.h>

#include <gtest/gtest.h>

#include <cmath>
#include <memory>

namespace {

/// Workspace with the observable "x", the morphing parameter "mu", and the
/// morph pdf "morph" with two RooHistPdf templates.
std::unique_ptr<RooWorkspace> makeMorphWorkspace(bool pdfMode)
{
   auto wsPtr = std::make_unique<RooWorkspace>();
   RooWorkspace &ws = *wsPtr;
   ws.factory("x[-5,15]");
   ws.factory("mu[0.5,0.0,1.0]");
   RooRealVar &x = *ws.var("x");
   RooRealVar &mu = *ws.var("mu");
   x.setBins(50);

   RooRealVar sigma("sigma", "sigma", 1.5, 1e-3, 10.0);
   sigma.setConstant(true);

   // Templates filled with 10000 events each. This means a huge normalization
   // integral, which makes the discrepancy easy to detect.
   RooMomentMorphFuncND::Grid grid(RooBinning(1, 0.0, 1.0));
   for (int i : {0, 1}) {
      RooRealVar mu_i(Form("mu%d", i), "", i);
      RooGaussian g(Form("g%d", i), "", x, mu_i, sigma);
      std::unique_ptr<RooDataHist> hist{g.generateBinned(x, 10000)};
      ws.import(RooHistPdf(Form("hp%d", i), "", x, *hist, 1), RooFit::Silence(true));
      grid.addPdf(*ws.pdf(Form("hp%d", i)), i);
   }

   RooMomentMorphFuncND morph_func("morph_func", "", RooArgList(mu), RooArgList(x), grid, RooMomentMorphFuncND::Linear);
   if (pdfMode) {
      morph_func.setPdfMode();
   }
   RooWrapperPdf morph("morph", "morph", morph_func, /*selfNormalized=*/true);
   ws.import(morph, RooFit::Silence(true));

   return wsPtr;
}

// For each morph configuration we expect: a normal `getVal(normSet)` on the
// uncompiled pdf and an Evaluator run on the same pdf compiled for that
// normSet must agree. Whether the morph is in pdf mode (sum is a RooAddPdf
// whose components must be normalized for the value to be a proper density)
// or in func mode (sum is a RooRealSumFunc, the value is the raw weighted
// sum of bin contents) is internal to compileForNormSet -- both paths must
// produce the same final value.
void checkBackendConsistency(bool pdfMode)
{
   // FATAL silences a transient logEvalError that the morph emits from its
   // slope/offset RooFormulaVars at cache-construction time (in func mode the
   // initial rms-fractions are zero, so slope = inf for one self-evaluation
   // before calculateFractions runs). It does not affect correctness.
   // TODO: its not elegant that this happens and should be fixed.
   RooHelpers::LocalChangeMsgLevel changeMsgLevel{RooFit::FATAL};

   std::unique_ptr<RooWorkspace> ws = makeMorphWorkspace(pdfMode);
   RooRealVar &x = *ws->var("x");
   RooRealVar &mu = *ws->var("mu");

   RooAbsPdf &morphPdf = *ws->pdf("morph");
   RooArgSet normSet(x);

   mu.setVal(0.5);

   const double valDirect = morphPdf.getVal(normSet);

   std::unique_ptr<RooAbsReal> compiled{RooFit::Detail::compileForNormSet(morphPdf, normSet)};
   RooFit::Evaluator evaluator{*compiled};
   const double valCompiled = evaluator.run()[0];

   EXPECT_NEAR(valCompiled, valDirect, 1e-6 * std::abs(valDirect))
      << "compileForNormSet path disagrees with getVal(normSet) for pdfMode=" << pdfMode;
}

} // namespace

TEST(RooMomentMorphFuncND, BackendConsistencyPdfMode)
{
   checkBackendConsistency(/*pdfMode=*/true);
}

TEST(RooMomentMorphFuncND, BackendConsistencyFuncMode)
{
   checkBackendConsistency(/*pdfMode=*/false);
}

/// The compiled computation graph of the morph pdf references the original
/// morph function and the RooHistPdf templates owned by the workspace. If the
/// workspace is deleted before an NLL object, the destructor of the NLL must
/// not touch the deleted nodes anymore. This happened at the exit of the
/// rf616_morphing.py tutorial, where the Python garbage collector deletes the
/// workspace first, and was observed as a `free(): invalid pointer` abort.
TEST(RooMomentMorphFuncND, WorkspaceDeletedBeforeNLL)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLevel{RooFit::FATAL};

   std::unique_ptr<RooWorkspace> ws = makeMorphWorkspace(/*pdfMode=*/true);
   RooAbsPdf &morphPdf = *ws->pdf("morph");
   std::unique_ptr<RooDataSet> data{ws->pdf("hp0")->generate(*ws->var("x"), 100)};
   std::unique_ptr<RooAbsReal> nll{morphPdf.createNLL(*data)};
   EXPECT_TRUE(std::isfinite(nll->getVal()));

   ws.reset();
   nll.reset();
}

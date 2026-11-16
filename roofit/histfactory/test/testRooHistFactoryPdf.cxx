// Tests for RooHistFactoryPdf and RooHistFactorySample
// Authors: Jonas Rembser, CERN 2026

#include <RooStats/HistFactory/HistoToWorkspaceFactoryFast.h>
#include <RooStats/HistFactory/Measurement.h>
#include <RooStats/HistFactory/RooHistFactoryPdf.h>
#include <RooStats/HistFactory/RooHistFactorySample.h>

#include <RooDataHist.h>
#include <RooFit/ModelConfig.h>
#include <RooHelpers.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TRandom3.h>

#include <gtest/gtest.h>

#include "../../roofitcore/test/gtest_wrapper.h"

#include <cmath>
#include <memory>

using namespace RooStats::HistFactory;

namespace {

TH1 *makeHist(TRandom3 &rng, const char *name, int dim, double scale)
{
   double xbins[] = {0., 0.5, 1.7, 2.0};
   TH1 *h = dim == 1 ? static_cast<TH1 *>(new TH1D(name, name, 3, xbins))
                     : static_cast<TH1 *>(new TH2D(name, name, 3, xbins, 2, 0, 1));
   h->SetDirectory(nullptr);
   for (int i = 1; i <= h->GetNbinsX(); ++i) {
      for (int j = 1; j <= h->GetNbinsY(); ++j) {
         const int bin = h->GetBin(i, j);
         const double v = scale * rng.Uniform(5, 50);
         h->SetBinContent(bin, v);
         h->SetBinError(bin, std::sqrt(v) * rng.Uniform(0.1, 0.5));
      }
   }
   return h;
}

TH1 *varied(TRandom3 &rng, TH1 const *h, const char *name, double lo, double hi)
{
   auto *out = static_cast<TH1 *>(h->Clone(name));
   out->SetDirectory(nullptr);
   for (int b = 0; b < out->GetNcells(); ++b) {
      out->SetBinContent(b, h->GetBinContent(b) * rng.Uniform(lo, hi));
   }
   return out;
}

/// A measurement with all types of modifiers, and with non-uniform bins.
std::unique_ptr<Measurement> makeMeasurement(int dim)
{
   TRandom3 rng{1337};
   auto meas = std::make_unique<Measurement>("meas", "meas");
   meas->SetPOI("mu");
   meas->SetLumi(1.0);
   meas->SetLumiRelErr(0.05);
   meas->AddLogNormSyst("lognorm", 0.1);

   Channel chan("ch");
   TH1 *data = makeHist(rng, "data", dim, 3.0);
   for (int b = 0; b < data->GetNcells(); ++b) {
      data->SetBinContent(b, std::round(data->GetBinContent(b)));
   }
   chan.SetData(data);
   chan.SetStatErrorConfig(0.01, "Poisson");

   Sample sig("signal");
   TH1 *hsig = makeHist(rng, "sig", dim, 1.0);
   sig.SetHisto(hsig);
   sig.AddNormFactor("mu", 1, 0, 5);
   sig.AddOverallSys("syst1", 0.97, 1.03);
   sig.AddOverallSys("lognorm", 0.9, 1.1);
   for (const char *name : {"shape1", "shape2"}) {
      HistoSys hs;
      hs.SetName(name);
      hs.SetHistoLow(varied(rng, hsig, (std::string{name} + "Lo").c_str(), 0.6, 1.0));
      hs.SetHistoHigh(varied(rng, hsig, (std::string{name} + "Hi").c_str(), 1.0, 1.4));
      sig.AddHistoSys(hs);
   }
   sig.ActivateStatError();
   chan.AddSample(sig);

   Sample bkg("background");
   TH1 *hbkg = makeHist(rng, "bkg", dim, 2.0);
   bkg.SetHisto(hbkg);
   bkg.AddOverallSys("syst2", 0.9, 1.2);
   bkg.ActivateStatError();
   ShapeSys ss;
   ss.SetName("bkgShape");
   TH1 *relErr = varied(rng, hbkg, "relErr", 0.0, 0.0);
   for (int b = 0; b < relErr->GetNcells(); ++b) {
      relErr->SetBinContent(b, rng.Uniform(0.02, 0.2));
   }
   ss.SetErrorHist(relErr);
   ss.SetConstraintType(Constraint::Poisson);
   bkg.AddShapeSys(ss);
   chan.AddSample(bkg);

   Sample free("free");
   free.SetHisto(makeHist(rng, "free", dim, 0.5));
   free.SetNormalizeByTheory(false);
   free.AddShapeFactor("freeShape", 1.0, 0.0, 5.0);
   chan.AddSample(free);

   meas->AddChannel(chan);
   return meas;
}

std::unique_ptr<RooWorkspace> makeWorkspace(Measurement &meas, bool useHistFactoryPdf)
{
   HistoToWorkspaceFactoryFast::Configuration cfg;
   cfg.useHistFactoryPdf = useHistFactoryPdf;
   cfg.createPerRegionWorkspaces = false;
   return std::unique_ptr<RooWorkspace>{HistoToWorkspaceFactoryFast::MakeCombinedModel(meas, cfg)};
}

/// Change the parameters in both workspaces to the same values, close to the current ones. The constrained parameters
/// are changed by at most their uncertainty, so that the constraint terms don't underflow to zero.
void setParameters(TRandom3 &rng, RooWorkspace &ws1, RooWorkspace &ws2)
{
   for (RooAbsArg *arg : ws1.allVars()) {
      auto *var1 = static_cast<RooRealVar *>(arg);
      auto *var2 = ws2.var(var1->GetName());
      if (var1->isConstant() || !var2) {
         continue;
      }
      const double shift = var1->getError() > 0. ? var1->getError() : 0.3;
      const double val = std::clamp(var1->getVal() + rng.Uniform(-shift, shift), var1->getMin(), var1->getMax());
      var1->setVal(val);
      var2->setVal(val);
   }
}

} // namespace

class RooHistFactoryPdfTest : public testing::TestWithParam<int> {};

/// The new model structure gives the same values and likelihoods as the one before ROOT 6.44.
TEST_P(RooHistFactoryPdfTest, SameAsLegacyStructure)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);
   const int dim = GetParam();
   auto meas = makeMeasurement(dim);
   auto wsOld = makeWorkspace(*meas, false);
   auto wsNew = makeWorkspace(*meas, true);

   auto *pdfOld = wsOld->pdf("ch_model");
   auto *pdfNew = dynamic_cast<RooHistFactoryPdf *>(wsNew->pdf("ch_model"));
   ASSERT_NE(pdfNew, nullptr);
   EXPECT_EQ(pdfNew->numSamples(), 3u);
   auto *mcOld = static_cast<RooStats::ModelConfig *>(wsOld->obj("ModelConfig"));
   auto *mcNew = static_cast<RooStats::ModelConfig *>(wsNew->obj("ModelConfig"));
   EXPECT_EQ(mcOld->GetNuisanceParameters()->size(), mcNew->GetNuisanceParameters()->size());
   EXPECT_EQ(mcOld->GetGlobalObservables()->size(), mcNew->GetGlobalObservables()->size());

   RooArgSet obsOld;
   RooArgSet obsNew;
   pdfOld->getObservables(mcOld->GetObservables(), obsOld);
   pdfNew->getObservables(mcNew->GetObservables(), obsNew);

   TRandom3 rng{42};
   for (int iPoint = 0; iPoint < 4; ++iPoint) {
      if (iPoint > 0) {
         setParameters(rng, *wsOld, *wsNew);
      }
      RooDataHist bins{"bins", "bins", obsOld};
      for (int i = 0; i < bins.numEntries(); ++i) {
         obsOld.assign(*bins.get(i));
         obsNew.assign(*bins.get(i));
         // The same operations in the same order give the same values.
         EXPECT_DOUBLE_EQ(pdfNew->getVal(), pdfOld->getVal()) << "bin " << i;
         EXPECT_NEAR(pdfNew->getVal(obsNew), pdfOld->getVal(obsOld), 1e-12 * pdfOld->getVal(obsOld)) << "bin " << i;
      }
      EXPECT_NEAR(pdfNew->expectedEvents(&obsNew), pdfOld->expectedEvents(&obsOld),
                  1e-12 * pdfOld->expectedEvents(&obsOld));

      for (bool binned : {true, false}) {
         pdfOld->setAttribute("BinnedLikelihood", binned);
         pdfNew->setAttribute("BinnedLikelihood", binned);
         using RooFit::GlobalObservables;
         std::unique_ptr<RooAbsReal> nllOld{
            mcOld->GetPdf()->createNLL(*wsOld->data("obsData"), GlobalObservables(*mcOld->GetGlobalObservables()))};
         std::unique_ptr<RooAbsReal> nllNew{
            mcNew->GetPdf()->createNLL(*wsNew->data("obsData"), GlobalObservables(*mcNew->GetGlobalObservables()))};
         ASSERT_TRUE(std::isfinite(nllOld->getVal())) << "point " << iPoint << ", binned = " << binned;
         EXPECT_NEAR(nllNew->getVal(), nllOld->getVal(), 1e-12 * std::abs(nllOld->getVal()))
            << "point " << iPoint << ", binned = " << binned;
      }
   }
}

#ifdef TEST_CODEGEN_AD
/// The likelihood with the code generation backend is the same as with the cpu backend.
TEST_P(RooHistFactoryPdfTest, Codegen)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);
   auto meas = makeMeasurement(GetParam());
   auto ws = makeWorkspace(*meas, true);
   auto *mc = static_cast<RooStats::ModelConfig *>(ws->obj("ModelConfig"));

   TRandom3 rng{7};
   for (bool binned : {true, false}) {
      ws->pdf("ch_model")->setAttribute("BinnedLikelihood", binned);
      using RooFit::GlobalObservables;
      std::unique_ptr<RooAbsReal> nllCpu{mc->GetPdf()->createNLL(
         *ws->data("obsData"), GlobalObservables(*mc->GetGlobalObservables()), RooFit::EvalBackend::Cpu())};
      std::unique_ptr<RooAbsReal> nllCodegen{mc->GetPdf()->createNLL(
         *ws->data("obsData"), GlobalObservables(*mc->GetGlobalObservables()), RooFit::EvalBackend::Codegen())};
      for (int iPoint = 0; iPoint < 3; ++iPoint) {
         if (iPoint > 0) {
            setParameters(rng, *ws, *ws);
         }
         ASSERT_TRUE(std::isfinite(nllCpu->getVal())) << "point " << iPoint << ", binned = " << binned;
         EXPECT_NEAR(nllCodegen->getVal(), nllCpu->getVal(), 1e-10 * std::abs(nllCpu->getVal()))
            << "point " << iPoint << ", binned = " << binned;
      }
      ws->loadSnapshot("NominalParamValues");
   }
}
#endif

/// The samples and their data survive the writing to a file, and the data is shared between clones.
TEST_P(RooHistFactoryPdfTest, PersistenceAndClones)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);
   auto meas = makeMeasurement(GetParam());
   auto ws = makeWorkspace(*meas, true);

   const std::string fileName = "testRooHistFactoryPdf_" + std::to_string(GetParam()) + ".root";
   {
      TFile file(fileName.c_str(), "RECREATE");
      file.WriteObject(ws.get(), "ws");
   }
   std::unique_ptr<TFile> file{TFile::Open(fileName.c_str())};
   std::unique_ptr<RooWorkspace> wsRead{file->Get<RooWorkspace>("ws")};
   ASSERT_NE(wsRead, nullptr);

   auto *pdf = dynamic_cast<RooHistFactoryPdf *>(ws->pdf("ch_model"));
   auto *pdfRead = dynamic_cast<RooHistFactoryPdf *>(wsRead->pdf("ch_model"));
   ASSERT_NE(pdfRead, nullptr);
   ASSERT_EQ(pdfRead->numSamples(), pdf->numSamples());

   TRandom3 rng{3};
   setParameters(rng, *ws, *wsRead);
   for (std::size_t i = 0; i < pdf->numSamples(); ++i) {
      RooHistFactorySample const &sample = pdf->sample(i);
      RooHistFactorySample const &sampleRead = pdfRead->sample(i);
      EXPECT_EQ(sampleRead.sampleName(), sample.sampleName());
      EXPECT_EQ(sampleRead.data().contents, sample.data().contents);
      EXPECT_EQ(sampleRead.data().histoSysHi, sample.data().histoSysHi);
      EXPECT_EQ(sampleRead.data().edges, sample.data().edges);
      std::vector<double> values(sample.numBins());
      std::vector<double> valuesRead(sample.numBins());
      sample.binValues(values, false);
      sampleRead.binValues(valuesRead, false);
      EXPECT_EQ(valuesRead, values) << sample.sampleName();
   }

   // A clone shares the data, until its interpolation codes are changed.
   auto const &signal = *pdf->sample("signal");
   std::unique_ptr<RooHistFactorySample> clone{static_cast<RooHistFactorySample *>(signal.clone())};
   EXPECT_EQ(&clone->data(), &signal.data());
   clone->setHistoSysInterpCode(*ws->var("alpha_shape1"), 0);
   EXPECT_NE(&clone->data(), &signal.data());
   EXPECT_EQ(clone->data().histoSysCodes, (std::vector<int>{0, 4}));
   EXPECT_EQ(signal.data().histoSysCodes, (std::vector<int>{4, 4}));
}

/// Integrals over ranges that don't align with the bins take the fraction of the bins in the range.
TEST(RooHistFactoryPdf, RangedIntegral)
{
   RooRealVar x{"x", "x", 0, 4};
   x.setBins(4);
   RooRealVar mu{"mu", "mu", 2.0};
   RooHistFactorySample::Spec spec;
   spec.sampleName = "sample";
   spec.contents = {1., 2., 3., 4.};
   spec.normFactors.push_back({"mu", &mu});
   RooHistFactorySample sample{"sample", "sample", RooArgList{x}, spec};
   RooHistFactoryPdf pdf{"pdf", "pdf", RooArgList{sample}};

   x.setRange("r", 0.5, 2.25);
   std::unique_ptr<RooAbsReal> integral{sample.createIntegral(x, "r")};
   EXPECT_DOUBLE_EQ(integral->getVal(), mu.getVal() * (0.5 * 1. + 2. + 0.25 * 3.));
   std::unique_ptr<RooAbsReal> full{sample.createIntegral(x)};
   EXPECT_DOUBLE_EQ(full->getVal(), mu.getVal() * 10.);
   RooArgSet xSet{x};
   EXPECT_DOUBLE_EQ(pdf.expectedEvents(&xSet), mu.getVal() * 10.);
   x.setVal(2.5);
   RooArgSet normSet{x};
   EXPECT_DOUBLE_EQ(pdf.getVal(normSet), 3. / 10.);
}

INSTANTIATE_TEST_SUITE_P(RooHistFactoryPdf, RooHistFactoryPdfTest, testing::Values(1, 2),
                         [](testing::TestParamInfo<int> const &paramInfo) {
                            return std::to_string(paramInfo.param) + "D";
                         });

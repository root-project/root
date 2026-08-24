// Tests for RooMCStudy
// Authors: Jonas Rembser, CERN 2026

#include <RooAbsPdf.h>
#include <RooArgSet.h>
#include <RooDataSet.h>
#include <RooHelpers.h>
#include <RooMCStudy.h>
#include <RooRandom.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include "gtest_wrapper.h"

#include <algorithm>
#include <stdexcept>
#include <string>
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

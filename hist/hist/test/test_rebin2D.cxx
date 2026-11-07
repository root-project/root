// Tests for rebinning TH2 and TProfile2D with variable bin edges (ROOT-5224).
// The rebinned histograms are compared against reference histograms created
// directly with the target binning and filled with the same pseudo-data.

#include "gtest/gtest.h"

#include "ROOT/TestSupport.hxx"

#include "TH2D.h"
#include "TProfile2D.h"
#include "TRandom3.h"

#include <cmath>
#include <memory>
#include <string>

namespace {

void fillOne(TH2D &h, double x, double y, double, double w)
{
   h.Fill(x, y, w);
}

void fillOne(TProfile2D &p, double x, double y, double z, double w)
{
   p.Fill(x, y, z, w);
}

// Fill both histograms with the same weighted pseudo-data, including entries
// that end up in the under- and overflow bins.
template <class Hist>
void fillSame(Hist &a, Hist &b)
{
   TRandom3 rng(42);
   for (int i = 0; i < 10000; ++i) {
      const double x = rng.Uniform(-10., 110.);
      const double y = rng.Uniform(-10., 110.);
      const double z = rng.Gaus(5., 1.);
      const double w = rng.Uniform(0.1, 2.);
      fillOne(a, x, y, z, w);
      fillOne(b, x, y, z, w);
   }
}

void expectSameBinsAndStats(const TH2 &h, const TH2 &ref, const std::string &ctx)
{
   ASSERT_EQ(h.GetNbinsX(), ref.GetNbinsX()) << ctx;
   ASSERT_EQ(h.GetNbinsY(), ref.GetNbinsY()) << ctx;
   for (Int_t j = 0; j <= ref.GetNbinsY() + 1; ++j) {
      for (Int_t i = 0; i <= ref.GetNbinsX() + 1; ++i) {
         const std::string binCtx = ctx + " bin (" + std::to_string(i) + "," + std::to_string(j) + ")";
         const Int_t bin = ref.GetBin(i, j);
         const double refContent = ref.GetBinContent(bin);
         const double refError = ref.GetBinError(bin);
         EXPECT_NEAR(h.GetBinContent(bin), refContent, 1e-6 * std::max(1., std::abs(refContent))) << binCtx;
         EXPECT_NEAR(h.GetBinError(bin), refError, 1e-6 * std::max(1., refError)) << binCtx;
      }
   }
   EXPECT_NEAR(h.GetEntries(), ref.GetEntries(), 1e-6) << ctx;
   EXPECT_NEAR(h.GetMean(1), ref.GetMean(1), 1e-9) << ctx;
   EXPECT_NEAR(h.GetMean(2), ref.GetMean(2), 1e-9) << ctx;
   EXPECT_NEAR(h.GetStdDev(1), ref.GetStdDev(1), 1e-9) << ctx;
}

const double xEdges[5] = {0., 10., 40., 70., 100.};
const double yEdges[5] = {0., 30., 50., 90., 100.};

} // namespace

// Rebin a uniform TH2 into variable bins on both axes.
TEST(Rebin2DVariable, TH2BothAxes)
{
   TH2D fine("fine", "fine", 100, 0., 100., 100, 0., 100.);
   fine.Sumw2();
   TH2D ref("ref", "ref", 4, xEdges, 4, yEdges);
   ref.Sumw2();
   fillSame(fine, ref);

   std::unique_ptr<TH2> hnew{fine.Rebin2D(4, 4, "hnew", xEdges, yEdges)};
   ASSERT_NE(hnew, nullptr);
   ASSERT_NE(hnew.get(), &fine);
   expectSameBinsAndStats(*hnew, ref, "TH2BothAxes");
}

// Variable bins on the x-axis only: the y-axis is grouped by a constant factor.
TEST(Rebin2DVariable, TH2SingleAxis)
{
   TH2D fine("fine", "fine", 100, 0., 100., 100, 0., 100.);
   fine.Sumw2();
   TH2D ref("ref", "ref", 4, xEdges, 50, 0., 100.);
   ref.Sumw2();
   fillSame(fine, ref);

   std::unique_ptr<TH2> hnew{fine.Rebin2D(4, 2, "hnew", xEdges, nullptr)};
   ASSERT_NE(hnew, nullptr);
   expectSameBinsAndStats(*hnew, ref, "TH2SingleAxis");

   // TH2::Rebin with bin edges must forward to Rebin2D with the TH1
   // conventions, leaving the y-axis untouched
   std::unique_ptr<TH2> viaRebin{fine.Rebin(4, "viaRebin", xEdges)};
   std::unique_ptr<TH2> viaRebin2D{fine.Rebin2D(4, 1, "viaRebin2D", xEdges, nullptr)};
   ASSERT_NE(viaRebin, nullptr);
   ASSERT_NE(viaRebin2D, nullptr);
   expectSameBinsAndStats(*viaRebin, *viaRebin2D, "TH2RebinForwards");
}

// Constant-group rebinning of a TH2 that has variable-width axes: the new
// edges are synthesized from the old axis.
TEST(Rebin2DVariable, TH2VariableSourceConstantGroups)
{
   TH2D fine("fine", "fine", 4, xEdges, 4, yEdges);
   fine.Sumw2();
   const double xCoarse[3] = {0., 40., 100.};
   const double yCoarse[3] = {0., 50., 100.};
   TH2D ref("ref", "ref", 2, xCoarse, 2, yCoarse);
   ref.Sumw2();
   fillSame(fine, ref);

   std::unique_ptr<TH2> hnew{fine.Rebin2D(2, 2, "hnew")};
   ASSERT_NE(hnew, nullptr);
   expectSameBinsAndStats(*hnew, ref, "TH2VariableSourceConstantGroups");
}

// Rebin a uniform TProfile2D into variable bins on both axes.
TEST(Rebin2DVariable, Profile2DBothAxes)
{
   TProfile2D fine("fine", "fine", 100, 0., 100., 100, 0., 100.);
   fine.Sumw2();
   TProfile2D ref("ref", "ref", 4, xEdges, 4, yEdges);
   ref.Sumw2();
   fillSame(fine, ref);

   std::unique_ptr<TProfile2D> pnew{fine.Rebin2D(4, 4, "pnew", xEdges, yEdges)};
   ASSERT_NE(pnew, nullptr);
   expectSameBinsAndStats(*pnew, ref, "Profile2DBothAxes");
   for (Int_t j = 0; j <= ref.GetNbinsY() + 1; ++j) {
      for (Int_t i = 0; i <= ref.GetNbinsX() + 1; ++i) {
         const Int_t bin = ref.GetBin(i, j);
         EXPECT_NEAR(pnew->GetBinEntries(bin), ref.GetBinEntries(bin), 1e-6)
            << "Profile2DBothAxes entries bin (" << i << "," << j << ")";
      }
   }
}

// Constant-group rebinning of a TProfile2D that has variable-width axes
// (regression test: this used to pass null bin edges to SetBins).
TEST(Rebin2DVariable, Profile2DVariableSourceConstantGroups)
{
   TProfile2D fine("fine", "fine", 4, xEdges, 4, yEdges);
   const double xCoarse[3] = {0., 40., 100.};
   const double yCoarse[3] = {0., 50., 100.};
   TProfile2D ref("ref", "ref", 2, xCoarse, 2, yCoarse);
   fillSame(fine, ref);

   std::unique_ptr<TProfile2D> pnew{fine.Rebin2D(2, 2, "pnew")};
   ASSERT_NE(pnew, nullptr);
   expectSameBinsAndStats(*pnew, ref, "Profile2DVariableSourceConstantGroups");
}

// When the group count does not divide the number of bins, the top bins move
// to the overflow and the statistics must be recomputed from the bin contents.
TEST(Rebin2DVariable, TH2NonDivisorGroupStats)
{
   TH2D h("h", "h", 10, 0., 10., 10, 0., 10.);
   TRandom3 rng(43);
   for (int i = 0; i < 10000; ++i)
      h.Fill(rng.Uniform(0., 10.), rng.Uniform(0., 10.));

   {
      ROOT::TestSupport::CheckDiagsRAII checkDiag;
      checkDiag.requiredDiag(kWarning, "TH2D::Rebin2D", "is not an exact divider", false);
      ASSERT_EQ(h.Rebin2D(3, 3), &h); // in-place
   }
   ASSERT_EQ(h.GetNbinsX(), 3);
   ASSERT_DOUBLE_EQ(h.GetXaxis()->GetXmax(), 9.);

   // reference statistics recomputed from the rebinned contents
   double sw = 0., swx = 0., swx2 = 0.;
   for (Int_t j = 1; j <= h.GetNbinsY(); ++j) {
      for (Int_t i = 1; i <= h.GetNbinsX(); ++i) {
         const double c = h.GetBinContent(i, j);
         const double x = h.GetXaxis()->GetBinCenter(i);
         sw += c;
         swx += c * x;
         swx2 += c * x * x;
      }
   }
   EXPECT_NEAR(h.GetMean(1), swx / sw, 1e-9);
   EXPECT_NEAR(h.GetStdDev(1), std::sqrt(swx2 / sw - (swx / sw) * (swx / sw)), 1e-9);
}

// Variable-bin rebinning requires a new name, and warns when a new bin edge
// does not line up with an old bin edge.
TEST(Rebin2DVariable, Diagnostics)
{
   TH2D h("h", "h", 100, 0., 100., 100, 0., 100.);
   {
      // an empty name must be rejected like a null one, otherwise Clone("")
      // would create a second histogram registered under the original name
      ROOT::TestSupport::CheckDiagsRAII checkDiag(kError, "TH2D::Rebin2D", "newname must be given", false);
      EXPECT_EQ(h.Rebin2D(4, 4, nullptr, xEdges, yEdges), nullptr);
      EXPECT_EQ(h.Rebin2D(4, 4, "", xEdges, yEdges), nullptr);
   }
   {
      ROOT::TestSupport::CheckDiagsRAII checkDiag(kWarning, "TH2D::Rebin2D", "does not match any bin edges", false);
      const double misaligned[3] = {0., 10.5, 100.};
      std::unique_ptr<TH2> hnew{h.Rebin2D(2, 2, "hnew", misaligned, nullptr)};
      EXPECT_NE(hnew, nullptr);
   }
   {
      // a new top edge that splits an old bin between range and overflow must
      // warn as well
      ROOT::TestSupport::CheckDiagsRAII checkDiag(kWarning, "TH2D::Rebin2D", "does not match any bin edges", false);
      const double splitTop[3] = {0., 50., 99.5};
      std::unique_ptr<TH2> hnew{h.Rebin2D(2, 2, "hnew", splitTop, nullptr)};
      EXPECT_NE(hnew, nullptr);
   }
}

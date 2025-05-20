// Tests for the RooExtendedBinding
// Authors: Jonas Rembser, CERN  05/2025

#include <RooRealVar.h>
#include <RooUniform.h>
#include <RooExtendedBinding.h>
#include <RooRealSumPdf.h>

#include <gtest/gtest.h>

/// Check that the normalization set is propagated correctly to the call of
/// RooAbsPdf::expectedEvents().
TEST(RooExtendedBinding, Normalization)
{
   RooRealVar x{"x", "x", -10, 10};

   const double normRef = x.getMax() - x.getMin();

   RooUniform unif{"unif", "unif", x};

   // Use a RooRealSumPdf, because the expectedEvents() implementation returns
   // the normalization integral depending on the normalization set. If
   // expectedEvents() would not depend on the normSet, this test would be
   // ineffective.
   RooRealSumPdf pdf{"pdf", "pdf", {unif}, {}};

   RooArgSet normSet{x};

   std::cout << pdf.getVal() << std::endl;

   // without normalization set
   RooExtendedBinding ext1{"ext1", "ext1", pdf};
   // with normalization set
   RooExtendedBinding ext2{"ext2", "ext2", pdf, normSet};

   EXPECT_DOUBLE_EQ(ext1.getVal(), 1.0);     // unnormalized
   EXPECT_DOUBLE_EQ(ext2.getVal(), normRef); // normalized
}

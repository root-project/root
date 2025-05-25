// Tests for the RooStepFunc
// Authors: Jonas Rembser, CERN 2025

#include <RooStepFunction.h>
#include <RooWrapperPdf.h>
#include <RooRealVar.h>
#include <RooRealIntegral.h>

#include <gtest/gtest.h>

// Test that RooStepFunction can be wrapped into a pdf object, where it is
// normalized without any analytical integrals.
TEST(RooStepFunc, InPdfWrapper)
{
   RooRealVar x{"x", "x", 5.0, 0.0, 10.0};
   RooRealVar a{"a", "a", 3.0, 0.0, 10.0};
   RooRealVar b{"b", "b", 7.0, 0.0, 10.0};

   RooStepFunction stepFunc{"step", "", x, RooArgList{1.0}, {a, b}};
   RooWrapperPdf stepPdf{"pdf", "", stepFunc};

   RooArgSet normSet{x};

   RooArgSet intSet{x};
   RooArgSet numSet{};

   RooRealIntegral integ{"integ", "", stepPdf, x};

   // Make sure that the normalization integral for the pdf is analytical
   EXPECT_EQ(integ.anaIntVars().size(), 1);
   EXPECT_DOUBLE_EQ(integ.getVal(), 4.0);

   // Check that the pdf in correctly normalized
   EXPECT_DOUBLE_EQ(stepPdf.getVal(normSet), 0.25);
}

// Tests for the RooHistPdf
// Authors: Jonas Rembser, CERN 03/2023

#include <RooDataHist.h>
#include <RooHistPdf.h>
#include <RooLinearVar.h>
#include <RooRealIntegral.h>
#include <RooRealVar.h>

#include <gtest/gtest.h>

// Verify that RooFit correctly uses analytic integration when having a
// RooLinearVar as the observable of a RooHistPdf.
TEST(RooHistPdf, AnalyticIntWithRooLinearVar)
{
    RooRealVar x{"x", "x", 0, -10, 10};
    x.setBins(10);

    RooDataHist dataHist("dataHist", "dataHist", x);
    for(int i = 0; i < x.numBins(); ++i) {
       dataHist.set(i, 10.0, 0.0);
    }

    RooRealVar shift{"shift", "shift", 2.0, -10, 10};
    RooRealVar slope{"slope", "slope", 1.0};
    RooLinearVar xShifted{"x_shifted", "x_shifted", x, slope, shift};

    RooHistPdf pdf{"pdf", "pdf", xShifted, x, dataHist};

    RooRealIntegral integ{"integ", "integ", pdf, x};

    EXPECT_DOUBLE_EQ(integ.getVal(), 90.);
    EXPECT_EQ(integ.anaIntVars().size(), 1);
}

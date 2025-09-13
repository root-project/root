/// \file
/// \ingroup RooFit_Tests
/// \brief Unit test for the new RooUniform constructor with fittable bounds.

#include "RooUniform.h"
#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooAddPdf.h"
#include "RooDataSet.h"
#include "RooFitResult.h"
#include "RooConstVar.h"
#include "gtest/gtest.h"
#include <memory>

// Use the RooFit namespace for clarity in the test
using namespace RooFit;

////////////////////////////////////////////////////////////////////////////////
/// This test validates the functionality of the new RooUniform constructor that
/// accepts fittable lower and upper bounds.
TEST(RooUniform, FitWithBoundedConstructor)
{
    // --- Define the "Truth" Model ---
    RooRealVar x("x", "x", 0, 20);
    
    RooConstVar mean_truth("mean_truth", "mean_truth", 10.0);
    RooConstVar width_truth("width_truth", "width_truth", 0.5);
    RooGaussian signal_truth("signal_truth", "signal_truth", x, mean_truth, width_truth);
    
    RooRealVar bkg_low_truth("bkg_low_truth", "bkg_low_truth", 4.0);
    RooRealVar bkg_high_truth("bkg_high_truth", "bkg_high_truth", 16.0);
    RooUniform background_truth("background_truth", "background_truth", x, bkg_low_truth, bkg_high_truth);
    
    RooRealVar n_sig_truth("n_sig_truth", "n_sig_truth", 300);
    RooRealVar n_bkg_truth("n_bkg_truth", "n_bkg_truth", 700);
    RooAddPdf model_truth("model_truth", "model_truth", {signal_truth, background_truth}, {n_sig_truth, n_bkg_truth});

    // --- Generate the Data ---
    std::unique_ptr<RooDataSet> data{model_truth.generate(x, 1000)};

    // --- Define the "Fit" Model ---
    RooRealVar mean_fit("mean_fit", "mean_fit", 10.0, 8.0, 12.0);
    RooGaussian signal_fit("signal_fit", "signal_fit", x, mean_fit, width_truth); // Can reuse width_truth
    
    RooRealVar bkg_low_fit("bkg_low_fit", "bkg_low_fit", 3.0, 0.0, 8.0);
    RooRealVar bkg_high_fit("bkg_high_fit", "bkg_high_fit", 17.0, 12.0, 20.0);
    RooUniform background_fit("background_fit", "background_fit", x, bkg_low_fit, bkg_high_fit);
    
    RooRealVar n_sig_fit("n_sig_fit", "n_sig_fit", 300, 0, 10000);
    RooRealVar n_bkg_fit("n_bkg_fit", "n_bkg_fit", 700, 0, 10000);
    RooAddPdf model_fit("model_fit", "model_fit", {signal_fit, background_fit}, {n_sig_fit, n_bkg_fit});

    // --- Perform the Fit ---
    std::unique_ptr<RooFitResult> fitResult{model_fit.fitTo(*data, Save(), PrintLevel(-1))};
    
    // --- Check the Results (Assertions) ---
    ASSERT_TRUE(fitResult) << "Fit result pointer should not be null";
    EXPECT_EQ(fitResult->status(), 0) << "Fit should converge successfully";

    EXPECT_NEAR(bkg_low_fit.getVal(),  4.0, 0.5);
    EXPECT_NEAR(bkg_high_fit.getVal(), 16.0, 0.5);
}
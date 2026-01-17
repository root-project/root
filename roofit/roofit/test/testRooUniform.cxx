/// \file
/// \ingroup RooFit_Tests
/// \brief Unit test for the new N-dimensional RooUniform with fittable bounds.

#include "RooUniform.h"
#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooProdPdf.h"
#include "RooAddPdf.h"
#include "RooDataSet.h"
#include "RooDataHist.h"
#include "RooFitResult.h"
#include "RooConstVar.h"
#include "RooArgList.h"
#include "RooArgSet.h"
#include "RooRandom.h"
#include "gtest/gtest.h"
#include <memory>

// Use RooFit symbols for options
using namespace RooFit;

////////////////////////////////////////////////////////////////////////////////
/// \brief Validates the RooUniform in a realistic signal+background fit.
///
/// This test validates the new N-dimensional RooUniform with fittable bounds by
/// using it as a background component in a complex model. It creates a model
/// of a Gaussian signal peak on a 2D uniform background, generates toy data,
/// and performs a BINNED likelihood fit.
TEST(RooUniform, FitAsBackgroundComponent)
{
   // deterministic seed for CI
   RooRandom::randomGenerator()->SetSeed(12345);

   // Define Observables and True Model Parameters
   RooRealVar x("x", "x", 0.0, 20.0);
   RooRealVar y("y", "y", 0.0, 20.0);

   // Set binning for stable binned fits
   x.setBins(40);
   y.setBins(40);

   RooArgSet observables(x, y);

   // Truth parameters for the signal (a 2D Gaussian)
   RooConstVar mean_x_truth("mean_x_truth", "", 10.0);
   RooConstVar mean_y_truth("mean_y_truth", "", 10.0);
   RooConstVar sigma_truth("sigma_truth", "", 1.0);
   RooGaussian gauss_x_truth("gauss_x_truth", "", x, mean_x_truth, sigma_truth);
   RooGaussian gauss_y_truth("gauss_y_truth", "", y, mean_y_truth, sigma_truth);
   RooArgList signal_components(gauss_x_truth, gauss_y_truth);
   RooProdPdf signal_truth("signal_truth", "2D Gaussian Signal", signal_components);

   // Truth parameters for the background (2D RooUniform)
   RooRealVar x_low_truth("x_low_truth", "", 5.0);
   RooRealVar x_high_truth("x_high_truth", "", 15.0);
   RooRealVar y_low_truth("y_low_truth", "", 4.0);
   RooRealVar y_high_truth("y_high_truth", "", 16.0);

   RooArgSet bkg_low(x_low_truth, y_low_truth);
   RooArgSet bkg_high(x_high_truth, y_high_truth);

   RooUniform background_truth("background_truth", "2D Uniform Background",
                               observables, bkg_low, bkg_high);

   // Yields
   RooRealVar n_sig_truth("n_sig_truth", "", 500.0);
   RooRealVar n_bkg_truth("n_bkg_truth", "", 2000.0);
   RooArgList pdfs_truth(signal_truth, background_truth);
   RooArgList yields_truth(n_sig_truth, n_bkg_truth);
   RooAddPdf model_truth("model_truth", "2D Truth Model", pdfs_truth, yields_truth);

   // Generate and Bin Data (binned likelihood) 
   // Generate unbinned dataset; model.generate returns a RooDataSet*
   std::unique_ptr<RooDataSet> unbinnedData(model_truth.generate(observables, 2500));
   ASSERT_TRUE(unbinnedData) << "Data generation failed";

   // Create binned data (RooDataHist) from unbinnedData
   RooDataHist binnedData("binnedData", "Binned Data", observables, *unbinnedData);

   // Define the Fit Model with Floating Parameters
   RooRealVar mean_x_fit("mean_x_fit", "", 10.1, 8.0, 12.0);
   RooRealVar mean_y_fit("mean_y_fit", "", 9.9, 8.0, 12.0);
   RooGaussian gx_fit("gx_fit", "", x, mean_x_fit, sigma_truth);
   RooGaussian gy_fit("gy_fit", "", y, mean_y_fit, sigma_truth);
   RooArgList signal_fit_comps(gx_fit, gy_fit);
   RooProdPdf signal_fit("signal_fit", "Signal Fit", signal_fit_comps);

   // Fit bounds (floating)
   RooRealVar x_low_fit("x_low_fit", "x_low_fit", 4.5, 0.0, 8.0);
   RooRealVar x_high_fit("x_high_fit", "x_high_fit", 15.5, 12.0, 18.0);
   RooRealVar y_low_fit("y_low_fit", "y_low_fit", 3.5, 0.0, 8.0);
   RooRealVar y_high_fit("y_high_fit", "y_high_fit", 16.5, 12.0, 18.0);

   RooArgSet bkg_low_fit(x_low_fit, y_low_fit);
   RooArgSet bkg_high_fit(x_high_fit, y_high_fit);

   RooUniform background_fit("background_fit", "Background Fit", observables, bkg_low_fit, bkg_high_fit);

   RooRealVar n_sig_fit("n_sig_fit", "Signal Yield", 450.0, 0.0, 1e6);
   RooRealVar n_bkg_fit("n_bkg_fit", "Background Yield", 2050.0, 0.0, 1e6);
   RooArgList pdfs_fit(signal_fit, background_fit);
   RooArgList yields_fit(n_sig_fit, n_bkg_fit);
   RooAddPdf model_fit("model_fit", "2D Fit Model", pdfs_fit, yields_fit);

   // Perform the Binned Fit
   // Use Save() to get RooFitResult; PrintLevel(-1) quiets output
   std::unique_ptr<RooFitResult> fitResult(model_fit.fitTo(binnedData, Save(), PrintLevel(-1)));

   // Check / Assertions 
   ASSERT_TRUE(fitResult) << "Fit result pointer should not be null";
   ASSERT_EQ(fitResult->status(), 0) << "Fit should converge successfully";

   // Fit should recover bounds within tolerance
   EXPECT_NEAR(x_low_fit.getVal(), 5.0, 0.5);
   EXPECT_NEAR(x_high_fit.getVal(), 15.0, 0.5);
   EXPECT_NEAR(y_low_fit.getVal(), 4.0, 0.5);
   EXPECT_NEAR(y_high_fit.getVal(), 16.0, 0.5);
}
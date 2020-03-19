#include "RooAddPdf.h"
#include "RooDataSet.h"
#include "RooGaussian.h"
#include "RooRealVar.h"
#include "RooVoigtian.h"
#include "TFile.h"
#include "TTree.h"
#include "RooFitResult.h"
#include "RooNumIntConfig.h"

#include <gtest/gtest.h>

// Test that a simple fit with Gauss and Voigt yields the same values as with ROOT 6.18.
// The fit requires a numeric integration for the Voigt. To suppress false positives,
// the precision for the numeric integrals was increased by 1000x.
TEST(Stability, ROOT_10615) {
  RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING);

  RooRealVar dt("dt", "#Deltat [ps]", -10, 10);

  RooRealVar bkg_dt_voigt_mu("bkg_dt_voigt_mu", "v_{#mu}", -0.303, -1, 1);
  RooRealVar bkg_dt_voigt_sigma("bkg_dt_voigt_sigma", "v_{#sigma}", 2.323, 0, 10);
  RooRealVar bkg_dt_voigt_width("bkg_dt_voigt_width", "v_{w}", 0.851, 0, 10);
  RooVoigtian bkg_dt_voigt("bkg_dt_voigt", "bkg_dt_voigt", dt, bkg_dt_voigt_mu,
                           bkg_dt_voigt_width, bkg_dt_voigt_sigma);
  RooNumIntConfig* voigtIntegrator = bkg_dt_voigt.specialIntegratorConfig(true);
  voigtIntegrator->setEpsAbs(1.E-10);
  voigtIntegrator->setEpsRel(1.E-10);
  voigtIntegrator->getConfigSection("RooIntegrator1D").setRealValue("maxSteps", 30);

  RooRealVar bkg_dt_gaus_mu_("bkg_dt_gaus_mu", "g_{#mu}", -0.161 - 1, 1);
  RooRealVar bkg_dt_gaus_sigma_("bkg_dt_gaus_sigma", "g_{#sigma}", 1.096, 0, 10);
  RooGaussian bkg_dt_gaus("bkg_dt_gaus", "bkg_dt_gaus", dt, bkg_dt_gaus_mu_, bkg_dt_gaus_sigma_);

  RooRealVar bkg_dt_f("bkg_dt_f", "f_{v/g}", 0.631, 0, 1);

  RooAddPdf bkg_dt_model_("bkg_dt_model", "bkg_dt_model", RooArgList(bkg_dt_voigt, bkg_dt_gaus),
                          RooArgList(bkg_dt_f));

  TFile file("stabilityTests_data_1.root");
  ASSERT_TRUE(file.IsOpen());

  TTree* input_tree = nullptr;
  file.GetObject("h2000", input_tree);
  ASSERT_NE(input_tree, nullptr);
  ASSERT_NE(input_tree->GetEntries(), 0);

  RooDataSet dataset("dataset", "dataset", input_tree, dt);

  auto fitResult = bkg_dt_model_.fitTo(dataset, RooFit::Minimizer("Minuit2"), RooFit::Save(), RooFit::Hesse(false), RooFit::PrintLevel(-1));

  // These are with numeric integrals at default precision:
//  EXPECT_NEAR(bkg_dt_f.getVal(), 0.588041, 1.E-6);
//  EXPECT_NEAR(bkg_dt_f.getError(), 0.0975393, 1.E-7);
//  EXPECT_NEAR(fitResult->minNll(), 2200.69528025409727, 1.E-6);
//  EXPECT_NEAR(fitResult->edm(), 0.000109836916104149446, 1.E-6);

  EXPECT_NEAR(bkg_dt_f.getVal(), 0.5863848853184781, 1.E-9);
  EXPECT_NEAR(bkg_dt_f.getError(), 0.097259271172036643, 1.E-9);
  EXPECT_NEAR(fitResult->minNll(), 2200.69610626874464, 1.E-9);
  EXPECT_NEAR(fitResult->edm(), 0.0002573480023165511, 1.E-11);

}

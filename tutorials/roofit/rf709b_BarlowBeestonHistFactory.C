/// \file
/// \ingroup tutorial_roofit
/// \notebook -js
/// Implementing the Barlow-Beeston method for taking into account the statistical
/// uncertainty of a Monte-Carlo fit template.
///
/// \macro_image
/// \macro_code
/// \macro_output
///
/// New implementation of the original rf709_BarlowBeeston tutorial, using the
/// same kind of RooFit models that are built by HistFactory. Therefore, this
/// tutorial is a good starting point if you want to understand how HistFactory
/// works.
//
/// \date June 2023
/// \author Jonas Rembser, CERN

#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooUniform.h"
#include "RooDataSet.h"
#include "RooDataHist.h"
#include "RooHistFunc.h"
#include "RooRealSumPdf.h"
#include "RooParamHistFunc.h"
#include "RooHistConstraint.h"
#include "RooProdPdf.h"
#include "RooPlot.h"

#include "TCanvas.h"
#include "TPaveText.h"

#include <iostream>
#include <memory>

RooArgSet createConstraints(RooArgList const &gammas, RooArgList const &samples, RooArgSet &globalObservables)
{
   RooArgSet constraints;
   for (std::size_t i = 0; i < gammas.size(); ++i) {
      RooAbsReal *gamma = static_cast<RooAbsReal *>(&gammas[i]);

      auto nomName = "nom_" + std::string(gamma->GetName());
      auto tauName = std::string(gamma->GetName()) + "_tau";
      auto meanName = std::string(gamma->GetName()) + "_mean";
      auto poissonName = std::string(gamma->GetName()) + "_constraint";

      double nomVal = 0.0;
      for (auto *func : static_range_cast<RooHistFunc *>(samples)) {
         nomVal += func->dataHist().weight(i);
      }

      auto nominal = std::make_unique<RooRealVar>(nomName.c_str(), nomName.c_str(), nomVal, 0.0, RooNumber::infinity());
      nominal->setConstant(true);
      globalObservables.add(*nominal);

      auto tau = std::make_unique<RooConstVar>(tauName.c_str(), tauName.c_str(), nominal->getVal());

      auto mean = std::make_unique<RooProduct>(meanName.c_str(), meanName.c_str(), *gamma, *tau);
      mean->addOwnedComponents(std::move(tau));

      auto poisson = std::make_unique<RooPoisson>(poissonName.c_str(), poissonName.c_str(), *nominal, *mean);
      poisson->addOwnedComponents(std::move(nominal));
      poisson->addOwnedComponents(std::move(mean));

      constraints.addOwned(std::move(poisson));
   }
   return constraints;
}

void rf709b_BarlowBeestonHistFactory()
{
   using namespace RooFit;

   RooMsgService::instance().getStream(1).removeTopic(NumIntegration);

   // First, construct a likelihood model with a Gaussian signal on top of a uniform background
   RooRealVar x("x", "x", -20, 20);
   x.setBins(25);

   RooRealVar meanG("meanG", "meanG", 1, -10, 10);
   RooRealVar sigG("sigG", "sigG", 1.5, -10, 10);
   RooGaussian g("g", "Gauss", x, meanG, sigG);
   RooUniform u("u", "Uniform", x);

   // Generate the data to be fitted
   std::unique_ptr<RooDataSet> sigData(g.generate(x, 50));
   std::unique_ptr<RooDataSet> bkgData(u.generate(x, 1000));

   RooDataSet sumData("sumData", "Gauss + Uniform", x);
   sumData.append(*sigData);
   sumData.append(*bkgData);

   // Make histogram templates for signal and background.
   // Let's take a signal distribution with low statistics and a more accurate
   // background distribution.
   // Normally, these come from Monte Carlo simulations, but we will just generate them.
   std::unique_ptr<RooDataHist> dh_sig(g.generateBinned(x, 50));
   std::unique_ptr<RooDataHist> dh_bkg(u.generateBinned(x, 10000));

   // ***** Case 0 - 'Rigid templates' *****

   // Construct histogram shapes for signal and background
   RooHistFunc p_h_sig("p_h_sig", "p_h_sig", x, *dh_sig);
   RooHistFunc p_h_bkg("p_h_bkg", "p_h_bkg", x, *dh_bkg);

   // Construct scale factors for adding the two distributions
   RooRealVar Asig0("Asig", "Asig", 1, 0.01, 5000);
   RooRealVar Abkg0("Abkg", "Abkg", 1, 0.01, 5000);

   // Construct the sum model
   RooRealSumPdf model0("model0", "model0", {p_h_sig, p_h_bkg}, {Asig0, Abkg0}, true);

   // ***** Case 1 - 'Barlow Beeston' *****

   // Construct parameterized histogram shapes for signal and background
   RooArgList gammas_sig1{ParamHistFunc::createParamSet("p_ph_sig_gamma", x.numBins(), 0, 10)};
   RooArgList gammas_bkg1{ParamHistFunc::createParamSet("p_ph_bkg_gamma", x.numBins(), 0, 10)};

   ParamHistFunc p_phf_sig1{"p_phf_sig1", "p_phf_sig1", x, gammas_sig1};
   ParamHistFunc p_phf_bkg1{"p_phf_bkg1", "p_phf_bkg1", x, gammas_bkg1};

   RooProduct p_p_sig1{"p_p_sig1", "p_p_sig1", p_h_sig, p_phf_sig1};
   RooProduct p_p_bkg1{"p_p_bkg1", "p_p_bkg1", p_h_bkg, p_phf_bkg1};

   RooRealVar Asig1("Asig", "Asig", 1, 0.01, 5000);
   RooRealVar Abkg1("Abkg", "Abkg", 1, 0.01, 5000);

   // Construct the sum of these
   RooRealSumPdf model1_tmp("sp_ph", "sp_ph", {p_p_sig1, p_p_bkg1}, {Asig1, Abkg1}, true);

   // Construct the subsidiary poisson measurements constraining the histogram parameters
   // These ensure that the bin contents of the histograms are only allowed to vary within
   // the statistical uncertainty of the Monte Carlo.
   RooArgSet globs1;
   RooArgSet constraints_sig1{createConstraints(gammas_sig1, p_h_sig, globs1)};
   RooArgSet constraints_bkg1{createConstraints(gammas_bkg1, p_h_bkg, globs1)};

   // Construct the joint model with template PDFs and constraints
   RooProdPdf model1("model1", "model1", {model1_tmp, constraints_sig1, constraints_bkg1});

   // ***** Case 2 - 'Barlow Beeston' light (one parameter per bin for all samples) *****

   // Construct the histogram shapes, using the same parameters for signal and background
   // This requires passing the first histogram to the second, so that their common parameters
   // can be re-used.
   // The first ParamHistFunc will create one parameter per bin, such as `p_ph_sig2_gamma_bin_0`.
   // This allows bin 0 to fluctuate up and down.
   // Then, the SAME parameters are connected to the background histogram, so the bins flucutate
   // synchronously. This reduces the number of parameters.
   RooArgList gammas2{ParamHistFunc::createParamSet("p_ph_gamma", x.numBins(), 0, 10)};

   ParamHistFunc p_phf2{"p_phf2", "p_phf2", x, gammas2};

   RooProduct p_p_sig2{"p_p_sig2", "p_p_sig2", p_h_sig, p_phf2};
   RooProduct p_p_bkg2{"p_p_bkg2", "p_p_bkg2", p_h_bkg, p_phf2};

   RooRealVar Asig2("Asig", "Asig", 1, 0.01, 5000);
   RooRealVar Abkg2("Abkg", "Abkg", 1, 0.01, 5000);

   // As before, construct the sum of signal2 and background2
   RooRealSumPdf model2_tmp("sp_ph", "sp_ph", {p_p_sig2, p_p_bkg2}, {Asig2, Abkg2}, true);

   // Construct the subsidiary poisson measurements constraining the statistical fluctuations
   RooArgSet globs2;
   RooArgSet constraints2{createConstraints(gammas2, {p_h_sig, p_h_bkg}, globs2)};

   // Construct the joint model
   RooProdPdf model2("model2", "model2", {model2_tmp, constraints2});

   // ************ Fit all models to data and plot *********************

   std::unique_ptr<RooFitResult> result0{model0.fitTo(sumData, PrintLevel(0), Save())};
   std::unique_ptr<RooFitResult> result1{model1.fitTo(sumData, PrintLevel(0), Save(), GlobalObservables(globs1))};
   std::unique_ptr<RooFitResult> result2{model2.fitTo(sumData, PrintLevel(0), Save(), GlobalObservables(globs2))};

   TCanvas *can = new TCanvas("can", "", 1500, 600);
   can->Divide(3, 1);

   TPaveText pt(-19.5, 1, -2, 25);
   pt.SetFillStyle(0);
   pt.SetBorderSize(0);

   can->cd(1);
   auto frame = x.frame(Title("No template uncertainties"));
   // Plot data to enable automatic determination of model0 normalisation:
   sumData.plotOn(frame);
   model0.plotOn(frame, LineColor(kBlue), VisualizeError(*result0));
   // Plot data again to show it on top of model0 error bands:
   sumData.plotOn(frame);
   // Plot model components
   model0.plotOn(frame, LineColor(kBlue));
   model0.plotOn(frame, Components(p_h_sig), LineColor(kAzure));
   model0.plotOn(frame, Components(p_h_bkg), LineColor(kRed));
   model0.paramOn(frame);

   sigData->plotOn(frame, MarkerColor(kBlue));
   frame->Draw();

   for (auto text : {"No template uncertainties", "are taken into account.", "This leads to low errors",
                     "for the parameters A, since", "the only source of errors", "are the data statistics."}) {
      pt.AddText(text);
   }
   pt.DrawClone();

   can->cd(2);
   frame = x.frame(Title("Barlow Beeston for Sig & Bkg separately"));
   sumData.plotOn(frame);
   model1.plotOn(frame, LineColor(kBlue), VisualizeError(*result1));
   // Plot data again to show it on top of error bands:
   sumData.plotOn(frame);
   model1.plotOn(frame, LineColor(kBlue));
   model1.plotOn(frame, Components(p_p_sig1), LineColor(kAzure));
   model1.plotOn(frame, Components(p_p_bkg1), LineColor(kRed));
   model1.paramOn(frame, Parameters({Asig1, Abkg1}));

   sigData->plotOn(frame, MarkerColor(kBlue));
   frame->Draw();

   pt.Clear();
   for (auto text : {"With gamma parameters, the", "signal & background templates", "can adapt to the data.",
                     "Note how the blue signal", "template changes its shape.", "This leads to higher errors",
                     "of the scale parameters A."}) {
      pt.AddText(text);
   }
   pt.DrawClone();

   can->cd(3);
   frame = x.frame(Title("Barlow Beeston light for (Sig+Bkg)"));
   sumData.plotOn(frame);
   model2.plotOn(frame, LineColor(kBlue), VisualizeError(*result2));
   // Plot data again to show it on top of model0 error bands:
   sumData.plotOn(frame);
   model2.plotOn(frame, LineColor(kBlue));
   model2.plotOn(frame, Components(p_p_sig2), LineColor(kAzure));
   model2.plotOn(frame, Components(p_p_bkg2), LineColor(kRed));
   model2.paramOn(frame, Parameters({Asig2, Abkg2}));

   sigData->plotOn(frame, MarkerColor(kBlue));
   frame->Draw();

   pt.Clear();
   for (auto text : {"When signal and background", "template share one gamma para-", "meter per bin, they adapt less.",
                     "The errors of the A parameters", "also shrink slightly."}) {
      pt.AddText(text);
   }
   pt.DrawClone();

   std::cout << "Asig [normal ] = " << Asig0.getVal() << " +/- " << Asig0.getError() << std::endl;
   std::cout << "Asig [BB     ] = " << Asig1.getVal() << " +/- " << Asig1.getError() << std::endl;
   std::cout << "Asig [BBlight] = " << Asig2.getVal() << " +/- " << Asig2.getError() << std::endl;
}

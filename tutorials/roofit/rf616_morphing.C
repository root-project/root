/// \file
/// \ingroup tutorial_roofit
/// \notebook -js
/// Use Morphing in RooFit.
///
/// This tutorial shows how to use template morphing inside RooFit. As input we have several
/// Gaussian distributions. The output is one gaussian, with a specific mean value.
/// Since likelihoods are often used within the framework of morphing, we provide a
/// way to estimate the negative log likelihood (nll).
///
/// Based on example of Kyle Cranmer https://gist.github.com/cranmer/b67830e46d53d5f7cf2d.
///
/// \macro_image
/// \macro_code
/// \macro_output
///
/// \date August 2024
/// \author Robin Syring

#include "RooRealVar.h"
#include "RooRealVar.h"
#include "RooWorkspace.h"
#include "RooGaussian.h"
#include "RooUniform.h"
#include "RooDataSet.h"
#include "RooPlot.h"
#include "RooMomentMorphFuncND.h"
#include "RooAbsPdf.h"

using namespace RooFit;

// Number of samples to fill the histograms
const int n_samples = 1000;

// Kills warning massages
RooMsgService::instance().setGlobalKillBelow(RooFit::WARNING);

// Define the morphing routine
RooPlot *perform_morphing(RooWorkspace &ws, RooMomentMorphFuncND::Setting setting, double sigma)
{
   // Get Variables from the workspace
   RooRealVar *x_var = ws.var("x");
   RooRealVar *mu_var = ws.var("mu");
   RooAbsPdf *gauss = ws.pdf("gauss");

   // Initialize a plot
   RooPlot *frame1 = x_var->frame();

   // Define binning for morphing
   RooMomentMorphFuncND::Grid grid(RooBinning(4, 0.0, 4.0));

   // Set binning of histograms, has to be customized for optimal results
   x_var->setBins(50);

   std::vector<int> parampoints = {0, 1, 2, 3, 4};

   for (auto i : parampoints) {
      // Define the sampled gaussians
      RooRealVar mu_help(Form("mu%d", i), Form("mu%d", i), i);
      // Use * because RooGaussian expects objects no pointers
      RooGaussian help(Form("g%d", i), Form("g%d", i), *x_var, mu_help, sigma);
      ws.import(help, Silence(true));

      // Fill the histograms use a unique pointer to prevent memory leaks
      std::unique_ptr<RooDataHist> hist1{
         dynamic_cast<RooDataHist *>(ws.pdf(Form("g%d", i))->generateBinned(*x_var, 100 * n_samples))};

      // Add the value 1 to each bin
      for (int i_bin = 0; i_bin < hist1->numEntries(); ++i_bin) {
         const RooArgSet *binContent = hist1->get(i_bin);
         hist1->add(*binContent, 1.0);
      }

      // Add the pdf to the workspace, the inOrder of 1 is necessary for calculation of the nll
      // Adjust it to 0 to see binning
      ws.import(RooHistPdf(Form("histpdf%d", i), Form("histpdf%d", i), *x_var, *hist1, 1), Silence(true));

      // Plot and add the pdf to the grid
      RooAbsPdf *pdf = ws.pdf(Form("histpdf%d", i));
      pdf->plotOn(frame1);
      grid.addPdf(*pdf, i);
   }

   // Create the morphing
   RooMomentMorphFuncND morph_func("morpf_func", "morph_func", RooArgList(*mu_var), RooArgList(*x_var), grid, setting);

   // Normalizing the morphed object to be a pdf, set it false to prevent warning messages and gain computational speed
   // up
   morph_func.setPdfMode();

   // Creating the morphed pdf
   RooWrapperPdf morph("morph", "morph", morph_func, true);
   ws.import(morph, Silence(true));
   RooAbsPdf *morph_ = ws.pdf("morph");
   morph_->plotOn(frame1, LineColor(kRed));

   return frame1;
}

// Define the workspace
std::unique_ptr<RooWorkspace> build_ws(double mu_observed, double sigma)
{
   auto ws = std::make_unique<RooWorkspace>();
   ws->factory(Form("Gaussian::gauss(x[-5,15],mu[%f,0,4], %f)", mu_observed, sigma));
   return ws;
}

// Do the example
void rf616_morphing()
{
   // Define the 'observed' mu
   double mu_observed = 2.5;
   double sigma = 1.5;

   // Import variables from workspace
   std::unique_ptr<RooWorkspace> ws = build_ws(mu_observed, sigma);

   RooPlot *frame1 = perform_morphing(*ws, RooMomentMorphFuncND::Linear, sigma);

   RooRealVar *x_var = ws->var("x");
   RooRealVar *mu_var = ws->var("mu");
   RooAbsPdf *gauss = ws->pdf("gauss");
   RooDataSet *obs_data = gauss->generate(*x_var, n_samples);

   // Create the exact negative log likelihood function for Gaussian model
   RooAbsReal *nll_gauss = gauss->createNLL(*obs_data);

   // Create the morphed negative log likelihood function
   // TODO: Fix RooAddPdf::fixCoefNormalization(nset) warnings with new CPU backend
   RooAbsReal *nll_morph = ws->pdf("morph")->createNLL(*obs_data, RooFit::EvalBackend("legacy"));

   // Plot the negative logarithmic summed likelihood
   RooPlot *frame2 = mu_var->frame(Title("Negative log Likelihood"));
   nll_gauss->plotOn(frame2, LineColor(kBlue), ShiftToZero(), Name("gauss"));
   nll_morph->plotOn(frame2, LineColor(kRed), ShiftToZero(), Name("morph"));

   TCanvas *c = new TCanvas("rf616_morphing", "rf616_morphing", 800, 400);
   c->Divide(2);
   c->cd(1);
   gPad->SetLeftMargin(0.15);
   frame1->GetYaxis()->SetTitleOffset(1.4);
   frame1->Draw();
   c->cd(2);
   gPad->SetLeftMargin(0.15);
   frame2->GetYaxis()->SetTitleOffset(1.8);
   frame2->Draw();

   // Compute the minimum of the nll via minuit
   std::vector<RooAbsReal *> nlls = {nll_gauss, nll_morph};
   for (auto nll : nlls) {
      RooMinimizer minimizer(*nll);
      minimizer.setPrintLevel(-1);
      minimizer.minimize("Minuit2");
      RooFitResult *result = minimizer.save();
      result->Print();
   }
}

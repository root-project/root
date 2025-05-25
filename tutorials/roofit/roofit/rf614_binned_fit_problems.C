/// \file
/// \ingroup tutorial_roofit_main
/// \notebook -js
/// A tutorial that explains you how to solve problems with binning effects and
/// numerical stability in binned fits.
///
/// ### Introduction
///
/// In this tutorial, you will learn three new things:
///
///  1. How to reduce the bias in binned fits by changing the definition of the
///     normalization integral
///
///  2. How to completely get rid of binning effects by integrating the pdf over each bin
///
///  3. How to improve the numeric stability of fits with a greatly different
///     number of events per bin, using a constant per-bin counterterm
///
/// \macro_code
/// \macro_output
///
/// \date January 2023
/// \author Jonas Rembser

// Generate binned Asimov dataset for a continuous pdf.
// One should in principle be able to use
// pdf.generateBinned(x, nEvents, RooFit::ExpectedData()).
// Unfortunately it has a problem: it also has the bin bias that this tutorial
// demonstrates, to if we would use it, the biases would cancel out.
std::unique_ptr<RooDataHist> generateBinnedAsimov(RooAbsPdf const &pdf, RooRealVar &x, int nEvents)
{
   auto dataH = std::make_unique<RooDataHist>("dataH", "dataH", RooArgSet{x});
   RooAbsBinning &xBinning = x.getBinning();
   for (int iBin = 0; iBin < x.numBins(); ++iBin) {
      x.setRange("bin", xBinning.binLow(iBin), xBinning.binHigh(iBin));
      std::unique_ptr<RooAbsReal> integ{pdf.createIntegral(x, RooFit::NormSet(x), RooFit::Range("bin"))};
      integ->getVal();
      dataH->set(iBin, nEvents * integ->getVal(), -1);
   }
   return dataH;
}

// Force numeric integration and do this numeric integration with the
// RooBinIntegrator, which sums the function values at the bin centers.
void enableBinIntegrator(RooAbsReal &func, int numBins)
{
   RooNumIntConfig customConfig(*func.getIntegratorConfig());
   customConfig.method1D().setLabel("RooBinIntegrator");
   customConfig.getConfigSection("RooBinIntegrator").setRealValue("numBins", numBins);
   func.setIntegratorConfig(customConfig);
   func.forceNumInt(true);
}

// Reset the integrator config to disable the RooBinIntegrator.
void disableBinIntegrator(RooAbsReal &func)
{
   func.setIntegratorConfig();
   func.forceNumInt(false);
}

void rf614_binned_fit_problems()
{
   using namespace RooFit;

   // Silence info output for this tutorial
   RooMsgService::instance().getStream(1).removeTopic(Minimization);
   RooMsgService::instance().getStream(1).removeTopic(Fitting);
   RooMsgService::instance().getStream(1).removeTopic(Generation);

   // Exponential example
   // -------------------

   // Set up the observable
   RooRealVar x{"x", "x", 0.1, 5.1};
   x.setBins(10); // fewer bins so we have larger binning effects for this demo

   // Let's first look at the example of an exponential function
   RooRealVar c{"c", "c", -1.8, -5, 5};
   RooExponential expo{"expo", "expo", x, c};

   // Generate an Asimov dataset such that the only difference between the fit
   // result and the true parameters comes from binning effects.
   std::unique_ptr<RooAbsData> expoData{generateBinnedAsimov(expo, x, 10000)};

   // If you do the fit the usual was in RooFit, you will get a bias in the
   // result. This is because the continuous, normalized pdf is evaluated only
   // at the bin centers.
   std::unique_ptr<RooFitResult> fit1{expo.fitTo(*expoData, Save(), PrintLevel(-1), SumW2Error(false))};
   fit1->Print();

   // In the case of an exponential function, the bias that you get by
   // evaluating the pdf only at the bin centers is a constant scale factor in
   // each bin. Here, we can do a trick to get rid of the bias: we also
   // evaluate the normalization integral for the pdf the same way, i.e.,
   // summing the values of the unnormalized pdf at the bin centers. Like this
   // the bias cancels out. You can achieve this by customizing the way how the
   // pdf is integrated (see also the rf901_numintconfig tutorial).
   enableBinIntegrator(expo, x.numBins());
   std::unique_ptr<RooFitResult> fit2{expo.fitTo(*expoData, Save(), PrintLevel(-1), SumW2Error(false))};
   fit2->Print();
   disableBinIntegrator(expo);

   // Power law example
   // -----------------

   // Let's not look at another example: a power law \f[x^a\f].
   RooRealVar a{"a", "a", -0.3, -5.0, 5.0};
   RooPowerSum powerlaw{"powerlaw", "powerlaw", x, RooConst(1.0), a};
   std::unique_ptr<RooAbsData> powerlawData{generateBinnedAsimov(powerlaw, x, 10000)};

   // Again, if you do a vanilla fit, you'll get a bias
   std::unique_ptr<RooFitResult> fit3{powerlaw.fitTo(*powerlawData, Save(), PrintLevel(-1), SumW2Error(false))};
   fit3->Print();

   // This time, the bias is not the same factor in each bin! This means our
   // trick by sampling the integral in the same way doesn't cancel out the
   // bias completely. The average bias is canceled, but there are per-bin
   // biases that remain. Still, this method has some value: it is cheaper than
   // rigurously correcting the bias by integrating the pdf in each bin. So if
   // you know your per-bin bias variations are small or performance is an
   // issue, this approach can be sufficient.
   enableBinIntegrator(powerlaw, x.numBins());
   std::unique_ptr<RooFitResult> fit4{powerlaw.fitTo(*powerlawData, Save(), PrintLevel(-1), SumW2Error(false))};
   fit4->Print();
   disableBinIntegrator(powerlaw);

   // To get rid of the binning effects in the general case, one can use the
   // IntegrateBins() command argument. Now, the pdf is not evaluated at the
   // bin centers, but numerically integrated over each bin and divided by the
   // bin width. The parameter for IntegrateBins() is the required precision
   // for the numeric integrals. This is computationally expensive, but the
   // bias is now not a problem anymore.
   std::unique_ptr<RooFitResult> fit5{
      powerlaw.fitTo(*powerlawData, IntegrateBins(1e-3), Save(), PrintLevel(-1), SumW2Error(false))};
   fit5->Print();

   // Improving numerical stability
   // -----------------------------

   // There is one more problem with binned fits that is related to the binning
   // effects because often, a binned fit is affected by both problems.
   //
   // The issue is numerical stability for fits with a greatly different number
   // of events in each bin. For each bin, you have a term \f[n\log(p)\f] in
   // the NLL, where \f[n\f] is the number of observations in the bin, and
   // \f[p\f] the predicted probability to have an event in that bin. The
   // difference in the logarithms for each bin is small, but the difference in
   // \f[n\f] can be orders of magnitudes! Therefore, when summing these terms,
   // lots of numerical precision is lost for the bins with less events.

   // We can study this with the example of an exponential plus a Gaussian. The
   // Gaussian is only a faint signal in the tail of the exponential where
   // there are not so many events. And we can't afford any precision loss for
   // these bins, otherwise we can't fit the Gaussian.

   x.setBins(100); // It's not about binning effects anymore, so reset the number of bins.

   RooRealVar mu{"mu", "mu", 3.0, 0.1, 5.1};
   RooRealVar sigma{"sigma", "sigma", 0.5, 0.01, 5.0};
   RooGaussian gauss{"gauss", "gauss", x, mu, sigma};

   RooRealVar nsig{"nsig", "nsig", 10000, 0, 1e9};
   RooRealVar nbkg{"nbkg", "nbkg", 10000000, 0, 1e9};
   RooRealVar frac{"frac", "frac", nsig.getVal() / (nsig.getVal() + nbkg.getVal()), 0.0, 1.0};

   RooAddPdf model{"model", "model", {gauss, expo}, {nsig, nbkg}};

   std::unique_ptr<RooAbsData> modelData{model.generateBinned(x)};

   // Set the starting values for the Gaussian parameters away from the true
   // value such that the fit is not trivial.
   mu.setVal(2.0);
   sigma.setVal(1.0);

   std::unique_ptr<RooFitResult> fit6{model.fitTo(*modelData, Save(), PrintLevel(-1), SumW2Error(false))};
   fit6->Print();

   // You should see in the previous fit result that the fit did not converge:
   // the `MINIMIZE` return code should be -1 (a successful fit has status code zero).

   // To improve the situation, we can apply a numeric trick: if we subtract in
   // each bin a constant counterterm \f[n\log(n/N)\f], we get terms for each
   // bin that are closer to each other in order of magnitude as long as the
   // initial model is not extremely off. Proving this mathematically is left
   // as an exercise to the reader.

   // This counterterms can be enabled by passing the Offset("bin") option to
   // RooAbsPdf::fitTo() or RooAbsPdf::createNLL().

   std::unique_ptr<RooFitResult> fit7{
      model.fitTo(*modelData, Offset("bin"), Save(), PrintLevel(-1), SumW2Error(false))};
   fit7->Print();

   // You should now see in the last fit result that the fit has converged.
}

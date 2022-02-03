/// \file
/// \ingroup tutorial_roofit
/// \notebook -js
/// This tutorial explains the concept of global observables in RooFit, and
/// showcases how their values can be stored either in the model or in the
/// dataset.
///
/// ### Introduction
///
/// Note: in this tutorial, we are multiplying the likelihood with an additional
/// likelihood to constrain the parameters with auxiliary measurements. This is
/// different from the `rf604_constraints` tutorial, where the likelihood is
/// multiplied with a Bayesian prior to constrain the parameters.
///
///
/// With RooFit, you usually optimize some model parameters `p` to maximize the
/// likelihood `L` given the per-event or per-bin ## observations `x`:
///
/// \f[ L( x | p ) \f]
///
/// Often, the parameters are constrained with some prior likelihood `C`, which
/// doesn't depend on the observables `x`:
///
/// \f[ L'( x | p ) = L( x | p ) * C( p ) \f]
///
/// Usually, these constraint terms depend on some auxiliary measurements of
/// other observables `g`. The constraint term is then the likelihood of the
/// so-called global observables:
///
/// \f[ L'( x | p ) = L( x | p ) * C( g | p ) \f]
///
/// For example, think of a model where the true luminosity `lumi` is a
/// nuisance parameter that is constrained by an auxiliary measurement
/// `lumi_obs` with uncertainty `lumi_obs_sigma`:
///
/// \f[ L'(data | mu, lumi) = L(data | mu, lumi) * Gauss(lumi_obs | lumi, lumi_obs_sigma) \f]
///
/// As a Gaussian is symmetric under exchange of the observable and the mean
/// parameter, you can also sometimes find this equivalent but less conventional
/// formulation for Gaussian constraints:
///
/// \f[ L'(data | mu, lumi) = L(data | mu, lumi) * Gauss(lumi | lumi_obs, lumi_obs_sigma) \f]
///
/// If you wanted to constrain a parameter that represents event counts, you
/// would use a Poissonian constraint, e.g.:
///
/// \f[ L'(data | mu, count) = L(data | mu, count) * Poisson(count_obs | count) \f]
///
/// Unlike a Guassian, a Poissonian is not symmetric under exchange of the
/// observable and the parameter, so here you need to be more careful to follow
/// the global observable prescription correctly.
///
///
/// \macro_code
///
/// \date January 2022
/// \author Jonas Rembser

#include <RooArgSet.h>
#include <RooConstVar.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooGaussian.h>
#include <RooProdPdf.h>
#include <RooRealVar.h>

void rf613_global_observables()
{
   using namespace RooFit;

   // Setting up the model and creating toy dataset
   // ---------------------------------------------

   // l'(x | mu, sigma) = l(x | mu, sigma) * Gauss(mu_obs | mu, 0.2)

   // event observables
   RooRealVar x("x", "x", -10, 10);

   // parameters
   RooRealVar mu("mu", "mu", 0.0, -10, 10);
   RooRealVar sigma("sigma", "sigma", 1.0, 0.1, 2.0);

   // Gaussian model for event observables
   RooGaussian gauss("gauss", "gauss", x, mu, sigma);

   // global observables (which are not parameters so they are constant)
   RooRealVar mu_obs("mu_obs", "mu_obs", 1.0, -10, 10);
   mu_obs.setConstant();
   // note: alternatively, one can create a constant with default limits using `RooRealVar("mu_obs", "mu_obs", 1.0)`

   // constraint pdf
   RooGaussian constraint("constraint", "constraint", mu_obs, mu, RooConst(0.2));

   // full pdf including constraint pdf
   RooProdPdf model("model", "model", {gauss, constraint});

   // Generating toy data with randomized global observables
   // ------------------------------------------------------

   // For most toy-based statistical procedures, it is necessary to also
   // randomize the global observable when generating toy datasets.
   //
   // To that end, let's generate a single event from the model and take the
   // global observable value (the same is done in the RooStats:ToyMCSampler
   // class):

   std::unique_ptr<RooDataSet> dataGlob{model.generate({mu_obs}, 1)};

   // Next, we temporarily set the value of `mu_obs` to the randomized value for
   // generating our toy dataset:
   double mu_obs_orig_val = mu_obs.getVal();

   RooArgSet{mu_obs}.assign(*dataGlob->get(0));

   // actually generate the toy dataset
   std::unique_ptr<RooDataSet> data{model.generate({x}, 1000)};

   // When fitting the toy dataset, it is important to set the global
   // observables in the fit to the values that were used to generate the toy
   // dataset. To facilitate the bookkeeping of global observable values, you
   // can attach a snapshot with the current global observable values to the
   // dataset like this (new feature introduced in ROOT 6.26):

   data->setGlobalObservables({mu_obs});

   // reset original mu_obs value
   mu_obs.setVal(mu_obs_orig_val);

   // Fitting a model with global observables
   // ---------------------------------------

   // Create snapshot of original parameters to reset parameters after fitting
   RooArgSet modelParameters;
   model.getParameters(data->get(), modelParameters);
   RooArgSet origParameters;
   modelParameters.snapshot(origParameters);

   using FitRes = std::unique_ptr<RooFitResult>;

   // When you fit a model that includes global observables, you need to
   // specify them in the call to RooAbsPdf::fitTo with the
   // RooFit::GlobalObservables command argument. By default, the global
   // observable values attached to the dataset will be prioritized over the
   // values in the model, so the following fit correctly uses the randomized
   // global observable values from the toy dataset:
   std::cout << "1. model.fitTo(*data, GlobalObservables(mu_obs))\n";
   std::cout << "------------------------------------------------\n\n";
   FitRes res1{model.fitTo(*data, GlobalObservables(mu_obs), PrintLevel(-1), Save())};
   res1->Print();
   modelParameters.assign(origParameters);

   // In our example, the set of global observables is attached to the toy
   // dataset. In this case, you can actually drop the GlobalObservables()
   // command argument, because the global observables are automatically
   // figured out from the data set (this fit result should be identical to the
   // previous one).
   std::cout << "2. model.fitTo(*data)\n";
   std::cout << "---------------------\n\n";
   FitRes res2{model.fitTo(*data, PrintLevel(-1), Save())};
   res2->Print();
   modelParameters.assign(origParameters);

   // If you want to explicitly ignore the global observables in the dataset,
   // you can do that by specifying GlobalObservablesSource("model"). Keep in
   // mind that now it's also again your responsability to define the set of
   // global observables.
   std::cout << "3. model.fitTo(*data, GlobalObservables(mu_obs), GlobalObservablesSource(\"model\"))\n";
   std::cout << "------------------------------------------------\n\n";
   FitRes res3{model.fitTo(*data, GlobalObservables(mu_obs), GlobalObservablesSource("model"), PrintLevel(-1), Save())};
   res3->Print();
   modelParameters.assign(origParameters);
}

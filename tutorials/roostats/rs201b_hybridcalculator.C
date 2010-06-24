#include "RooRandom.h"
#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooPolynomial.h"
#include "RooArgSet.h"
#include "RooAddPdf.h"
#include "RooDataSet.h"
#include "RooExtendPdf.h"
#include "RooConstVar.h"
#include "RooGlobalFunc.h"
#include "RooStats/HybridCalculator.h"
#include "RooStats/ToyMCSampler.h"
#include "RooStats/ProfileLikelihoodTestStat.h"
#include "RooStats/SimpleLikelihoodRatioTestStat.h"
#include "RooStats/RatioOfProfiledLikelihoodsTestStat.h"
#include "RooStats/HypoTestPlot.h"
#include "RooStats/HypoTestResult.h"


using namespace RooFit;
using namespace RooStats;

void rs201b_hybridcalculator(int ntoys = 1000)
{
  //***********************************************************************//
  // This macro show an example on how to use RooStats/HybridCalculator    //
  // Tutorial by Kyle Cranmer and Sven Kreiss, 
  //  adapted from example by Gregory Schott
  //***********************************************************************//
  //
  // The model is a Gaussian signal of width 1 sitting on top of a flat background.
  // The expected signal is 20 events and an uncertain background.
  // The backround rate is uncertain, but the sideband helps constrain it.
  //
  // In addition, 40 events are seen in another control sample where
  // one expects to see +/- 10 events around the true background rate.
  //
  // With the  informaton in the control  sample and a prior, one 
  // can form a posterior for the background rate.  This is not
  // yet supported within the HybridCalculator, but it will be soon.
  // In additon, one can force the HybridCalculator to use
  // a given prior to randomize the background rate in the toy monte carlo.
  //
  // The HybridCalculator gives you control over what "test statistic"
  // you want to use to distinguish the null and alternate hypotheses.
  // There are few sensible things that can be done here.

  // The first is to fix the background for any particular
  // pseudo experiment, but fluctuate it together with the background 
  // used to generate the pseduo experiments. This is the equivalent of 
  // the number counting experiment where this no sideband.  
  // To do this, the background rate should be set constant so that 
  // it will not float in any pseudo experiment.
  // If one also does not provide a force a prior in the hybrid calculator, 
  // then the background will not fluctuate across pseudo experiments and
  // this will reduce to having no uncertainty on the background.

  // The second is to let the background rate float in each fit
  // and only use the sideband to constrain the background.  This
  // should be more powerful if the sideband is large.

  // The third option is to let the background float and also use the 
  // information in the control measurement to reduce the background
  // uncertainty over what the sideband does on its own.
  //
  // In each of these examples the control measurement is fixed to 40.
  // It is also possible let that fluctuate if it is included
  // in the list of observables included in the model config. 
  // However, if this is done the control measurent  should also 
  // be included in the model config's list of global observables as
  // there is only one control measurement and several measurements 
  // of per pseudo-experiment.
  // 

  // set RooFit random seed for reproducible results
  RooRandom::randomGenerator()->SetSeed(3008);

  // B U I L D    M O D E L
  RooWorkspace w("example");
  w.factory("x[-3,3]"); // observable is called x

  // Gaussian signal
  w.factory("Gaussian::sig_pdf(x,sig_mean[0],sig_sigma[1])");
  // flat background 
  w.factory("Uniform::bkg_pdf(x)");
  // total model with signal and background yields as parameters
  w.factory("SUM::main_pdf(sig_yield[20,0,300]*sig_pdf,bkg_yield[50,0,300]*bkg_pdf)");
  //w.var("bkg_yield")->setConstant(); // if you want the background fixed

  // The model for the control sample that constrains the background.
  w.factory("Gaussian::control_pdf(control_meas[50],bkg_yield,10.)");

  // The total model including the main measurement and the control sample
  w.factory("PROD::main_with_control(main_pdf,control_pdf)");

  // choose which pdf you want to use
  RooAbsPdf* pdfToUse = w.pdf("main_pdf"); // only use main measurement
  //RooAbsPdf* pdfToUse = w.pdf("main_with_control"); // also include control sample
  
  // define sets for reference later
  w.defineSet("obs","x");
  w.defineSet("poi","sig_yield");
  w.defineSet("nuis","bkg_yield");

  // M A K E   T O Y  D A T A 
  RooDataSet* data = w.pdf("main_pdf")->generate(*w.set("obs"),RooFit::Extended());
  data->Print();

  // D E F I N E  N U L L  &  A L T E R N A T I V E   H Y P O T H E S E S  
  ModelConfig b_model("B_model", &w);
  b_model.SetPdf(*pdfToUse);
  b_model.SetObservables(*w.set("obs"));
  b_model.SetParametersOfInterest(*w.set("poi"));
  b_model.SetNuisanceParameters(*w.set("nuis")); 
  w.var("sig_yield")->setVal(0.0);
  b_model.SetSnapshot(*w.set("poi"));  

  ModelConfig sb_model("S+B_model", &w);
  sb_model.SetPdf(*pdfToUse);
  sb_model.SetObservables(*w.set("obs"));
  sb_model.SetParametersOfInterest(*w.set("poi"));
  sb_model.SetNuisanceParameters(*w.set("nuis")); 
  w.var("sig_yield")->setVal(20.0);
  sb_model.SetSnapshot(*w.set("poi"));  

  // inspect workspace if you wish
  //  w.Print();

  // C H O O S E   T E S T  S T A T I S T I C  F O R  T E S T
  // by default the hybrid calculator will use the ToyMCSampler
  // and the ratio of profiled likleihoods as a test statistic
  // but the user can have full control of the choice of test 
  // statistic and the method for sampling it.
  //
  // here we demonstrait the creation of several approriate test statistics

  // a simple likelihood ratio with background fixed to nominal value
  SimpleLikelihoodRatioTestStat slrts(*b_model.GetPdf(),*sb_model.GetPdf());
  slrts.SetNullParameters(*b_model.GetSnapshot());
  slrts.SetAltParameters(*sb_model.GetSnapshot());

  // ratio of alt and null likelihoods with background yiled profiled
  RatioOfProfiledLikelihoodsTestStat ropl(*b_model.GetPdf(), *sb_model.GetPdf(), sb_model.GetSnapshot());

  // profile likelihood where alternate is best fit value of signal yield
  ProfileLikelihoodTestStat profll(*b_model.GetPdf());

  // just use the maximum likelihood estimate of signal yield
  MaxLikelihoodEstimateTestStat mlets(*sb_model.GetPdf(), *w.var("sig_yield"));

  // Now we create toyMCSampler with the chosen test statistic
  //  ToyMCSampler toymcsampler(slrts, ntoys);
  ToyMCSampler toymcsampler(ropl, ntoys);
  //  ToyMCSampler toymcsampler(profll, ntoys);
  //  ToyMCSampler toymcsampler(mlets, ntoys);

  // and we can set some options for the toy mc sampler
  // toymcsampler.SetGenerateBinned(true); // can speed thing sup

  // U S E   H Y B R I D   C A L C U L A T O R
  // this is the tool that does the actual hypothesis test
  HybridCalculator myH2(*data,sb_model, b_model, &toymcsampler);
  // if toymcsampler is not provided in constructor, it will use defaults
  //  HybridCalculator myH2(*data,sb_model, b_model);

  // the hybrid calculator needs a Bayesian distribution \pi on the nuisance parameters
  // if a prior is included in the ModelConfig, then it is possible 
  // to create it from the posterior from a constraint term and the prior, but
  // this is not yet implemented.  
  // For now, one must force the hybrid calculator to use a 
  // specific prior for generating the toys.
  // In general, there are two different models for signal and background
  // and they can have different nuisance parameters and different priors.
  myH2.ForcePriorNuisanceNull(*w.pdf("control_pdf")); 
  myH2.ForcePriorNuisanceAlt(*w.pdf("control_pdf"));  

  // now get the result
  HypoTestResult *res = myH2.GetHypoTest();
  res->Print();

  // and make a plot. number of bins is optional (default: 100)
  HypoTestPlot *plot = new HypoTestPlot(*res, 80); 
  plot->Draw();

}



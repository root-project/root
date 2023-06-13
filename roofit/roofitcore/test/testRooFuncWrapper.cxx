/*
 * Project: RooFit
 * Authors:
 *   Garima Singh, CERN 2023
 *   Jonas Rembser, CERN 2023
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooAbsPdf.h>
#include <RooAbsData.h>
#include <RooAddPdf.h>
#include <RooBinWidthFunction.h>
#include <RooCategory.h>
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooExponential.h>
#include <RooFitResult.h>
#include <RooFuncWrapper.h>
#include <RooGaussian.h>
#include <RooHistFunc.h>
#include <RooHistPdf.h>
#include <RooHelpers.h>
#include <RooMinimizer.h>
#include <RooPoisson.h>
#include <RooPolynomial.h>
#include <RooProduct.h>
#include <RooRealVar.h>
#include <RooRealSumPdf.h>
#include <RooSimultaneous.h>
#include <RooWorkspace.h>
#include <RooStats/HistFactory/ParamHistFunc.h>

#include <ROOT/StringUtils.hxx>
#include <TROOT.h>
#include <TSystem.h>
#include <TMath.h>

#include <functional>
#include <random>

#include <gtest/gtest.h>

// Backward compatibility for gtest version < 1.10.0
#ifndef INSTANTIATE_TEST_SUITE_P
#define INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#endif

namespace {

// Function to get the derivative of pdf wrt var.
double getNumDerivative(const RooAbsReal &pdf, RooRealVar &var, const RooArgSet &normSet, double eps = 1e-8)
{
   double orig = var.getVal();
   if (!var.inRange(orig + eps, nullptr)) {
      throw std::runtime_error("getNumDerivative(): positive variation outside of range!");
   }
   if (!var.inRange(orig - eps, nullptr)) {
      throw std::runtime_error("getNumDerivative(): negative variation outside of range!");
   }
   var.setVal(orig + eps);
   double plus = pdf.getVal(normSet);
   var.setVal(orig - eps);
   double minus = pdf.getVal(normSet);
   var.setVal(orig);

   return (plus - minus) / (2 * eps);
}

void randomizeParameters(const RooArgSet &parameters)
{
   double lowerBound = -0.1;
   double upperBound = 0.1;
   std::uniform_real_distribution<double> unif(lowerBound, upperBound);
   std::default_random_engine re;

   for (auto *param : parameters) {
      double mul = unif(re);

      auto par = dynamic_cast<RooAbsRealLValue *>(param);
      if (!par)
         continue;
      double val = par->getVal();
      val = val + mul * (mul > 0 ? (par->getMax() - val) : (val - par->getMin()));

      par->setVal(val);
   }
}

} // namespace

TEST(RooFuncWrapper, GaussianNormalizedHardcoded)
{
   using namespace RooFit;
   auto inf = std::numeric_limits<double>::infinity();

   RooWorkspace ws;
   ws.import(RooRealVar{"x", "x", 0, -inf, inf});
   ws.factory("Gaussian::gauss(x, mu[0, -10, 10], sigma[2.0, 0.01, 10])");

   RooAbsPdf &gauss = *ws.pdf("gauss");
   RooRealVar &x = *ws.var("x");
   RooRealVar &mu = *ws.var("mu");
   RooRealVar &sigma = *ws.var("sigma");

   RooArgSet normSet{x};
   RooArgSet paramsGauss;
   RooArgSet paramsMyGauss;

   std::string func = "const double arg = params[0] - params[1];"
                      "const double sig = params[2];"
                      "double out = std::exp(-0.5 * arg * arg / (sig * sig));"
                      "return 1. / (std::sqrt(TMath::TwoPi()) * sig) * out;";
   RooFuncWrapper gaussFunc("myGauss1", "myGauss1", func, {x, mu, sigma}, {});

   // Check if functions results are the same even after changing parameters.
   EXPECT_NEAR(gauss.getVal(normSet), gaussFunc.getVal(), 1e-8);

   mu.setVal(1);
   EXPECT_NEAR(gauss.getVal(normSet), gaussFunc.getVal(), 1e-8);

   // Check if the parameter layout and size is the same.
   gauss.getParameters(&normSet, paramsGauss);
   gaussFunc.getParameters(&normSet, paramsMyGauss);

   EXPECT_TRUE(paramsMyGauss.hasSameLayout(paramsGauss));
   EXPECT_EQ(paramsMyGauss.size(), paramsGauss.size());

   // Get AD based derivative
   // Get number of actual parameters directly from the wrapper as not always will they be the same as paramsMyGauss.
   std::vector<double> dMyGauss(gaussFunc.getNumParams(), 0);
   gaussFunc.getGradient(dMyGauss.data());

   // Check if derivatives are equal
   EXPECT_NEAR(getNumDerivative(gauss, x, normSet), dMyGauss[0], 1e-8);
   EXPECT_NEAR(getNumDerivative(gauss, mu, normSet), dMyGauss[1], 1e-8);
   EXPECT_NEAR(getNumDerivative(gauss, sigma, normSet), dMyGauss[2], 1e-8);
}

TEST(RooFuncWrapper, GaussianNormalized)
{
   RooWorkspace ws;
   ws.import(RooRealVar{"x", "x", 0, -10, std::numeric_limits<double>::infinity()});
   ws.factory("sum::mu_shifted(mu[0, -10, 10], shift[1.0, -10, 10])");
   ws.factory("prod::sigma_scaled(sigma[2.0, 0.01, 10], 1.5)");
   ws.factory("Gaussian::gauss(x, mu_shifted, sigma_scaled)");

   RooAbsPdf &gauss = *ws.pdf("gauss");
   RooRealVar &x = *ws.var("x");
   RooRealVar &mu = *ws.var("mu");

   RooArgSet normSet{x};

   RooFuncWrapper gaussFunc("myGauss3", "myGauss3", gauss, normSet);

   RooArgSet paramsGauss;
   gauss.getParameters(nullptr, paramsGauss);

   // Check if functions results are the same even after changing parameters.
   EXPECT_NEAR(gauss.getVal(normSet), gaussFunc.getVal(), 1e-8);

   mu.setVal(1);
   EXPECT_NEAR(gauss.getVal(normSet), gaussFunc.getVal(), 1e-8);

   // Get AD based derivative
   std::vector<double> dMyGauss(gaussFunc.getNumParams(), 0);
   gaussFunc.getGradient(dMyGauss.data());

   // Check if derivatives are equal
   for (std::size_t i = 0; i < paramsGauss.size(); ++i) {
      EXPECT_NEAR(getNumDerivative(gauss, static_cast<RooRealVar &>(*paramsGauss[i]), normSet), dMyGauss[i], 1e-8);
   }
}

TEST(RooFuncWrapper, Exponential)
{
   RooWorkspace ws;
   ws.factory("Exponential::expo(x[1.0, 0, 10], c[0.1, 0, 10])");

   RooAbsPdf &expo = *ws.pdf("expo");
   RooRealVar &x = *ws.var("x");

   RooArgSet normSet{x};

   RooFuncWrapper expoFunc("expo", "expo", expo, normSet);

   RooArgSet params;
   expo.getParameters(nullptr, params);

   EXPECT_NEAR(expo.getVal(normSet), expoFunc.getVal(), 1e-8);

   // Get AD based derivative
   std::vector<double> dExpo(expoFunc.getNumParams(), 0);
   expoFunc.getGradient(dExpo.data());

   // Check if derivatives are equal
   for (std::size_t i = 0; i < params.size(); ++i) {
      EXPECT_NEAR(getNumDerivative(expo, static_cast<RooRealVar &>(*params[i]), normSet), dExpo[i], 1e-8)
         << params[i]->GetName();
   }
}

using CreateNLLFunc = std::function<std::unique_ptr<RooAbsReal>(RooAbsPdf &, RooAbsData &, RooWorkspace &)>;
using WorkspaceSetupFunc = std::function<void(RooWorkspace &)>;

class FactoryTestParams {
public:
   FactoryTestParams() = default;
   FactoryTestParams(std::string const &name, WorkspaceSetupFunc setupWorkspace, CreateNLLFunc createNLL,
                     double fitResultTolerance, bool randomizeParameters)
      : _name{name},
        _setupWorkspace{setupWorkspace},
        _createNLL{createNLL},
        _fitResultTolerance{fitResultTolerance},
        _randomizeParameters{randomizeParameters}
   {
   }

   std::string _name;
   WorkspaceSetupFunc _setupWorkspace;
   CreateNLLFunc _createNLL;
   double _fitResultTolerance = 1e-4;
   bool _randomizeParameters = true;
};

class FactoryTest : public testing::TestWithParam<FactoryTestParams> {
   void SetUp() override
   {
      _changeMsgLvl = std::make_unique<RooHelpers::LocalChangeMsgLevel>(RooFit::WARNING);
      _params = GetParam();
   }

   void TearDown() override { _changeMsgLvl.reset(); }

protected:
   FactoryTestParams _params;

private:
   std::unique_ptr<RooHelpers::LocalChangeMsgLevel> _changeMsgLvl;
};

TEST_P(FactoryTest, NLLFit)
{

   RooWorkspace ws;

   std::unique_ptr<RooAbsData> ownedData;
   _params._setupWorkspace(ws);
   RooArgSet const &observables = *ws.set("observables");
   RooAbsData *data = ws.data("data");

   RooAbsPdf &model = *ws.pdf("model");

   // Figure out if this is a simultaneous fit. The RooFuncWrapper needs to
   // know about this to figure out how to split the dataset and fill the
   // observables.
   auto simPdf = dynamic_cast<RooSimultaneous *>(&model);

   std::size_t nEvents = 100;
   if (!data) {
      std::unique_ptr<RooDataSet> data0{model.generate(observables, nEvents)};
      ownedData = std::unique_ptr<RooAbsData>{data0->binnedClone()};
      data = ownedData.get();
   }
   std::unique_ptr<RooAbsReal> nllRef = _params._createNLL(model, *data, ws);
   auto nllRefResolved = static_cast<RooAbsReal *>(nllRef->servers()[0]);

   static int funcWrapperCounter = 0;
   std::string wrapperName = "func_wrapper_" + std::to_string(funcWrapperCounter++);
   RooFuncWrapper nllFunc(wrapperName.c_str(), wrapperName.c_str(), *nllRefResolved, observables, data, simPdf);

   // Check if functions results are the same even after changing parameters.
   EXPECT_NEAR(nllRef->getVal(observables), nllFunc.getVal(), 1e-8);

   // Check if the parameter layout and size is the same.
   RooArgSet paramsRefNll;
   nllRef->getParameters(nullptr, paramsRefNll);
   RooArgSet paramsMyNLL;
   nllFunc.getParameters(&observables, paramsMyNLL);

   if (_params._randomizeParameters) {
      randomizeParameters(paramsMyNLL);
   }

   EXPECT_TRUE(paramsMyNLL.hasSameLayout(paramsRefNll));
   EXPECT_EQ(paramsMyNLL.size(), paramsRefNll.size());

   // Get AD based derivative
   std::vector<double> dMyNLL(nllFunc.getNumParams(), 0);
   nllFunc.getGradient(dMyNLL.data());

   // Check if derivatives are equal
   for (std::size_t i = 0; i < paramsMyNLL.size(); ++i) {
      EXPECT_NEAR(getNumDerivative(*nllRef, static_cast<RooRealVar &>(*paramsMyNLL[i]), observables), dMyNLL[i], 1e-4);
   }

   // Remember parameter state before minimization
   RooArgSet parametersOrig;
   paramsRefNll.snapshot(parametersOrig);

   auto runMinimizer = [&](RooAbsReal &absReal, RooMinimizer::Config cfg = {}) -> std::unique_ptr<RooFitResult> {
      RooMinimizer m{absReal, cfg};
      m.setPrintLevel(-1);
      m.setStrategy(0);
      m.minimize("Minuit2");
      auto result = std::unique_ptr<RooFitResult>{m.save()};
      // reset parameters
      paramsRefNll.assign(parametersOrig);
      return result;
   };

   // Minimize the RooFuncWrapper Implementation
   auto result = runMinimizer(nllFunc);

   // Minimize the RooFuncWrapper Implementation with AD
   RooMinimizer::Config minimizerCfgAd;
   std::size_t nGradientCalls = 0;
   minimizerCfgAd.gradFunc = [&](double *out) {
      nllFunc.getGradient(out);
      ++nGradientCalls;
   };
   auto resultAd = runMinimizer(nllFunc, minimizerCfgAd);
   EXPECT_GE(nGradientCalls, 1); // make sure the gradient function was actually called

   // Minimize the reference NLL
   auto resultRef = runMinimizer(*nllRef);

   // Compare minimization results
   double tol = _params._fitResultTolerance;
   // Same tolerance for parameter values and error, don't compare correlations
   // because for very small correlations it's usually not the same withing the
   // relative tolerance because you would compare two small values that are
   // only different from zero because of noise.
   EXPECT_TRUE(result->isIdenticalNoCov(*resultRef, tol, tol));
   EXPECT_TRUE(resultAd->isIdenticalNoCov(*resultRef, tol, tol));
}

/// Initial minimization that was not based on any other tutorial/test.
FactoryTestParams param1{"Gaussian",
                         [](RooWorkspace &ws) {
                            ws.factory("sum::mu_shifted(mu[0, -10, 10], shift[1.0, -10, 10])");
                            ws.factory("prod::sigma_scaled(sigma[3.0, 0.01, 10], 1.5)");
                            ws.factory("Gaussian::model(x[0, -10, 10], mu_shifted, sigma_scaled)");

                            ws.defineSet("observables", "x");
                         },
                         [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &) {
                            return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, RooFit::BatchMode("cpu"))};
                         },
                         1e-4,
                         /*randomizeParameters=*/false};

/// Test based on the rf301 tutorial.
FactoryTestParams param2{"PolyVar",
                         [](RooWorkspace &ws) {
                            ws.factory("PolyVar::fy(y[-5, 5], {a0[-0.5, -5, 5], a1[-0.5, -1, 1], y})");
                            ws.factory("Gaussian::model(x[-5, 5], fy, sigma[0.5, 0.01, 10])");

                            ws.defineSet("observables", "x,y");
                         },
                         [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &ws) {
                            using namespace RooFit;
                            RooRealVar &y = *ws.var("y");
                            return std::unique_ptr<RooAbsReal>{
                               pdf.createNLL(data, ConditionalObservables(y), BatchMode("cpu"))};
                         },
                         1e-4,
                         /*randomizeParameters=*/false};

/// Test based on the rf201 tutorial.
FactoryTestParams param3{"AddPdf",
                         [](RooWorkspace &ws) {
                            ws.factory("Gaussian::sig1(x[0, 10], mean[5, -10, 10], sigma1[0.50, .01, 10])");
                            ws.factory("Gaussian::sig2(x, mean, sigma2[1.0, .01, 10])");
                            ws.factory("Chebychev::bkg(x, {a0[0.3, 0., 0.5], a1[0.2, 0., 0.5]})");
                            ws.factory("SUM::sig(sig1frac[0.8, 0.0, 1.0] * sig1, sig2)");
                            ws.factory("SUM::model(bkgfrac[0.5, 0.0, 1.0] * bkg, sig)");

                            ws.defineSet("observables", "x");
                         },
                         [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &) {
                            return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, RooFit::BatchMode("cpu"))};
                         },
                         5e-3,
                         /*randomizeParameters=*/true};

/// Test based on the rf604 tutorial.
FactoryTestParams param4{"ConstraintSum",
                         [](RooWorkspace &ws) {
                            ws.factory("RealSumFunc::mu_func({mu[-1, -10, 10], 4.0, 5.0}, {1.1, 0.3, 0.2})");
                            ws.factory("Gaussian::gauss(x[-10, 10], mu_func, sigma[2, 0.1, 10])");
                            ws.factory("Polynomial::poly(x)");
                            ws.factory("SUM::model(f[0.5, 0.0, 1.0] * gauss, poly)");
                            ws.factory("Gaussian::fconstext(f, 0.2, 0.1)");

                            ws.defineSet("observables", "x");
                         },
                         [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &ws) {
                            using namespace RooFit;
                            return std::unique_ptr<RooAbsReal>{
                               pdf.createNLL(data, ExternalConstraints(*ws.pdf("fconstext")), BatchMode("cpu"))};
                         },
                         1e-4,
                         /*randomizeParameters=*/true};

namespace {

std::unique_ptr<RooAbsPdf> createSimPdfModel(RooRealVar &x, std::string const &channelName)
{
   auto prefix = [&](const char *name) { return name + std::string("_") + channelName; };

   RooRealVar c(prefix("c").c_str(), "c", -0.5, -0.8, 0.2);

   RooExponential expo(prefix("expo").c_str(), "expo", x, c);

   // Create two Gaussian PDFs g1(x,mean1,sigma) anf g2(x,mean2,sigma) and their parameters
   RooRealVar mean1(prefix("mean1").c_str(), "mean of gaussians", 3, 0, 5);
   RooRealVar sigma1(prefix("sigma1").c_str(), "width of gaussians", 0.8, .01, 3.0);
   RooRealVar mean2(prefix("mean2").c_str(), "mean of gaussians", 6, 5, 10);
   RooRealVar sigma2(prefix("sigma2").c_str(), "width of gaussians", 1.0, .01, 3.0);

   RooGaussian sig1(prefix("sig1").c_str(), "Signal component 1", x, mean1, sigma1);
   RooGaussian sig2(prefix("sig2").c_str(), "Signal component 2", x, mean2, sigma2);

   // Sum the signal components
   RooRealVar sig1frac(prefix("sig1frac").c_str(), "fraction of signal 1", 0.5, 0.0, 1.0);
   RooAddPdf sig(prefix("sig").c_str(), "g1+g2", {sig1, sig2}, {sig1frac});

   // Sum the composite signal and background
   RooRealVar sigfrac(prefix("sigfrac").c_str(), "fraction of signal", 0.4, 0.0, 1.0);
   RooAddPdf model(prefix("model").c_str(), "g1+g2+a", {sig, expo}, {sigfrac});

   return std::unique_ptr<RooAbsPdf>{static_cast<RooAbsPdf *>(model.cloneTree())};
}

void getSimPdfModel(RooWorkspace &ws)
{
   using namespace RooFit;
   RooCategory channelCat{"channel_cat", ""};

   std::map<std::string, RooAbsPdf *> pdfMap;
   std::map<std::string, std::unique_ptr<RooAbsData>> dataMap;

   RooArgSet models;
   RooArgSet observables;

   auto nChannels = 2;
   auto nEvents = 1000;

   for (int i = 0; i < nChannels; ++i) {
      std::string suffix = "_" + std::to_string(i + 1);
      auto obsName = "x" + suffix;
      auto x = std::make_unique<RooRealVar>(obsName.c_str(), obsName.c_str(), 0, 10.);
      x->setBins(20);

      std::unique_ptr<RooAbsPdf> model{createSimPdfModel(*x, std::to_string(i + 1))};

      pdfMap.insert({"channel" + suffix, model.get()});
      channelCat.defineType("channel" + suffix, i);
      dataMap.insert({"channel" + suffix, std::unique_ptr<RooAbsData>{model->generateBinned(*x, nEvents)}});

      observables.addOwned(std::move(x));
      models.addOwned(std::move(model));
   }

   RooSimultaneous model{"model", "model", pdfMap, channelCat};

   ws.import(RooDataSet("data", "data", {observables, channelCat}, Index(channelCat), Import(dataMap)));

   ws.import(model);
   ws.defineSet("observables", observables);
}
} // namespace

/// Test based on the simultaneous fit shown in CHEP'23 results
FactoryTestParams param5{"SimPdf", getSimPdfModel,
                         [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &) {
                            return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, RooFit::BatchMode("cpu"))};
                         },
                         5e-3,
                         /*randomizeParameters=*/true};

FactoryTestParams param6{"GaussianExtended",
                         [](RooWorkspace &ws) {
                            ws.factory("Gaussian::gauss(x[0, -10, 10], mu[0, -10, 10], sigma[3.0, 0.01, 10])");
                            ws.factory("ExtendPdf::model(gauss, n[100, 0, 10000])");
                            ws.defineSet("observables", "x");
                         },
                         [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &) {
                            return std::unique_ptr<RooAbsReal>{
                               pdf.createNLL(data, RooFit::BatchMode("cpu"), RooFit::Extended(true))};
                         },
                         1e-4,
                         /*randomizeParameters=*/false};
namespace {
void getDataHistModel(RooWorkspace &ws)
{
   RooRealVar x("x", "x", 6, 0, 20);
   RooPolynomial p("p", "p", x, RooArgList(0.01, -0.01, 0.0004));

   // Sample 500 events from p
   x.setBins(10);
   std::unique_ptr<RooDataSet> data1{p.generate(x, 500)};

   // Create a binned dataset with 10 bins and 500 events
   std::unique_ptr<RooDataHist> hist1{p.generateBinned(x, 500)};

   // Represent data in dh as pdf in x
   RooHistPdf histpdf("histpdf", "histpdf", x, *hist1, 0);

   RooRealVar mean("mean", "mean of gaussian", 6, 5, 10);
   RooRealVar sigma("sigma", "width of gaussian", 1.0, .01, 3.0);

   RooGaussian gauss("gauss", "gauss", x, mean, sigma);
   RooRealVar frac("frac", "faction of histpdf", 0.5, 0, 1);
   RooAddPdf model("model", "model", {histpdf, gauss}, frac);

   ws.import(model);
   ws.defineSet("observables", {x});
}
} // namespace

/// Test based on rf706 tutorial
FactoryTestParams param7{"HistPdf", getDataHistModel,
                         [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &) {
                            return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, RooFit::BatchMode("cpu"))};
                         },
                         1e-4,
                         /*randomizeParameters=*/true};

namespace {
void getDataHistFuncModel(RooWorkspace &ws)
{
   using namespace RooFit;

   int nEvents = 1000;
   int numBins = 5;

   ws.factory("x[0, 0, " + std::to_string(3 * numBins) + "]");

   auto &x = *ws.var("x");
   x.setBins(numBins);

   // The parameters represent the expected events in each bin
   RooArgList params = ParamHistFunc::createParamSet(ws, "gamma", x, 0, 1000);
   for (auto *param : static_range_cast<RooRealVar *>(params)) {
      param->setVal(nEvents / numBins);
   }
   // Constructing a bin width function is a bit tedious. But including it is
   // required to use the binned likelihood optimization.
   RooDataHist dataHist{"data_hist", "data_hist", x};
   RooHistFunc histFunc{"hist_func", "hist_func", x, dataHist};
   RooBinWidthFunction bwf{"bwf", "bwf", histFunc, true};
   ParamHistFunc phf{"phf", "phf", x, params};
   RooProduct prod{"prod", "prod", phf, bwf};
   RooRealSumPdf pdf{"pdf", "pdf", prod, RooArgList{1.0}};

   // Enable the binned likelihood optimization to avoid integrals
   // (like in HistFactory).
   pdf.setAttribute("BinnedLikelihood");

   ws.import(pdf);

   // We have to wrap the pdf in a RooSimultaneous, otherwise the
   // BinnedLikelihood optimization doesn't work.
   ws.factory("SIMUL::model( cat[A=0], A=pdf)");
   auto &model = *ws.pdf("model");
   auto &cat = *ws.cat("cat");

   std::unique_ptr<RooDataHist> data{model.generateBinned({x, cat}, nEvents)};
   data->SetName("data");
   ws.import(*data);

   ws.defineSet("observables", *model.getObservables(*data));
}
} // namespace

FactoryTestParams param8{"HistFuncPdf", getDataHistFuncModel,
                         [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &) {
                            return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, RooFit::BatchMode("cpu"))};
                         },
                         1e-4,
                         /*randomizeParameters=*/true};

FactoryTestParams param9{"Poisson",
                         [](RooWorkspace &ws) {
                            ws.factory("sum::mu_shifted(mu[0, -10, 10], shift[1.0, -10, 10])");
                            ws.factory("Poisson::model(x[5, -10, 10], mu_shifted)");
                            ws.defineSet("observables", "x");
                         },
                         [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &) {
                            return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, RooFit::BatchMode("cpu"))};
                         },
                         1e-4,
                         /*randomizeParameters=*/false};

// A RooPoisson where x is not rounded, like it is used in HistFactory
FactoryTestParams param10{"PoissonNoRounding",
                          [](RooWorkspace &ws) {
                             ws.factory("sum::mu_shifted(mu[0, -10, 10], shift[1.0, -10, 10])");
                             ws.factory("Poisson::model(x[5, -10, 10], mu_shifted)");
                             auto poisson = static_cast<RooPoisson *>(ws.pdf("model"));
                             poisson->setNoRounding(true);
                             ws.defineSet("observables", "x");
                          },
                          [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &) {
                             return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, RooFit::BatchMode("cpu"))};
                          },
                          1e-4,
                          /*randomizeParameters=*/false};

INSTANTIATE_TEST_SUITE_P(RooFuncWrapper, FactoryTest,
                         testing::Values(param1, param2, param3, param4, param5, param6, param7, param8, param9,
                                         param10),
                         [](testing::TestParamInfo<FactoryTest::ParamType> const &paramInfo) {
                            return paramInfo.param._name;
                         });

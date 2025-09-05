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

#include <RooAbsData.h>
#include <RooAbsPdf.h>
#include <RooAddPdf.h>
#include <RooBinWidthFunction.h>
#include <RooCategory.h>
#include <RooClassFactory.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooExponential.h>
#include <RooFitResult.h>
#include <../src/RooEvaluatorWrapper.h>
#include <RooGaussian.h>
#include <RooHelpers.h>
#include <RooHistFunc.h>
#include <RooHistPdf.h>
#include <RooMinimizer.h>
#include <RooPoisson.h>
#include <RooPolynomial.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooWorkspace.h>

#include <ROOT/StringUtils.hxx>
#include <TROOT.h>
#include <TSystem.h>
#include <TMath.h>

#include <functional>
#include <random>

#include "gtest_wrapper.h"

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

using CreateNLLFunc =
   std::function<std::unique_ptr<RooAbsReal>(RooAbsPdf &, RooAbsData &, RooWorkspace &, RooFit::EvalBackend)>;
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

std::unique_ptr<RooFitResult> runMinimizer(RooAbsReal &absReal, bool useGradient = true)
{
   RooMinimizer::Config cfg;
   cfg.useGradient = useGradient;
   RooMinimizer m{absReal, cfg};
   m.setPrintLevel(-1);
   m.setStrategy(0);
   m.minimize("Minuit2");
   return std::unique_ptr<RooFitResult>{m.save()};
}

TEST_P(FactoryTest, NLLFit)
{

   RooWorkspace ws;

   std::unique_ptr<RooAbsData> ownedData;
   _params._setupWorkspace(ws);
   RooArgSet const &observables = *ws.set("observables");
   RooAbsData *data = ws.data("data");
   RooAbsPdf &model = *ws.pdf("model");

   std::size_t nEvents = 100;
   if (!data) {
      std::unique_ptr<RooDataSet> data0{model.generate(observables, nEvents)};
      ownedData = std::unique_ptr<RooAbsData>{data0->binnedClone()};
      data = ownedData.get();
   }

   std::unique_ptr<RooAbsReal> nllRef = _params._createNLL(model, *data, ws, RooFit::EvalBackend::Cpu());
   std::unique_ptr<RooAbsReal> nllFunc = _params._createNLL(model, *data, ws, RooFit::EvalBackend::Codegen());

   // We want to use the generated code also for the nominal likelihood. Like
   // this, we make sure to validate also the NLL values of the generated code.
   static_cast<RooEvaluatorWrapper &>(*nllFunc).setUseGeneratedFunctionCode(true);

   double tol = _params._fitResultTolerance;

   EXPECT_NEAR(nllRef->getVal(observables), nllFunc->getVal(), tol);

   // Check if the parameter layout and size is the same.
   RooArgSet paramsRefNll;
   nllRef->getParameters(nullptr, paramsRefNll);
   RooArgSet paramsMyNLL;
   nllFunc->getParameters(&observables, paramsMyNLL);

   if (_params._randomizeParameters) {
      randomizeParameters(paramsMyNLL);
      // Check if functions results are the same even after changing parameters.
      EXPECT_NEAR(nllRef->getVal(observables), nllFunc->getVal(), tol);
   }

   EXPECT_TRUE(paramsMyNLL.hasSameLayout(paramsRefNll));
   EXPECT_EQ(paramsMyNLL.size(), paramsRefNll.size());

   // Get AD based derivative
   std::vector<double> dMyNLL(paramsMyNLL.size(), 0);
   nllFunc->gradient(dMyNLL.data());

   // Check if derivatives are equal
   for (std::size_t i = 0; i < paramsMyNLL.size(); ++i) {
      EXPECT_NEAR(getNumDerivative(*nllRef, static_cast<RooRealVar &>(*paramsMyNLL[i]), observables), dMyNLL[i], tol);
   }

   // Remember parameter state before minimization
   RooArgSet parametersOrig;
   paramsRefNll.snapshot(parametersOrig);

   // Minimize the RooFuncWrapper Implementation
   auto result = runMinimizer(*nllFunc, false);
   paramsRefNll.assign(parametersOrig);

   // Minimize the RooFuncWrapper Implementation with AD
   auto resultAd = runMinimizer(*nllFunc);
   paramsRefNll.assign(parametersOrig);

   // Minimize the reference NLL
   auto resultRef = runMinimizer(*nllRef);
   paramsRefNll.assign(parametersOrig);

   // Compare minimization results
   // Same tolerance for parameter values and error, don't compare correlations
   // because for very small correlations it's usually not the same within the
   // relative tolerance because you would compare two small values that are
   // only different from zero because of noise.
   EXPECT_TRUE(result->isIdenticalNoCov(*resultRef, tol, tol));
   EXPECT_TRUE(resultAd->isIdenticalNoCov(*resultRef, tol, tol));
}

/// Initial minimization that was not based on any other tutorial/test.
FactoryTestParams param1{"Gaussian",
                         [](RooWorkspace &ws) {
                            constexpr double inf = std::numeric_limits<double>::infinity();
                            ws.import(RooRealVar{"mu", "mu", -inf, inf});
                            ws.factory("sum::mu_shifted(mu, shift[1.0, -10, 10])");
                            ws.factory("prod::sigma_scaled(sigma[3.0, 0.01, 10], 1.5)");
                            ws.factory("Gaussian::model(x[0, -10, 10], mu_shifted, sigma_scaled)");

                            ws.defineSet("observables", "x");
                         },
                         [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &, RooFit::EvalBackend backend) {
                            using namespace RooFit;
                            return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, backend)};
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
                         [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &ws, RooFit::EvalBackend backend) {
                            using namespace RooFit;
                            RooRealVar &y = *ws.var("y");
                            return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, ConditionalObservables(y), backend)};
                         },
                         1e-4,
                         /*randomizeParameters=*/false};

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
                         [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &ws, RooFit::EvalBackend backend) {
                            using namespace RooFit;
                            return std::unique_ptr<RooAbsReal>{
                               pdf.createNLL(data, ExternalConstraints(*ws.pdf("fconstext")), backend)};
                         },
                         1e-4,
                         /*randomizeParameters=*/true};

namespace {

std::unique_ptr<RooAbsPdf> createSimPdfModel(RooRealVar &x, std::string const &channelName)
{
   auto prefix = [&](const char *name) { return name + std::string("_") + channelName; };

   RooRealVar c(prefix("c").c_str(), "c", -0.5, -0.8, 0.2);

   RooExponential expo(prefix("expo").c_str(), "expo", x, c);

   // Create two Gaussian PDFs g1(x,mean1,sigma) and g2(x,mean2,sigma) and their parameters
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
                         [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &, RooFit::EvalBackend backend) {
                            return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, backend)};
                         },
                         5e-3,
                         /*randomizeParameters=*/true};

FactoryTestParams param6{"GaussianExtended",
                         [](RooWorkspace &ws) {
                            ws.factory("Gaussian::gauss(x[0, -10, 10], mu[0, -10, 10], sigma[3.0, 0.01, 10])");
                            ws.factory("ExtendPdf::model(gauss, n[100, 0, 10000])");
                            ws.defineSet("observables", "x");
                         },
                         [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &, RooFit::EvalBackend backend) {
                            return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, backend)};
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
                         [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &, RooFit::EvalBackend backend) {
                            return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, backend)};
                         },
                         1e-4,
                         /*randomizeParameters=*/true};

FactoryTestParams param8{"Lognormal",
                         [](RooWorkspace &ws) {
                            ws.factory("Lognormal::model(x[1.0, 1.1, 10], mu[2.0, 1.1, 10], k[2.0, 1.1, 5.0])");
                            ws.defineSet("observables", "x");
                         },
                         [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &, RooFit::EvalBackend backend) {
                            return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, backend)};
                         },
                         1e-4,
                         /*randomizeParameters=*/true};

FactoryTestParams param8p1{"LognormalStandard",
                           [](RooWorkspace &ws) {
                              ws.factory(
                                 "Lognormal::model(x[1.0, 1.1, 10], mu[0.7, 0.1, 2.3], k[0.7, 0.1, 0.95], true)");
                              ws.defineSet("observables", "x");
                           },
                           [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &, RooFit::EvalBackend backend) {
                              return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, backend)};
                           },
                           3e-4,
                           /*randomizeParameters=*/true};

FactoryTestParams param9{"Poisson",
                         [](RooWorkspace &ws) {
                            ws.factory("Poisson::model(x[5, 0, 10], mu[5, 0, 10])");
                            ws.defineSet("observables", "x");
                         },
                         [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &, RooFit::EvalBackend backend) {
                            return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, backend)};
                         },
                         1e-4,
                         /*randomizeParameters=*/true};

// A RooPoisson where x is not rounded, like it is used in HistFactory
FactoryTestParams param10{"PoissonNoRounding",
                          [](RooWorkspace &ws) {
                             constexpr double inf = std::numeric_limits<double>::infinity();
                             RooRealVar mu{"mu", "mu", 5., 0., inf};
                             ws.import(mu);
                             ws.factory("Poisson::model(x[5, 0, 10], mu)");
                             auto poisson = static_cast<RooPoisson *>(ws.pdf("model"));
                             poisson->setNoRounding(true);
                             ws.defineSet("observables", "x");
                          },
                          [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &, RooFit::EvalBackend backend) {
                             return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, backend)};
                          },
                          1e-4,
                          /*randomizeParameters=*/true};

FactoryTestParams param11{"ClassFactory1D",
                          [](RooWorkspace &ws) {
                             RooRealVar x{"x", "x", 4.0, 0, 10};
                             RooRealVar mu{"mu", "mu", 5, 0, 10};
                             RooRealVar sigma{"sigma", "sigma", 2.0, 0.1, 10};

                             // TODO: When Clad issue #635 is solved, we can
                             // actually use a complete Gaussian here, also
                             // with sigma.
                             std::unique_ptr<RooAbsPdf> pdf{RooClassFactory::makePdfInstance(
                                //"model", "std::exp(-0.5 * (x - mu)*(x - mu) / (sigma * sigma))", {x, mu, sigma})};
                                "model", "std::exp(-0.5 * (x - mu)*(x - mu))", {x, mu})};
                             ws.import(*pdf);
                             ws.defineSet("observables", "x");
                          },
                          [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &, RooFit::EvalBackend backend) {
                             return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, backend)};
                          },
                          5e-3, // increase tolerance because the numeric integration algos are still different
                          /*randomizeParameters=*/true};

FactoryTestParams makeTestParams(const char *name, std::vector<std::string> const &expressions,
                                 double fitResultTolerance, bool randomizeParameters = true)
{
   return {name,
           [=](RooWorkspace &ws) {
              for (std::string const &expr : expressions) {
                 ws.factory(expr.c_str());
              }
              ws.defineSet("observables", "x");
           },
           [](RooAbsPdf &pdf, RooAbsData &data, RooWorkspace &, RooFit::EvalBackend backend) {
              using namespace RooFit;
              return std::unique_ptr<RooAbsReal>{pdf.createNLL(data, backend)};
           },
           fitResultTolerance, randomizeParameters};
}

auto testValues = testing::Values(
   param1, param2, param4, param5, param6, param7, param8, param8p1, param9, param10, param11,
   makeTestParams(
      "BifurGauss",
      {"x[0, -10, 10]", "mu[0, -10, 10]", "BifurGauss::model(x, mu, sigmaL[3.0, 0.01, 10], sigmaR[2.0, 0.01, 10])"},
      1e-4, false),
   makeTestParams("RooFormulaVar",
                  {"expr::mu_shifted('mu+shift',{mu[0, -10, 10], shift[1.0, -10, 10]})",
                   "expr::sigma_scaled('sigma*1.5',{sigma[3.0, 0.01, 10]})",
                   "Gaussian::model(x[0, -10, 10], mu_shifted, sigma_scaled)"},
                  1e-4, false),
   // Test for the uniform pdf. Since it doesn't depend on any parameters, we need
   // to add it to some other model like a Gaussian to get a meaningful fit model.
   makeTestParams("Uniform",
                  {"Gaussian::sig(x[0, 10], mean[5, -10, 10], sigma1[0.50, .01, 10])", "Uniform::bkg(x)",
                   "SUM::model(bkgfrac[0.5, 0.0, 1.0] * bkg, sig)"},
                  5e-3, true),
   // Test for RooRecursiveFraction.
   makeTestParams("RecursiveFraction",
                  {"Gaussian::sig1(x[0, 10], 5.0, sigma1[0.50, .01, 10])",
                   "Gaussian::sig2(x, 2.0, sigma2[1.0, .01, 10])", "Gaussian::sig3(x, 7.0, sigma3[1.5, .01, 10])",
                   "Gaussian::sig4(x, 6.0, sigma4[2.0, .01, 10])",
                   "RecursiveFraction::recfrac({a1[0.25, 0.0, 1.0], a2[0.25, 0.0, 1.0]})",
                   "SUM::model(a1 * sig1, a2 * sig2, recfrac * sig3, sig4)"},
                  5e-3, true),
   makeTestParams("RooCBShape",
                  {"x[0., -200., 200.]", "x0[100., -200., 200.]",
                   "CBShape::model(x, x0, sigma[2., 1.E-6, 100.], alpha[1., 1.E-6, 100.], n[1., 1.E-6, 100.])"},
                  6e-3, true),
   makeTestParams("RooBernstein",
                  {"Bernstein::model(x[0., 100.], {c0[0.3, 0., 10.], c1[0.7, 0., 10.], c2[0.2, 0., 10.]})"}, 6e-3,
                  true),
   // We're testing several Landau configurations, because the underlying
   // ROOT::Math::landau_cdf is defined piecewise. Like this, we're covering
   // all possible code paths in the pullback.
   makeTestParams("RooLandau1", {"Landau::model(x[5., 0., 30.], ml[6., 1., 30.], sl[1., 0.01, 50.])"}, 6e-3, false),
   makeTestParams("RooLandau2", {"Landau::model(x[5., 0., 30.], ml[6., 1., 30.], sl[2.1, 0.01, 50.])"}, 6e-3, false),
   makeTestParams("RooLandau3", {"Landau::model(x[5., 0., 30.], ml[6., 1., 30.], sl[10., 0.01, 50.])"}, 6e-3, false),
   makeTestParams("RooLandau4", {"Landau::model(x[5., 0., 30.], ml[6., 1., 30.], sl[0.3, 0.01, 50.])"}, 6e-3, false),
   makeTestParams("RooLandau5", {"Landau::model(x[5., 0., 30.], ml[6., 1., 30.], sl[0.07, 0.01, 50.])"}, 6e-3, false),
   makeTestParams(
      "RooRealSumPdf1",
      {"Gaussian::gx(x[-10,10],m[0],1.0)", "Chebychev::ch(x,{0.1,0.2,-0.3})", "RealSumPdf::model({gx, ch}, {f[0,1]})"},
      6e-3, false),
   // Express a Gaussian as a RooFormulaVar/expr and put in a RooWrapper pdf to
   // get test coverage for RooWrapperPdf.
   makeTestParams("RooWrapperPdf",
                  {"x[-10., 10.]", "mean[1., -10., 10.]", "sigma[1., 0.1, 10.]",
                   "expr::gauss_func('std::exp(-0.5*(x - mean) * (x - mean) / (sigma * sigma))', {x, mean, sigma})",
                   "WrapperPdf::model(gauss_func)"},
                  6e-3, true));

INSTANTIATE_TEST_SUITE_P(RooFuncWrapper, FactoryTest, testValues,
                         [](testing::TestParamInfo<FactoryTest::ParamType> const &paramInfo) {
                            return paramInfo.param._name;
                         });

// Tests for the RooProdPdf
// Authors: Stephan Hageboeck, CERN  02/2019
//          Jonas Rembser, CERN, June 2021

#include <RooAddPdf.h>
#include <RooArgList.h>
#include <RooArgSet.h>
#include <RooConstVar.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooGenericPdf.h>
#include <RooProdPdf.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>
#include <RooHelpers.h>
#include <RooRandom.h>

#include <Math/PdfFuncMathCore.h>

#include "gtest_wrapper.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

TEST(RooProdPdf, TestGetPartIntList)
{
   RooHelpers::LocalChangeMsgLevel chmsglvl1{RooFit::ERROR, 0u, RooFit::InputArguments, true};
   RooHelpers::LocalChangeMsgLevel chmsglvl2{RooFit::WARNING, 0u, RooFit::NumericIntegration, true};

   // This test checks if RooProdPdf::getPartIntList factorizes the integrals
   // as expected, for the example of a three dimensional RooProdPdf.

   RooWorkspace ws;

   auto &x = static_cast<RooRealVar &>(*ws.factory("x[0, 0, 10.]"));
   auto &y = static_cast<RooRealVar &>(*ws.factory("y[0, 0, 4.]"));
   auto &z = static_cast<RooRealVar &>(*ws.factory("z[0, 0, 2.5]"));

   double a = x.getMax();
   double b = y.getMax();
   double c = z.getMax();

   // Factorize the product in one 1D and one 2D pdf to get a more complicated
   // and complete test case.
   ws.factory("Uniform::pdf1({x})");
   ws.factory("Uniform::pdf2({y, z})");

   // Product of all the pdfs.
   auto &prod = static_cast<RooProdPdf &>(*ws.factory("PROD::prod(pdf1, pdf2)"));

   RooArgSet normSetNada{};
   RooArgSet normSetX{x};
   RooArgSet normSetY{y};
   RooArgSet normSetZ{z};
   RooArgSet normSetXY{x, y};
   RooArgSet normSetXZ{x, z};
   RooArgSet normSetYZ{y, z};
   RooArgSet normSetXYZ{x, y, z};

   EXPECT_DOUBLE_EQ(prod.getVal(normSetNada), 1.0);
   EXPECT_DOUBLE_EQ(prod.getVal(normSetX), 1. / a);
   EXPECT_DOUBLE_EQ(prod.getVal(normSetY), 1. / b);
   EXPECT_DOUBLE_EQ(prod.getVal(normSetZ), 1. / c);
   EXPECT_DOUBLE_EQ(prod.getVal(normSetXY), 1. / a / b);
   EXPECT_DOUBLE_EQ(prod.getVal(normSetXZ), 1. / a / c);
   EXPECT_DOUBLE_EQ(prod.getVal(normSetYZ), 1. / b / c);
   EXPECT_DOUBLE_EQ(prod.getVal(normSetXYZ), 1. / a / b / c);
}

TEST(RooProdPdf, TestDepsAreCond)
{
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::Minimization, true};

   using namespace RooFit;

   RooWorkspace ws;

   ws.factory("GaussModel::gm(x[0, 1], 0.0, xErr[0.0001, 0.1])");
   ws.factory("Decay::decayPdf(x, tau[0.4, 0, 1], gm, RooDecay::SingleSided)");
   ws.factory("Gamma::errPdf(xErr, 4.0, 0.005, 0.0)");

   RooRealVar &x = *ws.var("x");
   RooRealVar &xErr = *ws.var("xErr");
   RooRealVar &tau = *ws.var("tau");
   RooAbsPdf &decayPdf = *ws.pdf("decayPdf");
   RooAbsPdf &errPdf = *ws.pdf("errPdf");

   // What we want: decayPdf(x|xErr)*errPdf(xErr):
   RooProdPdf pdf1("pdf1", "", {errPdf}, Conditional(decayPdf, x, false));

   // Should be the same as pdf1:
   RooProdPdf pdf2("pdf2", "", {errPdf}, Conditional(decayPdf, xErr, true));

   std::unique_ptr<RooDataSet> data{pdf1.generate({x, xErr}, NumEvents(10000))};

   auto resetParameters = [&]() {
      tau.setVal(0.4);
      tau.setError(0.0);
   };

   using ResultPtr = std::unique_ptr<RooFitResult>;

   resetParameters();
   ResultPtr result2{pdf1.fitTo(*data, Save(), EvalBackend::Cpu(), PrintLevel(-1))};
   resetParameters();
   ResultPtr result4{pdf2.fitTo(*data, Save(), EvalBackend::Cpu(), PrintLevel(-1))};

   EXPECT_TRUE(result4->isIdentical(*result2)) << "alternative model fit is inconsistent!";

#ifdef ROOFIT_LEGACY_EVAL_BACKEND
   resetParameters();
   ResultPtr result1{pdf1.fitTo(*data, Save(), EvalBackend(EvalBackend::Value::Legacy), PrintLevel(-1))};
   resetParameters();
   ResultPtr result3{pdf2.fitTo(*data, Save(), EvalBackend(EvalBackend::Value::Legacy), PrintLevel(-1))};

   EXPECT_TRUE(result2->isIdentical(*result1)) << "legacy fit is inconsistent!";
   EXPECT_TRUE(result4->isIdentical(*result1)) << "alternative model legacy fit is inconsistent!";
#endif
}

// The RooGenProdProj that normalizes factors with overlapping normalization
// observables must integrate the unnormalized pdfs. Then, the cpu backend
// agrees with the scalar evaluation to high precision, also if the
// normalization of a factor depends on an observable. Covers the HS3 test
// suite model rf305_condcorrprod.
TEST(RooProdPdf, NonFactorizingProductLikelihood)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooRandom::randomGenerator()->SetSeed(1337);

   RooWorkspace ws;
   ws.factory("Gaussian::gaussy(y[-5, 5], 0.0, 3.0)");
   ws.factory("Gaussian::gaussx(x[-5, 5], PolyVar::fy(y, {a0[-1.5, -5, 5], a1[-0.5, -1, 1]}), sigma[0.5, 0.1, 3])");
   ws.factory("PROD::joint(gaussy, gaussx)");
   // Here, gaussx is normalized over y only, so its normalization depends on x.
   ws.factory("PROD::condOnX(gaussy, gaussx|x)");

   RooRealVar &x = *ws.var("x");
   RooRealVar &y = *ws.var("y");
   RooRealVar &a0 = *ws.var("a0");

   for (const char *name : {"joint", "condOnX"}) {
      RooAbsPdf &model = *ws.pdf(name);
      RooArgSet obs{x, y};

      std::unique_ptr<RooDataSet> data{model.generate(obs, 1000)};
      std::unique_ptr<RooAbsReal> nll{model.createNLL(*data, RooFit::EvalBackend::Cpu())};

      auto scalarNll = [&]() {
         double out = 0.0;
         for (int i = 0; i < data->numEntries(); ++i) {
            RooArgSet const *row = data->get(i);
            x.setVal(row->getRealValue("x"));
            y.setVal(row->getRealValue("y"));
            out -= std::log(model.getVal(obs));
         }
         return out;
      };

      for (double a0Val : {-1.5, 0.0, -1.5}) {
         a0.setVal(a0Val);
         const double ref = scalarNll();
         EXPECT_NEAR(nll->getVal(), ref, 1e-10 * std::abs(ref)) << name << ", a0 = " << a0Val;
      }
   }
}

namespace {

/// Checks a likelihood value with the cpu and codegen backends, and, if Clad
/// is available, the codegen gradient against numeric derivatives.
void checkLikelihoodBackends(std::function<std::unique_ptr<RooAbsReal>(RooFit::EvalBackend)> const &createNLL,
                             RooAbsData const &data, double ref, std::string const &label)
{
   // The underflow of the product must not cause evaluation errors anywhere.
   const auto prevErrorLoggingMode = RooAbsReal::evalErrorLoggingMode();
   RooAbsReal::setEvalErrorLoggingMode(RooAbsReal::CollectErrors);
   RooAbsReal::clearEvalErrorLog();

   std::unique_ptr<RooAbsReal> nll = createNLL(RooFit::EvalBackend::Cpu());
   EXPECT_NEAR(nll->getVal(), ref, 1e-10 * std::abs(ref)) << label << ", cpu backend";

   std::unique_ptr<RooAbsReal> nllCodegen = createNLL(RooFit::EvalBackend::CodegenNoGrad());
   EXPECT_NEAR(nllCodegen->getVal(), ref, 1e-10 * std::abs(ref)) << label << ", codegen backend";

#ifdef ROOFIT_CLAD
   std::unique_ptr<RooAbsReal> nllGrad = createNLL(RooFit::EvalBackend::Codegen());
   // Like in the wrapped function, all variables that are not in the dataset
   // are parameters, including the global observables.
   RooArgSet params;
   nllGrad->getParameters(data.get(), params);
   std::vector<double> gradient(params.size());
   nllGrad->gradient(gradient.data());
   for (std::size_t i = 0; i < params.size(); ++i) {
      auto &param = static_cast<RooRealVar &>(*params[i]);
      const double val = param.getVal();
      constexpr double eps = 1e-6;
      param.setVal(val + eps);
      const double up = nll->getVal();
      param.setVal(val - eps);
      const double down = nll->getVal();
      param.setVal(val);
      const double numDerivative = (up - down) / (2 * eps);
      EXPECT_NEAR(gradient[i], numDerivative, 1e-5 * std::max(std::abs(numDerivative), 1.0))
         << label << ", codegen gradient by " << param.GetName();
   }
#endif

   EXPECT_EQ(RooAbsReal::numEvalErrors(), 0) << label;
   RooAbsReal::clearEvalErrorLog();
   RooAbsReal::setEvalErrorLoggingMode(prevErrorLoggingMode);
}

} // namespace

/// The joint probability of a product of many pdfs, like an HS3 array
/// distribution, underflows, but the likelihood should still be finite. Covers
/// factors that can evaluate their logarithm directly (Gaussian), factors that
/// can't (Exponential), and a mix of both. Covers GitHub issue #24202.
TEST(RooProdPdf, ManyFactorsLikelihood)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   constexpr int nFactors = 100;

   for (std::string kind : {"Gaussian", "Exponential", "Mixed"}) {
      RooWorkspace ws;
      RooArgList factors;
      RooArgSet obs;
      for (int i = 0; i < nFactors; ++i) {
         const std::string idx = std::to_string(i);
         if (kind == "Gaussian" || (kind == "Mixed" && i % 2 == 0)) {
            ws.factory("Gaussian::f" + idx + "(x" + idx + "[0, -10, 10], theta[0, -2, 2], 1.0)");
         } else {
            ws.factory("Exponential::f" + idx + "(x" + idx + "[1, 0, 10], c[-1, -2, -0.5])");
         }
         factors.add(*ws.pdf("f" + idx));
         obs.add(*ws.var("x" + idx));
      }
      RooProdPdf prod{"prod", "prod", factors};

      RooRealVar w{"w", "w", 1.0};
      RooArgSet vars{obs};
      vars.add(w);
      RooDataSet data{"data", "data", vars, RooFit::WeightVar(w)};
      for (double val : {8.0, 9.0}) {
         for (auto *x : static_range_cast<RooRealVar *>(obs)) {
            x->setVal(val);
         }
         data.add(obs, val);
      }

      double ref = 0.0;
      for (int i = 0; i < data.numEntries(); ++i) {
         obs.assign(*data.get(i));
         for (auto *factor : static_range_cast<RooAbsPdf *>(factors)) {
            ref -= data.weight() * factor->getLogVal(&obs);
         }
      }

      auto createNLL = [&](RooFit::EvalBackend backend) {
         return std::unique_ptr<RooAbsReal>{prod.createNLL(data, backend)};
      };
      checkLikelihoodBackends(createNLL, data, ref, kind);
   }
}

/// Like ManyFactorsLikelihood, but with the product as a constraint term.
TEST(RooProdPdf, ManyFactorsConstraint)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   constexpr int nFactors = 100;

   RooWorkspace ws;
   ws.factory("Gaussian::model(x[0, -10, 10], theta[0.5, -2, 2], 1.0)");
   RooArgList constraints;
   RooArgSet globs;
   for (int i = 0; i < nFactors; ++i) {
      const std::string idx = std::to_string(i);
      ws.factory("Gaussian::c" + idx + "(g" + idx + "[8.5, -10, 10], theta, 1.0)");
      constraints.add(*ws.pdf("c" + idx));
      globs.add(*ws.var("g" + idx));
   }
   RooProdPdf constraint{"constraint", "constraint", constraints};

   RooRealVar &x = *ws.var("x");
   RooDataSet data{"data", "data", x};
   data.add(x);

   using namespace RooFit;
   RooAbsPdf &model = *ws.pdf("model");
   std::unique_ptr<RooAbsReal> nll{model.createNLL(data, EvalBackend::Cpu())};

   double ref = nll->getVal();
   for (auto *c : static_range_cast<RooAbsPdf *>(constraints)) {
      ref -= c->getLogVal(&globs);
   }

   auto createNLL = [&](RooFit::EvalBackend backend) {
      return std::unique_ptr<RooAbsReal>{
         model.createNLL(data, backend, ExternalConstraints(constraint), GlobalObservables(globs))};
   };
   checkLikelihoodBackends(createNLL, data, ref, "constraint");
}

/// This test covers a potential problem with the custom normalization ranges
/// of PDFs: the value of a RooProdPdf should not depend on the normalization
/// ranges of its clients. TODO: this is still a problem! It needs to be fixed,
/// and then the test can be enabled and this note should be removed.
TEST(RooProdPdf, DISABLED_ChangeServerNormSetForProdPdfInAddPdf)
{
   // Define the model
   RooWorkspace ws;
   ws.factory("PROD::sig(Gaussian(x[-10, 10], 1.0, 1.0), Gaussian(y[-10, 10], 1.0, 1.0))");
   ws.factory("PROD::bkg(Polynomial(x, {}), Polynomial(y, {}))");
   ws.factory("SUM::model(f[0.5, 0, 1] * sig, bkg)");
   ws.factory("SUM::modelFixed(f * sig, bkg)");
   ws.factory("EXPR::modelRef('f * 3.0/2000. * x * x + (1 - f) / 20.', {x, f})");

   RooRealVar &x = *ws.var("x");
   RooRealVar &y = *ws.var("y");

   RooAddPdf &model = static_cast<RooAddPdf &>(*ws.pdf("model"));

   // Define signal and sideband regions
   x.setRange("SB1", -10, +10);
   y.setRange("SB1", -10, 0);

   x.setRange("SB2", -10, 0);
   y.setRange("SB2", 0, +10);

   x.setRange("SIG", 0, +10);
   y.setRange("SIG", 0, +10);

   x.setRange("FULL", -10, +10);
   y.setRange("FULL", -10, +10);

   // Try different normalization sets to check if there is a false cache hit
   // after changing the normalization range of the servers.
   RooArgSet normSet1{x, y};
   RooArgSet normSet2{x, y};

   // The fit range needs to be a multi-range to trigger the problem that is covered by this test
   const char *fitRange = "SB1,SB2";

   // Set the normalization range for the top-level model
   model.setNormRange(fitRange);

   const double val1 = model.getVal(normSet1);

   // Now set the normalization range for all other PDFs, which should not
   // change the value of the top-level PDF.
   for (auto *pdf : static_range_cast<RooAbsPdf *>(ws.allPdfs())) {
      pdf->setNormRange(fitRange);
   }

   const double val2 = model.getVal(normSet1);
   const double val3 = model.getVal(normSet2);

   EXPECT_FLOAT_EQ(val2, val1);
   EXPECT_FLOAT_EQ(val3, val1);
}

// A RooGenProdProj normalizes factors with overlapping normalization
// observables together, integrating them with an empty normalization set. The
// cpu backend must agree with the scalar evaluation also if these integrals
// reach normalized pdfs inside composite factors, like a nested RooProdPdf or
// a RooExtendPdf. A jointly normalized product nested in another one is
// compared to the equivalent flat product instead.
TEST(RooProdPdf, JointlyNormalizedCompositeFactors)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooRandom::randomGenerator()->SetSeed(1337);

   RooWorkspace ws;
   ws.factory("Gaussian::gaussy(y[-5, 5], 0.0, 3.0)");
   ws.factory("Gaussian::gaussx(x[-5, 5], PolyVar::fy(y, {a0[-1.5, -5, 5], a1[-0.5, -1, 1]}), sigma[0.5, 0.1, 3])");
   ws.factory("Gaussian::gaussz(z[-5, 5], 0.0, 1.0)");
   ws.factory("PROD::nested(gaussy, PROD::inner(gaussx, gaussz))");
   ws.factory("PROD::extended(gaussy, ExtendPdf::gaussxExt(gaussx, n[1000]))");
   ws.factory("Gaussian::gaussw(w[-5, 5], PolyVar::fw(y, {0.0, b[0.3, -1, 1]}), 1.0)");
   ws.factory("PROD::flat(gaussy, gaussx, gaussw)");
   ws.factory("PROD::nestedFirst(PROD::innerFirst(gaussy, gaussx), gaussw)");

   RooRealVar &x = *ws.var("x");
   RooRealVar &y = *ws.var("y");
   RooRealVar &z = *ws.var("z");
   RooRealVar &w = *ws.var("w");
   RooRealVar &a0 = *ws.var("a0");
   RooRealVar &sigma = *ws.var("sigma");

   auto check = [&](RooAbsPdf &model, RooArgSet const &obs) {
      std::unique_ptr<RooDataSet> data{model.generate(obs, 1000)};
      std::unique_ptr<RooAbsReal> nll{model.createNLL(*data, RooFit::EvalBackend::Cpu(), RooFit::Extended(false))};

      auto scalarNll = [&]() {
         double out = 0.0;
         for (int i = 0; i < data->numEntries(); ++i) {
            RooArgSet const *row = data->get(i);
            for (RooAbsArg *arg : obs) {
               static_cast<RooRealVar *>(arg)->setVal(row->getRealValue(arg->GetName()));
            }
            out -= std::log(model.getVal(obs));
         }
         return out;
      };

      for (double a0Val : {-1.5, 0.0, -1.5}) {
         a0.setVal(a0Val);
         const double ref = scalarNll();
         EXPECT_NEAR(nll->getVal(), ref, 1e-6 * std::abs(ref)) << model.GetName() << ", a0 = " << a0Val;
      }
   };

   check(*ws.pdf("nested"), {x, y, z});
   check(*ws.pdf("extended"), {x, y});

   // The nested product only agrees with the flat product to about 1e-3. This
   // tolerance still catches integrals of the inner product that are
   // inconsistent with its value, which double the NLL.
   std::unique_ptr<RooDataSet> data{ws.pdf("flat")->generate({x, y, w}, 1000)};
   std::unique_ptr<RooAbsReal> nllFlat{ws.pdf("flat")->createNLL(*data, RooFit::EvalBackend::Cpu())};
   std::unique_ptr<RooAbsReal> nllNested{ws.pdf("nestedFirst")->createNLL(*data, RooFit::EvalBackend::Cpu())};
   for (double sigmaVal : {0.5, 1.0}) {
      sigma.setVal(sigmaVal);
      const double ref = nllFlat->getVal();
      EXPECT_NEAR(nllNested->getVal(), ref, 1e-2 * std::abs(ref)) << "sigma = " << sigmaVal;
   }
}

// Make sure that the pdf that provides the expected number of events in a
// RooProdPdf is not considered as a constraint. Covers JIRA issue ROOT-7604.
TEST(RooProdPdf, RooProdPdfWithExtendedTerm)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws;
   ws.factory("Gaussian::L_constraint(L_nom[10],L[10,0,20],0.2)");
   ws.factory("PROD::full_model( RooExtendedTerm::model( L ) , L_constraint )");

   ws.factory("weightVar[10,0,100]");
   ws.factory("dummy_obs[1,0,1]");
   ws.defineSet("obs", "dummy_obs,weightVar");
   RooDataSet data("data", "data", *ws.set("obs"), RooFit::WeightVar("weightVar"));
   data.add(*ws.set("obs"), 5);

   std::unique_ptr<RooAbsReal> nll1{ws.pdf("full_model")->createNLL(data)};
   double refVal = -std::log(ROOT::Math::gaussian_pdf(10, 0.2, 10)) + (10. - 5 * std::log(10.));
   double nll1Val = nll1->getVal();
   EXPECT_FLOAT_EQ(nll1Val, refVal);
}

// A RooProdPdf that just wraps a (extended) RooAddPdf must give the same result
// in a ranged fit as the bare RooAddPdf. In particular, the RooAddPdf yield must
// be reinterpreted with respect to the full range in both cases. This is the
// common workspace pattern of attaching constraint terms to a model via a
// RooProdPdf. Regression test for GitHub issue #16673.
TEST(RooProdPdf, RangedFitWrappingRooAddPdf)
{
   using namespace RooFit;
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooRandom::randomGenerator()->SetSeed(42);

   RooRealVar x("x", "x", 0, 100);
   x.setRange("LEFT", 0, 20);
   x.setRange("RIGHT", 60, 100);

   RooRealVar alphaGen("alpha", "alpha", -0.04, -0.1, -0.0);
   RooGenericPdf modelGen("model", "exp(alpha*x)", {x, alphaGen});
   std::unique_ptr<RooDataSet> data{modelGen.generate(x, 10000)};

   // A nuisance parameter with a Gaussian constraint. Its observable does not
   // know about the "LEFT"/"RIGHT" ranges, so the fit range must not be forced
   // onto the constraint pdf (otherwise the ranged normalization throws).
   RooRealVar theta("theta", "theta", 0, -5, 5);
   RooRealVar thetaGlob("thetaGlob", "thetaGlob", 0);
   RooRealVar thetaErr("thetaErr", "thetaErr", 1.0);
   RooGenericPdf constraint("constraint", "exp(-0.5*(theta-thetaGlob)^2/thetaErr^2)", {theta, thetaGlob, thetaErr});

   auto fitYield = [&](bool wrapInProd, bool withConstraint) {
      RooRealVar alpha("alpha", "alpha", -0.04, -0.1, -0.0);
      RooGenericPdf model("model", "exp(alpha*x)", {x, alpha});
      RooRealVar nBkg("Nbkg", "Nbkg", 10000, 0, 20000);
      RooAddPdf add("pdfadd", "", RooArgList{model}, RooArgList{nBkg});
      std::unique_ptr<RooAbsPdf> owned;
      RooAbsPdf *pdf = &add;
      if (wrapInProd) {
         RooArgList factors{add};
         if (withConstraint) {
            factors.add(constraint);
         }
         owned = std::make_unique<RooProdPdf>("pdfprod", "", factors);
         pdf = owned.get();
      }
      pdf->fitTo(*data, Range("LEFT,RIGHT"), PrintLevel(-1));
      return nBkg.getVal();
   };

   const double yieldAdd = fitYield(false, false);
   const double yieldProd = fitYield(true, false);
   const double yieldProdConstr = fitYield(true, true);

   // The yield must refer to the full range, i.e. be clearly larger than the
   // number of events actually inside the fit range.
   const double nInFitRange = std::unique_ptr<RooAbsData>{data->reduce(CutRange("LEFT,RIGHT"))}->sumEntries();
   EXPECT_GT(yieldAdd, 1.2 * nInFitRange);

   // Wrapping in a RooProdPdf (with or without a constraint term) must not
   // change the fitted yield.
   EXPECT_NEAR(yieldProd, yieldAdd, 1e-3 * yieldAdd);
   EXPECT_NEAR(yieldProdConstr, yieldAdd, 1e-3 * yieldAdd);
}

// Tests for the RooJSONFactoryWSTool
// Authors: Carsten D. Burgard, DESY/ATLAS, 12/2021
//          Jonas Rembser, CERN 12/2022

#include <RooFitHS3/JSONIO.h>
#include <RooFitHS3/RooJSONFactoryWSTool.h>

#include <RooAddPdf.h>
#include <RooCategory.h>
#include <RooConstVar.h>
#include <RooDataSet.h>
#include <RooExponential.h>
#include <RooGenericPdf.h>
#include <RooGaussian.h>
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooHistFunc.h>
#include <RooHistPdf.h>
#include <RooSpline.h>
#include <RooLognormal.h>
#include <RooMultiVarGaussian.h>
#include <RooPoisson.h>
#include <RooProdPdf.h>
#include <RooRealIntegral.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooWorkspace.h>
#include <RooFormulaVar.h>
#include <RooFit/ModelConfig.h>

#include <cmath>

#include <TROOT.h>

#include <gtest/gtest.h>

namespace {

// If the JSON files should be written out for debugging purpose.
const bool writeJsonFiles = true;

// Validate the JSON IO for a given RooAbsReal in a RooWorkspace. The workspace
// will be written out and read back, and then the values of the old and new
// RooAbsReal will be compared for equality in each bin of the observable that
// is called "x" by convention.
int validate(RooWorkspace &ws1, std::string const &argName, bool exact = true)
{
   RooWorkspace ws2;

   const std::string json1 = RooJSONFactoryWSTool{ws1}.exportJSONtoString();

   if (writeJsonFiles) {
      RooJSONFactoryWSTool{ws1}.exportJSON(argName + "_1.json");
   }

   RooJSONFactoryWSTool{ws2}.importJSONfromString(json1);
   if (writeJsonFiles) {
      RooJSONFactoryWSTool{ws2}.exportJSON(argName + "_2.json");
   }

   // Export the re-imported workspace back to JSON, and compare the first JSON
   // with the second one. They should be identical.
   const std::string json2 = RooJSONFactoryWSTool{ws2}.exportJSONtoString();
   EXPECT_EQ(json2, json1) << argName;

   // It would be nice to do a similar closure check for the original and for
   // the re-imported workspace. However, there is no way to compare workspaces
   // for equality. But we can still check that the objects in the workspace
   // have at least the same name.
   RooArgSet comps1 = ws1.components();
   RooArgSet comps2 = ws2.components();
   EXPECT_EQ(comps2.size(), comps1.size());

   comps1.sort();
   comps2.sort();

   for (std::size_t i = 0; i < comps1.size(); ++i) {
      EXPECT_STREQ(comps1[i]->GetName(), comps2[i]->GetName());
   }

   RooRealVar *x1 = ws1.var("x");
   RooRealVar *x2 = ws2.var("x");

   if (!x1 || !x2)
      return 1;

   TObject *arg1 = ws1.obj(argName);
   TObject *arg2 = ws2.obj(argName);

   if (!arg1 || !arg2)
      return 1;

   RooArgSet nset1{*x1};
   RooArgSet nset2{*x2};

   RooAbsReal *r1 = dynamic_cast<RooAbsReal *>(arg1);
   RooAbsReal *r2 = dynamic_cast<RooAbsReal *>(arg2);

   if (r1 && !r2)
      return 1;

   if (r1 && r2) {
      bool allGood = true;
      for (int i = 0; i < x1->numBins(); ++i) {
         x1->setBin(i);
         x2->setBin(i);
         const double val1 = r1->getVal(nset1);
         const double val2 = r1->getVal(nset2);
         allGood &= (exact ? (val1 == val2) : std::abs(val1 - val2) < 1e-10);
      }

      return allGood ? 0 : 1;
   }

   return 0;
}

int validate(std::vector<std::string> const &expressions, bool exact = true)
{
   RooWorkspace ws;
   for (std::size_t iExpr = 0; iExpr < expressions.size() - 1; ++iExpr) {
      ws.factory(expressions[iExpr]);
   }
   const std::string argName = ws.factory(expressions.back())->GetName();
   return validate(ws, argName, exact);
}

int validate(RooAbsArg const &arg, bool exact = true)
{
   RooWorkspace ws;
   ws.import(arg, RooFit::Silence());
   return validate(ws, arg.GetName(), exact);
}

} // namespace

// Test that the IO of attributes and string attributes works.
TEST(RooFitHS3, AttributesIO)
{

   std::string jsonString;

   // Export to JSON
   {
      RooWorkspace ws{"workspace"};
      ws.factory("Gaussian::pdf(x[0, 10], mean[5], sigma[1.0, 0.1, 10])");
      RooAbsPdf &pdf = *ws.pdf("pdf");

      // set attributes
      pdf.setAttribute("attr0");
      pdf.setStringAttribute("key0", "val0");

      jsonString = RooJSONFactoryWSTool{ws}.exportJSONtoString();
   }

   // Import JSON
   RooWorkspace ws{"workspace"};
   RooJSONFactoryWSTool{ws}.importJSONfromString(jsonString);
   RooAbsPdf &pdf = *ws.pdf("pdf");

   EXPECT_TRUE(pdf.getAttribute("attr0")) << "IO of attribute didn't work!";
   EXPECT_FALSE(pdf.getAttribute("attr1")) << "unexpected attribute found!";

   EXPECT_STREQ(pdf.getStringAttribute("key0"), "val0") << "IO of string attribute didn't work!";
   EXPECT_STREQ(pdf.getStringAttribute("key1"), nullptr) << "unexpected string attribute found!";
}

TEST(RooFitHS3, RooAddPdf)
{
   int status = validate({"Gaussian::sig(x[5.20, 5.30], sigmean[5.28, 5.20, 5.30], sigwidth[0.0027, 0.001, 1.])",
                          "ArgusBG::bkg(x, 5.291, argpar[-20.0, -100., -1.])",
                          "SUM::model(nsig[200, 0., 10000] * sig, nbkg[800, 0., 10000] * bkg)"});
   EXPECT_EQ(status, 0);

   // With the next part of the test, we want to cover the closure of
   // coefficient normalization reference observables.
   RooWorkspace ws;
   ws.factory("Gaussian::sig_1(x[5.20, 5.30], sigmean[5.28, 5.20, 5.30], sigwidth[0.0027, 0.001, 1.])");
   ws.factory("Uniform::sig_2(x_2[0, 10])");

   ws.factory("ArgusBG::bkg_1(x, 5.291, argpar[-20.0, -100., -1.])");
   // Some pdf in x_2 needs to be non linear, otherwise the reference
   // normalization set makes no difference.
   ws.factory("Polynomial::bkg_2(x_2, {a2[1.0, 0.0, 2.0]}, 2)");

   ws.factory("PROD::sig(sig_1, sig_2)");
   ws.factory("PROD::bkg(bkg_1, bkg_2)");

   ws.factory("nsig[200, 0., 10000]");
   ws.factory("nbkg[800, 0., 10000]");
   RooAddPdf addPdf{"model_cond", "model_cond", {*ws.pdf("sig"), *ws.pdf("bkg")}, {*ws.var("nsig"), *ws.var("nbkg")}};
   addPdf.fixCoefNormalization({*ws.var("x"), *ws.var("x_2")});
   status = validate(addPdf);
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooArgusBG)
{
   int status = validate({"ArgusBG::argusBG(x[0, 20], x0[10], c[-1], p[0.5])"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooBifurGauss)
{
   int status = validate({"BifurGauss::bifurGauss(x[0, 10], mean[5], sigmaL[1.0, 0.1, 10], sigmaR[2.0, 0.1, 10])"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooCBShape)
{
   int status = validate({"CBShape::cbShape(x[-10, 10], x0[0], sigma[2.0], alpha[1.4], n[1.2])"});
   EXPECT_EQ(status, 0);
}

/// Test that the IO of pdfs that contain literal RooConstVars works.
TEST(RooFitHS3, RooConstVar)
{
   RooRealVar x{"x", "x", 100, 0, 1000};
   int status = validate(RooPoisson{"pdf_with_const_var", "pdf_with_const_var", x, RooFit::RooConst(100.)});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooExponential)
{
   int status = validate({"Exponential::exponential_1(x[0, 10], c[-0.1])"});
   EXPECT_EQ(status, 0);
   RooWorkspace ws;
   ws.factory("x[0, 10]");
   ws.factory("c[-0.1]");
   RooExponential exponential2{"exponential_2", "exponential_2", *ws.var("x"), *ws.var("c"), true};
   status = validate(exponential2);
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooLegacyExpPoly)
{
   // To silence the numeric integration
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   // Test different values for "lowestOrder"
   int status = 0;
   status = validate({"LegacyExpPoly::exppoly0(x[0, 10], {a_0[3.0], a_1[-0.3, -10, 10], a_2[0.01, -10, 10]}, 0)"});
   EXPECT_EQ(status, 0);
   status = validate({"LegacyExpPoly::exppoly1(x[0, 10], {a_1[-0.1, -10, 10], a_2[0.003, -10, 10]}, 1)"});
   EXPECT_EQ(status, 0);
   status = validate({"LegacyExpPoly::exppoly1(x[0, 10], {a_2[0.003, -10, 10]}, 2)"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooGamma)
{
   int status = validate({"Gamma::gamma_dist(x[5.0, 10.0], gamma[1.0, 0.1, 10.0], beta[1.0, 0.1, 10.0], mu[5.0])"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooGaussian)
{
   int status = validate({"Gaussian::gaussian(x[0, 10], mean[5], sigma[1.0, 0.1, 10])"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooBernstein)
{
   int status = validate({"RooBernstein::bernstein(x[0, 10], { a[1], 3, b[5, 0, 20] })"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooGenericPdf)
{
   // To silence the numeric integration
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   // At this point, only basic arithmetic operations with +, -, * and / are
   // defined in the HS3 standard.
   int status = validate({"x[0, 10]", "c[5]", "a[1.0, 0.1, 10]", "EXPR::genericPdf1('a * x + c', {x, a, c})"});
   EXPECT_EQ(status, 0);

   // Test that it also works with index notation builtin to TFormula
   status = validate({"x[0, 10]", "c[5]", "a[1.0, 0.1, 10]", "EXPR::genericPdf2('x[1] * x[0] + x[2]', {x, a, c})"});
   EXPECT_EQ(status, 0);

   // Test for ordinal notation
   status = validate({"x[0, 10]", "c[5]", "a[1.0, 0.1, 10]", "EXPR::genericPdf3('@1 * @0 + @2', {x, a, c})"});
   EXPECT_EQ(status, 0);

   // Test for variable names with numbers and extra whitespaces in it
   status = validate({"m_001_mu[1.0, 0.1, 10]", "x[0, 5]", "m_003_mu[5]",
                      "EXPR::genericPdf4('@0   *  2  *      @1 +   @2', {m_001_mu, x, m_003_mu})"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooHistPdf)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooRealVar x{"x", "x", 0.0, 0.02};
   x.setBins(2);

   RooDataHist dataHist{"myDataHist", "myDataHist", x};
   dataHist.set(0, 23, -1);
   dataHist.set(1, 17, -1);

   int status = validate(RooHistPdf{"histPdf", "histPdf", x, dataHist});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooLandau)
{
   int status = validate({"Landau::landau(x[0, 10], mean[5], sigma[1.0, 0.1, 10])"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooLognormal)
{
   RooWorkspace ws;
   int status = validate({"Lognormal::lognormal_1(x[1.0, 1.1, 10], mu_2[2.0, 1.1, 10], k_1[2.0, 1.1, 5.0])"});
   EXPECT_EQ(status, 0);
   ws.factory("x[1.0, 1.1, 10]");
   ws.factory("mu_2[0.7, 0.1, 2.3]");
   ws.factory("k_2[0.7, 0.1, 1.6]");
   RooLognormal lognormal2{"lognormal_2", "lognormal_2", *ws.var("x"), *ws.var("mu_2"), *ws.var("k_2"), true};
   status = validate(lognormal2);
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooMultiVarGaussian)
{
   // To silence the numeric differentiation
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   using RooFit::RooConst;

   RooRealVar x{"x", "x", 0, 10};
   RooRealVar y{"y", "y", 0, 10};
   RooRealVar mean{"mean", "mean", 3};
   TMatrixDSym cov{2};
   cov(0, 0) = 1.0;
   cov(0, 1) = 0.2;
   cov(1, 0) = 0.2;
   cov(1, 1) = 1.0;
   RooMultiVarGaussian multiVarGauss{"multi_var_gauss", "", {x, y}, {mean, RooConst(5.0)}, cov};
   int status = validate(multiVarGauss);
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooPoisson)
{
   int status = 0;

   for (auto noRounding : {false, true}) {

      std::string name = "poisson";
      name += noRounding ? "_true" : "_false";

      RooRealVar x{"x", "x", 0, 10};
      RooRealVar mean{"mean", "mean", 5};
      RooPoisson poisson{name.c_str(), name.c_str(), x, mean, noRounding};
      status = validate(poisson);
      EXPECT_EQ(status, 0);
   }
}

TEST(RooFitHS3, RooPolynomial)
{
   // Test different values for "lowestOrder"
   int status = 0;
   status = validate({"Polynomial::poly0(x[0, 10], {a_0[3.0], a_1[-0.3, -10, 10], a_2[0.01, -10, 10]}, 0)"});
   EXPECT_EQ(status, 0);
   status = validate({"Polynomial::poly1(x[0, 10], {a_1[-0.1, -10, 10], a_2[0.003, -10, 10]}, 1)"});
   EXPECT_EQ(status, 0);
   status = validate({"Polynomial::poly1(x[0, 10], {a_2[0.003, -10, 10]}, 2)"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooPowerSum)
{
   int status = 0;
   status = validate({"PowerSum::power(x[0, 10], {a_0[3.0], a_1[-0.3, -10, 10]}, {1.0, 2.0})"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooRealIntegral)
{
   int status = 0;

   RooRealVar v1("v1", "v1", 0.6, 0., 1.);
   RooRealVar v2("v2", "v2", 1.0, 0., 2.);
   RooGenericPdf formula{"formula", "2 * x[0] * x[1]", {v1, v2}};
   RooArgSet funcNormSet{v1, v2};
   RooRealIntegral integ{"integ", "integ", formula, v2};
   RooRealIntegral integWithNormSet{"integ_with_norm_set", "integ_with_norm_set", formula, v2, &funcNormSet};

   RooRealVar x("x", "x", -8, 8);
   RooRealVar sigma("sigma", "sigma", 0.3, 0.1, 10);

   RooGaussian pdfContainingIntegralA("pdf_containing_integral_a", "pdf_containing_integral_a", x, integ, sigma);
   status = validate(pdfContainingIntegralA);
   EXPECT_EQ(status, 0);

   RooGaussian pdfContainingIntegralB("pdf_containing_integral_b", "pdf_containing_integral_b", x, integWithNormSet,
                                      sigma);
   status = validate(pdfContainingIntegralB);
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooUniform)
{
   int status = 0;
   status = validate({"Uniform::uniform({x[0.0, 10.0], y[0.0, 5.0]})"});
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, SimultaneousGaussians)
{
   // Create a test model: RooSimultaneous with Gaussian in one component, and
   // product of two Gaussians in the other.
   RooRealVar x("x", "x", -8, 8);
   RooRealVar mean("mean", "mean", 0, -8, 8);
   RooRealVar sigma("sigma", "sigma", 0.3, 0.1, 10);
   RooGaussian g1("g1", "g1", x, mean, sigma);
   RooGaussian g2("g2", "g2", x, mean, 0.3);
   RooProdPdf model("model", "model", RooArgList{g1, g2});
   RooGaussian model_ctl("model_ctl", "model_ctl", x, mean, sigma);
   RooCategory sample("sample", "sample", {{"physics", 0}, {"control", 1}});
   RooSimultaneous simPdf("simPdf", "simultaneous pdf", sample);
   simPdf.addPdf(model, "physics");
   simPdf.addPdf(model_ctl, "control");

   int status = validate(simPdf);
   EXPECT_EQ(status, 0);
}

// https://github.com/root-project/root/issues/14637
TEST(RooFitHS3, ScientificNotation)
{
   RooRealVar v1("v1", "v1", 1.0);
   RooRealVar v2("v2", "v2", 1.0);

   // make a formula that is some parameters times some numbers
   auto thestring = "@0*0.2e-6 + @1*0.1";
   RooArgList arglist;
   arglist.add(v1);
   arglist.add(v2);

   RooFormulaVar fvBad("fvBad", "fvBad", thestring, arglist);

   // make gaussian with mean as that formula
   RooRealVar x("x", "x", 0.0, -5.0, 5.0);
   RooGaussian g("g", "g", x, fvBad, 1.0);

   RooWorkspace ws("ws");
   ws.import(g);
   // std::cout << (fvBad.expression()) << std::endl;

   // export to json
   RooJSONFactoryWSTool t(ws);
   auto jsonStr = t.exportJSONtoString();

   // try to import, before the fix, it threw RooJSONFactoryWSTool::DependencyMissingError because of problem reading
   // the exponential char
   RooWorkspace newws("newws");
   RooJSONFactoryWSTool t2(newws);
   ASSERT_TRUE(t2.importJSONfromString(jsonStr));
}

// Workspace with ONLY a dataset (here: RooDataHist to avoid extra includes).
// -----------------------------------------------------------------------------
TEST(RooFitHS3, WorkspaceOnlyDataset_RooDataHist)
{
   RooWorkspace ws1{"ws_dataset_only"};

   // Observable with explicit binning
   RooRealVar x{"x", "x", 0.0, 1.0};
   x.setBins(3);
   // Build a tiny RooDataHist
   RooDataHist dh{"dh", "dataset-only (hist)", RooArgList{x}};
   // Fill deterministic contents
   x.setVal(0.1666667);
   dh.set(0, 10.0, 0.0); // bin 0
   x.setVal(0.5000000);
   dh.set(1, 20.0, 0.0); // bin 1
   x.setVal(0.8333333);
   dh.set(2, 15.0, 0.0); // bin 2

   ws1.import(dh, RooFit::Silence());

   // Round-trip and strict checks (no numeric comparison needed here)
   // Use the dataset name for object tracking
   const int status = validate(ws1, "dh");
   EXPECT_EQ(status, 0);
}

// -----------------------------------------------------------------------------
// Workspace with ONLY a function (no dataset, no pdfs).
// -----------------------------------------------------------------------------
TEST(RooFitHS3, WorkspaceOnlyFunction)
{
   int status = validate({std::string("x[-3, 3]"), std::string("RooFormulaVar::myfunc(\"sin(x) + 0.5*x*x\",x)")});
   EXPECT_EQ(status, 0);
}

// -----------------------------------------------------------------------------
// Workspace with a ModelConfig that points to a multivariate Gaussian pdf.
// -----------------------------------------------------------------------------
TEST(RooFitHS3, ModelConfigWithMultiVarGaussian)
{
   using RooFit::RooConst;

   // Observables
   RooRealVar x{"x", "x", -5.0, 5.0};
   RooRealVar y{"y", "y", -5.0, 5.0};

   // Means
   RooRealVar mx{"mx", "mx", 0.5};
   RooRealVar my{"my", "my", -0.3};

   // Covariance
   TMatrixDSym cov{2};
   cov(0, 0) = 1.2;
   cov(0, 1) = 0.25;
   cov(1, 0) = 0.25;
   cov(1, 1) = 0.9;

   RooMultiVarGaussian mv{"mvgauss", "mvgauss", RooArgList{x, y}, RooArgList{mx, my}, cov};

   RooWorkspace ws1{"ws_mc"};
   ws1.import(mv, RooFit::Silence(), RooFit::RecycleConflictNodes());

   // Build a ModelConfig referencing the pdf and its observables
   RooFit::ModelConfig mc{"mc", &ws1};
   mc.SetPdf(*ws1.pdf("mvgauss"));
   mc.SetObservables("x,y");
   ws1.import(mc);

   int status = validate(ws1, "mc");
   EXPECT_EQ(status, 0);
}

TEST(RooFitHS3, RooSpline)
{
   // Observable must be called "x" because validate() assumes that convention.
   RooWorkspace ws;

   // Use an observable with bins to enable the per-bin closure check.
   auto *x = ws.factory("x[0,10]");
   ASSERT_NE(x, nullptr);
   ws.var("x")->setBins(50);

   // Define knots. Keep it simple but nontrivial (nonlinear).
   const std::vector<double> x0{0.0, 1.5, 3.0, 6.0, 10.0};
   const std::vector<double> y0{1.0, 2.0, 1.0, 4.0, 3.0};

   RooSpline spline{"spline", "spline", *ws.var("x"), x0, y0, /*order=*/3, /*logx=*/false, /*logy=*/false};

   // Import the object into the workspace and validate JSON IO.
   ws.import(spline, RooFit::Silence());

   const int status = validate(ws, "spline", /*exact=*/true);
   EXPECT_EQ(status, 0);
}

namespace {

class TestExporterA final : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override
   {
      static const std::string k{"unit_test_exporter_A"};
      return k;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *, RooFit::Detail::JSONNode &) const override
   {
      callCounter()++;
      return true; // do nothing, just for test
   }
   static int &callCounter()
   {
      static int counter = 0;
      return counter;
   }
};

template <int N>
class TestExporter final : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override
   {
      static const std::string k{"unit_test_exporter"};
      return k;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *, RooFit::Detail::JSONNode &) const override
   {
      callCounter()++;
      return true; // do nothing, just for test
   }
   static int &callCounter()
   {
      static int counter = 0;
      return counter;
   }
};

} // namespace

// Test the custom exporter registration mechanism.
TEST(RooFitHS3, RegisterExporterByClassName)
{
   using RooFit::JSONIO::registerExporter;

   constexpr const char *className = "RooGaussian";
   TClass *klass = TClass::GetClass(className);
   ASSERT_NE(klass, nullptr);

   RooWorkspace ws{"ws"};
   ws.factory("RooGaussian::model(x[-10, 10], mu[-10, 10], sigma[2., 0.01, 10])");

   // 1. Add new exporter by class pointer with top priority.
   //    We expect this to get used.
   registerExporter<TestExporter<1>>(klass, /*topPriotiry=*/true);
   RooJSONFactoryWSTool{ws}.exportJSONtoString();
   EXPECT_EQ(TestExporter<1>::callCounter()--, 1);

   // 2. Add new exporter by class pointer with bottom priority.
   //    We expect the previous TestExporter<1> to still be used.
   registerExporter<TestExporter<2>>(klass, /*topPriotiry=*/false);
   RooJSONFactoryWSTool{ws}.exportJSONtoString();
   EXPECT_EQ(TestExporter<1>::callCounter()--, 1);

   // 3. Add new exporter by name with top priority.
   //    We expect this to get used.
   registerExporter<TestExporter<3>>(std::string{className}, /*topPriotiry=*/true);
   RooJSONFactoryWSTool{ws}.exportJSONtoString();
   EXPECT_EQ(TestExporter<3>::callCounter()--, 1);

   // 4. Add new exporter by name with bottom priority.
   //    We expect the previous TestExporter<3> to still be used.
   registerExporter<TestExporter<4>>(std::string{className}, /*topPriotiry=*/false);
   RooJSONFactoryWSTool{ws}.exportJSONtoString();
   EXPECT_EQ(TestExporter<3>::callCounter()--, 1);

   // Cleanup for other tests, also making sure the expected number of
   // exporters is removed.
   EXPECT_EQ(RooFit::JSONIO::removeExporters("TestExporter"), 4);
}

// Round-trip an unbinned RooDataSet and verify that the observable's range
// (min/max) is preserved through JSON. The "axes" node of an unbinned dataset
// is read back via min/max/nbins fields, so non-constant variables must export
// these fields directly on the variable node.
TEST(RooFitHS3, UnbinnedDatasetAxisRange)
{
   constexpr double xMin = -2.5;
   constexpr double xMax = 7.5;

   RooWorkspace ws1{"ws_unbinned"};
   {
      RooRealVar x{"x", "x", xMin, xMax};
      RooDataSet ds{"ds", "unbinned dataset", RooArgSet{x}};
      for (double val : {-1.0, 0.5, 2.0, 3.5, 6.0}) {
         x.setVal(val);
         ds.add(RooArgSet{x});
      }
      ws1.import(ds, RooFit::Silence());
   }

   const std::string json1 = RooJSONFactoryWSTool{ws1}.exportJSONtoString();

   RooWorkspace ws2{"ws_unbinned_2"};
   ASSERT_TRUE(RooJSONFactoryWSTool{ws2}.importJSONfromString(json1));

   auto *ds2 = dynamic_cast<RooDataSet *>(ws2.data("ds"));
   ASSERT_NE(ds2, nullptr);
   EXPECT_EQ(ds2->numEntries(), 5);

   RooRealVar *x2 = ws2.var("x");
   ASSERT_NE(x2, nullptr);
   EXPECT_DOUBLE_EQ(x2->getMin(), xMin);
   EXPECT_DOUBLE_EQ(x2->getMax(), xMax);

   // The exported "axes" node of an unbinned dataset must carry the
   // observable range so the file is self-describing. Before the fix, only
   // the variable name and current value were written there (the range was
   // only present in the separate "domains" block).
   const auto axesPos = json1.find("\"axes\":[{");
   ASSERT_NE(axesPos, std::string::npos) << json1;
   const auto axesEnd = json1.find("}]", axesPos);
   ASSERT_NE(axesEnd, std::string::npos) << json1;
   const std::string axesNode = json1.substr(axesPos, axesEnd - axesPos);
   EXPECT_NE(axesNode.find("\"min\":-2.5"), std::string::npos) << axesNode;
   EXPECT_NE(axesNode.find("\"max\":7.5"), std::string::npos) << axesNode;
}

// HistFactory channels with samples that have a zero-yield bin together with a
// staterror modifier used to produce NaN gamma errors because the relative
// error is computed as sqrt(sumW2)/sumW. Importing such a channel should now
// produce a finite (zero) error for that bin.
TEST(RooFitHS3, HistFactoryZeroYieldBin)
{
   const std::string jsonStr = R"({
      "metadata": {"hs3_version": "0.1.90"},
      "distributions": [
         {
            "name": "model_channel0",
            "type": "histfactory_dist",
            "axes": [
               {"name": "obs_channel0", "min": 0.0, "max": 2.0, "nbins": 2}
            ],
            "samples": [
               {
                  "name": "sig",
                  "data": {"contents": [10.0, 0.0]},
                  "modifiers": [
                     {"name": "mu", "type": "normfactor"},
                     {"name": "mcstat", "type": "staterror"}
                  ]
               }
            ]
         }
      ],
      "data": [
         {
            "name": "obsData_channel0",
            "type": "binned",
            "axes": [
               {"name": "obs_channel0", "min": 0.0, "max": 2.0, "nbins": 2}
            ],
            "contents": [10.0, 0.0]
         }
      ]
   })";

   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws{"ws_zero_yield"};
   ASSERT_TRUE(RooJSONFactoryWSTool{ws}.importJSONfromString(jsonStr));

   // The mc_stat ParamHistFunc is created with one gamma per bin. Their nominal
   // constraint values are derived from the relative bin error. For the
   // zero-yield bin the new behaviour avoids the 0/0 NaN and uses 0 instead.
   bool foundFiniteNomGamma = false;
   for (auto *arg : ws.allVars()) {
      const std::string name = arg->GetName();
      if (name.find("nom_gamma_stat_channel0") == std::string::npos)
         continue;
      auto *rrv = static_cast<RooRealVar *>(arg);
      EXPECT_TRUE(std::isfinite(rrv->getVal())) << "Non-finite nominal gamma value for " << name;
      foundFiniteNomGamma = true;
   }
   EXPECT_TRUE(foundFiniteNomGamma) << "No nominal gamma stat parameters were created";

   // The gamma stat parameters themselves must be finite as well.
   for (auto *arg : ws.allVars()) {
      const std::string name = arg->GetName();
      if (name.rfind("gamma_stat_channel0", 0) != 0)
         continue;
      auto *rrv = static_cast<RooRealVar *>(arg);
      EXPECT_TRUE(std::isfinite(rrv->getVal())) << "Non-finite gamma value for " << name;
      EXPECT_TRUE(std::isfinite(rrv->getMin())) << "Non-finite gamma min for " << name;
      EXPECT_TRUE(std::isfinite(rrv->getMax())) << "Non-finite gamma max for " << name;
   }
}

// Snapshot export must keep all variables that any pdf depends on, even when
// the variable is not in the set of separately exported objects. Global
// observables of HistFactory constraint pdfs (the nominal "nom_*" parameters)
// are exactly such variables: the HistFactory exporter explicitly skips them
// when collecting parameters to export, but pdfs still depend on them.
TEST(RooFitHS3, SnapshotKeepsGlobalObservables)
{
   const std::string jsonStr = R"({
      "metadata": {"hs3_version": "0.1.90"},
      "distributions": [
         {
            "name": "model_channel0",
            "type": "histfactory_dist",
            "axes": [
               {"name": "obs_channel0", "min": 0.0, "max": 2.0, "nbins": 2}
            ],
            "samples": [
               {
                  "name": "sig",
                  "data": {"contents": [10.0, 20.0], "errors": [1.0, 2.0]},
                  "modifiers": [
                     {"name": "mu", "type": "normfactor"},
                     {"name": "mcstat", "type": "staterror"}
                  ]
               }
            ]
         }
      ],
      "data": [
         {
            "name": "obsData_channel0",
            "type": "binned",
            "axes": [
               {"name": "obs_channel0", "min": 0.0, "max": 2.0, "nbins": 2}
            ],
            "contents": [10.0, 20.0]
         }
      ]
   })";

   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws1{"ws_snap"};
   ASSERT_TRUE(RooJSONFactoryWSTool{ws1}.importJSONfromString(jsonStr));

   // Collect the "nom_*" global observables created on import. The constraint
   // pdfs of the staterror modifier depend on them, but the HistFactory
   // exporter does not list them as top-level exported objects.
   RooArgSet globs;
   for (auto *arg : ws1.allVars()) {
      const std::string name = arg->GetName();
      if (name.rfind("nom_", 0) == 0) {
         globs.add(*arg);
      }
   }
   ASSERT_GT(globs.size(), 0u) << "No nominal global observables found in workspace";

   // Save a snapshot containing only the global observables. With the old
   // filter (require name in exportedObjectNames AND pdf dependence), this
   // snapshot would be dropped on export.
   const char *snapName = "globsSnap";
   ws1.saveSnapshot(snapName, globs, true);

   const std::string exported = RooJSONFactoryWSTool{ws1}.exportJSONtoString();

   // The exported JSON should mention the snapshot name and at least one of
   // the global observables.
   EXPECT_NE(exported.find(snapName), std::string::npos) << "Snapshot name missing from exported JSON";
   EXPECT_NE(exported.find("nom_gamma"), std::string::npos) << "Global observable missing from exported snapshot block";

   // Re-import and check that the snapshot survived the round-trip with all
   // global observables included.
   RooWorkspace ws2{"ws_snap_2"};
   ASSERT_TRUE(RooJSONFactoryWSTool{ws2}.importJSONfromString(exported));

   const RooArgSet *snap = ws2.getSnapshot(snapName);
   ASSERT_NE(snap, nullptr) << "Snapshot was not preserved through JSON round-trip";

   for (auto *arg : globs) {
      EXPECT_NE(snap->find(arg->GetName()), nullptr) << "Snapshot is missing pdf-dependent variable " << arg->GetName();
   }
}

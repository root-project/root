// Tests for the RooJSONFactoryWSTool
// Authors: Carsten D. Burgard, DESY/ATLAS, 12/2021
//          Jonas Rembser, CERN 12/2022

#include "HS3TestUtils.h"

#include <RooFitHS3/JSONIO.h>
#include <RooFitHS3/RooJSONFactoryWSTool.h>

#include <RooAddPdf.h>
#include <RooAddition.h>
#include <RooBinWidthFunction.h>
#include <RooBinning.h>
#include <RooCBShape.h>
#include <RooAbsCategoryLValue.h>
#include <RooCategory.h>
#include <RooBinSamplingPdf.h>
#include <RooConstVar.h>
#include <RooCrystalBall.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooExtendPdf.h>
#include <RooExponential.h>
#include <RooFit/ModelConfig.h>
#include <RooFitResult.h>
#include <RooFormulaVar.h>
#include <RooGaussian.h>
#include <RooGenericPdf.h>
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooHistFunc.h>
#include <RooHistPdf.h>
#include <RooLandau.h>
#include <RooLognormal.h>
#include <RooMultiVarGaussian.h>
#include <RooNumber.h>
#include <RooPoisson.h>
#include <RooProdPdf.h>
#include <RooProduct.h>
#include <RooRealIntegral.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooSuperCategory.h>
#include <RooSpline.h>
#include <RooUniformBinning.h>
#include <RooStats/HistFactory/ParamHistFunc.h>
#include <RooStats/HistFactory/FlexibleInterpVar.h>
#include <RooStats/HistFactory/PiecewiseInterpolation.h>
#include <RooStats/HistFactory/MakeModelAndMeasurementsFast.h>
#include <RooStats/HistFactory/Measurement.h>
#include <RooTFnBinding.h>
#include <RooWorkspace.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <functional>
#include <map>
#include <regex>
#include <memory>
#include <set>
#include <sstream>
#include <string_view>
#include <tuple>
#include <vector>

#include <Math/PdfFuncMathCore.h>
#include <Math/ProbFuncMathCore.h>
#include <TF3.h>
#include <TFile.h>
#include <TH1D.h>
#include <TMath.h>
#include <TROOT.h>

#include <gtest/gtest.h>

using namespace HS3Test;
using testing::HasSubstr;
using testing::Not;

namespace {

const testing::Environment *const quietEnvironment = testing::AddGlobalTestEnvironment(new QuietEnvironment);

std::string jsonString(RooFit::Detail::JSONTree &tree)
{
   std::ostringstream stream;
   tree.rootnode().writeJSON(stream);
   return stream.str();
}

// Builds a single-bin-observable RooDataHist filled with the given bin contents. Returns an owning pointer so that it
// can be handed to the RooHistFunc constructor that takes ownership (the const-reference overload keeps only an unowned
// pointer, which would dangle for a temporary).
std::unique_ptr<RooDataHist> makeDataHist(std::string const &name, RooRealVar &obs, std::vector<double> const &contents)
{
   auto dh = std::make_unique<RooDataHist>(name.c_str(), name.c_str(), RooArgList{obs});
   for (std::size_t i = 0; i < contents.size(); ++i) {
      dh->set(i, contents[i], -1.0);
   }
   return dh;
}

// Imports a "model_channel0" RooRealSumPdf with a single sample whose shape carries two histosys variations sharing the
// same parameter (i.e. a duplicate histosys). The first variation always lives on a 2-bin observable "x"; the second
// variation lives on `obs2` if given, or on the same "x" otherwise. Used to check that incompatible duplicates are
// rejected on export.
void importDuplicateHistoSysModel(RooWorkspace &ws, int interpCode, std::vector<double> const &low2Contents,
                                  std::vector<double> const &high2Contents, RooRealVar *obs2 = nullptr)
{
   RooRealVar x{"x", "x", 0.0, 2.0};
   x.setBins(2);
   RooRealVar &var2Obs = obs2 ? *obs2 : x;

   RooHistFunc nominal{"nominal", "nominal", RooArgSet{x}, makeDataHist("nominalData", x, {10.0, 20.0})};
   RooHistFunc low1{"low1", "low1", RooArgSet{x}, makeDataHist("lowData1", x, {8.0, 18.0})};
   RooHistFunc high1{"high1", "high1", RooArgSet{x}, makeDataHist("highData1", x, {12.0, 22.0})};
   RooHistFunc low2{"low2", "low2", RooArgSet{var2Obs}, makeDataHist("lowData2", var2Obs, low2Contents)};
   RooHistFunc high2{"high2", "high2", RooArgSet{var2Obs}, makeDataHist("highData2", var2Obs, high2Contents)};

   RooRealVar alphaShape{"alpha_shape", "alpha_shape", 0.0, -5.0, 5.0};
   alphaShape.setConstant(true);
   RooArgList parameters;
   parameters.add(alphaShape);
   parameters.add(alphaShape);
   RooArgList lowVariations{low1, low2};
   RooArgList highVariations{high1, high2};
   PiecewiseInterpolation interpolation{"duplicate_shape", "duplicate_shape", nominal,
                                        lowVariations,     highVariations,    parameters};
   interpolation.setAllInterpCodes(interpCode);

   RooProduct sampleShapes{"sample_shapes", "sample_shapes", RooArgList{interpolation}};
   RooConstVar sampleScale{"sample_scale", "sample_scale", 1.0};
   RooRealSumPdf model{"model_channel0", "model_channel0", RooArgList{sampleShapes}, RooArgList{sampleScale}, true};

   ws.import(model, RooFit::Silence());
}

// A two-channel HistFactory model with a normsys shared by the channels, a staterror in both channels, a shapefactor,
// and binned data, with the constraints either written one per parameter or collected into one array distribution
// per type (`arrays`), as RooFit writes them.
std::string twoChannelHistFactoryJSON(bool arrays)
{
   const std::vector<std::string> gammas{"gamma_c1_0", "gamma_c1_1", "gamma_c2_0", "gamma_c2_1", "gs_0", "gs_1"};
   std::string constraintDists;
   std::string components;
   std::string functions;
   std::string values;
   std::string domainAxes;
   for (auto const &g : gammas) {
      functions += R"({"name": ")" + g + R"(_poisMean", "type": "product", "factors": [")" + g + R"(", 100.0]},)";
      values +=
         R"({"name": "nom_)" + g + R"(", "value": 100.0, "const": true}, {"name": ")" + g + R"(", "value": 1.0},)";
      domainAxes += R"({"name": ")" + g + R"(", "min": 0.0, "max": 5.0}, {"name": "nom_)" + g +
                    R"(", "min": 0.0, "max": 1000.0},)";
      constraintDists += R"({"name": ")" + g + R"(_constraint", "type": "poisson_continuous_dist", "x": "nom_)" + g +
                         R"(", "mean": ")" + g + R"(_poisMean"},)";
      components += R"({"x": "nom_)" + g + R"(", "mean": ")" + g + R"(_poisMean"},)";
   }
   components.pop_back();
   auto constraint = [&](std::string const &individual, std::string const &array) {
      return "\"" + (arrays ? array : individual) + "\"";
   };
   auto gammaConstraints = [&](std::string const &prefix) {
      return constraint(prefix + "_0_constraint", "pois_constraints") + ", " +
             constraint(prefix + "_1_constraint", "pois_constraints");
   };
   const std::string distributions =
      arrays
         ? R"({"name": "gauss_constraints", "type": "array_dist", "component_type": "gaussian_dist",
                   "components": [{"x": "alpha_a", "mean": 0.0, "sigma": 1.0}]},
                  {"name": "pois_constraints", "type": "array_dist", "component_type": "poisson_continuous_dist",
                   "components": [)" +
              components + "]}"
         : R"({"name": "alpha_a_constraint", "type": "gaussian_dist", "x": "alpha_a", "mean": 0.0, "sigma": 1.0},)" +
              constraintDists.substr(0, constraintDists.size() - 1);
   // The likelihood lists the constraints that the modifiers name (HS3 0.3).
   std::string aux = arrays ? R"("gauss_constraints", "pois_constraints")" : R"("alpha_a_constraint")";
   if (!arrays) {
      for (auto const &g : gammas) {
         aux += ", \"" + g + "_constraint\"";
      }
   }
   return R"({
      "metadata": {"hs3_version": "0.3"},
      "distributions": [
         {"name": "model_c1", "type": "histfactory_dist",
          "axes": [{"name": "x_c1", "min": 0.0, "max": 2.0, "nbins": 2}],
          "default_interpolation": {"type": "mult", "in": "poly6", "out": "exp"},
          "samples": [
            {"name": "sig", "data": {"contents": [10.0, 20.0], "errors": [1.0, 2.0]},
             "modifiers": [
               {"type": "normfactor", "parameters": ["mu"]},
               {"name": "a", "type": "normsys", "parameters": ["alpha_a"], "data": {"lo": 0.9, "hi": 1.1},
                "constraints": [)" +
          constraint("alpha_a_constraint", "gauss_constraints") + R"(]},
               {"type": "staterror", "parameters": ["gamma_c1_0", "gamma_c1_1"], "constraints": [)" +
          gammaConstraints("gamma_c1") + R"(]}]},
            {"name": "bkg", "data": {"contents": [5.0, 5.0], "errors": [1.0, 1.0]},
             "modifiers": [
               {"name": "shape", "type": "shapefactor", "parameters": ["gs_0", "gs_1"], "constraints": [)" +
          gammaConstraints("gs") + R"(]},
               {"type": "staterror", "parameters": ["gamma_c1_0", "gamma_c1_1"], "constraints": [)" +
          gammaConstraints("gamma_c1") + R"(]}]}]},
         {"name": "model_c2", "type": "histfactory_dist",
          "axes": [{"name": "x_c2", "min": 0.0, "max": 2.0, "nbins": 2}],
          "default_interpolation": {"type": "mult", "in": "poly6", "out": "exp"},
          "samples": [
            {"name": "sig", "data": {"contents": [30.0, 10.0], "errors": [2.0, 1.0]},
             "modifiers": [
               {"type": "normfactor", "parameters": ["mu"]},
               {"name": "a", "type": "normsys", "parameters": ["alpha_a"], "data": {"lo": 0.8, "hi": 1.2},
                "constraints": [)" +
          constraint("alpha_a_constraint", "gauss_constraints") + R"(]},
               {"type": "staterror", "parameters": ["gamma_c2_0", "gamma_c2_1"], "constraints": [)" +
          gammaConstraints("gamma_c2") + R"(]}]}]},
         )" +
          distributions + R"(
      ],
      "functions": [)" +
          functions.substr(0, functions.size() - 1) + R"(],
      "data": [
         {"name": "obs_c1", "type": "binned", "axes": [{"name": "x_c1", "min": 0.0, "max": 2.0, "nbins": 2}],
          "contents": [12.0, 19.0]},
         {"name": "obs_c2", "type": "binned", "axes": [{"name": "x_c2", "min": 0.0, "max": 2.0, "nbins": 2}],
          "contents": [28.0, 11.0]}
      ],
      "domains": [{"name": "default_domain", "type": "product_domain", "axes": [
         {"name": "mu", "min": 0.0, "max": 5.0}, {"name": "alpha_a", "min": -5.0, "max": 5.0}, )" +
          domainAxes.substr(0, domainAxes.size() - 1) + R"(]}],
      "parameter_points": [{"name": "default_values", "parameters": [
         {"name": "mu", "value": 1.0}, {"name": "alpha_a", "value": 0.0}, )" +
          values.substr(0, values.size() - 1) + R"(]}],
      "likelihoods": [{"name": "lh", "distributions": ["model_c1", "model_c2"], "data": ["obs_c1", "obs_c2"],
                       "aux_distributions": [)" +
          aux + R"(]}],
      "analyses": [{"name": "an", "likelihood": "lh", "domains": ["default_domain"],
                    "parameters_of_interest": ["mu"]}]
   })";
}

// The NLL of the two-channel model of twoChannelHistFactoryJSON() at a parameter point. The importer merges the data
// sets of the two terms of the likelihood into "an_data", with the index of the RooSimultaneous of the terms.
double twoChannelNLL(RooWorkspace &ws, std::map<std::string, double> const &point)
{
   auto *mc = dynamic_cast<RooStats::ModelConfig *>(ws.obj("an"));
   if (!mc || !dynamic_cast<RooSimultaneous *>(mc->GetPdf()) || !ws.data("an_data")) {
      return std::numeric_limits<double>::quiet_NaN();
   }
   for (auto const &[name, value] : point) {
      ws.var(name)->setVal(value);
   }
   std::unique_ptr<RooAbsReal> nll{mc->GetPdf()->createNLL(*ws.data("an_data"))};
   return nll->getVal();
}

// The two-channel model with its data as a data set of the workspace, and with an observed value of a constraint
// that differs from the default. The importer merges the data sets of the two terms of the likelihood into "an_data",
// the data set of the RooSimultaneous of the ModelConfig.
void importTwoChannelsWithData(RooWorkspace &ws)
{
   importInto(ws, twoChannelHistFactoryJSON(false));
   ws.var("nom_gamma_c1_0")->setVal(120.0);
}

const std::map<std::string, double> twoChannelPoint{
   {"mu", 1.3}, {"alpha_a", 0.7}, {"gamma_c1_0", 0.9}, {"gamma_c2_1", 1.1}, {"gs_0", 1.2}};

std::vector<double> twoChannelLikelihood(RooWorkspace &ws)
{
   return {twoChannelNLL(ws, twoChannelPoint)};
}

} // namespace

// ---------------------------------------------------------------------------------------------------------------
// Round trips of workspaces, see expectRoundTrip(). The suite is instantiated with the models of each topic.

class RoundTrip : public testing::TestWithParam<RoundTripModel> {};

TEST_P(RoundTrip, Workspace)
{
   RooWorkspace ws;
   GetParam().build(ws);
   expectRoundTrip(ws, GetParam());
}

std::string roundTripName(testing::TestParamInfo<RoundTripModel> const &info)
{
   return info.param.name;
}

// The helpers find the differences that the tests rely on them to find.
TEST(HS3TestUtils, Matches)
{
   const Doc doc{R"({"list": [{"name": "a", "x": 1.5, "tags": ["p", null], "on": true}, {"name": "b"}]})"};
   EXPECT_TRUE(matches(doc, "list/a", R"({"x": 1.5, "tags": ["p", null], "on": true, "y": "<absent>"})"));
   EXPECT_FALSE(matches(doc, "list/a", R"({"on": "true"})"));
   EXPECT_FALSE(matches(doc, "list/a", R"({"on": 1})"));
   EXPECT_TRUE(matches(doc, "list/1", R"({"name": "b"})"));
   EXPECT_FALSE(matches(doc, "list/a", R"({"x": 1.6})"));
   EXPECT_FALSE(matches(doc, "list/a", R"({"x": "1.5"})"));
   EXPECT_FALSE(matches(doc, "list/a", R"({"x": "<absent>"})"));
   EXPECT_FALSE(matches(doc, "list/a", R"({"y": 1})"));
   EXPECT_FALSE(matches(doc, "list/a", R"({"tags": ["p"]})"));
   EXPECT_FALSE(matches(doc, "list/a", R"({"tags": ["p", "q"]})"));
   EXPECT_FALSE(matches(doc, "list/a", R"({"tags": null})"));
   EXPECT_FALSE(matches(doc, "list/c", R"({})"));
   EXPECT_TRUE(doc.has("list/a/x"));
   EXPECT_FALSE(doc.has("list/a/y"));
   EXPECT_THROW(doc.has("list/c/x"), std::runtime_error);
   EXPECT_THROW(doc["list/2"], std::runtime_error);
}

TEST(HS3TestUtils, SameValues)
{
   const RoundTripModel model{"model", RoundTripModel::Builder{}};
   RooWorkspace ws;
   ws.factory("Gaussian::g(x[0, 10], m[5, 0, 10], 1.0)");
   RooWorkspace same;
   same.factory("Gaussian::g(x[0, 10], m[5, 0, 10], 1.0)");
   EXPECT_TRUE(sameValues(ws, same, model, /*sameComponents=*/true));

   RooWorkspace wider;
   wider.factory("Gaussian::g(x[0, 10], m[5, 0, 10], 1.001)");
   EXPECT_FALSE(sameValues(ws, wider, model, /*sameComponents=*/true));
   // A difference that only shows with moved parameters.
   RooWorkspace other;
   other.factory("Gaussian::g(x[0, 10], expr::mean('m * m / 5', m[5, 0, 10]), 1.0)");
   EXPECT_FALSE(sameValues(ws, other, model, /*sameComponents=*/false));
   EXPECT_FALSE(sameValues(ws, other, model, /*sameComponents=*/true));

   EXPECT_TRUE(sameNumbers({1.0, 2.0}, {1.0, 2.0}, 0.));
   EXPECT_FALSE(sameNumbers({1.0, 2.0}, {1.0, 2.0 + 1e-12}, 0.));
   EXPECT_TRUE(sameNumbers({1.0, 2.0}, {1.0, 2.0 + 1e-12}, 1e-10));
   EXPECT_FALSE(sameNumbers({1.0, 2.0}, {1.0}, 1e-10));
}

// Pdfs and functions made with the workspace factory.
INSTANTIATE_TEST_SUITE_P(
   Factory, RoundTrip,
   testing::ValuesIn(std::vector<RoundTripModel>{
      {"AddPdf",
       {"Gaussian::sig(x[5.20, 5.30], sigmean[5.28, 5.20, 5.30], sigwidth[0.0027, 0.001, 1.])",
        "ArgusBG::bkg(x, 5.291, argpar[-20.0, -100., -1.])",
        "SUM::model(nsig[200, 0., 10000] * sig, nbkg[800, 0., 10000] * bkg)"}},
      {"ArgusBG", {"ArgusBG::argusBG(x[0, 20], x0[10], c[-1], p[0.5])"}},
      {"BifurGauss", {"BifurGauss::bifurGauss(x[0, 10], mean[5], sigmaL[1.0, 0.1, 10], sigmaR[2.0, 0.1, 10])"}},
      {"CBShape", {"CBShape::cbShape(x[-10, 10], x0[0], sigma[2.0], alpha[1.4], n[1.2])"}},
      {"Chebychev", {"Chebychev::chebychev(x[-1, 1], {a_0[0.1], 0.2, a_2[-0.3, -1, 1]})"}},
      {"EffProd",
       {"Gaussian::gauss(x[0, 10], mean[5], sigma[1.0, 0.1, 10])", "expr::eff('0.5 + 0.05 * x', x)",
        "EffProd::effProd(gauss, eff)"}},
      {"Exponential", {"Exponential::exponential_1(x[0, 10], c[-0.1])"}},
      // Different values for "lowestOrder"
      {"ExpPolyLowestOrder0",
       {"LegacyExpPoly::exppoly0(x[0, 10], {a_0[3.0], a_1[-0.3, -10, 10], a_2[0.01, -10, 10]}, 0)"}},
      {"ExpPolyLowestOrder1", {"LegacyExpPoly::exppoly1(x[0, 10], {a_1[-0.1, -10, 10], a_2[0.003, -10, 10]}, 1)"}},
      {"ExpPolyLowestOrder2", {"LegacyExpPoly::exppoly1(x[0, 10], {a_2[0.003, -10, 10]}, 2)"}},
      {"Gamma", {"Gamma::gamma_dist(x[5.0, 10.0], gamma[1.0, 0.1, 10.0], beta[1.0, 0.1, 10.0], mu[5.0])"}},
      {"Gaussian", {"Gaussian::gaussian(x[0, 10], mean[5], sigma[1.0, 0.1, 10])"}},
      {"Bernstein", {"RooBernstein::bernstein(x[0, 10], { a[1], 3, b[5, 0, 20] })"}},
      // Only +, -, * and / are defined in the HS3 standard. The variables are given by name, by the index notation of
      // TFormula and by ordinal notation, also with numbers in the names and extra whitespace.
      {"GenericPdfNames", {"x[0, 10]", "c[5]", "a[1.0, 0.1, 10]", "EXPR::genericPdf1('a * x + c', {x, a, c})"}},
      {"GenericPdfIndices",
       {"x[0, 10]", "c[5]", "a[1.0, 0.1, 10]", "EXPR::genericPdf2('x[1] * x[0] + x[2]', {x, a, c})"}},
      {"GenericPdfOrdinals", {"x[0, 10]", "c[5]", "a[1.0, 0.1, 10]", "EXPR::genericPdf3('@1 * @0 + @2', {x, a, c})"}},
      {"GenericPdfWhitespace",
       {"m_001_mu[1.0, 0.1, 10]", "x[0, 5]", "m_003_mu[5]",
        "EXPR::genericPdf4('@0   *  2  *      @1 +   @2', {m_001_mu, x, m_003_mu})"}},
      {"Landau", {"Landau::landau(x[0, 10], mean[5], sigma[1.0, 0.1, 10])"}},
      {"Lognormal", {"Lognormal::lognormal_1(x[1.0, 1.1, 10], mu_2[2.0, 1.1, 10], k_1[2.0, 1.1, 5.0])"}},
      // Different values for "lowestOrder"
      {"PolynomialLowestOrder0",
       {"Polynomial::poly0(x[0, 10], {a_0[3.0], a_1[-0.3, -10, 10], a_2[0.01, -10, 10]}, 0)"}},
      {"PolynomialLowestOrder1", {"Polynomial::poly1(x[0, 10], {a_1[-0.1, -10, 10], a_2[0.003, -10, 10]}, 1)"}},
      {"PolynomialLowestOrder2", {"Polynomial::poly1(x[0, 10], {a_2[0.003, -10, 10]}, 2)"}},
      {"PowerSum", {"PowerSum::power(x[0, 10], {a_0[3.0], a_1[-0.3, -10, 10]}, {1.0, 2.0})"}},
      {"Projection",
       {"Gaussian::gx(x[0, 10], mx[5], sx[1.0, 0.1, 10])", "Gaussian::gy(y[0, 10], my[4], sy[2.0, 0.1, 10])",
        "PROD::prod(gx, gy)", "PROJ::projected(prod, y)"}},
      {"Uniform", {"Uniform::uniform({x[0.0, 10.0], y[0.0, 5.0]})"}},
      // A workspace with only a function
      {"FormulaVar", {"x[-3, 3]", "RooFormulaVar::myfunc(\"sin(x) + 0.5*x*x\",x)"}},
   }),
   roundTripName);

// Pdfs, functions and data that are built here.
INSTANTIATE_TEST_SUITE_P(
   Objects, RoundTrip,
   testing::ValuesIn(std::vector<RoundTripModel>{
      // Closure of the coefficient normalization reference observables.
      RoundTripModel{
         "AddPdfWithFixedCoefNormalization",
         [](RooWorkspace &ws) {
            ws.factory("Gaussian::sig_1(x[5.20, 5.30], sigmean[5.28, 5.20, 5.30], sigwidth[0.0027, 0.001, 1.])");
            ws.factory("Uniform::sig_2(x_2[0, 10])");
            ws.factory("ArgusBG::bkg_1(x, 5.291, argpar[-20.0, -100., -1.])");
            // Some pdf in x_2 needs to be non linear, otherwise the reference normalization set makes no
            // difference.
            ws.factory("Polynomial::bkg_2(x_2, {a2[1.0, 0.0, 2.0]}, 2)");
            ws.factory("PROD::sig(sig_1, sig_2)");
            ws.factory("PROD::bkg(bkg_1, bkg_2)");
            ws.factory("nsig[200, 0., 10000]");
            ws.factory("nbkg[800, 0., 10000]");
            RooAddPdf addPdf{
               "model_cond", "model_cond", {*ws.pdf("sig"), *ws.pdf("bkg")}, {*ws.var("nsig"), *ws.var("nbkg")}};
            addPdf.fixCoefNormalization({*ws.var("x"), *ws.var("x_2")});
            ws.import(addPdf, RooFit::Silence());
         }}
         .over({"x", "x_2"}),
      // A pdf that contains a literal RooConstVar.
      {"ConstVar",
       [](RooWorkspace &ws) {
          RooRealVar x{"x", "x", 100, 0, 1000};
          ws.import(RooPoisson{"pdf_with_const_var", "pdf_with_const_var", x, RooFit::RooConst(100.)},
                    RooFit::Silence());
       }},
      {"ExponentialNegateCoefficient",
       [](RooWorkspace &ws) {
          RooRealVar x{"x", "x", 0, 10};
          RooRealVar c{"c", "c", -0.1};
          ws.import(RooExponential{"exponential_2", "exponential_2", x, c, true}, RooFit::Silence());
       }},
      {"HistPdf",
       [](RooWorkspace &ws) {
          RooRealVar x{"x", "x", 0.0, 0.02};
          x.setBins(2);
          RooDataHist dataHist{"myDataHist", "myDataHist", x};
          dataHist.set(0, 23, -1);
          dataHist.set(1, 17, -1);
          ws.import(RooHistPdf{"histPdf", "histPdf", x, dataHist}, RooFit::Silence());
       }},
      {"LognormalStandardParametrization",
       [](RooWorkspace &ws) {
          RooRealVar x{"x", "x", 1.0, 1.1, 10};
          RooRealVar mu{"mu_2", "mu_2", 0.7, 0.1, 2.3};
          RooRealVar k{"k_2", "k_2", 0.7, 0.1, 1.6};
          ws.import(RooLognormal{"lognormal_2", "lognormal_2", x, mu, k, true}, RooFit::Silence());
       }},
      RoundTripModel{"MultiVarGaussian",
                     [](RooWorkspace &ws) {
                        RooRealVar x{"x", "x", 0, 10};
                        RooRealVar y{"y", "y", 0, 10};
                        RooRealVar mean{"mean", "mean", 3};
                        TMatrixDSym cov{2};
                        cov(0, 0) = 1.0;
                        cov(0, 1) = 0.2;
                        cov(1, 0) = 0.2;
                        cov(1, 1) = 1.0;
                        ws.import(
                           RooMultiVarGaussian{"multi_var_gauss", "", {x, y}, {mean, RooFit::RooConst(5.0)}, cov},
                           RooFit::Silence());
                     }}
         .over({"x", "y"}),
      {"Poisson",
       [](RooWorkspace &ws) {
          RooRealVar x{"x", "x", 0, 10};
          RooRealVar mean{"mean", "mean", 5};
          ws.import(RooPoisson{"poisson_false", "poisson_false", x, mean, /*noRounding=*/false}, RooFit::Silence());
       }},
      {"PoissonNoRounding",
       [](RooWorkspace &ws) {
          RooRealVar x{"x", "x", 0, 10};
          RooRealVar mean{"mean", "mean", 5};
          ws.import(RooPoisson{"poisson_true", "poisson_true", x, mean, /*noRounding=*/true}, RooFit::Silence());
       }},
      {"RealIntegral",
       [](RooWorkspace &ws) {
          RooRealVar v1("v1", "v1", 0.6, 0., 1.);
          RooRealVar v2("v2", "v2", 1.0, 0., 2.);
          RooGenericPdf formula{"formula", "2 * x[0] * x[1]", {v1, v2}};
          RooRealIntegral integ{"integ", "integ", formula, v2};
          RooRealVar x("x", "x", -8, 8);
          RooRealVar sigma("sigma", "sigma", 0.3, 0.1, 10);
          ws.import(RooGaussian{"pdf_containing_integral_a", "pdf_containing_integral_a", x, integ, sigma},
                    RooFit::Silence());
       }},
      {"RealIntegralWithNormSet",
       [](RooWorkspace &ws) {
          RooRealVar v1("v1", "v1", 0.6, 0., 1.);
          RooRealVar v2("v2", "v2", 1.0, 0., 2.);
          RooGenericPdf formula{"formula", "2 * x[0] * x[1]", {v1, v2}};
          RooArgSet funcNormSet{v1, v2};
          RooRealIntegral integ{"integ_with_norm_set", "integ_with_norm_set", formula, v2, &funcNormSet};
          RooRealVar x("x", "x", -8, 8);
          RooRealVar sigma("sigma", "sigma", 0.3, 0.1, 10);
          ws.import(RooGaussian{"pdf_containing_integral_b", "pdf_containing_integral_b", x, integ, sigma},
                    RooFit::Silence());
       }},
      // A RooSimultaneous with a Gaussian in one component, and a product of two Gaussians in the other.
      RoundTripModel{"SimultaneousGaussians",
                     [](RooWorkspace &ws) {
                        RooRealVar x("x", "x", -8, 8);
                        RooRealVar mean("mean", "mean", 0, -8, 8);
                        RooRealVar sigma("sigma", "sigma", 0.3, 0.1, 10);
                        RooGaussian g1("g1", "g1", x, mean, sigma);
                        RooGaussian g2("g2", "g2", x, mean, 0.3);
                        RooProdPdf model("model", "model", RooArgList{g1, g2});
                        // The components have their own observables: HS3 requires distinct variates of the
                        // components.
                        RooRealVar xCtl("x_ctl", "x_ctl", -8, 8);
                        RooGaussian model_ctl("model_ctl", "model_ctl", xCtl, mean, sigma);
                        RooCategory sample("sample", "sample", {{"physics", 0}, {"control", 1}});
                        RooSimultaneous simPdf("simPdf", "simultaneous pdf", sample);
                        simPdf.addPdf(model, "physics");
                        simPdf.addPdf(model_ctl, "control");
                        ws.import(simPdf, RooFit::Silence());
                     }}
         .over({"x", "x_ctl"}),
      // A workspace with only a data set.
      {"DataHistOnly",
       [](RooWorkspace &ws) {
          RooRealVar x{"x", "x", 0.0, 1.0};
          x.setBins(3);
          RooDataHist dh{"dh", "dataset-only (hist)", RooArgList{x}};
          dh.set(0, 10.0, 0.0);
          dh.set(1, 20.0, 0.0);
          dh.set(2, 15.0, 0.0);
          ws.import(dh, RooFit::Silence());
       }},
      // A ModelConfig that points to a multivariate Gaussian pdf.
      RoundTripModel{"ModelConfigWithMultiVarGaussian",
                     [](RooWorkspace &ws) {
                        RooRealVar x{"x", "x", -5.0, 5.0};
                        RooRealVar y{"y", "y", -5.0, 5.0};
                        RooRealVar mx{"mx", "mx", 0.5};
                        RooRealVar my{"my", "my", -0.3};
                        TMatrixDSym cov{2};
                        cov(0, 0) = 1.2;
                        cov(0, 1) = 0.25;
                        cov(1, 0) = 0.25;
                        cov(1, 1) = 0.9;
                        RooMultiVarGaussian mv{"mvgauss", "mvgauss", RooArgList{x, y}, RooArgList{mx, my}, cov};
                        ws.import(mv, RooFit::Silence(), RooFit::RecycleConflictNodes());
                        RooFit::ModelConfig mc{"mc", &ws};
                        mc.SetPdf(*ws.pdf("mvgauss"));
                        mc.SetObservables("x,y");
                        ws.import(mc);
                     }}
         .over({"x", "y"}),
      // A nonlinear spline, with bins of the observable for the comparison of the values.
      {"Spline",
       [](RooWorkspace &ws) {
          RooRealVar x{"x", "x", 0, 10};
          x.setBins(50);
          const std::vector<double> x0{0.0, 1.5, 3.0, 6.0, 10.0};
          const std::vector<double> y0{1.0, 2.0, 1.0, 4.0, 3.0};
          ws.import(RooSpline{"spline", "spline", x, x0, y0, /*order=*/3, /*logx=*/false, /*logy=*/false},
                    RooFit::Silence());
       }},
   }),
   roundTripName);

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

      jsonString = toJSON(ws);
   }

   // Import JSON
   Imported ws{jsonString};
   RooAbsPdf &pdf = *ws.pdf("pdf");

   EXPECT_TRUE(pdf.getAttribute("attr0")) << "IO of attribute didn't work!";
   EXPECT_FALSE(pdf.getAttribute("attr1")) << "unexpected attribute found!";

   EXPECT_STREQ(pdf.getStringAttribute("key0"), "val0") << "IO of string attribute didn't work!";
   EXPECT_STREQ(pdf.getStringAttribute("key1"), nullptr) << "unexpected string attribute found!";
}

TEST(RooFitHS3, ParameterPointsDoNotExportRanges)
{
   RooWorkspace ws{"workspace"};
   ws.factory("Gaussian::pdf(x[0, 10], mu[1, -5, 5], sigma[2, 0.1, 10])");

   const std::string points = Doc{ws}.dump("parameter_points");
   EXPECT_THAT(points, HasSubstr(R"({"name":"mu","value":1.0})"));
   EXPECT_THAT(points, Not(HasSubstr("\"min\"")));
   EXPECT_THAT(points, Not(HasSubstr("\"max\"")));
}

// Evaluating a pdf with a normalization set, or creating an integral over it,
// registers an integral that lives outside the workspace as a client of the
// pdf. This must not demote the pdf from being a top-level object of the
// export. Covers https://github.com/root-project/root/issues/23221.
TEST(RooFitHS3, TopLevelPdfExportAfterNormalizedEvaluation)
{
   const std::vector<std::string> expected{"gauss"};

   {
      RooWorkspace ws{"ws"};
      ws.factory("Gaussian::gauss(x[0, -5, 5], mean[1, -5, 5], sigma[2, 0.1, 10])");
      RooAbsPdf &pdf = *ws.pdf("gauss");
      RooArgSet normSet{*ws.var("x")};
      pdf.getVal(normSet);
      // Make sure the test covers the scenario from the issue: the cached
      // normalization integral is registered as a client of the pdf.
      ASSERT_TRUE(pdf.hasClients());

      EXPECT_EQ(Doc{ws}.names("distributions"), expected);
      // cleanWS() uses the same top-level criterion
      EXPECT_NE(RooJSONFactoryWSTool::cleanWS(ws).pdf("gauss"), nullptr);
      // ... and the full round trip should still work after the evaluation
      expectRoundTrip(ws);
   }
   {
      RooWorkspace ws{"ws"};
      ws.factory("Gaussian::gauss(x[0, -5, 5], mean[1, -5, 5], sigma[2, 0.1, 10])");
      std::unique_ptr<RooAbsReal> integral{ws.pdf("gauss")->createIntegral(*ws.var("x"))};
      ASSERT_TRUE(ws.pdf("gauss")->hasClients());

      EXPECT_EQ(Doc{ws}.names("distributions"), expected);
      EXPECT_NE(RooJSONFactoryWSTool::cleanWS(ws).pdf("gauss"), nullptr);
   }
}

TEST(RooFitHS3, ProductDomainEntriesExportExplicitBounds)
{
   RooRealVar x{"x", "x", 0.0, -10.0, 10.0};
   RooRealVar mean{"mean", "mean", 0.0};
   RooRealVar sigma{"sigma", "sigma", 1.0, 0.1, 10.0};
   RooGaussian gauss{"gauss", "gauss", x, mean, sigma};

   RooWorkspace ws{"workspace"};
   ws.import(gauss, RooFit::Silence());

   Doc doc{ws};
   EXPECT_TRUE(matches(doc, "domains/default_domain/axes/x", R"({"min": -10.0, "max": 10.0})"));
   // Infinite bounds are expressed by omitting them, since JSON has no literal for infinity.
   EXPECT_TRUE(matches(doc, "domains/default_domain/axes/mean", R"({"min": "<absent>", "max": "<absent>"})"));

   Imported imported{doc};
   auto *importedMean = object<RooRealVar>(imported, "mean");
   EXPECT_TRUE(std::isinf(importedMean->getMin()));
   EXPECT_LT(importedMean->getMin(), 0.0);
   EXPECT_TRUE(std::isinf(importedMean->getMax()));
   EXPECT_GT(importedMean->getMax(), 0.0);

   // Files written by older RooFit versions express infinite bounds as null, which is still understood.
   Imported legacy{
      document("0.2", {defaultDomain({R"({"name": "a", "min": null, "max": 3.0}, {"name": "b", "min": 1.0})"})})};
   ASSERT_NE(legacy.var("a"), nullptr);
   EXPECT_TRUE(std::isinf(legacy.var("a")->getMin()));
   EXPECT_DOUBLE_EQ(legacy.var("a")->getMax(), 3.0);
   ASSERT_NE(legacy.var("b"), nullptr);
   EXPECT_DOUBLE_EQ(legacy.var("b")->getMin(), 1.0);
   EXPECT_TRUE(std::isinf(legacy.var("b")->getMax()));
}

TEST(RooFitHS3, ProductDomainEntriesExportBinning)
{
   RooRealVar uniform{"uniform", "uniform", 0.0, 1.0};
   uniform.setBins(7);

   RooRealVar nonuniform{"nonuniform", "nonuniform", 0.0, 3.0};
   RooBinning nonuniformBinning{0.0, 3.0};
   nonuniformBinning.addBoundary(1.0);
   nonuniformBinning.addBoundary(1.5);
   nonuniform.setBinning(nonuniformBinning);

   RooAddition sum{"sum", "sum", RooArgList{uniform, nonuniform}};

   RooWorkspace ws{"workspace"};
   ws.import(sum, RooFit::Silence());

   Doc doc{ws};
   EXPECT_TRUE(matches(doc, "domains/default_domain/axes/uniform", R"({"nbins": 7, "edges": "<absent>"})"));
   EXPECT_TRUE(matches(doc, "domains/default_domain/axes/nonuniform",
                       R"({"edges": [0.0, 1.0, 1.5, 3.0], "nbins": "<absent>"})"));

   Imported imported{doc};
   auto *importedUniform = object<RooRealVar>(imported, "uniform");
   EXPECT_EQ(importedUniform->getBins(), 7);

   auto *importedNonuniform = object<RooRealVar>(imported, "nonuniform");
   auto const &importedBinning = importedNonuniform->getBinning();
   EXPECT_FALSE(importedBinning.isUniform());
   ASSERT_EQ(importedBinning.numBins(), 3);
   EXPECT_DOUBLE_EQ(importedBinning.binLow(0), 0.0);
   EXPECT_DOUBLE_EQ(importedBinning.binHigh(0), 1.0);
   EXPECT_DOUBLE_EQ(importedBinning.binHigh(1), 1.5);
   EXPECT_DOUBLE_EQ(importedBinning.binHigh(2), 3.0);
}

// Named fit ranges are stored as RooRangeBinning, which has no bin structure.
// They must be exported as plain min/max bounds, not as bin edges — in
// particular for half-infinite ranges, where an infinite edge would not be
// representable as a JSON number and broke the round trip.
TEST(RooFitHS3, NamedRangesRoundTrip)
{
   RooWorkspace ws{"workspace"};
   ws.factory("Gaussian::pdf(x[0, 10], mu[1, -5, 5], sigma[2, 0.1, 10])");
   RooRealVar &x = *ws.var("x");
   x.setRange("tail", 1.0, RooNumber::infinity());
   x.setRange("core", 2.0, 5.0);

   const std::string json = toJSON(ws);
   EXPECT_THAT(json, Not(HasSubstr("\"edges\"")));

   Imported imported{json};
   auto *importedX = object<RooRealVar>(imported, "x");

   ASSERT_TRUE(importedX->hasRange("tail"));
   EXPECT_DOUBLE_EQ(importedX->getMin("tail"), 1.0);
   EXPECT_TRUE(std::isinf(importedX->getMax("tail")));

   ASSERT_TRUE(importedX->hasRange("core"));
   EXPECT_DOUBLE_EQ(importedX->getMin("core"), 2.0);
   EXPECT_DOUBLE_EQ(importedX->getMax("core"), 5.0);
}

TEST(RooFitHS3, ParameterStepWidthsModelConfigRoundTrip)
{
   RooWorkspace ws1{"workspace"};
   ws1.factory("Gaussian::sig(x[-5, 5], mu[0, -10, 10], sigma[1, 0.1, 10])");
   ws1.factory("Polynomial::bkg(x, {theta[0, -1, 1]})");
   ws1.factory("SUM::model(fsig[0.5, 0, 1] * sig, bkg)");

   RooRealVar &x = *ws1.var("x");
   RooDataSet data{"data", "data", RooArgSet{x}};
   for (double val : {-1.0, 0.5, 1.5}) {
      x.setVal(val);
      data.add(RooArgSet{x});
   }
   ws1.import(data);

   ws1.var("x")->setError(9.0);
   ws1.var("mu")->setError(0.12);
   ws1.var("theta")->setError(0.33);
   ws1.var("sigma")->setError(0.20);
   ws1.var("sigma")->setAsymError(-0.18, 0.25);

   addModelConfig(ws1, "mc", "model", "x", "mu", /*nuisance=*/"sigma");

   // Neither the observable "x" nor the asymmetric errors are written.
   Doc doc{ws1};
   EXPECT_EQ(
      doc.dump("misc/minimization/parameter_stepwidths"),
      R"([{"name":"mu","step_width":0.12},{"name":"sigma","step_width":0.2},{"name":"theta","step_width":0.33}])");

   Imported ws2{doc};

   ASSERT_NE(ws2.var("mu"), nullptr);
   ASSERT_NE(ws2.var("theta"), nullptr);
   ASSERT_NE(ws2.var("sigma"), nullptr);
   ASSERT_NE(ws2.var("x"), nullptr);
   EXPECT_TRUE(ws2.var("mu")->hasError());
   EXPECT_DOUBLE_EQ(ws2.var("mu")->getError(), 0.12);
   EXPECT_TRUE(ws2.var("theta")->hasError());
   EXPECT_DOUBLE_EQ(ws2.var("theta")->getError(), 0.33);
   EXPECT_TRUE(ws2.var("sigma")->hasError());
   EXPECT_DOUBLE_EQ(ws2.var("sigma")->getError(), 0.20);
   EXPECT_FALSE(ws2.var("sigma")->hasAsymError());
   EXPECT_FALSE(ws2.var("x")->hasError());
}

TEST(RooFitHS3, ParameterStepWidthsFallbackExcludesDataAxes)
{
   RooWorkspace ws1{"workspace"};
   ws1.factory("Gaussian::model(x[-5, 5], mu[0, -10, 10], sigma[1, 0.1, 10])");

   RooRealVar &x = *ws1.var("x");
   RooDataSet data{"data", "data", RooArgSet{x}};
   for (double val : {-1.0, 0.5, 1.5}) {
      x.setVal(val);
      data.add(RooArgSet{x});
   }
   ws1.import(data);

   ws1.var("x")->setError(9.0);
   ws1.var("mu")->setError(0.12);
   ws1.var("sigma")->setError(0.20);

   Doc doc{ws1};
   EXPECT_EQ(doc.dump("misc/minimization/parameter_stepwidths"),
             R"([{"name":"mu","step_width":0.12},{"name":"sigma","step_width":0.2}])");

   Imported ws2{doc};

   ASSERT_NE(ws2.var("mu"), nullptr);
   ASSERT_NE(ws2.var("sigma"), nullptr);
   ASSERT_NE(ws2.var("x"), nullptr);
   EXPECT_DOUBLE_EQ(ws2.var("mu")->getError(), 0.12);
   EXPECT_DOUBLE_EQ(ws2.var("sigma")->getError(), 0.20);
   EXPECT_FALSE(ws2.var("x")->hasError());
}

TEST(RooFitHS3, FixedRangeParameterExportsConst)
{
   RooWorkspace ws{"ws_fixed_range"};
   RooRealVar x{"x", "x", 0.0, -5.0, 5.0};
   RooRealVar fixed{"fixed", "fixed", 1.0, 1.0, 1.0};
   RooRealVar sigma{"sigma", "sigma", 1.0, 0.1, 10.0};
   RooGaussian gauss{"gauss", "gauss", x, fixed, sigma};
   fixed.setConstant(false);
   ws.import(gauss, RooFit::Silence());

   EXPECT_TRUE(matches(Doc{ws}, "parameter_points/default_values/parameters/fixed",
                       R"({"const": true, "min": "<absent>", "max": "<absent>"})"));
}

TEST(RooFitHS3, ParameterStepWidthsImportAfterDefaultSnapshot)
{
   const std::string stepWidths = R"("misc": {"minimization": {"parameter_stepwidths": [
      {"name": "mu", "step_width": 0.42}, {"name": "missing", "step_width": 1.0}]}})";
   Imported ws{document("0.1.90", {defaultValues({R"({"name": "mu", "value": 0.0, "err": 0.01})"}), stepWidths})};

   ASSERT_NE(ws.var("mu"), nullptr);
   EXPECT_TRUE(ws.var("mu")->hasError());
   EXPECT_DOUBLE_EQ(ws.var("mu")->getError(), 0.42);
}

TEST(RooFitHS3, RooGaussianConstVarSigmaExport)
{

   RooRealVar x{"x", "x", 0.0, -10.0, 10.0};
   RooRealVar mean{"mean", "mean", 0.0};
   mean.setConstant(true);

   RooConstVar sigmaConst{"sigma_const", "sigma_const", 1.0};
   RooGaussian gaussConst{"gauss_const", "gauss_const", x, mean, sigmaConst};

   RooGaussian gaussLiteral{"gauss_literal", "gauss_literal", x, mean, RooFit::RooConst(2.0)};

   RooRealVar sigmaReal{"sigma_real", "sigma_real", 1.0, 0.1, 10.0};
   sigmaReal.setConstant(true);
   RooGaussian gaussReal{"gauss_real", "gauss_real", x, mean, sigmaReal};

   RooWorkspace ws;
   ws.import(gaussConst, RooFit::Silence());
   ws.import(gaussLiteral, RooFit::RecycleConflictNodes(), RooFit::Silence());
   ws.import(gaussReal, RooFit::RecycleConflictNodes(), RooFit::Silence());

   Doc doc{ws};
   // A RooConstVar with a name is referred to by its name, a literal one is written as a number.
   EXPECT_TRUE(matches(doc, "distributions/gauss_const", R"({"sigma": "sigma_const"})"));
   EXPECT_TRUE(matches(doc, "distributions/gauss_literal", R"({"sigma": 2.0})"));
   EXPECT_TRUE(matches(doc, "distributions/gauss_real", R"({"sigma": "sigma_real"})"));
   EXPECT_THAT(doc.json(), Not(HasSubstr("is_const_var")));
   // The named RooConstVar is a domain axis with a single value, and not in the parameter points.
   EXPECT_EQ(doc.dump("domains/default_domain/axes/sigma_const"), R"({"name":"sigma_const","value":1.0})");
   EXPECT_THAT(doc.dump("parameter_points"), Not(HasSubstr("sigma_const")));
   // The unbounded constant RooRealVar is still a RooRealVar, so it gets a domain axis without bounds that
   // distinguishes it from a RooConstVar.
   EXPECT_TRUE(doc.has("domains/default_domain/axes/sigma_real"));
   EXPECT_TRUE(matches(doc, "domains/default_domain/axes/mean", R"({"min": "<absent>", "max": "<absent>"})"));

   Imported imported{doc};
   EXPECT_NE(dynamic_cast<RooConstVar *>(imported.obj("sigma_const")), nullptr);
   EXPECT_EQ(imported.var("sigma_const"), nullptr);
   EXPECT_NE(dynamic_cast<RooRealVar *>(imported.obj("sigma_real")), nullptr);
   EXPECT_NE(dynamic_cast<RooRealVar *>(imported.obj("mean")), nullptr);

   Doc roundTrip{imported};
   EXPECT_TRUE(matches(roundTrip, "distributions/gauss_const", R"({"sigma": "sigma_const"})"));
   EXPECT_THAT(roundTrip.json(), Not(HasSubstr("is_const_var")));
   EXPECT_EQ(roundTrip.dump("domains/default_domain/axes/sigma_const"), R"({"name":"sigma_const","value":1.0})");

   // Before HS3 0.3, a constant without domain was a parameter.
   Imported legacyImport{document(
      "0.2",
      {defaultValues({values({{"x", 0.0}, {"mean", 0.0}}), R"({"name": "sigma_const", "value": 1.0, "const": true})"}),
       distributions(
          {R"({"name": "gauss", "type": "gaussian_dist", "x": "x", "mean": "mean", "sigma": "sigma_const"})"})})};
   EXPECT_NE(dynamic_cast<RooRealVar *>(legacyImport.obj("sigma_const")), nullptr);
   EXPECT_EQ(dynamic_cast<RooConstVar *>(legacyImport.obj("sigma_const")), nullptr);
}

TEST(RooFitHS3, RooConstVarCollectionProxyExport)
{

   RooRealVar x{"x", "x", 0.0, -10.0, 10.0};
   RooRealVar mean1{"mean1", "mean1", -1.0};
   RooRealVar mean2{"mean2", "mean2", 1.0};
   RooRealVar sigma{"sigma", "sigma", 1.0, 0.1, 10.0};

   RooGaussian g1{"g1", "g1", x, mean1, sigma};
   RooGaussian g2{"g2", "g2", x, mean2, sigma};
   RooConstVar frac{"frac_const", "frac_const", 0.25};
   RooAddPdf model{"model", "model", RooArgList{g1, g2}, RooArgList{frac}};

   RooWorkspace ws;
   ws.import(model, RooFit::Silence());

   Doc doc{ws};
   EXPECT_TRUE(matches(doc, "distributions/model", R"({"coefficients": ["frac_const"]})"));
   EXPECT_THAT(doc.json(), Not(HasSubstr("is_const_var")));
   EXPECT_EQ(doc.dump("domains/default_domain/axes/frac_const"), R"({"name":"frac_const","value":0.25})");

   Imported imported{doc};
   EXPECT_NE(dynamic_cast<RooConstVar *>(imported.obj("frac_const")), nullptr);
   EXPECT_EQ(imported.var("frac_const"), nullptr);
}

TEST(RooFitHS3, BinnedGenericPdfUniformRoundTrip)
{
   RooRealVar x{"x", "x", 0.0, 10.0};
   RooGenericPdf pdf{"binnedPdf", "binnedPdf", "floor(x / 2.0) + 1.0", RooArgList{x}};
   pdf.setBinning(x, RooUniformBinning{0.0, 10.0, 5}, /*checkFlatness=*/false);

   RooWorkspace source{"source"};
   source.import(pdf, RooFit::Silence());

   Doc doc{source};
   EXPECT_TRUE(matches(doc, "distributions/binnedPdf/axes/x", R"({"min": 0.0, "max": 10.0, "edges": "<absent>"})"));
   // With JSONNode::val_int(), which the tests don't cover otherwise.
   EXPECT_EQ(doc["distributions/binnedPdf/axes/x/nbins"].val_int(), 5);

   Imported imported{doc};
   auto *importedPdf = object<RooGenericPdf>(imported, "binnedPdf");
   auto *importedX = object<RooRealVar>(imported, "x");
   ASSERT_TRUE(importedPdf->isBinnedDistribution(RooArgSet{*importedX}));
   const RooAbsBinning *binning = importedPdf->getBinning(*importedX);
   ASSERT_NE(binning, nullptr);
   EXPECT_TRUE(binning->isUniform());
   EXPECT_EQ(binning->numBins(), 5);
   EXPECT_DOUBLE_EQ(binning->lowBound(), 0.0);
   EXPECT_DOUBLE_EQ(binning->highBound(), 10.0);
}

TEST(RooFitHS3, BinnedFormulaVarUniformRoundTripAndTypeAliases)
{
   RooRealVar x{"x", "x", 0.0, 6.0};
   RooFormulaVar formula{"binnedFunction", "binnedFunction", "floor(x / 2.0)", RooArgList{x}};
   formula.setBinning(x, RooUniformBinning{0.0, 6.0, 3}, /*checkFlatness=*/false);

   RooWorkspace source{"source"};
   source.import(formula, RooFit::Silence());

   Doc doc{source};
   EXPECT_TRUE(matches(doc, "functions/binnedFunction", R"({"type": "generic"})"));
   ASSERT_TRUE(doc.has("functions/binnedFunction/axes"));

   Imported legacyImport{doc};
   auto *legacyFormula = object<RooFormulaVar>(legacyImport, "binnedFunction");
   ASSERT_NE(legacyFormula->getBinning(*legacyImport.var("x")), nullptr);

   Imported standardImport{replaced(doc, R"("type":"generic")", R"("type":"generic_function")")};
   auto *standardFormula = object<RooFormulaVar>(standardImport, "binnedFunction");
   auto *standardX = object<RooRealVar>(standardImport, "x");
   ASSERT_TRUE(standardFormula->isBinnedDistribution(RooArgSet{*standardX}));
   ASSERT_NE(standardFormula->getBinning(*standardX), nullptr);
   EXPECT_EQ(standardFormula->getBinning(*standardX)->numBins(), 3);
}

TEST(RooFitHS3, BinnedGenericPdfMultipleAxesAndExplicitEdgesRoundTrip)
{
   RooRealVar x{"x", "x", 2.0, 8.0};
   RooRealVar y{"y", "y", 0.0, 6.0};
   RooGenericPdf pdf{"binnedPdf", "binnedPdf", "floor(x) + floor(y)", RooArgList{x, y}};

   const double xEdges[] = {1.0, 3.0, 5.0, 9.0};
   pdf.setBinning(x, RooBinning{3, xEdges}, /*checkFlatness=*/false);
   pdf.setBinning(y, RooUniformBinning{0.0, 6.0, 3}, /*checkFlatness=*/false);

   RooWorkspace source{"source"};
   source.import(pdf, RooFit::Silence());

   Doc doc{source};
   EXPECT_EQ(doc["distributions/binnedPdf/axes"].num_children(), 2u);
   EXPECT_TRUE(matches(doc, "distributions/binnedPdf/axes/x", R"({"edges": [1.0, 3.0, 5.0, 9.0]})"));

   Imported imported{doc};
   auto *importedPdf = object<RooGenericPdf>(imported, "binnedPdf");
   auto *importedX = imported.var("x");
   auto *importedY = imported.var("y");
   ASSERT_NE(importedX, nullptr);
   ASSERT_NE(importedY, nullptr);
   EXPECT_DOUBLE_EQ(importedX->getMin(), 2.0);
   EXPECT_DOUBLE_EQ(importedX->getMax(), 8.0);
   EXPECT_TRUE(importedPdf->isBinnedDistribution(RooArgSet{*importedX, *importedY}));

   const RooAbsBinning *xBinning = importedPdf->getBinning(*importedX);
   ASSERT_NE(xBinning, nullptr);
   ASSERT_EQ(xBinning->numBoundaries(), 4);
   for (int i = 0; i < xBinning->numBoundaries(); ++i) {
      EXPECT_DOUBLE_EQ(xBinning->array()[i], xEdges[i]);
   }
   EXPECT_DOUBLE_EQ(xBinning->lowBound(), 1.0);
   EXPECT_DOUBLE_EQ(xBinning->highBound(), 9.0);
}

TEST(RooFitHS3, GenericFormulaWithoutBinningHasNoAxes)
{
   RooRealVar x{"x", "x", 0.0, 10.0};
   RooGenericPdf pdf{"ordinaryPdf", "ordinaryPdf", "floor(x) + 1.0", RooArgList{x}};
   RooFormulaVar function{"ordinaryFunction", "ordinaryFunction", "floor(x)", RooArgList{x}};

   RooWorkspace source{"source"};
   source.import(pdf, RooFit::Silence());
   source.import(function, RooFit::Silence());

   Doc doc{source};
   EXPECT_FALSE(doc.has("distributions/ordinaryPdf/axes"));
   EXPECT_FALSE(doc.has("functions/ordinaryFunction/axes"));
}

TEST(RooFitHS3, BinnedGenericRejectsMalformedAxes)
{
   RooRealVar x{"x", "x", 0.0, 10.0};
   RooGenericPdf pdf{"binnedPdf", "binnedPdf", "floor(x)", RooArgList{x}};
   pdf.setBinning(x, RooUniformBinning{0.0, 10.0, 5}, /*checkFlatness=*/false);

   RooWorkspace source{"source"};
   source.import(pdf, RooFit::Silence());
   const std::string validJson = toJSON(source);

   // What is rejected in the place of the exported "axes" of the pdf.
   const std::vector<std::string> malformedAxes{
      R"("not-a-sequence")",
      // An axis twice.
      R"([{"name": "x", "min": 0.0, "max": 10.0, "nbins": 5}, {"name": "x", "min": 0.0, "max": 10.0, "nbins": 5}])",
      R"([{"name": "notADependency", "min": 0.0, "max": 10.0, "nbins": 5}])",
      R"([{"name": "x", "min": 0.0, "max": 10.0, "nbins": 2.5}])",
      R"([{"name": "x", "min": 0.0, "max": 10.0, "nbins": 0}])",
      R"([{"name": "x", "min": 0.0, "max": 0.0, "nbins": 5}])",
      R"([{"name": "x", "edges": [0.0]}])",
      R"([{"name": "x", "edges": [0.0, 2.0, 1.0]}])",
      // Non-numeric bounds must be rejected cleanly rather than throwing a backend-specific exception or being
      // silently coerced to zero.
      R"([{"name": "x", "min": "not-a-number", "max": 10.0, "nbins": 5}])",
      R"([{"name": "x", "edges": ["not-a-number", 1.0, 2.0]}])",
   };
   const std::string validAxes = R"("axes":[{"max":10.0,"min":0.0,"name":"x","nbins":5}])";
   for (auto const &axes : malformedAxes) {
      SCOPED_TRACE(axes);
      // Outside of EXPECT_THROW, such that the test fails if the exported "axes" are not found.
      const std::string malformed = replaced(validJson, validAxes, "\"axes\":" + axes);
      EXPECT_THROW(Imported{malformed}, std::runtime_error);
   }
}

TEST(RooFitHS3, BinnedGenericAcceptsFloatEncodedNBins)
{
   RooRealVar x{"x", "x", 0.0, 10.0};
   RooGenericPdf pdf{"binnedPdf", "binnedPdf", "floor(x)", RooArgList{x}};
   pdf.setBinning(x, RooUniformBinning{0.0, 10.0, 5}, /*checkFlatness=*/false);

   RooWorkspace source{"source"};
   source.import(pdf, RooFit::Silence());

   // Encode an integer bin count as a JSON float; its textual form is "1e+06",
   // which the previous strict integer parser wrongly rejected.
   Imported imported{replaced(toJSON(source), R"("nbins":5)", R"("nbins":1000000.0)")};
   auto *importedPdf = object<RooGenericPdf>(imported, "binnedPdf");
   const RooAbsBinning *binning = importedPdf->getBinning(*imported.var("x"));
   ASSERT_NE(binning, nullptr);
   EXPECT_TRUE(binning->isUniform());
   EXPECT_EQ(binning->numBins(), 1000000);
}

namespace {

// The round trip of a formula whose expression is cleaned up. It is read back from another expression than the
// original one, so the values are compared up to a relative 2 epsilon: at most the 4 ULP that EXPECT_DOUBLE_EQ accepts.
const RoundTripModel cleanedUpFormula =
   RoundTripModel{"formula", RoundTripModel::Builder{}}.within(2 * std::numeric_limits<double>::epsilon());

} // namespace

TEST(RooFitHS3, GenericExpressionCleanup)
{
   RooRealVar x{"x", "x", 0.5, -1.0, 1.0};
   RooFormulaVar formula{"formula", "formula",
                         "TMath::Floor(x) + TMath::Ceil(x) + TMath::Abs(x) + TMath::Tan(x) + "
                         "TMath::ASin(x / 2.) + TMath::ACos(x / 2.) + TMath::ATan(x) + TMath::Pi() + TMath::E()",
                         RooArgList{x}};

   RooWorkspace ws1{"ws_expr_cleanup"};
   ws1.import(formula, RooFit::Silence());

   Doc doc{ws1};
   EXPECT_EQ(doc["functions/formula/expression"].val(),
             "floor(x) + ceil(x) + abs(x) + tan(x) + asin(x / 2.) + acos(x / 2.) + atan(x) + PI + EULER");
   EXPECT_THAT(doc.json(), Not(HasSubstr("TMath::Pi")));
   EXPECT_THAT(doc.json(), Not(HasSubstr("TMath::E")));
   expectRoundTrip(ws1, cleanedUpFormula);
}

// The expression cleanup must not corrupt identifiers that merely share a
// prefix with a replaced one, like TMath::PiOver2 ("TMath::Pi") or
// TMath::TanH ("TMath::Tan").
TEST(RooFitHS3, GenericExpressionCleanupKeepsLongerIdentifiers)
{
   RooRealVar x{"x", "x", 0.5, -1.0, 1.0};
   RooFormulaVar formula{"formula", "formula",
                         "TMath::PiOver2() + TMath::TanH(x) + TMath::SinH(x) + TMath::Log10(2. + x)", RooArgList{x}};

   RooWorkspace ws1{"ws_expr_cleanup_prefixes"};
   ws1.import(formula, RooFit::Silence());

   Doc doc{ws1};
   EXPECT_EQ(doc["functions/formula/expression"].val(), "TMath::PiOver2() + tanh(x) + sinh(x) + log10(2. + x)");
   for (const char *token : {"PIOver2", "tanH", "sinH"}) {
      EXPECT_THAT(doc.json(), Not(HasSubstr(token)));
   }
   expectRoundTrip(ws1, cleanedUpFormula);
}

TEST(RooFitHS3, RooBinWidthFunctionUsesBinVolumeKeys)
{
   RooRealVar x{"x", "x", 0.0, 2.0};
   x.setBins(2);

   RooDataHist dataHist{"dataHist", "dataHist", x};
   dataHist.set(0, 2.0, -1);
   dataHist.set(1, 4.0, -1);

   RooHistFunc histFunc{"histFunc", "histFunc", x, dataHist};
   RooBinWidthFunction binVolume{"binVolume", "binVolume", histFunc, false};
   RooBinWidthFunction inverseBinVolume{"inverseBinVolume", "inverseBinVolume", histFunc, true};

   RooWorkspace ws1{"ws_binvolume"};
   ws1.import(binVolume, RooFit::Silence());
   ws1.import(inverseBinVolume, RooFit::Silence(), RooFit::RecycleConflictNodes());

   Doc doc{ws1};
   EXPECT_TRUE(matches(doc, "functions/binVolume", R"({"type": "binvolume"})"));
   EXPECT_TRUE(matches(doc, "functions/inverseBinVolume", R"({"type": "inverse_binvolume"})"));
   EXPECT_THAT(doc.json(), Not(HasSubstr("divideByBinWidth")));
   EXPECT_THAT(doc.json(), Not(HasSubstr("\"type\":\"binwidth\"")));

   Imported ws2{doc};
   auto *importedBinVolume = dynamic_cast<RooBinWidthFunction *>(ws2.function("binVolume"));
   auto *importedInverseBinVolume = dynamic_cast<RooBinWidthFunction *>(ws2.function("inverseBinVolume"));
   ASSERT_NE(importedBinVolume, nullptr);
   ASSERT_NE(importedInverseBinVolume, nullptr);
   EXPECT_FALSE(importedBinVolume->divideByBinWidth());
   EXPECT_TRUE(importedInverseBinVolume->divideByBinWidth());
}

TEST(RooFitHS3, StepDispatchesToRooHistFuncAndParamHistFunc)
{
   RooRealVar x{"x", "x", 0.0, 2.0};
   x.setBins(2);

   RooDataHist dataHist{"dataHist", "dataHist", x};
   dataHist.set(0, 3.0, -1);
   dataHist.set(1, 5.0, -1);

   RooHistFunc histFunc{"histFunc", "histFunc", x, dataHist};
   RooRealVar p0{"p0", "p0", 1.0};
   RooRealVar p1{"p1", "p1", 2.0};
   ParamHistFunc paramHistFunc{"paramHistFunc", "paramHistFunc", RooArgList{x}, RooArgList{p0, p1}};

   RooWorkspace ws1{"ws_step"};
   ws1.import(histFunc, RooFit::Silence());
   ws1.import(paramHistFunc, RooFit::Silence());

   Doc doc{ws1};
   EXPECT_TRUE(matches(doc, "functions/histFunc", R"({"type": "step"})"));
   EXPECT_THAT(doc.json(), Not(HasSubstr("\"name\":\"histFunc\",\"type\":\"histogram\"")));
   EXPECT_TRUE(matches(doc, "functions/paramHistFunc", R"({"type": "step"})"));

   Imported ws2{doc};
   EXPECT_NE(dynamic_cast<RooHistFunc *>(ws2.function("histFunc")), nullptr);
   EXPECT_NE(dynamic_cast<ParamHistFunc *>(ws2.function("paramHistFunc")), nullptr);
}

// HS3 0.3 splits the Poisson distribution into the discrete "poisson_dist" and the "poisson_continuous_dist" (HS3
// #152), which are a RooPoisson without and with noRounding. Documents before HS3 0.3 used "poisson_dist" for both,
// distinguished by ROOT's own key "integer", which must not be present from HS3 0.3 on.
TEST(RooFitHS3, PoissonContinuous)
{
   RooWorkspace ws;
   ws.factory("Poisson::discrete(n[3, 0, 10], lambda[2.5, 0, 10])");
   ws.factory("Poisson::continuous(k[3.5, 0, 10], lambda)");
   static_cast<RooPoisson *>(ws.pdf("continuous"))->setNoRounding(true);

   Doc exported{ws};
   EXPECT_TRUE(matches(exported, "distributions/discrete", R"({"type": "poisson_dist", "integer": "<absent>"})"));
   EXPECT_TRUE(
      matches(exported, "distributions/continuous", R"({"type": "poisson_continuous_dist", "integer": "<absent>"})"));

   Imported ws2{exported};
   EXPECT_FALSE(static_cast<RooPoisson *>(ws2.pdf("discrete"))->getNoRounding());
   EXPECT_TRUE(static_cast<RooPoisson *>(ws2.pdf("continuous"))->getNoRounding());
   EXPECT_DOUBLE_EQ(ws2.pdf("continuous")->getVal(), ws.pdf("continuous")->getVal());
   EXPECT_EQ(toJSON(ws2), exported.json());

   auto poisson = [](std::string const &version, std::string const &type, std::string const &integer = "") {
      return document(version, {distributions({R"({"name": "pois", "type": ")" + type + R"(", "x": "n", "mean": 2.5)" +
                                               integer + "}"}),
                                defaultDomain({axes({"n"}, 0, 10)})});
   };

   // Before HS3 0.3: "integer" false is the continuous form, true or no "integer" the discrete form.
   EXPECT_EQ(reexport(poisson("0.2", "poisson_dist", R"(, "integer": false)")),
             reexport(poisson("0.3", "poisson_continuous_dist")));
   EXPECT_EQ(reexport(poisson("0.2", "poisson_dist", R"(, "integer": true)")),
             reexport(poisson("0.3", "poisson_dist")));
   EXPECT_EQ(reexport(poisson("0.2", "poisson_dist")), reexport(poisson("0.3", "poisson_dist")));

   // From HS3 0.3 on, and for the continuous type, which was never written with it, "integer" is an error.
   const std::string reason = "has the key \"integer\", which is not part of HS3";
   expectImportThrowsWithError(poisson("0.3", "poisson_dist", R"(, "integer": false)"), reason);
   expectImportThrowsWithError(poisson("0.3", "poisson_dist", R"(, "integer": true)"), reason);
   expectImportThrowsWithError(poisson("0.2", "poisson_continuous_dist", R"(, "integer": false)"), reason);
}

TEST(RooFitHS3, RooPolynomial)
{
   RooWorkspace ws;
   ws.factory("Polynomial::poly1(x[0, 10], {a_2[0.003, -10, 10]}, 2)");
   // The coefficients below the lowest order are numbers, not strings.
   Doc doc{ws};
   EXPECT_TRUE(matches(doc, "distributions/poly1", R"({"coefficients": [1.0, 0.0, "a_2"]})"));
   EXPECT_THAT(doc.json(), Not(HasSubstr("\"coefficients\":[\"1.0\"")));
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

   // try to import, before the fix, it threw RooJSONFactoryWSTool::DependencyMissingError because of problem reading
   // the exponential char
   Imported newws{toJSON(ws)};
}

namespace {

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
   registerExporter<TestExporter<1>>(klass, /*topPriority=*/true);
   toJSON(ws);
   EXPECT_EQ(TestExporter<1>::callCounter()--, 1);

   // 2. Add new exporter by class pointer with bottom priority.
   //    We expect the previous TestExporter<1> to still be used.
   registerExporter<TestExporter<2>>(klass, /*topPriority=*/false);
   toJSON(ws);
   EXPECT_EQ(TestExporter<1>::callCounter()--, 1);

   // 3. Add new exporter by name with top priority.
   //    We expect this to get used.
   registerExporter<TestExporter<3>>(std::string{className}, /*topPriority=*/true);
   toJSON(ws);
   EXPECT_EQ(TestExporter<3>::callCounter()--, 1);

   // 4. Add new exporter by name with bottom priority.
   //    We expect the previous TestExporter<3> to still be used.
   registerExporter<TestExporter<4>>(std::string{className}, /*topPriority=*/false);
   toJSON(ws);
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

   Doc doc{ws1};
   Imported ws2{doc};

   auto *ds2 = object<RooDataSet>(ws2, "ds");
   EXPECT_EQ(ds2->numEntries(), 5);

   auto *x2 = object<RooRealVar>(ws2, "x");
   EXPECT_DOUBLE_EQ(x2->getMin(), xMin);
   EXPECT_DOUBLE_EQ(x2->getMax(), xMax);

   // The exported "axes" node of an unbinned dataset must carry the
   // observable range so the file is self-describing. Before the fix, only
   // the variable name and current value were written there (the range was
   // only present in the separate "domains" block).
   EXPECT_TRUE(matches(doc, "data/ds/axes", R"([{"name": "x", "min": -2.5, "max": 7.5, "value": "<absent>"}])"));
}

TEST(RooFitHS3, StandaloneFlexibleInterpVarStructuredInterpolations)
{
   RooRealVar alpha{"alpha", "alpha", 0.25, -2.0, 2.0};
   RooRealVar beta{"beta", "beta", -0.4, -2.0, 2.0};
   RooArgList parameters{alpha, beta};
   RooStats::HistFactory::FlexibleInterpVar shared{"fiv_shared", "", parameters, 1.0, {0.8, 0.7}, {1.2, 1.4}, {4, 5}};
   RooStats::HistFactory::FlexibleInterpVar mixed{"fiv_mixed", "", parameters, 1.0, {0.8, 0.7}, {1.2, 1.4}, {0, 1}};

   RooWorkspace source;
   source.import(shared, RooFit::Silence());
   source.import(mixed, RooFit::RecycleConflictNodes(), RooFit::Silence());

   // One interpolation for all parameters, or one per parameter.
   Doc doc{source};
   EXPECT_TRUE(matches(doc, "functions/fiv_shared", R"({
      "interpolationCodes": "<absent>", "interpolations": [{"type": "mult", "in": "poly6", "out": "exp"}]})"));
   EXPECT_TRUE(matches(doc, "functions/fiv_mixed", R"({
      "interpolationCodes": "<absent>",
      "interpolations": [{"type": "add", "in": "poly1", "out": null}, {"type": "mult", "in": "exp", "out": null}]})"));

   Imported imported{doc};
   auto *importedShared = dynamic_cast<RooStats::HistFactory::FlexibleInterpVar *>(imported.function("fiv_shared"));
   auto *importedMixed = dynamic_cast<RooStats::HistFactory::FlexibleInterpVar *>(imported.function("fiv_mixed"));
   ASSERT_NE(importedShared, nullptr);
   ASSERT_NE(importedMixed, nullptr);
   EXPECT_EQ(importedShared->interpolationCodes(), (std::vector<int>{4, 4}));
   EXPECT_EQ(importedMixed->interpolationCodes(), (std::vector<int>{0, 1}));
   expectRoundTrip(source);
}

TEST(RooFitHS3, StandalonePiecewiseInterpolationStructuredInterpolations)
{
   RooRealVar alpha{"alpha", "alpha", 0.25, -2.0, 2.0};
   RooRealVar beta{"beta", "beta", -0.4, -2.0, 2.0};
   RooRealVar nominal{"nominal", "nominal", 10.0};
   RooRealVar low1{"low1", "low1", 8.0};
   RooRealVar low2{"low2", "low2", 7.0};
   RooRealVar high1{"high1", "high1", 12.0};
   RooRealVar high2{"high2", "high2", 14.0};
   RooArgList lows{low1, low2};
   RooArgList highs{high1, high2};

   PiecewiseInterpolation shared{"pip_shared", "", nominal, lows, highs, RooArgList{alpha, beta}, {4, 4}};
   PiecewiseInterpolation mixedRepeated{"pip_mixed_repeated",     "",    nominal, lows, highs,
                                        RooArgList{alpha, alpha}, {0, 2}};
   mixedRepeated.setPositiveDefinite();

   RooWorkspace source;
   source.import(shared, RooFit::Silence());
   source.import(mixedRepeated, RooFit::RecycleConflictNodes(), RooFit::Silence());

   Doc doc{source};
   EXPECT_TRUE(matches(doc, "functions/pip_shared", R"({
      "interpolationCodes": "<absent>", "interpolations": [{"type": "add", "in": "poly6", "out": "poly1"}]})"));
   EXPECT_TRUE(matches(doc, "functions/pip_mixed_repeated", R"({
      "interpolationCodes": "<absent>",
      "interpolations": [{"type": "add", "in": "poly1", "out": null},
                         {"type": "add", "in": "poly2", "out": "poly1"}]})"));

   Imported imported{doc};
   auto *importedShared = dynamic_cast<PiecewiseInterpolation *>(imported.function("pip_shared"));
   auto *importedMixed = dynamic_cast<PiecewiseInterpolation *>(imported.function("pip_mixed_repeated"));
   ASSERT_NE(importedShared, nullptr);
   ASSERT_NE(importedMixed, nullptr);
   EXPECT_EQ(importedShared->interpolationCodes(), (std::vector<int>{4, 4}));
   EXPECT_EQ(importedMixed->interpolationCodes(), (std::vector<int>{0, 2}));
   ASSERT_EQ(importedMixed->paramList().size(), 2u);
   EXPECT_STREQ(importedMixed->paramList().at(0)->GetName(), "alpha");
   EXPECT_STREQ(importedMixed->paramList().at(1)->GetName(), "alpha");
   EXPECT_TRUE(importedMixed->positiveDefinite());
   expectRoundTrip(source);
}

namespace {

// A document with the given functions, and with values for the parameters of standalone interpolations.
std::string standaloneInterpolationDoc(std::string const &functionList)
{
   return document("0.2", {defaultValues({values({{"alpha", 0.25},
                                                  {"beta", -0.4},
                                                  {"nominal", 10.0},
                                                  {"low1", 8.0},
                                                  {"low2", 7.0},
                                                  {"high1", 12.0},
                                                  {"high2", 14.0}})}),
                           functions({functionList})});
}

} // namespace

TEST(RooFitHS3, StandaloneInterpolationLegacyAndDefaults)
{
   const std::string functionList = R"(
      {
         "name":"legacy_fiv", "type":"interpolation0d", "vars":["alpha","beta"], "nom":1.0,
         "low":[0.8,0.7], "high":[1.2,1.4], "interpolationCodes":[5,3]
      },
      {
         "name":"default_fiv", "type":"interpolation0d", "vars":["alpha","beta"], "nom":1.0,
         "low":[0.8,0.7], "high":[1.2,1.4]
      },
      {
         "name":"precedence_fiv", "type":"interpolation0d", "vars":["alpha","beta"], "nom":1.0,
         "low":[0.8,0.7], "high":[1.2,1.4],
         "interpolations":[{"type":"add","in":"poly1","out":null}], "interpolationCodes":[4,4]
      },
      {
         "name":"legacy_pip", "type":"interpolation", "vars":["alpha","alpha"], "nom":"nominal",
         "low":["low1","low2"], "high":["high1","high2"], "positiveDefinite":true,
         "interpolationCodes":[0,2]
      },
      {
         "name":"default_pip", "type":"interpolation", "vars":["alpha","beta"], "nom":"nominal",
         "low":["low1","low2"], "high":["high1","high2"], "positiveDefinite":false
      })";

   Imported imported{standaloneInterpolationDoc(functionList)};

   auto *legacyFiv = dynamic_cast<RooStats::HistFactory::FlexibleInterpVar *>(imported.function("legacy_fiv"));
   auto *defaultFiv = dynamic_cast<RooStats::HistFactory::FlexibleInterpVar *>(imported.function("default_fiv"));
   auto *precedenceFiv = dynamic_cast<RooStats::HistFactory::FlexibleInterpVar *>(imported.function("precedence_fiv"));
   auto *legacyPip = dynamic_cast<PiecewiseInterpolation *>(imported.function("legacy_pip"));
   auto *defaultPip = dynamic_cast<PiecewiseInterpolation *>(imported.function("default_pip"));
   ASSERT_NE(legacyFiv, nullptr);
   ASSERT_NE(defaultFiv, nullptr);
   ASSERT_NE(precedenceFiv, nullptr);
   ASSERT_NE(legacyPip, nullptr);
   ASSERT_NE(defaultPip, nullptr);
   EXPECT_EQ(legacyFiv->interpolationCodes(), (std::vector<int>{4, 2}));
   EXPECT_EQ(defaultFiv->interpolationCodes(), (std::vector<int>{0, 0}));
   EXPECT_EQ(precedenceFiv->interpolationCodes(), (std::vector<int>{0, 0}));
   EXPECT_EQ(legacyPip->interpolationCodes(), (std::vector<int>{0, 2}));
   EXPECT_EQ(defaultPip->interpolationCodes(), (std::vector<int>{0, 0}));
   EXPECT_TRUE(legacyPip->positiveDefinite());

   Doc exported{imported};
   EXPECT_THAT(exported.json(), Not(HasSubstr("interpolationCodes")));
   EXPECT_EQ(exported["functions/legacy_fiv/interpolations"].num_children(), 2u);
   EXPECT_EQ(exported["functions/legacy_pip/interpolations"].num_children(), 2u);
}

TEST(RooFitHS3, StandaloneInterpolationRejectsMalformedLengths)
{
   const std::string descriptor = R"({"type":"add","in":"poly1","out":null})";
   const std::string fiv = R"({"name":"bad","type":"interpolation0d","vars":["alpha","beta"],"nom":1.0,)"
                           R"("low":[0.8,0.7],"high":[1.2,1.4],)";
   const std::string pip = R"({"name":"bad","type":"interpolation","vars":["alpha","beta"],"nom":"nominal",)"
                           R"("low":["low1","low2"],"high":["high1","high2"],"positiveDefinite":false,)";
   const std::string wrongNumber =
      "component 'interpolations' must contain either one descriptor or one descriptor per parameter";

   expectImportThrowsWithError(standaloneInterpolationDoc(fiv + R"("interpolations":[]})"), wrongNumber);
   expectImportThrowsWithError(
      standaloneInterpolationDoc(pip + R"("interpolations":[)" + join({descriptor, descriptor, descriptor}) + "]}"),
      wrongNumber);
   expectImportThrowsWithError(standaloneInterpolationDoc(fiv + R"("interpolationCodes":[4]})"),
                               "legacy component 'interpolationCodes' must contain one code per parameter");
}

TEST(RooFitHS3, HistFactoryInterpolationCodeMapping)
{
   struct TestCase {
      const char *modifierType;
      int inputCode;
      const char *interpolation;
      int canonicalCode;
   };
   const TestCase testCases[]{
      {"normsys", 0, R"({"type": "add", "in": "poly1", "out": null})", 0},
      {"normsys", 1, R"({"type": "mult", "in": "exp", "out": null})", 1},
      {"normsys", 2, R"({"type": "add", "in": "poly2", "out": "poly1"})", 2},
      {"normsys", 3, R"({"type": "add", "in": "poly2", "out": "poly1"})", 2},
      {"normsys", 4, R"({"type": "mult", "in": "poly6", "out": "exp"})", 4},
      {"normsys", 5, R"({"type": "mult", "in": "poly6", "out": "exp"})", 4},
      {"histosys", 0, R"({"type": "add", "in": "poly1", "out": null})", 0},
      {"histosys", 1, R"({"type": "mult", "in": "exp", "out": null})", 1},
      {"histosys", 2, R"({"type": "add", "in": "poly2", "out": "poly1"})", 2},
      {"histosys", 3, R"({"type": "add", "in": "poly2", "out": "poly1"})", 2},
      {"histosys", 4, R"({"type": "add", "in": "poly6", "out": "poly1"})", 4},
      {"histosys", 5, R"({"type": "mult", "in": "poly6", "out": "exp"})", 5},
      {"histosys", 6, R"({"type": "mult", "in": "poly6", "out": "poly1"})", 6},
   };

   for (const auto &testCase : testCases) {
      SCOPED_TRACE(std::string(testCase.modifierType) + " code " + std::to_string(testCase.inputCode));
      const bool isNormSys = std::string_view{testCase.modifierType} == "normsys";
      const std::string data =
         isNormSys ? R"({"lo":0.8,"hi":1.2})" : R"({"lo":{"contents":[8.0,18.0]},"hi":{"contents":[12.0,22.0]}})";
      const std::string modifier = R"({"name":"modifier","parameter":"alpha_modifier","type":")" +
                                   std::string(testCase.modifierType) + R"(","interpolation":)" +
                                   std::to_string(testCase.inputCode) + R"(,"data":)" + data + "}";

      // The code is written as the default interpolation of the channel, which the only modifier takes.
      const Doc exported{reexport(HistFactoryDoc{"0.2", modifier})};
      EXPECT_TRUE(matches(exported, "distributions/model_channel0/default_interpolation", testCase.interpolation));
      EXPECT_FALSE(exported.has("distributions/model_channel0/samples/sig/modifiers/modifier/interpolation"));

      Imported roundTripped{exported};
      if (isNormSys) {
         auto *interpolation = object<RooStats::HistFactory::FlexibleInterpVar>(roundTripped, "sig_channel0_epsilon");
         EXPECT_EQ(interpolation->interpolationCodes(), std::vector<int>{testCase.canonicalCode});
      } else {
         auto *interpolation = object<PiecewiseInterpolation>(roundTripped, "histoSys_model_channel0_sig");
         EXPECT_EQ(interpolation->interpolationCodes(), std::vector<int>{testCase.canonicalCode});
      }
   }
}

TEST(RooFitHS3, HistFactoryInterpolationDefaultsAndOverrides)
{
   const std::string modifiers = R"(
      {
         "name":"norm_default_1", "parameter":"alpha_norm_default_1", "type":"normsys",
         "interpolation":4, "data":{"lo":0.8,"hi":1.2}
      },
      {
         "name":"norm_default_2", "parameter":"alpha_norm_default_2", "type":"normsys",
         "interpolation":5, "data":{"lo":0.9,"hi":1.1}
      },
      {
         "name":"norm_linear", "parameter":"alpha_norm_linear", "type":"normsys",
         "interpolation":0, "data":{"lo":0.85,"hi":1.15}
      },
      {
         "name":"shape_default", "parameter":"alpha_shape_default", "type":"histosys",
         "interpolation":4,
         "data":{"lo":{"contents":[8.0,18.0]},"hi":{"contents":[12.0,22.0]}}
      },
      {
         "name":"shape_quadratic", "parameter":"alpha_shape_quadratic", "type":"histosys",
         "interpolation":2,
         "data":{"lo":{"contents":[9.0,17.0]},"hi":{"contents":[11.0,23.0]}}
      })";

   const std::string channel = "distributions/model_channel0";
   // The document that RooFit writes for the channel with the given modifiers.
   auto exportOf = [](std::string const &modifierList) { return Doc{reexport(HistFactoryDoc{"0.2", modifierList})}; };

   const Doc exported = exportOf(modifiers);
   EXPECT_TRUE(matches(exported, channel, R"({
      "default_interpolation": {"type": "mult", "in": "poly6", "out": "exp"},
      "samples": [{"modifiers": [
         {"name": "norm_default_1", "interpolation": "<absent>"},
         {"name": "norm_default_2", "interpolation": "<absent>"},
         {"name": "norm_linear", "interpolation": {"type": "add", "in": "poly1", "out": null}},
         {"name": "shape_default", "interpolation": {"type": "add", "in": "poly6", "out": "poly1"}},
         {"name": "shape_quadratic", "interpolation": {"type": "add", "in": "poly2", "out": "poly1"}}]}]})"));

   Imported roundTripped{exported};
   auto *shapeInterpolation = object<PiecewiseInterpolation>(roundTripped, "histoSys_model_channel0_sig");
   ASSERT_EQ(shapeInterpolation->paramList().size(), 2u);
   for (std::size_t i = 0; i < shapeInterpolation->paramList().size(); ++i) {
      const std::string parameter = shapeInterpolation->paramList().at(i)->GetName();
      if (parameter == "alpha_shape_default") {
         EXPECT_EQ(shapeInterpolation->interpolationCodes()[i], 4);
      } else if (parameter == "alpha_shape_quadratic") {
         EXPECT_EQ(shapeInterpolation->interpolationCodes()[i], 2);
      } else {
         FAIL() << "Unexpected histosys parameter " << parameter;
      }
   }

   const std::string implicitModifiers = R"(
      {"name":"norm","parameter":"alpha_norm","type":"normsys","data":{"lo":0.8,"hi":1.2}},
      {
         "name":"shape","parameter":"alpha_shape","type":"histosys",
         "data":{"lo":{"contents":[8.0,18.0]},"hi":{"contents":[12.0,22.0]}}
      })";
   EXPECT_TRUE(matches(exportOf(implicitModifiers), channel, R"({
      "default_interpolation": {"type": "mult", "in": "poly6", "out": "exp"},
      "samples": [{"modifiers": [
         {"name": "norm", "interpolation": "<absent>"},
         {"name": "shape", "interpolation": {"type": "add", "in": "poly6", "out": "poly1"}}]}]})"));

   const std::string stableTieModifiers = R"(
      {"name":"norm","parameter":"alpha_norm","type":"normsys","interpolation":0,"data":{"lo":0.8,"hi":1.2}},
      {
         "name":"shape","parameter":"alpha_shape","type":"histosys","interpolation":2,
         "data":{"lo":{"contents":[8.0,18.0]},"hi":{"contents":[12.0,22.0]}}
      })";
   EXPECT_TRUE(matches(exportOf(stableTieModifiers), channel + "/default_interpolation",
                       R"({"type": "add", "in": "poly1", "out": null})"));

   const Doc noInterpolation = exportOf(R"({"name":"mu","parameter":"mu","type":"normfactor"})");
   EXPECT_FALSE(noInterpolation.has(channel + "/default_interpolation"));
}

namespace {

// A channel with a normsys that takes the default interpolation of the channel, and a histosys with its own one.
std::string structuredInterpolationJSON()
{
   const std::string modifiers = R"(
      {
         "name":"norm", "parameter":"alpha_norm", "type":"normsys",
         "data":{"lo":0.8,"hi":1.3}
      },
      {
         "name":"shape", "parameter":"alpha_shape", "type":"histosys",
         "interpolation":{"type":"add","in":"poly2","out":"poly1"},
         "data":{"lo":{"contents":[8.0,17.0]},"hi":{"contents":[12.0,23.0]}}
      })";
   return HistFactoryDoc{"0.2", modifiers}.interpolation(R"({"type":"mult","in":"poly6","out":"exp"})");
}

// The yields of the sample in its bins, with the parameters inside of, at the border of and outside of the range
// that the "in" functions interpolate in.
std::vector<double> structuredInterpolationYields(RooWorkspace &ws)
{
   std::vector<double> yields;
   RooArgSet observables{*ws.var("x")};
   for (double theta : {-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0}) {
      ws.var("alpha_norm")->setVal(theta);
      ws.var("alpha_shape")->setVal(theta);
      for (int bin = 0; bin < 2; ++bin) {
         ws.var("x")->setBin(bin);
         yields.push_back(ws.function("model_channel0_sig_shapes")->getVal(observables) *
                          ws.function("model_channel0_sig_scaleFactors")->getVal());
      }
   }
   return yields;
}

} // namespace

// The yields take the place of a likelihood, which the model doesn't have.
INSTANTIATE_TEST_SUITE_P(HistFactoryStructuredInterpolation, RoundTrip,
                         testing::Values(RoundTripModel::fromDocument("NormSysAndHistoSys",
                                                                      structuredInterpolationJSON())
                                            .with(constraintStyles())
                                            .comparing(structuredInterpolationYields)
                                            .within(1e-13)),
                         roundTripName);

TEST(RooFitHS3, HistFactoryStructuredInterpolationRoundTrip)
{
   Imported ws{structuredInterpolationJSON()};
   auto *norm = dynamic_cast<RooStats::HistFactory::FlexibleInterpVar *>(ws.function("sig_channel0_epsilon"));
   auto *shape = dynamic_cast<PiecewiseInterpolation *>(ws.function("histoSys_model_channel0_sig"));
   ASSERT_NE(norm, nullptr);
   ASSERT_NE(shape, nullptr);
   EXPECT_EQ(norm->interpolationCodes()[0], 4);
   EXPECT_EQ(shape->interpolationCodes()[0], 2);
}

TEST(RooFitHS3, HistFactoryStructuredInterpolationValidation)
{
   const std::string normsys =
      R"({"name":"norm","parameter":"alpha_norm","type":"normsys","data":{"lo":0.8,"hi":1.2}})";

   expectImportThrowsWithError(HistFactoryDoc{"0.2", normsys}.interpolation(R"({"type":"mult","in":"poly6"})"),
                               "does not define the required 'out' component");
   expectImportThrowsWithError(
      HistFactoryDoc{"0.2", normsys}.interpolation(R"({"type":"mult","in":"spline","out":null})"),
      "unknown interpolation function 'spline'");
   expectImportThrowsWithError(
      HistFactoryDoc{"0.2", R"({"name":"norm","parameter":"alpha_norm","type":"normsys",)"
                            R"("interpolation":{"type":"add","in":"poly6","out":"poly1"},"data":{"lo":0.8,"hi":1.2}})"},
      "cannot be represented by FlexibleInterpVar");
   expectImportThrowsWithError(
      HistFactoryDoc{"0.2", R"({"name":"shape","parameter":"alpha_shape","type":"histosys",)"
                            R"("interpolation":{"type":"add","in":"exp","out":null},)"
                            R"("data":{"lo":{"contents":[8.0,18.0]},"hi":{"contents":[12.0,22.0]}}})"},
      "cannot be represented by PiecewiseInterpolation");
   expectImportThrowsWithError(
      HistFactoryDoc{"0.2", R"({"name":"norm","parameter":"alpha_norm","type":"normsys","interpolation":"four",)"
                            R"("data":{"lo":0.8,"hi":1.2}})"},
      "invalid legacy interpolation code 'four'");
}

// HistFactory channels with samples that have a zero-yield bin together with a
// staterror modifier used to produce NaN gamma errors because the relative
// error is computed as sqrt(sumW2)/sumW. Importing such a channel should now
// produce a finite (zero) error for that bin.
TEST(RooFitHS3, HistFactoryZeroYieldBin)
{
   Imported ws{
      HistFactoryDoc{"0.1.90", R"({"name": "mu", "type": "normfactor"}, {"name": "mcstat", "type": "staterror"})"}
         .sampleData(R"({"contents": [10.0, 0.0]})")};

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

TEST(RooFitHS3, HistFactoryDuplicateModifiersAreCombined)
{
   // The two "poly" modifiers deliberately use implicit code 4 and explicit
   // code 5, which are semantic aliases in FlexibleInterpVar.
   Imported ws1{HistFactoryDoc{"0.1.90", R"(
   {"name": "norm", "parameter": "alpha_norm", "type": "normsys", "interpolation": 1, "data": {"lo": 0.8, "hi": 1.2}},
   {"name": "norm", "parameter": "alpha_norm", "type": "normsys", "interpolation": 1, "data": {"lo": 0.9, "hi": 1.1}},
   {"name": "poly", "parameter": "alpha_poly", "type": "normsys", "data": {"lo": 0.8, "hi": 1.2}},
   {"name": "poly", "parameter": "alpha_poly", "type": "normsys", "interpolation": 5, "data": {"lo": 0.75, "hi": 1.25}},
   {"name": "shape_input_1", "parameter": "alpha_shape", "type": "histosys",
    "data": {"lo": {"contents": [8.0, 18.0]}, "hi": {"contents": [12.0, 22.0]}}},
   {"name": "shape_input_2", "parameter": "alpha_shape", "type": "histosys",
    "data": {"lo": {"contents": [9.0, 16.0]}, "hi": {"contents": [11.0, 24.0]}}})"}};

   Doc exported{ws1};
   EXPECT_THAT(exported.warnings(), HasSubstr("combined 2 duplicate modifiers named 'norm' of type 'normsys'"));
   EXPECT_THAT(exported.warnings(), HasSubstr("combined 2 duplicate modifiers named 'poly' of type 'normsys'"));
   EXPECT_THAT(exported.warnings(), HasSubstr("combined 2 duplicate modifiers named 'shape' of type 'histosys'"));
   EXPECT_EQ(exported.names("distributions/model_channel0/samples/sig/modifiers"),
             (std::vector<std::string>{"norm", "poly", "shape"}));

   Imported ws2{exported};

   auto *normInterp = object<RooStats::HistFactory::FlexibleInterpVar>(ws2, "sig_channel0_epsilon");
   ASSERT_EQ(normInterp->variables().size(), 2u);
   for (std::size_t i = 0; i < normInterp->variables().size(); ++i) {
      const std::string parameterName = normInterp->variables().at(i)->GetName();
      if (parameterName == "alpha_norm") {
         EXPECT_DOUBLE_EQ(normInterp->low()[i], 0.8 * 0.9);
         EXPECT_DOUBLE_EQ(normInterp->high()[i], 1.2 * 1.1);
         EXPECT_EQ(normInterp->interpolationCodes()[i], 1);
      } else if (parameterName == "alpha_poly") {
         EXPECT_DOUBLE_EQ(normInterp->low()[i], 0.8 * 0.75);
         EXPECT_DOUBLE_EQ(normInterp->high()[i], 1.2 * 1.25);
         EXPECT_EQ(normInterp->interpolationCodes()[i], 4);
      } else {
         FAIL() << "Unexpected normsys parameter " << parameterName;
      }
   }

   auto *shapeInterp = object<PiecewiseInterpolation>(ws2, "histoSys_model_channel0_sig");
   ASSERT_EQ(shapeInterp->paramList().size(), 1u);
   EXPECT_EQ(shapeInterp->interpolationCodes()[0], 4);
   auto *shapeLow = dynamic_cast<RooHistFunc *>(shapeInterp->lowList().at(0));
   auto *shapeHigh = dynamic_cast<RooHistFunc *>(shapeInterp->highList().at(0));
   ASSERT_NE(shapeLow, nullptr);
   ASSERT_NE(shapeHigh, nullptr);
   EXPECT_DOUBLE_EQ(shapeLow->dataHist().weight(0), 7.0);
   EXPECT_DOUBLE_EQ(shapeLow->dataHist().weight(1), 14.0);
   EXPECT_DOUBLE_EQ(shapeHigh->dataHist().weight(0), 13.0);
   EXPECT_DOUBLE_EQ(shapeHigh->dataHist().weight(1), 26.0);

   auto samplePrediction = [](RooWorkspace &ws, int bin, double norm, double poly, double shape) {
      auto &x = *ws.var("x");
      x.setBin(bin);
      ws.var("alpha_norm")->setVal(norm);
      ws.var("alpha_poly")->setVal(poly);
      ws.var("alpha_shape")->setVal(shape);
      auto &shapeFunction = *ws.function("model_channel0_sig_shapes");
      auto &scaleFunction = *ws.function("model_channel0_sig_scaleFactors");
      RooArgSet observables{x};
      return shapeFunction.getVal(observables) * scaleFunction.getVal();
   };

   struct ParameterPoint {
      double norm;
      double poly;
      double shape;
   };
   // The code-4 normsys product is exactly representable by its combined
   // anchors at -1, 0, and +1. The code-1 normsys and additive histosys are
   // also tested away from their anchors. Between the anchors, the combined code-4 normsys is not the product (the
   // relative difference is 7e-4 at alpha_poly = 0.5), so expectRoundTrip() cannot make this comparison.
   const ParameterPoint points[]{{-0.6, -1.0, 0.35}, {0.8, 0.0, -0.4}, {1.3, 1.0, 1.2}};
   for (const auto &point : points) {
      for (int bin = 0; bin < 2; ++bin) {
         const double before = samplePrediction(ws1, bin, point.norm, point.poly, point.shape);
         const double after = samplePrediction(ws2, bin, point.norm, point.poly, point.shape);
         EXPECT_NEAR(after, before, std::abs(before) * 1e-13)
            << "bin=" << bin << ", norm=" << point.norm << ", poly=" << point.poly << ", shape=" << point.shape;
      }
   }

   // The operation must be logged as a warning, not as an error.
   const std::string sentinelErrors = captureMessages(RooFit::ERROR, RooFit::ObjectHandling, [] {
      RooJSONFactoryWSTool::warning("duplicate-modifier warning-level sentinel");
   });
   EXPECT_TRUE(sentinelErrors.empty()) << sentinelErrors;
}

// Duplicate modifiers that cannot be combined into one are rejected on export.
TEST(RooFitHS3, HistFactoryIncompatibleDuplicateModifiers)
{
   // Two normsys modifiers, "input_1" and "input_2", with the given further keys.
   auto normsys = [](std::string const &keys1, std::string const &keys2) {
      return R"({"name": "input_1", "type": "normsys", "data": {"lo": 0.8, "hi": 1.2}, )" + keys1 + R"(},
                {"name": "input_2", "type": "normsys", "data": {"lo": 0.9, "hi": 1.1}, )" +
             keys2 + "}";
   };
   struct Case {
      std::string name;
      std::string modifiers;
      std::string expectedReason;
      std::string extraDistributions;
   };
   const std::vector<Case> cases{
      {"Parameter",
       normsys(R"("parameter": "alpha_dup", "constraint": "sharedConstraint", "interpolation": 1)",
               R"("parameter": "dup", "constraint": "sharedConstraint", "interpolation": 1)"),
       "parameter metadata differs",
       R"({"name": "sharedConstraint", "type": "gaussian_dist", "x": "alpha_dup", "mean": "dup", "sigma": 1.0})"},
      {"Interpolation",
       normsys(R"("parameter": "alpha_dup", "interpolation": 1)", R"("parameter": "alpha_dup", "interpolation": 4)"),
       "interpolation behaviours differ", ""},
      {"Constraint",
       normsys(R"("parameter": "alpha_dup", "constraint": "Gauss", "interpolation": 1)",
               R"("parameter": "dup", "constraint": "Gauss", "interpolation": 1)"),
       "constraint metadata differs", ""},
      {"UnrepresentableNormFactor", R"({"name": "mu", "type": "normfactor"}, {"name": "mu", "type": "normfactor"})",
       "cannot be combined without changing its meaning", ""},
      {"LinearSpaceNormSys",
       normsys(R"("parameter": "alpha_dup", "interpolation": 0)", R"("parameter": "alpha_dup", "interpolation": 0)"),
       "multiplicative combination is only valid for log-space", ""},
   };
   for (auto const &testCase : cases) {
      SCOPED_TRACE(testCase.name);
      Imported ws{HistFactoryDoc{"0.1.90", testCase.modifiers}
                     .distributions(testCase.extraDistributions)
                     .sections(defaultDomain({axes({"alpha_dup", "dup"}, -5, 5)}))
                     .sections(defaultValues({values({{"alpha_dup", 0.0}, {"dup", 0.0}})}))};
      expectExportThrowsWithError(ws, testCase.expectedReason);
   }
}

TEST(RooFitHS3, HistFactoryDuplicateHistoSysWithDifferentBinningFails)
{
   RooRealVar y{"y", "y", 0.0, 3.0};
   y.setBins(3);

   RooWorkspace ws{"ws_incompatible_histosys_binning"};
   importDuplicateHistoSysModel(ws, /*interpCode=*/4, /*low2Contents=*/{7.0, 8.0, 9.0},
                                /*high2Contents=*/{13.0, 14.0, 15.0}, &y);

   expectExportThrowsWithError(ws, "histogram binning differs");
}

TEST(RooFitHS3, HistFactoryDuplicateHistoSysWithNonDefaultInterpolationFails)
{
   RooWorkspace ws{"ws_incompatible_histosys_interpolation"};
   importDuplicateHistoSysModel(ws, /*interpCode=*/2, /*low2Contents=*/{9.0, 16.0}, /*high2Contents=*/{11.0, 24.0});

   expectExportThrowsWithError(ws, "this interpolation cannot currently be combined for duplicate histosys modifiers");
}

// Older versions of RooFit wrote a parameter that interpolates a sample more than once as several histosys modifiers
// of the same name. Documents before HS3 0.3 are still read like this, and the modifiers are combined on export.
TEST(RooFitHS3, HistFactoryHistoSysOfTheSameNameInOlderDocuments)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::ERROR);

   const std::string modifiers = R"(
   {"name": "shape", "parameter": "alpha_shape", "type": "histosys",
    "data": {"lo": {"contents": [8.0, 18.0]}, "hi": {"contents": [12.0, 22.0]}}},
   {"name": "shape", "parameter": "alpha_shape", "type": "histosys",
    "data": {"lo": {"contents": [9.0, 16.0]}, "hi": {"contents": [11.0, 24.0]}}})";

   expectImportThrowsWithError(HistFactoryDoc{"0.3", modifiers},
                               "there is another histosys modifier named 'shape' in the sample");

   Imported ws{HistFactoryDoc{"0.2", modifiers}};
   auto *interp = object<PiecewiseInterpolation>(ws, "histoSys_model_channel0_sig");
   ASSERT_EQ(interp->paramList().size(), 2u);
   EXPECT_EQ(interp->paramList().at(0), interp->paramList().at(1));
   auto high = [&](int i, int bin) {
      return static_cast<RooHistFunc *>(interp->highList().at(i))->dataHist().weight(bin);
   };
   EXPECT_DOUBLE_EQ(high(0, 1), 22.0);
   EXPECT_DOUBLE_EQ(high(1, 1), 24.0);

   // The export combines them into one modifier, which is valid in the current version.
   Doc exported{ws};
   EXPECT_EQ(exported.names("distributions/model_channel0/samples/sig/modifiers"), std::vector<std::string>{"shape"});
   Imported ws2{exported};
   auto *interp2 = object<PiecewiseInterpolation>(ws2, "histoSys_model_channel0_sig");
   ASSERT_EQ(interp2->paramList().size(), 1u);
   // Both functions give the same for a shifted parameter. (expectRoundTrip() cannot make this comparison: the
   // variations of the first modifier and of the combined one have the same names.)
   ws.var("alpha_shape")->setVal(0.7);
   ws2.var("alpha_shape")->setVal(0.7);
   ws.var("x")->setBin(1);
   ws2.var("x")->setBin(1);
   EXPECT_NEAR(interp->getVal(), interp2->getVal(), 1e-9 * interp->getVal());
}

TEST(RooFitHS3, HistFactoryConstraintKeyMigration)
{
   Imported ws{
      HistFactoryDoc{"0.1.90",
                     R"({"name": "mu", "parameter": "mu", "type": "normfactor", "constraint_name": "muConstraint"})"}
         .distributions(
            R"({"name": "muConstraint", "type": "gaussian_dist", "x": "mu", "mean": "nom_mu", "sigma": "sigma_mu"})")
         .sections(defaultDomain({axes({"nom_mu", "sigma_mu"}, 1, 1)}))
         .sections(defaultValues({R"({"name": "nom_mu", "value": 1.0, "const": true})",
                                  R"({"name": "sigma_mu", "value": 1.0, "const": true})"}))};

   Doc exported{ws};
   // The constraint is collected into the array distribution of its type and top-level pdf.
   EXPECT_TRUE(matches(exported, "distributions/model_channel0/samples/sig/modifiers/mu",
                       R"({"constraints": ["model_channel0_gaussian_constraints"]})"));
   EXPECT_THAT(exported.json(), Not(HasSubstr("\"constraint\":")));
   EXPECT_THAT(exported.json(), Not(HasSubstr("constraint_name")));
   EXPECT_THAT(exported.json(), Not(HasSubstr("constraint_type")));
}

TEST(RooFitHS3, HistFactoryLegacyConstraintTypeInConstraintKey)
{
   Imported ws{HistFactoryDoc{
      "0.1.90", R"({"name": "lumi", "type": "normsys", "constraint": "Gauss", "data": {"lo": 0.95, "hi": 1.05}})"}};

   object<RooGaussian>(ws, "alpha_lumiConstraint");
   auto *alpha = object<RooRealVar>(ws, "alpha_lumi");
   EXPECT_DOUBLE_EQ(alpha->getError(), 1.0);

   Doc exported{ws};
   EXPECT_TRUE(matches(exported, "distributions/model_channel0/samples/sig/modifiers/lumi",
                       R"({"constraints": ["model_channel0_gaussian_constraints"]})"));
   EXPECT_THAT(exported.json(), Not(HasSubstr("\"constraint\":")));
   EXPECT_THAT(exported.json(), Not(HasSubstr("constraint_name")));
   EXPECT_THAT(exported.json(), Not(HasSubstr("constraint_type")));
}

// Snapshot export must keep all variables that any pdf depends on, even when
// the variable is not in the set of separately exported objects. Global
// observables of HistFactory constraint pdfs (the nominal "nom_*" parameters)
// are exactly such variables: the HistFactory exporter explicitly skips them
// when collecting parameters to export, but pdfs still depend on them.
TEST(RooFitHS3, SnapshotKeepsGlobalObservables)
{
   Imported ws1{
      HistFactoryDoc{"0.1.90", R"({"name": "mu", "type": "normfactor"}, {"name": "mcstat", "type": "staterror"})"}
         .sampleData(R"({"contents": [10.0, 20.0], "errors": [1.0, 2.0]})")};

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

   Doc exported{ws1};

   // The exported snapshot should have at least one of the global observables.
   EXPECT_THAT(exported.dump(std::string("parameter_points/") + snapName + "/parameters"), HasSubstr("nom_gamma"))
      << "Global observable missing from exported snapshot block";

   // Re-import and check that the snapshot survived the round-trip with all
   // global observables included.
   Imported ws2{exported};

   const RooArgSet *snap = ws2.getSnapshot(snapName);
   ASSERT_NE(snap, nullptr) << "Snapshot was not preserved through JSON round-trip";

   for (auto *arg : globs) {
      EXPECT_NE(snap->find(arg->GetName()), nullptr) << "Snapshot is missing pdf-dependent variable " << arg->GetName();
   }
}

TEST(RooFitHS3, SimultaneousFit)
{
   using namespace RooFit;

   auto fit = [](RooWorkspace &ws) {
      return std::unique_ptr<RooFitResult>{ws.pdf("simPdf")->fitTo(*ws.data("obsData"), Save(), PrintLevel(-1),
                                                                   PrintEvalErrors(-1), Minimizer("Minuit2"))};
   };

   std::string jsonStr;
   std::unique_ptr<RooFitResult> res1;
   {
      RooWorkspace ws{"workspace"};

      // Build two channels for different observables where the distributions
      // share one parameter: the mean for the signal.

      // Channel 1: Gaussian signal and exponential background
      ws.factory("Gaussian::sig_1(x_1[0, 10], mean[5.0, 0, 10], sigma_1[0.5, 0.1, 10.0])");
      ws.factory("Exponential::bkg_1(x_1, c_1[-0.2, -100, -0.001])");
      ws.factory("SUM::model_1(n_sig_1[10000, 0, 10000000] * sig_1, nbkg_2[100000, 0, 10000000] * bkg_1)");

      // Channel 2: Crystal ball signal and polynomial background
      ws.factory("CBShape::sig_2(x_2[0, 10], mean[5.0, 0, 10], sigma_2[0.8, 0.1, 10.0], alpha[0.9, 0.1, 10.0], "
                 "ncb[1.0, 0.1, 10.0])");
      ws.factory("Polynomial::bkg_2(x_2, {3.0, a_1[-0.3, -10, 10], a_2[0.01, -10, 10]}, 0)");
      ws.factory("SUM::model_2(n_sig_2[30000, 0, 10000000] * sig_2, nbkg_2[100000, 0, 10000000] * bkg_2)");

      // Simultaneous PDF and model config
      ws.factory("SIMUL::simPdf(channelCat[channel_1=0, channel_2=1], channel_1=model_1, channel_2=model_2)");

      addModelConfig(ws, "ModelConfig", "simPdf", "x_1,x_2", "mean");

      RooRealVar &x1 = *ws.var("x_1");
      RooRealVar &x2 = *ws.var("x_2");
      x1.setBins(20);
      x2.setBins(20);

      std::map<std::string, std::unique_ptr<RooAbsData>> datasets;
      datasets["channel_1"] = std::unique_ptr<RooDataHist>{ws.pdf("model_1")->generateBinned(x1)};
      datasets["channel_2"] = std::unique_ptr<RooDataHist>{ws.pdf("model_2")->generateBinned(x2)};

      datasets["channel_1"]->SetName("obsData_channel_1");
      datasets["channel_2"]->SetName("obsData_channel_2");

      RooDataSet obsData{"obsData", "obsData", {x1, x2}, Index(*ws.cat("channelCat")), Import(datasets)};
      ws.import(obsData);

      // Export before fitting to keep the prefit values
      jsonStr = toJSON(ws);
      res1 = fit(ws);
   }

   Imported ws{jsonStr};

   // Make sure that there is exactly one dataset in the new workspace, and
   // that there are no spurious datasets left over from first importing the
   // channel datasets that later get merged to the combined dataset
   EXPECT_EQ(ws.allData().size(), 1) << "Unexpected number of datasets in the new workspace";

   // todo: also check the modelconfig for equality

   EXPECT_TRUE(fit(ws)->isIdentical(*res1));
}

TEST(RooFitHS3, GenericTypeNames)
{
   RooRealVar x{"x", "x", 0.5, -1.0, 1.0};
   RooRealVar y{"y", "y", 0.5, -1.0, 1.0};
   RooRealVar z{"z", "z", 0.5, -1.0, 1.0};
   RooRealVar c{"c", "c", -0.1};
   RooFormulaVar formula{"formula", "x + y", RooArgList{x, y}};
   RooGenericPdf genericPdf{"genericPdf", "x + 2.0", RooArgList{x}};
   TF3 tf3{"tf3", "x + y + z", -1.0, 1.0, -1.0, 1.0, -1.0, 1.0};
   RooTFnBinding binding{"binding", "binding", &tf3, RooArgList{x, y, z}};
   RooExponential exponential{"exponential", "exponential", x, c};

   RooWorkspace ws{"ws_generic_types"};
   ws.import(formula, RooFit::Silence());
   ws.import(genericPdf, RooFit::Silence(), RooFit::RecycleConflictNodes());
   ws.import(binding, RooFit::Silence(), RooFit::RecycleConflictNodes());
   ws.import(exponential, RooFit::Silence(), RooFit::RecycleConflictNodes());

   Doc doc{ws};
   EXPECT_TRUE(matches(doc, "functions/formula", R"({"type": "generic"})"));
   EXPECT_TRUE(matches(doc, "distributions/genericPdf", R"({"type": "generic_dist"})"));
   EXPECT_TRUE(matches(doc, "functions/binding", R"({"type": "generic"})"));
   EXPECT_TRUE(matches(doc, "functions/c_exponential_inverted", R"({"type": "generic"})"));
   EXPECT_THAT(doc.json(), Not(HasSubstr("\"type\":\"generic_function\"")));
}

// The hs3_version must be a string. Documents that are newer than what RooFit
// writes are imported with a warning.
TEST(RooFitHS3, VersionStringHandling)
{
   expectImportThrowsWithError(R"({"metadata": {}})", "The HS3 version is missing");
   expectImportThrowsWithError(R"({"metadata": {"hs3_version": 0.2}})", "must be a string");

   EXPECT_THAT(Imported{document("0.2")}.warnings, Not(HasSubstr("hs3_version")));
   EXPECT_THAT(Imported{document("99.1.4")}.warnings, HasSubstr("'99.1.4', which is newer"));
   EXPECT_THAT(Imported{document("latest")}.warnings, HasSubstr("cannot parse hs3_version 'latest'"));
}

// Names of top-level objects must be unique across the whole HS3 document.
TEST(RooFitHS3, DuplicateNamesAreRejected)
{
   // The same name used for a distribution and a function.
   expectImportThrowsWithError(
      document("0.3",
               {distributions({R"({"name": "dup", "type": "gaussian_dist", "x": "x", "mean": 0.0, "sigma": 1.0})"}),
                functions({R"({"name": "dup", "type": "sum", "summands": ["x"]})"})}),
      "duplicate name 'dup'");

   // The same name used twice within one section.
   expectImportThrowsWithError(document("0.3", {section("parameter_points", {R"({"name": "pp", "parameters": []})",
                                                                             R"({"name": "pp", "parameters": []})"})}),
                               "duplicate name 'pp'");

   // Exported likelihoods don't share their name with the analysis anymore.
   RooWorkspace ws;
   ws.factory("Gaussian::pdf(x[0, -5, 5], mu[0, -5, 5], 1.0)");
   addModelConfig(ws, "ModelConfig", "pdf", "x", "mu");
   Doc doc{ws};
   EXPECT_TRUE(matches(doc, "analyses/ModelConfig", R"({"likelihood": "ModelConfig_likelihood"})"));
   EXPECT_THAT(Imported{doc}.warnings, Not(HasSubstr("not allowed by the HS3 standard")));

   // Files written by older ROOT versions named the likelihood (and the analysis) after the pdf. They are still read,
   // to the same workspace.
   Imported legacy{replaced(replaced(doc, R"("likelihood":"ModelConfig_likelihood")", R"("likelihood":"pdf")"),
                            R"("name":"ModelConfig_likelihood")", R"("name":"pdf")")};
   EXPECT_THAT(legacy.warnings, HasSubstr(R"(the name 'pdf' is used in "distributions" and in "likelihoods")"));
   EXPECT_EQ(toJSON(legacy), doc.json());
}

// The RooFit-specific "default_domain" is optional on import. Variables then
// take the range of the domain that mentions them.
TEST(RooFitHS3, DefaultDomainIsOptional)
{
   const std::string myDomain =
      R"({"name": "my_domain", "type": "product_domain", "axes": [{"name": "mu", "min": -3.0, "max": 4.0}]})";
   Imported ws{document(
      "0.2", {section("domains", {myDomain}), defaultValues({values({{"mu", 0.5}})}),
              distributions({R"({"name": "pdf", "type": "gaussian_dist", "x": "mu", "mean": 0.0, "sigma": 1.0})"})})};
   auto *mu = object<RooRealVar>(ws, "mu");
   EXPECT_DOUBLE_EQ(mu->getMin(), -3.0);
   EXPECT_DOUBLE_EQ(mu->getMax(), 4.0);
   EXPECT_DOUBLE_EQ(mu->getVal(), 0.5);
   EXPECT_FALSE(mu->isConstant());

   // On export, the "default_domain" is still written.
   EXPECT_TRUE(Doc{ws}.has("domains/default_domain"));
}

// Parameters that only appear in a parameter point are created on import, and
// categorical parameters can be set with a string value. A parameter without domain is constant only with
// "const": true; otherwise it is free, with a warning (HS3 requires a domain or "const": true).
TEST(RooFitHS3, ParametersOnlyInParameterPoints)
{
   const std::string json = document("0.2", {defaultDomain({axes({"free", "freeButConst"}, -1, 1)}),
                                             R"("parameter_points": [{"name": "point", "parameters": [
                                                   {"name": "fixed", "value": 2.5},
                                                   {"name": "fixedExplicitFalse", "value": 1.5, "const": false},
                                                   {"name": "free", "value": 0.25},
                                                   {"name": "freeButConst", "value": 0.5, "const": true},
                                                   {"name": "channel", "value": "CR"}]}])"});

   RooWorkspace ws;
   RooCategory channel{"channel", "channel", {{"SR", 0}, {"CR", 1}}};
   ws.import(channel);
   const std::string warnings = importInto(ws, json);
   for (const char *name : {"fixed", "fixedExplicitFalse"}) {
      EXPECT_THAT(warnings, HasSubstr(std::string("the parameter '") + name + "' is not listed in any domain"));
   }

   auto *fixed = object<RooRealVar>(ws, "fixed");
   EXPECT_DOUBLE_EQ(fixed->getVal(), 2.5);
   EXPECT_FALSE(fixed->isConstant());

   auto *free = object<RooRealVar>(ws, "free");
   EXPECT_DOUBLE_EQ(free->getVal(), 0.25);
   EXPECT_FALSE(free->isConstant());

   // "const": false is the same as no "const".
   auto *fixedExplicitFalse = object<RooRealVar>(ws, "fixedExplicitFalse");
   EXPECT_FALSE(fixedExplicitFalse->isConstant());

   // "const": true makes a parameter constant also if a domain lists it.
   auto *freeButConst = object<RooRealVar>(ws, "freeButConst");
   EXPECT_TRUE(freeButConst->isConstant());

   const RooArgSet *snapshot = ws.getSnapshot("point");
   ASSERT_NE(snapshot, nullptr);
   auto *channelInSnapshot = dynamic_cast<RooCategory *>(snapshot->find("channel"));
   ASSERT_NE(channelInSnapshot, nullptr);
   EXPECT_EQ(channelInSnapshot->getCurrentLabel(), std::string("CR"));
}

// HS3 forbids binning information on the axes of unbinned data, regardless of
// the binning of the observable in RooFit.
TEST(RooFitHS3, UnbinnedDatasetAxesHaveNoBinning)
{
   RooWorkspace ws;
   RooRealVar x{"x", "x", 0.0, 10.0};
   RooRealVar y{"y", "y", 0.0, 3.0};
   x.setBins(5);
   std::vector<double> edges{0.0, 1.0, 3.0};
   y.setBinning(RooBinning{2, edges.data()});
   RooDataSet ds{"ds", "ds", RooArgSet{x, y}};
   for (double val : {1.0, 2.0, 2.5}) {
      x.setVal(val);
      y.setVal(val);
      ds.add(RooArgSet{x, y});
   }
   ws.import(ds, RooFit::Silence());

   EXPECT_TRUE(matches(Doc{ws}, "data/ds", R"({"type": "unbinned", "axes": [
      {"name": "x", "min": 0.0, "max": 10.0, "nbins": "<absent>", "edges": "<absent>"},
      {"name": "y", "min": 0.0, "max": 3.0, "nbins": "<absent>", "edges": "<absent>"}]})"));
}

// The snapshot of a ModelConfig is exported as the "init" parameter point of
// the analysis, and restored on import.
TEST(RooFitHS3, AnalysisInitRoundTrip)
{
   RooWorkspace ws1;
   ws1.factory("Gaussian::pdf(x[0, -5, 5], mu[0, -5, 5], 1.0)");
   RooFit::ModelConfig mc1{"ModelConfig", &ws1};
   mc1.SetPdf("pdf");
   mc1.SetParametersOfInterest("mu");
   mc1.SetObservables("x");
   ws1.var("mu")->setVal(1.5);
   RooArgSet initSet{*ws1.var("mu")};
   initSet.setName("POI");
   mc1.SetSnapshot(initSet);
   ws1.var("mu")->setVal(0.0);
   ws1.import(mc1);

   Doc doc1{ws1};
   EXPECT_TRUE(matches(doc1, "analyses/ModelConfig", R"({"init": "ModelConfig_POI_snapshot"})"));

   Imported ws2{doc1};
   auto *mc2 = object<RooFit::ModelConfig>(ws2, "ModelConfig");
   std::unique_ptr<const RooArgSet> snapshot{mc2->GetSnapshot()};
   ASSERT_NE(snapshot, nullptr);
   auto *mu = dynamic_cast<RooRealVar *>(snapshot->find("mu"));
   ASSERT_NE(mu, nullptr);
   EXPECT_DOUBLE_EQ(mu->getVal(), 1.5);

   // The snapshot name must be stable under repeated round trips.
   EXPECT_EQ(toJSON(ws2), doc1.json());

   // An "init" that refers to a missing parameter point is reported.
   Imported ws3{replaced(doc1, R"("init":"ModelConfig_POI_snapshot")", R"("init":"does_not_exist")")};
   EXPECT_THAT(ws3.warnings, HasSubstr("'does_not_exist' used as \"init\""));

   // The snapshot of an unnamed set is called "<mc name>__snapshot", which must be stable as well.
   RooWorkspace ws4;
   ws4.factory("Gaussian::pdf(x[0, -5, 5], mu[0, -5, 5], 1.0)");
   RooFit::ModelConfig mc4{"ModelConfig", &ws4};
   mc4.SetPdf("pdf");
   mc4.SetParametersOfInterest("mu");
   mc4.SetObservables("x");
   mc4.SetSnapshot(RooArgSet{*ws4.var("mu")});
   ws4.import(mc4);
   Doc doc4{ws4};
   EXPECT_TRUE(matches(doc4, "analyses/ModelConfig", R"({"init": "ModelConfig__snapshot"})"));
   EXPECT_EQ(reexport(doc4), doc4.json());
}

// Type keys follow the HS3 catalog. The old RooFit-specific keys are still accepted on import.
TEST(RooFitHS3, TypeKeysMatchTheCatalog)
{
   RooRealVar x{"x", "x", 0.0, 10.0};
   x.setBins(10);
   RooRealVar mean{"mean", "mean", 5.0, 0.0, 10.0};
   RooRealVar sigma{"sigma", "sigma", 1.0, 0.1, 5.0};

   // A document of HS3 0.2 with the given distributions in "m".
   auto olderDocument = [](std::vector<std::string> const &dists) {
      return document("0.2", {defaultDomain({axes({"m"}, 0, 10)}), distributions(dists)});
   };

   // argus_dist is the HS3 key, ARGUS_dist is deprecated: it is still read, and written with the HS3 key.
   RooWorkspace wsArgus;
   wsArgus.factory("ArgusBG::argus(x[0, 10], 9.0, -1.0, 0.5)");
   EXPECT_TRUE(matches(Doc{wsArgus}, "distributions/argus", R"({"type": "argus_dist"})"));
   Imported legacyArgus{olderDocument(
      {R"({"name": "argus", "type": "ARGUS_dist", "mass": "m", "resonance": 9.0, "slope": -1.0, "power": 0.5})"})};
   EXPECT_NE(legacyArgus.pdf("argus"), nullptr);
   EXPECT_THAT(legacyArgus.warnings, HasSubstr(R"(the type "ARGUS_dist" is deprecated)"));
   EXPECT_TRUE(matches(Doc{legacyArgus}, "distributions/argus",
                       R"({"type": "argus_dist", "mass": "m", "resonance": 9.0, "slope": -1.0, "power": 0.5})"));

   // RooBinSamplingPdf is exported and imported with the same key.
   RooGaussian gauss{"gauss", "", x, mean, sigma};
   RooBinSamplingPdf binSampling{"binSampling", "", x, gauss, 1e-3};
   // Evaluating the pdf leaves the named range "_binSampling_binSampling" on x, which is then exported as a domain.
   RooWorkspace wsBinSampling;
   wsBinSampling.import(binSampling, RooFit::Silence());
   expectRoundTrip(wsBinSampling, RoundTripModel{}.exportChangesAfterEvaluation());

   // rate_extended_dist uses the HS3 field names "rate" and "dist". The names "pdf" and "norm" that RooFit used to
   // write are still read.
   RooRealVar nEvents{"nEvents", "nEvents", 100.0, 0.0, 1000.0};
   RooWorkspace wsExtended;
   wsExtended.import(RooExtendPdf{"extended", "", gauss, nEvents}, RooFit::Silence());
   expectRoundTrip(wsExtended);
   Doc extended{wsExtended};
   EXPECT_TRUE(matches(extended, "distributions/extended", R"({"dist": "gauss", "rate": "nEvents"})"));
   EXPECT_THAT(extended.json(), Not(HasSubstr(R"("range")")));
   Imported legacyExtended{
      olderDocument({R"({"name": "ext", "type": "rate_extended_dist", "pdf": "g", "norm": 10.0})",
                     R"({"name": "g", "type": "gaussian_dist", "x": "m", "mean": 5.0, "sigma": 1.0})"})};
   EXPECT_NE(dynamic_cast<RooExtendPdf *>(legacyExtended.pdf("ext")), nullptr);
}

// A literal constant is written as a number, not as the name of the constant: that would refer to an object that is
// not in the document.
TEST(RooFitHS3, LiteralConstantsAreWrittenAsNumbers)
{
   using RooFit::RooConst;

   RooWorkspace ws;
   ws.factory("Poisson::pois(n[0, 10], 2.5)");
   // A number as observable, like the constraint terms of HistFactory.
   ws.factory("Poisson::poisOfNumber(400, expr::lambda('400 * gamma', gamma[1, 0, 5]))");
   ws.factory("Polynomial::poly(x[0, 10], {2.0, 0.5})");
   ws.factory("ExtendPdf::extended(poly, 10.0)");
   ws.factory("GaussModel::gm(t[-5, 5], 0.0, 1.0)");
   ws.factory("Decay::decay(t, 1.5, gm, SingleSided)");
   RooRealVar y{"y", "y", 1, 10};
   ws.import(RooLognormal{"logn", "logn", y, RooConst(2.0), RooConst(1.5), /*useStandardParametrization=*/true},
             RooFit::Silence());
   ws.import(RooLognormal{"lognOfNumber", "lognOfNumber", RooConst(3.0), y, RooConst(1.5), true}, RooFit::Silence(),
             RooFit::RecycleConflictNodes());
   ws.import(RooExponential{"expo", "expo", *ws.var("x"), RooConst(0.5), /*negateCoefficient=*/true}, RooFit::Silence(),
             RooFit::RecycleConflictNodes());
   RooRealVar z{"z", "z", 0, 2};
   z.setBins(2);
   RooRealVar p0{"p0", "p0", 1.0, 0, 5};
   ws.import(ParamHistFunc{"phf", "phf", RooArgList{z}, RooArgList{p0, RooConst(2.0)}}, RooFit::Silence());

   const Doc doc{ws};
   EXPECT_TRUE(matches(doc, "distributions/pois", R"({"mean": 2.5})"));
   EXPECT_TRUE(matches(doc, "distributions/poisOfNumber", R"({"x": 400.0})"));
   // The first coefficient is the implicit constant term.
   EXPECT_TRUE(matches(doc, "distributions/poly", R"({"coefficients": [1.0, 2.0, 0.5]})"));
   EXPECT_TRUE(matches(doc, "distributions/extended", R"({"rate": 10.0})"));
   EXPECT_TRUE(matches(doc, "functions/gm", R"({"mean": 0.0, "sigma": 1.0})"));
   EXPECT_TRUE(matches(doc, "distributions/decay", R"({"tau": 1.5})"));
   EXPECT_TRUE(matches(doc, "distributions/logn", R"({"mu": 2.0, "sigma": 1.5})"));
   EXPECT_TRUE(matches(doc, "distributions/lognOfNumber", R"({"x": 3.0})"));
   EXPECT_TRUE(matches(doc, "distributions/expo", R"({"c": 0.5})"));
   EXPECT_TRUE(matches(doc, "functions/phf", R"({"parameters": ["p0", 2.0]})"));

   expectRoundTrip(ws, RoundTripModel{}.over({"x", "y", "t", "n", "z"}));
}

// TFormula ternaries are exported as the HS3 conditional ifelse(), and ifelse()
// is translated back on import.
TEST(RooFitHS3, TernaryIsExportedAsIfElse)
{
   RooRealVar x{"x", "x", 0.5, -2.0, 2.0};
   RooRealVar y{"y", "y", -0.5, -2.0, 2.0};
   RooFormulaVar formula{"formula", "2 * (x > 0 ? (y > 0 ? x * y : -x) : 3 * x) + 1", RooArgList{x, y}};

   RooWorkspace ws1;
   ws1.import(formula, RooFit::Silence());
   Doc doc1{ws1};
   EXPECT_TRUE(matches(doc1, "functions/formula",
                       R"json({"expression": "2 * ifelse(x > 0,ifelse(y > 0,x * y,-x),3 * x) + 1"})json"));
   EXPECT_THAT(doc1.json(), Not(HasSubstr("?")));

   // The values along x for both signs of y cover all branches.
   for (double yVal : {-1.0, 0.5}) {
      ws1.var("y")->setVal(yVal);
      expectRoundTrip(ws1);
   }

   // Expressions without ternaries are not changed.
   RooFormulaVar plain{"plain", "(x + y) * (x - y)", RooArgList{x, y}};
   RooWorkspace ws3;
   ws3.import(plain, RooFit::Silence());
   EXPECT_TRUE(matches(Doc{ws3}, "functions/plain", R"json({"expression": "(x + y) * (x - y)"})json"));
}

namespace {

// A document with the mixture_dist "mix" of two Gaussians in "x", with the given coefficients and further keys.
std::string mixtureDocument(std::string const &version, std::string const &coefficients, std::string const &keys = "")
{
   return document(
      version, {defaultDomain({axes({"x"}, -5, 5)}), defaultValues({values({{"f", 0.3}, {"n1", 30}, {"n2", 70}})}),
                distributions({R"({"name": "g1", "type": "gaussian_dist", "x": "x", "mean": -1.0, "sigma": 1.0})",
                               R"({"name": "g2", "type": "gaussian_dist", "x": "x", "mean": 1.0, "sigma": 1.0})",
                               R"({"name": "mix", "type": "mixture_dist", "summands": ["g1", "g2"], "coefficients": )" +
                                  coefficients + keys + "}"})});
}

} // namespace

// The explicit "extended" flag of a mixture_dist has to be consistent with the number of coefficients.
TEST(RooFitHS3, MixtureDistExtendedFlag)
{
   const std::string fraction = R"(["f"])";
   const std::string yields = R"(["n1", "n2"])";
   const std::string extended = R"(, "extended": true)";
   const std::string notExtended = R"(, "extended": false)";
   auto isExtended = [](std::string const &json) {
      return Imported{json}.pdf("mix")->extendMode() != RooAbsPdf::CanNotBeExtended;
   };

   EXPECT_FALSE(isExtended(mixtureDocument("0.2", fraction, notExtended)));
   EXPECT_TRUE(isExtended(mixtureDocument("0.2", yields, extended)));
   expectImportThrowsWithError(mixtureDocument("0.2", fraction, extended),
                               "is declared as extended, but it has 1 coefficients");

   // An explicit "extended": false with one coefficient per summand is honoured: the mixture is not extendable, and its
   // shape is the one with the normalized coefficients.
   Imported ws{mixtureDocument("0.2", yields, notExtended)};
   auto *mix = object<RooAbsPdf>(ws, "mix");
   EXPECT_EQ(mix->extendMode(), RooAbsPdf::CanNotBeExtended);
   RooArgSet normSet{*ws.var("x")};
   for (double xVal : {-2.0, 0.0, 1.5}) {
      ws.var("x")->setVal(xVal);
      const double expected = (30. * ws.pdf("g1")->getVal(normSet) + 70. * ws.pdf("g2")->getVal(normSet)) / 100.;
      EXPECT_NEAR(mix->getVal(normSet), expected, 1e-12) << "x=" << xVal;
   }

   // HS3 0.3 requires the flag where it decides about the Poisson term, i.e. with one coefficient per summand.
   expectImportThrowsWithError(mixtureDocument("0.3", yields), "but no \"extended\" flag");
   EXPECT_FALSE(isExtended(mixtureDocument("0.3", fraction)));
   // Older documents are read as before: as an extended mixture.
   EXPECT_EQ(reexport(mixtureDocument("0.2", yields)), reexport(mixtureDocument("0.2", yields, extended)));
}

// Invalid documents are reported as std::runtime_error, also where the JSON library finds the error: invalid JSON, or a
// value of the wrong type that an importer reads.
TEST(RooFitHS3, ImportErrorsAreRuntimeErrors)
{
   auto expectRuntimeError = [](std::string const &json, std::string const &reason) {
      RooWorkspace ws;
      try {
         importInto(ws, json);
         ADD_FAILURE() << "no error for " << json;
      } catch (std::runtime_error const &err) {
         EXPECT_THAT(std::string{err.what()}, HasSubstr(reason)) << err.what();
      }
   };

   expectRuntimeError(R"({"distributions": [ })", "parse error");
   expectRuntimeError(mixtureDocument("0.2", R"(["f"])", R"(, "extended": "no")"),
                      R"(node "extended": expected a boolean)");
}

// The parameter of a normfactor is given by "parameter", the "name" is only a label.
// The HS3 spelling of the log-normal constraint type is recognized.
TEST(RooFitHS3, HistFactoryNormFactorParameterAndLogNormal)
{
   Imported ws{HistFactoryDoc{"0.2", R"({"name": "lumi_modifier", "parameter": "lumi", "type": "normfactor"})"}};
   EXPECT_NE(ws.var("lumi"), nullptr);
   EXPECT_EQ(ws.var("lumi_modifier"), nullptr);

   expectImportThrowsWithError(
      HistFactoryDoc{
         "0.2",
         R"({"name": "sys", "parameter": "alpha_sys", "type": "normsys", "constraint": "LogNormal", "data": {"lo": 0.9, "hi": 1.1}})"},
      "legacy constraint value 'LogNormal' for modifier 'sys' is a known constraint type");
}

// A normsys or histosys modifier without "constraint" is unconstrained, and
// it is exported without constraint.
TEST(RooFitHS3, HistFactoryModifierWithoutConstraintIsUnconstrained)
{
   const std::string modifiers = R"(
      {"name": "norm", "parameter": "alpha_norm", "type": "normsys", "data": {"lo": 0.8, "hi": 1.2}},
      {"name": "shape", "parameter": "alpha_shape", "type": "histosys",
       "data": {"lo": {"contents": [8.0, 18.0]}, "hi": {"contents": [12.0, 22.0]}}},
      {"name": "constrained", "parameter": "alpha_constrained", "type": "normsys", "constraint": "Gauss",
       "data": {"lo": 0.9, "hi": 1.1}})";

   Imported ws1{HistFactoryDoc{"0.2", modifiers}.interpolation(R"({"type": "mult", "in": "poly6", "out": "exp"})")};
   EXPECT_EQ(ws1.pdf("alpha_normConstraint"), nullptr);
   EXPECT_EQ(ws1.pdf("alpha_shapeConstraint"), nullptr);
   EXPECT_NE(ws1.pdf("alpha_constrainedConstraint"), nullptr);
   EXPECT_EQ(ws1.var("nom_alpha_norm"), nullptr);

   Doc doc1{ws1};
   const std::string modifier = "distributions/model_channel0/samples/sig/modifiers/";
   EXPECT_FALSE(doc1.has(modifier + "norm/constraints"));
   EXPECT_FALSE(doc1.has(modifier + "shape/constraints"));
   EXPECT_TRUE(doc1.has(modifier + "constrained/constraints"));

   Imported ws2{doc1};
   EXPECT_EQ(ws2.pdf("alpha_normConstraint"), nullptr);
   EXPECT_EQ(toJSON(ws2), doc1.json());
}

// The legacy constraint type of the staterror is read from the modifier, with
// the channel-level setting as fallback. On export, the constraint
// distributions are referenced by name.
TEST(RooFitHS3, HistFactoryStatErrorConstraint)
{
   auto makeJSON = [](std::string const &modifierConstraint, std::string const &channelConstraint) {
      return HistFactoryDoc{
         "0.2", R"({"name": "mcstat", "type": "staterror")" +
                   (modifierConstraint.empty() ? "" : R"(, "constraint": ")" + modifierConstraint + "\"") + "}"}
         .sampleData(R"({"contents": [10.0, 20.0], "errors": [1.0, 2.0]})")
         .channelKeys(channelConstraint.empty() ? ""
                                                : R"("staterror": {"constraint_type": ")" + channelConstraint + R"("})")
         .str();
   };
   auto hasGaussianStatConstraints = [](RooWorkspace &ws) {
      bool foundGauss = false;
      bool foundPoisson = false;
      for (RooAbsArg *pdf : ws.allPdfs()) {
         if (std::string(pdf->GetName()).find("gamma_stat_channel0") == std::string::npos)
            continue;
         foundGauss |= dynamic_cast<RooGaussian *>(pdf) != nullptr;
         foundPoisson |= dynamic_cast<RooPoisson *>(pdf) != nullptr;
      }
      EXPECT_NE(foundGauss, foundPoisson);
      return foundGauss;
   };

   {
      Imported ws{makeJSON("Gauss", "Poisson")};
      EXPECT_TRUE(hasGaussianStatConstraints(ws));

      // The constraints are exported as one array distribution that the staterror modifier references by name.
      Doc doc{ws};
      EXPECT_FALSE(doc.has("distributions/model_channel0/staterror"));
      const std::string modifier = "distributions/model_channel0/samples/sig/modifiers/staterror";
      EXPECT_FALSE(doc.has(modifier + "/constraint"));
      const std::vector<std::string> constraints = doc.strings(modifier + "/constraints");
      ASSERT_EQ(constraints.size(), 2u);
      for (auto const &constraint : constraints) {
         EXPECT_TRUE(matches(doc, "distributions/" + constraint,
                             R"({"type": "array_dist", "component_type": "gaussian_dist"})"));
      }

      Imported ws2{doc};
      EXPECT_TRUE(hasGaussianStatConstraints(ws2));
      EXPECT_EQ(toJSON(ws2), doc.json());
   }
   {
      Imported ws{makeJSON("", "Gauss")};
      EXPECT_TRUE(hasGaussianStatConstraints(ws));
   }
   {
      Imported ws{makeJSON("", "")};
      EXPECT_FALSE(hasGaussianStatConstraints(ws));
   }
   expectImportThrowsWithError(makeJSON("LogNormal", ""), "unsupported constraint 'LogNormal' of the staterror");
}

// HS3 forbids implementation-specific default interpolations. Documents from
// before HS3 v0.3 still get the implicit default of older RooFit versions.
TEST(RooFitHS3, HistFactoryMissingInterpolation)
{
   const std::string modifiers = R"(
      {"name": "norm", "parameter": "alpha_norm", "type": "normsys", "data": {"lo": 0.8, "hi": 1.2}},
      {"name": "norm2", "parameter": "alpha_norm2", "type": "normsys", "data": {"lo": 0.9, "hi": 1.1}})";

   expectImportThrowsWithError(HistFactoryDoc{"0.3", modifiers},
                               "interpolation for normsys modifier 'norm' in sample 'sig' of channel "
                               "'channel0' is not given, and the channel has no \"default_interpolation\"");

   Imported ws{HistFactoryDoc{"0.2", modifiers}};
   EXPECT_EQ(countOccurrences(ws.warnings, "implicit default of older RooFit versions"), 1u) << ws.warnings;
   auto *norm = object<RooStats::HistFactory::FlexibleInterpVar>(ws, "sig_channel0_epsilon");
   EXPECT_EQ(norm->interpolationCodes()[0], 4);

   // RooFit always writes a "default_interpolation" if a modifier relies on it.
   Doc exported{ws};
   EXPECT_TRUE(exported.has("distributions/model_channel0/default_interpolation"));
   EXPECT_TRUE(matches(exported, "metadata", R"({"hs3_version": "0.3"})"));
   Imported ws2{exported};
}

// Malformed matrices, error arrays and bin edges are rejected instead of being read out of bounds.
TEST(RooFitHS3, MalformedArraysAreRejected)
{
   auto binnedData = [](std::string const &axis, std::string const &extra) {
      return document("0.2", {section("data", {R"({"name": "data", "type": "binned", "axes": [)" + axis +
                                               R"(], "contents": [1.0, 2.0])" + extra + "}"})});
   };
   const std::string goodAxis = R"({"name": "x", "min": 0.0, "max": 2.0, "nbins": 2})";

   expectImportThrowsWithError(binnedData(goodAxis, R"(, "errors": [1.0])"),
                               "inconsistent number of errors and contents in 'data'");
   expectImportThrowsWithError(binnedData(R"({"name": "x", "edges": [0.0, 2.0, 1.0]})", ""),
                               "\"edges\" of axis 'x' must be strictly ascending");
   expectImportThrowsWithError(binnedData(R"({"name": "x", "edges": []})", ""),
                               "\"edges\" of axis 'x' must contain at least two values");

   expectImportThrowsWithError(
      document("0.2", {distributions({R"({"name": "mvn", "type": "multivariate_normal_dist", "x": ["x1", "x2"],
                                          "mean": [0.0, 0.0], "covariances": [[1.0, 0.0, 0.0], [0.0, 1.0]]})"})}),
      "\"covariances\" in 'mvn' must be a square matrix");
}

// The observables of unbinned data take their range from the axes, unless a domain defines it.
TEST(RooFitHS3, UnbinnedAxisRangesOnImport)
{
   auto makeJSON = [](std::string const &domainAxes) {
      return document(
         "0.2",
         {defaultDomain({domainAxes}), defaultValues({values({{"x", 0.0}})}),
          distributions({R"({"name": "pdf", "type": "gaussian_dist", "x": "x", "mean": 0.0, "sigma": 1.0})"}),
          section("data", {R"({"name": "data", "type": "unbinned", "axes": [{"name": "x", "min": -3.0, "max": 8.0}],
                               "entries": [[-2.5], [7.5]]})"})});
   };

   {
      // "x" is not in any domain, so it gets the range of the axis and no entry is clipped.
      Imported ws{makeJSON(axes({"other"}, 0, 1))};
      EXPECT_DOUBLE_EQ(ws.var("x")->getMin(), -3.0);
      EXPECT_DOUBLE_EQ(ws.var("x")->getMax(), 8.0);
      const RooArgSet *row = ws.data("data")->get(1);
      EXPECT_DOUBLE_EQ(static_cast<RooRealVar *>(row->find("x"))->getVal(), 7.5);
   }
   {
      // A disagreeing domain range is kept, with a warning.
      Imported ws{makeJSON(axes({"x"}, -10, 10))};
      EXPECT_DOUBLE_EQ(ws.var("x")->getMin(), -10.0);
      EXPECT_THAT(ws.warnings, HasSubstr("of the axis 'x' of the dataset 'data' differs"));
   }
}

// The analysis keys "parameters_of_interest" and "domains" are written. The
// singular keys of HS3 v0.2.9 are still read, with a deprecation warning.
TEST(RooFitHS3, AnalysisLegacySingularKeys)
{
   RooWorkspace ws1;
   ws1.factory("Gaussian::pdf(x[0, -5, 5], mu[0, -5, 5], sigma[1, 0.1, 5])");
   addModelConfig(ws1, "ModelConfig", "pdf", "x", "mu", /*nuisance=*/"sigma");

   Doc doc{ws1};
   EXPECT_TRUE(matches(doc, "analyses/ModelConfig", R"({
      "parameters_of_interest": ["mu"],
      "domains": ["ModelConfig_nuisance_parameters", "ModelConfig_parameters_of_interest"]})"));

   // The analysis in the singular form of HS3 v0.2.9, with only the nuisance parameter domain, is read to the same
   // workspace.
   Imported legacy{replaced(replaced(doc, R"("parameters_of_interest":["mu"])", R"("parameter_of_interest":["mu"])"),
                            R"("domains":["ModelConfig_nuisance_parameters","ModelConfig_parameters_of_interest"])",
                            R"("domain":"ModelConfig_nuisance_parameters")")};
   EXPECT_THAT(legacy.warnings, HasSubstr(R"(the key "parameter_of_interest" of analysis)"));
   EXPECT_THAT(legacy.warnings, HasSubstr(R"(the key "domain" of analysis)"));
   EXPECT_EQ(toJSON(legacy), doc.json());
}

// "crystalball_dist" is the single-sided Crystal Ball, "crystalball_doublesided_dist" the double-sided one. Before
// HS3 0.3, "crystalball_dist" with the "_L"/"_R" fields was the double-sided one, which is still read.
TEST(RooFitHS3, CrystalBallTypes)
{
   RooRealVar x{"x", "x", 0.0, 10.0};
   x.setBins(20);
   RooRealVar mean{"mean", "mean", 5.0, 0.0, 10.0};
   RooRealVar sigma{"sigma", "sigma", 1.0, 0.1, 5.0};
   RooRealVar sigmaR{"sigmaR", "sigmaR", 1.5, 0.1, 5.0};
   RooRealVar alphaL{"alphaL", "alphaL", 1.2, 0.1, 5.0};
   RooRealVar nL{"nL", "nL", 3.0, 0.5, 10.0};
   RooRealVar alphaR{"alphaR", "alphaR", 2.0, 0.1, 5.0};
   RooRealVar nR{"nR", "nR", 5.0, 0.5, 10.0};
   RooRealVar alphaNeg{"alphaNeg", "alphaNeg", -1.2, -5.0, -0.1};

   // The document that the pdf is exported to.
   auto exported = [](RooAbsArg const &arg) {
      RooWorkspace ws;
      ws.import(arg, RooFit::Silence());
      return Doc{ws};
   };

   // RooCBShape is the single-sided crystalball_dist.
   RooCBShape cbShape{"cbShape", "", x, mean, sigma, alphaL, nL};
   expectRoundTrip(cbShape);
   EXPECT_TRUE(matches(exported(cbShape), "distributions/cbShape", R"({"type": "crystalball_dist"})"));

   // A single-sided RooCrystalBall is a crystalball_dist too, for both signs of alpha. It is read back as a
   // RooCBShape, which is the same function.
   for (RooRealVar *alpha : {&alphaL, &alphaNeg}) {
      RooCrystalBall cbSingle{"cbSingle", "", x, mean, sigma, *alpha, nL, /*doubleSided=*/false};
      const Doc doc = exported(cbSingle);
      EXPECT_TRUE(matches(doc, "distributions/cbSingle",
                          R"({"type": "crystalball_dist", "alpha": ")" + std::string(alpha->GetName()) + "\"}"));
      EXPECT_THAT(doc.json(), Not(HasSubstr("alpha_L")));

      EXPECT_NE(dynamic_cast<RooCBShape *>(Imported{doc}.pdf("cbSingle")), nullptr);
      expectRoundTrip(cbSingle);
   }

   // Double-sided RooCrystalBall shapes are crystalball_doublesided_dist, with the tails always written for both
   // sides, and the width written as "sigma" if both sides share it.
   RooCrystalBall cbSymmetric{"cbSymmetric", "", x, mean, sigma, alphaL, nL, /*doubleSided=*/true};
   RooCrystalBall cbAsymmetric{"cbAsymmetric", "", x, mean, sigma, sigmaR, alphaL, nL, alphaR, nR};
   expectRoundTrip(cbSymmetric);
   expectRoundTrip(cbAsymmetric);
   const Doc symmetric = exported(cbSymmetric);
   EXPECT_TRUE(matches(symmetric, "distributions/cbSymmetric",
                       R"({"type": "crystalball_doublesided_dist", "sigma": "sigma", "alpha_R": "alphaL"})"));
   EXPECT_THAT(symmetric.json(), Not(HasSubstr(R"("alpha":)")));
   EXPECT_TRUE(matches(exported(cbAsymmetric), "distributions/cbAsymmetric",
                       R"({"type": "crystalball_doublesided_dist", "sigma_R": "sigmaR", "alpha_R": "alphaR"})"));

   // A document with the distribution "cb" of the given type in "m", with further keys.
   auto crystalBall = [](std::string const &type, std::string const &keys) {
      return document(
         "0.3", {defaultDomain({axes({"m"}, 0, 10)}),
                 distributions({R"({"name": "cb", "type": ")" + type + R"(", "m": "m", "m0": 5.0, )" + keys + "}"})});
   };
   const std::string single = "crystalball_dist";
   const std::string doubleSided = "crystalball_doublesided_dist";
   const std::string tails = R"("sigma": 1.0, "alpha_L": 1.2, "n_L": 3.0, "alpha_R": 2.0, "n_R": 5.0)";
   const std::string widths = R"("sigma_L": 1.0, "sigma_R": 1.5, )";

   // Import of the two standard types.
   Imported standardSingle{crystalBall(single, R"("sigma": 1.0, "alpha": -1.2, "n": 3.0)")};
   EXPECT_NE(dynamic_cast<RooCBShape *>(standardSingle.pdf("cb")), nullptr);
   Imported standardDoubleSided{crystalBall(doubleSided, tails)};
   EXPECT_NE(dynamic_cast<RooCrystalBall *>(standardDoubleSided.pdf("cb")), nullptr);

   // The double-sided form of crystalball_dist from before HS3 0.3 is still read like the double-sided type, with a
   // deprecation warning. This includes the symmetric tails given as "alpha"/"n" with asymmetric widths.
   Imported legacy{crystalBall(single, tails)};
   EXPECT_THAT(legacy.warnings, HasSubstr("please use \"crystalball_doublesided_dist\""));
   auto *legacyCb = object<RooCrystalBall>(legacy, "cb");
   ASSERT_NE(legacyCb->alphaR(), nullptr);
   EXPECT_DOUBLE_EQ(legacyCb->alphaR()->getVal(), 2.0);
   EXPECT_EQ(toJSON(legacy), toJSON(standardDoubleSided));
   Imported legacySymmetric{crystalBall(single, widths + R"("alpha": 1.2, "n": 3.0)")};
   auto *legacySymmetricCb = object<RooCrystalBall>(legacySymmetric, "cb");
   ASSERT_NE(legacySymmetricCb->alphaR(), nullptr);
   EXPECT_DOUBLE_EQ(legacySymmetricCb->alphaR()->getVal(), 1.2);
   EXPECT_EQ(toJSON(legacySymmetric),
             reexport(crystalBall(doubleSided, widths + R"("alpha_L": 1.2, "n_L": 3.0, "alpha_R": 1.2, "n_R": 3.0)")));

   // Invalid field combinations.
   expectImportThrowsWithError(crystalBall(doubleSided, R"("sigma": 1.0, "alpha": 1.2, "n": 3.0)"),
                               "must not have \"alpha\" or \"n\"");
   expectImportThrowsWithError(crystalBall(doubleSided, R"("sigma": 1.0, "alpha_L": 1.2, "n_L": 3.0, "alpha_R": 2.0)"),
                               "is missing \"n_R\"");
   expectImportThrowsWithError(crystalBall(doubleSided, R"("sigma_L": 1.0, "sigma_R": 1.0, )" + tails),
                               "must not combine \"sigma\"");
   expectImportThrowsWithError(
      crystalBall(doubleSided, R"("sigma_L": 1.0, "alpha_L": 1.2, "n_L": 3.0, "alpha_R": 2.0, "n_R": 5.0)"),
      "needs both \"sigma_L\" and \"sigma_R\"");
   expectImportThrowsWithError(
      crystalBall(single, R"("sigma": 1.0, "alpha": 1.2, "n": 3.0, "alpha_R": 2.0, "n_R": 5.0)"), "must not combine");
}

// HistFactory modifiers reference their parameters with "parameters" and their constraint distributions by name with
// "constraints", where null leaves a parameter unconstrained.
TEST(RooFitHS3, HistFactoryConstraintsByName)
{
   auto makeJSON = [](std::string const &statConstraints) {
      return HistFactoryDoc{"0.3", R"(
         {"type": "normfactor", "parameters": ["mu"]},
         {"type": "normsys", "parameters": ["alpha_a"], "constraints": ["alpha_a_constraint"],
          "data": {"lo": 0.9, "hi": 1.1}},
         {"type": "normsys", "parameters": ["alpha_b"], "constraints": [null], "data": {"lo": 0.8, "hi": 1.2}},
         {"type": "staterror", "parameters": ["gamma_0", "gamma_1"])" +
                                      statConstraints + "}"}
         .interpolation(R"({"type": "mult", "in": "poly6", "out": "exp"})")
         .sampleData(R"({"contents": [10.0, 20.0], "errors": [1.0, 2.0]})")
         .distributions(R"(
         {"name": "alpha_a_constraint", "type": "gaussian_dist", "x": "alpha_a", "mean": 0.0, "sigma": 1.0},
         {"name": "gamma_0_constraint", "type": "gaussian_dist", "x": "gamma_0", "mean": 1.0, "sigma": 0.1})")
         .sections(defaultDomain({axes({"alpha_a", "alpha_b"}, -5, 5), axes({"gamma_0", "gamma_1"}, 0, 5)}))
         .str();
   };

   Imported ws{makeJSON(R"(, "constraints": ["gamma_0_constraint", null])")};
   auto *model = object<RooProdPdf>(ws, "model_channel0");
   RooArgSet constraintNames;
   for (RooAbsArg *pdf : model->pdfList()) {
      if (!dynamic_cast<RooRealSumPdf *>(pdf)) {
         constraintNames.add(*pdf);
      }
   }
   // Exactly the two named constraints, and no constraints created by RooFit.
   EXPECT_EQ(constraintNames.size(), 2u);
   EXPECT_NE(constraintNames.find("alpha_a_constraint"), nullptr);
   EXPECT_NE(constraintNames.find("gamma_0_constraint"), nullptr);
   EXPECT_NE(ws.var("mu"), nullptr);

   // The export collects the constraints into the array distribution of their type (both are Gaussians), the
   // unconstrained gamma keeps its null, the unconstrained normsys has no "constraints", and the result is stable
   // under a round trip.
   Doc exported{ws};
   EXPECT_TRUE(matches(exported, "distributions/model_channel0/samples/sig/modifiers", R"([
      {"type": "normfactor", "parameters": ["mu"]},
      {"type": "normsys", "parameters": ["alpha_a"], "constraints": ["model_channel0_gaussian_constraints"]},
      {"type": "normsys", "parameters": ["alpha_b"], "constraints": "<absent>"},
      {"type": "staterror", "parameters": ["gamma_0", "gamma_1"],
       "constraints": ["model_channel0_gaussian_constraints", null]}])"));
   EXPECT_THAT(exported.json(), Not(HasSubstr(R"("parameter":)")));
   EXPECT_EQ(reexport(exported), exported.json());

   // In HS3 >= 0.3, a staterror without any constraint information is unconstrained.
   Imported ws3{makeJSON("")};
   auto *model3 = object<RooProdPdf>(ws3, "model_channel0");
   EXPECT_EQ(model3->pdfList().size(), 2u); // the sum and alpha_a_constraint
}

// An array_dist is imported as one distribution per component, named after its observable, held together by a
// RooProdPdf with the attribute hs3_array_dist that carries the name of the array. The component types are built by
// their own importers, with "default_parameters" filled in where a component omits a parameter.
TEST(RooFitHS3, ArrayDistImport)
{
   const std::string domain =
      defaultDomain({axes({"n_1", "n_2", "n_3"}, 0, 100), axes({"np_1", "np_2"}, 0, 5),
                     axes({"nom_a", "nom_b", "nom_c", "a", "b"}, -5, 5),
                     axes({"sigma_a", "y_1", "y_2", "mu_1"}, 0.1, 10), axes({"sigma_y"}, 1, 10)});
   const std::string parameterValues = defaultValues(
      {values({{"n_1", 20.0}, {"n_2", 10.0}, {"n_3", 8.0}, {"np_1", 1.1}, {"np_2", 0.9}}),
       values({{"nom_a", 0.3}, {"nom_b", -0.2}, {"nom_c", 1.0}, {"a", 0.1}, {"b", 0.4}, {"sigma_a", 0.5}}),
       values({{"y_1", 2.0}, {"y_2", 3.0}, {"mu_1", 1.5}, {"sigma_y", 1.2}})});
   Imported ws{document("0.3", {R"("distributions": [
         {"name": "pois", "type": "array_dist", "component_type": "poisson_continuous_dist",
          "default_parameters": {"mean": "lambda_default"},
          "components": [{"x": "n_1", "mean": "lambda_1"}, {"x": "n_2"}]},
         {"name": "pois_discrete", "type": "array_dist", "component_type": "poisson_dist",
          "components": [{"x": "n_3", "mean": 7.5}]},
         {"name": "gauss", "type": "array_dist", "component_type": "gaussian_dist",
          "default_parameters": {"sigma": 1.0},
          "components": [{"x": "nom_a", "mean": "a", "sigma": "sigma_a"}, {"x": "nom_b", "mean": "b"},
                         {"x": "nom_c", "mean": 0.5, "sigma": 2.0}]},
         {"name": "logn", "type": "array_dist", "component_type": "lognormal_dist",
          "components": [{"x": "y_1", "mu": "mu_1", "sigma": 1.5}, {"x": "y_2", "mu": 2.0, "sigma": "sigma_y"}]}],
       "functions": [
         {"name": "lambda_1", "type": "product", "factors": [20.0, "np_1"]},
         {"name": "lambda_default", "type": "product", "factors": [10.0, "np_2"]}])",
                                domain, parameterValues})};

   auto checkArray = [&](const char *name, std::vector<std::string> const &components) {
      auto *array = object<RooProdPdf>(ws, name);
      EXPECT_TRUE(array->getAttribute("hs3_array_dist")) << name;
      ASSERT_EQ(array->pdfList().size(), components.size()) << name;
      double product = 1.0;
      for (std::size_t i = 0; i < components.size(); ++i) {
         EXPECT_STREQ(array->pdfList()[i].GetName(), components[i].c_str()) << name;
         product *= static_cast<RooAbsReal &>(array->pdfList()[i]).getVal();
      }
      // The array is the product of its components.
      EXPECT_DOUBLE_EQ(array->getVal(), product) << name;
   };
   checkArray("pois", {"n_1_constraint", "n_2_constraint"});
   checkArray("pois_discrete", {"n_3_constraint"});
   checkArray("gauss", {"nom_a_constraint", "nom_b_constraint", "nom_c_constraint"});
   checkArray("logn", {"y_1_constraint", "y_2_constraint"});

   // Poisson: own parameter, default parameter, number. The component type decides between the continuous and the
   // discrete form.
   auto *p1 = dynamic_cast<RooPoisson *>(ws.pdf("n_1_constraint"));
   auto *p2 = dynamic_cast<RooPoisson *>(ws.pdf("n_2_constraint"));
   auto *p3 = dynamic_cast<RooPoisson *>(ws.pdf("n_3_constraint"));
   ASSERT_NE(p1, nullptr);
   ASSERT_NE(p2, nullptr);
   ASSERT_NE(p3, nullptr);
   EXPECT_STREQ(p1->getX().GetName(), "n_1");
   EXPECT_STREQ(p1->getMean().GetName(), "lambda_1");
   EXPECT_STREQ(p2->getMean().GetName(), "lambda_default");
   EXPECT_DOUBLE_EQ(p3->getMean().getVal(), 7.5);
   EXPECT_TRUE(p1->getNoRounding());
   EXPECT_TRUE(p2->getNoRounding());
   EXPECT_FALSE(p3->getNoRounding());
   EXPECT_DOUBLE_EQ(p1->getVal(), TMath::Poisson(20.0, 22.0));
   EXPECT_DOUBLE_EQ(p2->getVal(), TMath::Poisson(10.0, 9.0));

   // Gaussian: parameters by name, default sigma, numbers.
   auto *ga = dynamic_cast<RooGaussian *>(ws.pdf("nom_a_constraint"));
   auto *gb = dynamic_cast<RooGaussian *>(ws.pdf("nom_b_constraint"));
   auto *gc = dynamic_cast<RooGaussian *>(ws.pdf("nom_c_constraint"));
   ASSERT_NE(ga, nullptr);
   ASSERT_NE(gb, nullptr);
   ASSERT_NE(gc, nullptr);
   EXPECT_STREQ(ga->getX().GetName(), "nom_a");
   EXPECT_STREQ(ga->getMean().GetName(), "a");
   EXPECT_STREQ(ga->getSigma().GetName(), "sigma_a");
   EXPECT_STREQ(gb->getMean().GetName(), "b");
   EXPECT_DOUBLE_EQ(gb->getSigma().getVal(), 1.0);
   EXPECT_DOUBLE_EQ(gc->getMean().getVal(), 0.5);
   EXPECT_DOUBLE_EQ(gc->getSigma().getVal(), 2.0);
   EXPECT_DOUBLE_EQ(ga->getVal(), std::exp(-0.5 * std::pow((0.3 - 0.1) / 0.5, 2)));
   EXPECT_DOUBLE_EQ(gb->getVal(), std::exp(-0.5 * std::pow((-0.2 - 0.4) / 1.0, 2)));
   EXPECT_DOUBLE_EQ(gc->getVal(), std::exp(-0.5 * std::pow((1.0 - 0.5) / 2.0, 2)));

   // LogNormal in the standard parametrization.
   auto *l1 = dynamic_cast<RooLognormal *>(ws.pdf("y_1_constraint"));
   auto *l2 = dynamic_cast<RooLognormal *>(ws.pdf("y_2_constraint"));
   ASSERT_NE(l1, nullptr);
   ASSERT_NE(l2, nullptr);
   EXPECT_TRUE(l1->useStandardParametrization());
   EXPECT_STREQ(l1->getMedian().GetName(), "mu_1");
   EXPECT_DOUBLE_EQ(l1->getShapeK().getVal(), 1.5);
   EXPECT_DOUBLE_EQ(l2->getMedian().getVal(), 2.0);
   EXPECT_STREQ(l2->getShapeK().GetName(), "sigma_y");
   RooLognormal ref{"ref", "ref", *ws.var("y_1"), *ws.var("mu_1"), RooFit::RooConst(1.5), true};
   EXPECT_DOUBLE_EQ(l1->getVal(), ref.getVal());
}

// Invalid array distributions are rejected with explicit errors.
TEST(RooFitHS3, ArrayDistImportErrors)
{
   // The keys of the array "arr", the error, and the functions of the document.
   struct Case {
      std::string array;
      std::string reason;
      std::string functions = "";
   };
   // The keys of an array of Poisson distributions.
   auto poisson = [](std::string const &components) {
      return R"("component_type": "poisson_dist", "components": [)" + components + "]";
   };
   const std::vector<Case> cases{
      {R"("components": [{"x": "n_1", "mean": "mu_1"}])", "missing \"component_type\""},
      {R"("component_type": "exponential_dist", "components": [{"x": "n_1", "c": 1.0}])",
       "the component_type 'exponential_dist' is not supported"},
      {poisson(""), "\"components\" must be a non-empty list"},
      {R"("default_parameters": {"x": "n_1", "mean": "mu_1"}, )" + poisson(R"({"x": "n_1"})"),
       "\"x\" must be given per component, not in \"default_parameters\""},
      {poisson(R"({"x": "n_1", "mean": "mu_1"}, {"x": "n_1", "mean": "mu_2"})"),
       "the observable 'n_1' is used by two components"},
      {poisson(R"({"x": 3.0, "mean": "mu_1"})"), "\"x\" must be the name of the observable"},
      {poisson(R"({"mean": "mu_1"})"), "component 0 has no \"x\""},
      // A parameter given neither in the component nor in the defaults: the error of the component type's importer.
      {poisson(R"({"x": "n_1"})"), "no \"mean\" given in \"n_1_constraint\""},
      {poisson(R"({"x": "n_1", "mean": "n_2"}, {"x": "n_2", "mean": "mu_2"})"),
       "component 0 depends on the observable 'n_2' of component 1"},
      {poisson(R"({"x": "n_1", "mean": "f"})"), "the parameters of component 0 depend on its own observable 'n_1'",
       R"({"name": "f", "type": "product", "factors": ["n_1", 2.0]})"},
   };
   for (auto const &c : cases) {
      SCOPED_TRACE(c.reason);
      expectImportThrowsWithError(
         document("0.3", {distributions({R"({"name": "arr", "type": "array_dist", )" + c.array + "}"}),
                          functions({c.functions}), defaultDomain({axes({"n_1", "n_2", "mu_1", "mu_2"}, 0, 100)})}),
         c.reason);
   }
}

// A RooProdPdf with the attribute hs3_array_dist is written as array_dist: one component per factor with the fields
// of the standalone distribution, values shared by all components in "default_parameters". The written document
// has no standalone distribution for the components, and it is stable under a round trip.
TEST(RooFitHS3, ArrayDistExport)
{
   Imported ws{document("0.3", {R"("distributions": [
         {"name": "pois", "type": "array_dist", "component_type": "poisson_continuous_dist",
          "default_parameters": {"mean": "lambda_default"},
          "components": [{"x": "n_1", "mean": "lambda_1"}, {"x": "n_2"}, {"x": "n_3", "mean": 7.5}]},
         {"name": "gauss", "type": "array_dist", "component_type": "gaussian_dist",
          "default_parameters": {"sigma": 1.0},
          "components": [{"x": "nom_a", "mean": "a"}, {"x": "nom_b", "mean": "b"}, {"x": "nom_c", "mean": 0.5}]}],
       "functions": [
         {"name": "lambda_1", "type": "product", "factors": [20.0, "np_1"]},
         {"name": "lambda_default", "type": "product", "factors": [10.0, "np_2"]}])",
                                defaultDomain({axes({"n_1", "n_2", "n_3"}, 0, 100), axes({"np_1", "np_2"}, 0, 5),
                                               axes({"nom_a", "nom_b", "nom_c", "a", "b"}, -5, 5)})})};
   Doc exported{ws};

   // The components are not written on their own. The Poisson components have different means, so nothing is
   // shared.
   EXPECT_EQ(exported["distributions"].num_children(), 2u);
   EXPECT_TRUE(matches(exported, "distributions/pois", R"({
      "type": "array_dist", "component_type": "poisson_continuous_dist", "default_parameters": "<absent>",
      "components": [{"x": "n_1", "mean": "lambda_1", "integer": "<absent>"},
                     {"x": "n_2", "mean": "lambda_default", "integer": "<absent>"},
                     {"x": "n_3", "mean": 7.5, "integer": "<absent>"}]})"));
   EXPECT_TRUE(matches(exported, "distributions/gauss", R"({
      "component_type": "gaussian_dist", "default_parameters": {"sigma": 1.0},
      "components": [{"mean": "a", "sigma": "<absent>"}, {"sigma": "<absent>"}, {"mean": 0.5, "sigma": "<absent>"}]})"));

   // The tag is not written as an attribute; it is restored from the type on import.
   EXPECT_THAT(exported.json(), Not(HasSubstr("hs3_array_dist")));

   Imported ws2{exported};
   EXPECT_TRUE(ws2.pdf("gauss")->getAttribute("hs3_array_dist"));
   EXPECT_EQ(toJSON(ws2), exported.json());
}

// An array distribution built in RooFit (a tagged RooProdPdf), the opt-out that writes it as product_dist, and the
// products that cannot be written as array_dist.
TEST(RooFitHS3, ArrayDistExportOfTaggedProduct)
{
   RooRealVar nomA{"nom_a", "nom_a", 0.0, -5.0, 5.0};
   RooRealVar nomB{"nom_b", "nom_b", 0.0, -5.0, 5.0};
   RooRealVar a{"a", "a", 0.0, -5.0, 5.0};
   RooRealVar b{"b", "b", 0.0, -5.0, 5.0};
   RooGaussian gaussA{"gauss_a", "gauss_a", nomA, a, RooFit::RooConst(1.0)};
   RooGaussian gaussB{"gauss_b", "gauss_b", nomB, b, RooFit::RooConst(1.0)};
   RooProdPdf array{"constraints", "constraints", {gaussA, gaussB}};
   array.setAttribute("hs3_array_dist");

   RooWorkspace ws;
   ws.import(array, RooFit::Silence());

   Doc exported{ws};
   EXPECT_EQ(exported["distributions"].num_children(), 1u);
   EXPECT_TRUE(matches(exported, "distributions/constraints", R"({
      "type": "array_dist", "component_type": "gaussian_dist", "default_parameters": {"sigma": 1.0},
      "components": [{"x": "nom_a", "mean": "a"}, {"x": "nom_b"}]})"));

   // Round trip: the components get the names of their observables, the product is tagged again.
   Imported ws2{exported};
   auto *imported = object<RooProdPdf>(ws2, "constraints");
   EXPECT_TRUE(imported->getAttribute("hs3_array_dist"));
   EXPECT_NE(dynamic_cast<RooGaussian *>(ws2.pdf("nom_a_constraint")), nullptr);
   expectRoundTrip(ws, RoundTripModel{"tagged product", RoundTripModel::Builder{}}
                          .with(constraintStyles())
                          .over({"nom_a", "nom_b"})
                          .withOtherComponents());

   // Opt-out: no array_dist is written, the tagged product is an ordinary product_dist.
   Doc plain{ws, {exportArrayDist(false)}};
   EXPECT_EQ(plain["distributions"].num_children(), 3u);
   EXPECT_TRUE(matches(plain, "distributions/constraints", R"({"type": "product_dist"})"));
   EXPECT_TRUE(matches(plain, "distributions/gauss_a", R"({"type": "gaussian_dist"})"));
   EXPECT_THAT(plain.json(), Not(HasSubstr("array_dist")));

   // Products that cannot be written as array_dist: a factor of another class, two factors over the same
   // observable, and a tag on something that is not a product.
   {
      RooRealVar x{"x", "x", 1.0, 0.0, 10.0};
      RooRealVar c{"c", "c", -1.0, -10.0, 0.0};
      RooExponential expo{"expo", "expo", x, c};
      RooProdPdf mixed{"mixed", "mixed", {gaussA, expo}};
      mixed.setAttribute("hs3_array_dist");
      RooWorkspace wsMixed;
      wsMixed.import(mixed, RooFit::Silence());
      expectExportThrowsWithError(wsMixed, "'expo' cannot be a component");
   }
   {
      RooGaussian gaussA2{"gauss_a2", "gauss_a2", nomA, b, RooFit::RooConst(2.0)};
      RooProdPdf sameX{"same_x", "same_x", {gaussA, gaussA2}};
      sameX.setAttribute("hs3_array_dist");
      RooWorkspace wsSameX;
      wsSameX.import(sameX, RooFit::Silence());
      expectExportThrowsWithError(wsSameX, "two components have the observable 'nom_a'");
   }
   {
      RooGaussian tagged{"tagged", "tagged", nomA, a, RooFit::RooConst(1.0)};
      tagged.setAttribute("hs3_array_dist");
      RooWorkspace wsTagged;
      wsTagged.import(tagged, RooFit::Silence());
      expectExportThrowsWithError(wsTagged, "is not a RooProdPdf");
   }
}

// An array distribution paired with an unbinned data set with one entry (the form of the HS3 appendix on auxiliary
// measurements) gives a likelihood with one term per component.
TEST(RooFitHS3, ArrayDistLikelihood)
{
   const std::vector<std::string> observables{"nom_alpha_1", "nom_alpha_2", "nom_alpha_3"};
   Imported ws{document(
      "0.3",
      {R"("distributions": [
         {"name": "gauss_constraints", "type": "array_dist", "component_type": "gaussian_dist",
          "default_parameters": {"sigma": 1.0},
          "components": [{"x": "nom_alpha_1", "mean": "alpha_1"}, {"x": "nom_alpha_2", "mean": "alpha_2"},
                         {"x": "nom_alpha_3", "mean": "alpha_3", "sigma": 2.0}]}],
       "data": [{"name": "global_observables", "type": "unbinned", "entries": [[0.5, -0.5, 1.0]], "axes": [)" +
          axes(observables, -5, 5) + "]}]",
       defaultDomain({axes(observables, -5, 5), axes({"alpha_1", "alpha_2", "alpha_3"}, -5, 5)}),
       defaultValues({values({{"alpha_1", 0.2}, {"alpha_2", 0.1}, {"alpha_3", -0.3}})}),
       R"("likelihoods": [{"name": "likelihood", "distributions": ["gauss_constraints"], "data": ["global_observables"]}],
          "analyses": [{"name": "analysis", "likelihood": "likelihood", "domains": ["default_domain"]}])"})};
   auto *mc = object<RooStats::ModelConfig>(ws, "analysis");
   ASSERT_NE(mc->GetPdf(), nullptr);
   EXPECT_STREQ(mc->GetPdf()->GetName(), "gauss_constraints");
   auto *data = object<RooAbsData>(ws, "global_observables");
   ASSERT_EQ(data->numEntries(), 1);

   std::unique_ptr<RooAbsReal> nll{mc->GetPdf()->createNLL(*data)};
   double expected = 0.0;
   for (const char *name : {"nom_alpha_1_constraint", "nom_alpha_2_constraint", "nom_alpha_3_constraint"}) {
      auto *component = object<RooAbsPdf>(ws, name);
      RooArgSet observable;
      component->getObservables(data->get(), observable);
      expected -= std::log(component->getVal(observable));
   }
   EXPECT_NEAR(nll->getVal(), expected, 1e-10 * std::abs(expected));

   expectRoundTrip(ws, RoundTripModel{"array distribution with data", RoundTripModel::Builder{}}
                          .over(observables)
                          .likelihoodOf("analysis", "global_observables"));
}

// A two-channel HistFactory model, see twoChannelHistFactoryJSON(): with one constraint per parameter or with array
// distributions, and with its data in the workspace, such that the likelihood of the ModelConfig has data.
INSTANTIATE_TEST_SUITE_P(HistFactoryTwoChannels, RoundTrip,
                         testing::Values(RoundTripModel::fromDocument("Constraints", twoChannelHistFactoryJSON(false))
                                            .with(constraintStyles())
                                            .over({"x_c1", "x_c2"})
                                            .comparing(twoChannelLikelihood)
                                            .within(1e-10)
                                            .withOtherComponents(),
                                         RoundTripModel::fromDocument("Arrays", twoChannelHistFactoryJSON(true))
                                            .with(constraintStyles())
                                            .over({"x_c1", "x_c2"})
                                            .comparing(twoChannelLikelihood)
                                            .within(1e-10),
                                         RoundTripModel{"WithData", importTwoChannelsWithData}
                                            .with(constraintStyles())
                                            .over({"x_c1", "x_c2"})
                                            .likelihoodOf("an", "an_data")
                                            .within(1e-10)
                                            .withOtherComponents()),
                         roundTripName);

// A HistFactory modifier whose constraint is an array distribution gets the component that depends on its
// parameter. The model built from the array form equals the model built from one constraint per parameter, also
// for the constraint shared by both channels.
TEST(RooFitHS3, HistFactoryArrayDistConstraints)
{
   Imported individual{twoChannelHistFactoryJSON(false)};
   Imported arrays{twoChannelHistFactoryJSON(true)};
   EXPECT_TRUE(arrays.warnings.empty()) << arrays.warnings;

   // The channels hold the components, not the arrays.
   auto constraintNames = [](RooWorkspace &ws, const char *channel) {
      std::set<std::string> names;
      auto *model = dynamic_cast<RooProdPdf *>(ws.pdf(channel));
      if (model) {
         for (RooAbsArg *pdf : model->pdfList()) {
            if (!dynamic_cast<RooRealSumPdf *>(pdf)) {
               names.insert(pdf->GetName());
            }
         }
      }
      return names;
   };
   EXPECT_EQ(constraintNames(arrays, "model_c1"),
             (std::set<std::string>{"alpha_a_constraint", "nom_gamma_c1_0_constraint", "nom_gamma_c1_1_constraint",
                                    "nom_gs_0_constraint", "nom_gs_1_constraint"}));
   EXPECT_EQ(constraintNames(arrays, "model_c2"),
             (std::set<std::string>{"alpha_a_constraint", "nom_gamma_c2_0_constraint", "nom_gamma_c2_1_constraint"}));
   EXPECT_EQ(constraintNames(individual, "model_c1").size(), 5u);
   EXPECT_EQ(constraintNames(individual, "model_c2").size(), 3u);
   EXPECT_TRUE(arrays.pdf("pois_constraints")->getAttribute("hs3_array_dist"));
   // The shared constraint is one object in both channels, so it is counted once.
   auto countConstraints = [](RooWorkspace &ws) {
      auto *mc = dynamic_cast<RooStats::ModelConfig *>(ws.obj("an"));
      RooArgSet params;
      mc->GetPdf()->getParameters(mc->GetObservables(), params);
      return mc->GetPdf()->getAllConstraints(*mc->GetObservables(), params, true)->size();
   };
   EXPECT_EQ(countConstraints(arrays), 7u);
   EXPECT_EQ(countConstraints(individual), 7u);

   const double nllIndividual = twoChannelNLL(individual, twoChannelPoint);
   const double nllArrays = twoChannelNLL(arrays, twoChannelPoint);
   EXPECT_FALSE(std::isnan(nllIndividual));
   EXPECT_NEAR(nllArrays, nllIndividual, 1e-10 * std::abs(nllIndividual));
}

// Errors and warnings when an array distribution does not fit the parameters that name it.
TEST(RooFitHS3, HistFactoryArrayDistConstraintErrors)
{
   // A normsys that names the array "arr" with the given components as the constraint of alpha_a.
   auto withArray = [](std::string const &components) {
      return HistFactoryDoc{"0.3", R"({"name": "a", "type": "normsys", "parameters": ["alpha_a"],
                                       "data": {"lo": 0.9, "hi": 1.1}, "constraints": ["arr"]})"}
         .interpolation(R"({"type": "mult", "in": "poly6", "out": "exp"})")
         .distributions(R"({"name": "arr", "type": "array_dist", "component_type": "gaussian_dist",
                            "default_parameters": {"sigma": 1.0}, "components": [)" +
                        components + "]}")
         .sections(defaultDomain({axes({"alpha_a", "b", "nom_1", "nom_2"}, -5, 5)}))
         .str();
   };

   expectImportThrowsWithError(withArray(R"({"x": "nom_1", "mean": "b"})"),
                               "the array distribution 'arr' named as the constraint of 'alpha_a' (the normsys "
                               "modifier 'a' of sample 'sig') has no component that depends on this parameter");
   expectImportThrowsWithError(withArray(R"({"x": "nom_1", "mean": "alpha_a"}, {"x": "nom_2", "mean": "alpha_a"})"),
                               "has 2 components that depend on this parameter");

   // A component that no parameter uses is reported: by the standard, the whole array enters the likelihood.
   Imported ws{withArray(R"({"x": "nom_1", "mean": "alpha_a"}, {"x": "nom_2", "mean": "b"})")};
   EXPECT_THAT(ws.warnings,
               HasSubstr("the components 'nom_2_constraint' of the array distribution 'arr' are not used"));
   auto *model = object<RooProdPdf>(ws, "model_channel0");
   EXPECT_NE(model->pdfList().find("nom_1_constraint"), nullptr);
   EXPECT_EQ(model->pdfList().find("nom_2_constraint"), nullptr);
}

// The HistFactory export collects the constraints of the modifiers into one array distribution per type and
// top-level pdf, which the modifiers name for each constrained parameter. The shared constraint is one component.
TEST(RooFitHS3, HistFactoryArrayDistExport)
{
   Imported ws{twoChannelHistFactoryJSON(false)};
   Doc exported{ws};

   for (auto const &dist : exported["distributions"].children()) {
      EXPECT_NE(dist["type"].val(), "gaussian_dist");
      EXPECT_NE(dist["type"].val(), "poisson_continuous_dist");
   }
   EXPECT_TRUE(
      matches(exported, "distributions/an_simPdf_gaussian_constraints",
              R"({"type": "array_dist", "component_type": "gaussian_dist", "components": [{"x": "alpha_a"}]})"));
   // In the order of the channels, their samples and modifiers, and of the bins.
   std::vector<std::string> components;
   for (std::string const gamma : {"gs_0", "gs_1", "gamma_c1_0", "gamma_c1_1", "gamma_c2_0", "gamma_c2_1"}) {
      components.push_back(R"({"x": "nom_)" + gamma + R"(", "mean": ")" + gamma +
                           R"(_poisMean", "integer": "<absent>"})");
   }
   EXPECT_TRUE(matches(exported, "distributions/an_simPdf_poisson_continuous_constraints",
                       R"({"component_type": "poisson_continuous_dist", "default_parameters": "<absent>",
                           "components": [)" +
                          join(components) + "]}"));
   EXPECT_EQ(countOccurrences(exported.json(), R"("constraints":["an_simPdf_gaussian_constraints"])"), 2u);
   EXPECT_EQ(
      countOccurrences(
         exported.json(),
         R"("constraints":["an_simPdf_poisson_continuous_constraints","an_simPdf_poisson_continuous_constraints"])"),
      4u);

   // Opt-out: one constraint distribution per parameter, no array_dist at all.
   const std::string modifiers = "distributions/model_c1/samples/sig/modifiers/";
   Doc plain{ws, {exportArrayDist(false)}};
   EXPECT_THAT(plain.json(), Not(HasSubstr("array_dist")));
   EXPECT_TRUE(matches(plain, modifiers + "a", R"({"constraints": ["alpha_a_constraint"]})"));
   EXPECT_TRUE(matches(plain, modifiers + "staterror",
                       R"({"constraints": ["gamma_c1_0_constraint", "gamma_c1_1_constraint"]})"));
   // From the workspace with imported arrays, the components are written on their own as well.
   Imported ws2{exported};
   Doc plainFromArrays{ws2, {exportArrayDist(false)}};
   EXPECT_THAT(plainFromArrays.json(), Not(HasSubstr("array_dist")));
   EXPECT_TRUE(matches(plainFromArrays, modifiers + "staterror",
                       R"({"constraints": ["nom_gamma_c1_0_constraint", "nom_gamma_c1_1_constraint"]})"));
}

// The tutorial JSON imports and exports stably. Its Poisson constraints have a number as observable, so they cannot
// be components of an array distribution and are written on their own; the Gaussian constraints are collected.
TEST(RooFitHS3, TutorialRF515RoundTrip)
{
   const std::string path = std::string(gROOT->GetTutorialDir().Data()) + "/roofit/roofit/rf515_hfJSON.json";
   RooWorkspace ws;
   ASSERT_TRUE(RooJSONFactoryWSTool{ws}.importJSON(path));
   Doc exported{ws};

   EXPECT_TRUE(matches(exported, "distributions/gamma_bin_0_constraint", R"({"type": "poisson_continuous_dist"})"));
   // The top-level pdf is the RooSimultaneous "main" of the analysis.
   EXPECT_TRUE(matches(exported, "distributions/main_gaussian_constraints", R"({"type": "array_dist"})"));
   EXPECT_EQ(exported["distributions/main_gaussian_constraints/components"].num_children(), 3u);
   EXPECT_TRUE(matches(exported, "distributions/model_channel1/samples/background1/modifiers/staterror",
                       R"({"constraints": ["gamma_bin_0_constraint", "gamma_bin_1_constraint"]})"));

   EXPECT_EQ(reexport(exported), exported.json());

   // None of the constraints has a global observable to pair with data (the Poissons have a number as observable,
   // the Gaussians have the parameter as x and a number as mean), so all of them stay in "aux_distributions".
   Doc withData{ws, {exportConstraintData(true)}};
   EXPECT_TRUE(matches(withData, "likelihoods", R"([{
      "distributions": ["main"],
      "aux_distributions": ["main_gaussian_constraints", "gamma_bin_0_constraint", "gamma_bin_1_constraint"]}])"));
   // No data set for a constraint term: only the observed data (a combined data set and its component).
   EXPECT_EQ(withData.names("data"), (std::vector<std::string>{"observed", "observed_channel1"}));
}

// The constraint terms of the HistFactory channels are listed in the likelihood: in "aux_distributions" by default,
// with the observed values of their observables as constant parameters, or (config().exportConstraintData) in
// "distributions", each paired with a data set that holds the observed values. On import they enter the likelihood
// once, through the channels, and the observed values are taken from the data sets.
TEST(RooFitHS3, HistFactoryConstraintTermsInLikelihood)
{
   // With a combined data set for the RooSimultaneous of the ModelConfig, so that the likelihood has data: without
   // data, it would be written as a pure model, which has no data sets for the constraint terms either.
   RooWorkspace ws;
   importInto(ws, twoChannelHistFactoryJSON(false));
   // The importer merges the data sets of the two terms of the likelihood into one combined data set "an_data" for
   // the RooSimultaneous of the ModelConfig, so the likelihood is written with data.
   ASSERT_NE(ws.data("an_data"), nullptr);
   // An observed value that differs from the default, to see it transported.
   ws.var("nom_gamma_c1_0")->setVal(120.0);

   // The likelihood of the ModelConfig.
   const std::string likelihood = "likelihoods/an_an_data_likelihood";

   // Default: "aux_distributions".
   Doc exported{ws};
   EXPECT_TRUE(matches(exported, likelihood, R"({
      "distributions": ["an_simPdf"], "data": ["an_data"],
      "aux_distributions": ["an_simPdf_poisson_continuous_constraints", "an_simPdf_gaussian_constraints"]})"));
   {
      Imported ws2{exported};
      auto *mc = object<RooStats::ModelConfig>(ws2, "an");
      // Not added again as external constraints: they are in the channels already.
      EXPECT_TRUE(!mc->GetExternalConstraints() || mc->GetExternalConstraints()->empty());
      EXPECT_DOUBLE_EQ(ws2.var("nom_gamma_c1_0")->getVal(), 120.0);
      EXPECT_TRUE(ws2.var("nom_gamma_c1_0")->isConstant());
   }

   // Data route: the terms are paired with data sets holding the observed values.
   Doc withData{ws, {exportConstraintData(true)}};
   // The Gaussian of alpha_a has the parameter as x and a number as mean, so it has no global observable to pair
   // with data and stays in "aux_distributions".
   EXPECT_TRUE(matches(withData, likelihood, R"({
      "distributions": ["an_simPdf"], "data": ["an_data"], "aux_distributions": ["an_simPdf_gaussian_constraints"]})"));
   // The array with its data set is a component of the combined distribution, like the channels.
   EXPECT_TRUE(matches(withData, "distributions/an_simPdf/components/2", R"({
      "key": ["an_simPdf_poisson_continuous_constraints"], "dist": "an_simPdf_poisson_continuous_constraints"})"));
   EXPECT_TRUE(matches(withData, "data/an_data/components/2", R"({
      "key": ["an_simPdf_poisson_continuous_constraints"],
      "dataset": "an_data_an_simPdf_poisson_continuous_constraints"})"));
   EXPECT_TRUE(matches(withData, "misc/ROOT_internal/auxiliary_channels/an_simPdf",
                       R"(["an_simPdf_poisson_continuous_constraints"])"));
   // The observed values, in the order of the channels, their samples and modifiers, and of the bins.
   EXPECT_TRUE(
      matches(withData, "data/an_data_an_simPdf_poisson_continuous_constraints",
              R"({"type": "unbinned", "entries": [[100.0, 100.0, 120.0, 100.0, 100.0, 100.0]], "axes": [)" +
                 axes({"nom_gs_0", "nom_gs_1", "nom_gamma_c1_0", "nom_gamma_c1_1", "nom_gamma_c2_0", "nom_gamma_c2_1"},
                      0, 1000) +
                 "]}"));
   EXPECT_FALSE(withData.has("data/an_simPdf_gaussian_constraints_data"));
   {
      Imported ws3{withData};
      // The terms are not channels, and their data sets are not imported: the values live in the variables.
      auto *mc = object<RooStats::ModelConfig>(ws3, "an");
      auto *simPdf = dynamic_cast<RooSimultaneous *>(mc->GetPdf());
      ASSERT_NE(simPdf, nullptr);
      EXPECT_EQ(simPdf->indexCat().size(), 2);
      EXPECT_EQ(ws3.data("an_data_an_simPdf_poisson_continuous_constraints"), nullptr);
      EXPECT_DOUBLE_EQ(ws3.var("nom_gamma_c1_0")->getVal(), 120.0);
      EXPECT_TRUE(ws3.var("nom_gamma_c1_0")->isConstant());
      ASSERT_NE(mc->GetGlobalObservables(), nullptr);
      EXPECT_NE(mc->GetGlobalObservables()->find("nom_gamma_c1_0"), nullptr);
      EXPECT_TRUE(!mc->GetExternalConstraints() || mc->GetExternalConstraints()->empty());
   }

   // Without arrays, every constraint is paired with its own data set.
   Doc plainWithData{ws, {exportArrayDist(false), exportConstraintData(true)}};
   EXPECT_TRUE(matches(plainWithData, likelihood, R"({
      "distributions": ["an_simPdf", "gs_0_constraint", "gs_1_constraint", "gamma_c1_0_constraint",
                        "gamma_c1_1_constraint", "gamma_c2_0_constraint", "gamma_c2_1_constraint"],
      "data": ["an_data", "gs_0_constraint_data", "gs_1_constraint_data", "gamma_c1_0_constraint_data",
               "gamma_c1_1_constraint_data", "gamma_c2_0_constraint_data", "gamma_c2_1_constraint_data"],
      "aux_distributions": ["alpha_a_constraint"]})"));
   // The six constraint data sets, and the merged data of both channels with its two sub-data sets.
   EXPECT_EQ(plainWithData["data"].num_children(), 9u);
   {
      Imported ws4{plainWithData};
      const double nll = twoChannelNLL(ws, twoChannelPoint);
      EXPECT_NEAR(twoChannelNLL(ws4, twoChannelPoint), nll, 1e-10 * std::abs(nll));
   }
}

// The range of an irregular binning is given by its "edges": HS3 0.3 (#126) forbids "min" and "max" alongside them, in
// binned data and in domains. Documents before HS3 0.3 may have them; from HS3 0.3 on, they give a warning.
TEST(RooFitHS3, IrregularBinnedAxesWithoutRange)
{
   RooRealVar x{"x", "x", 0.0, 3.0};
   std::vector<double> edges{0.0, 1.0, 3.0};
   x.setBinning(RooBinning{2, edges.data()});
   RooDataHist dh{"dh", "dh", RooArgSet{x}};
   RooWorkspace ws;
   ws.import(dh, RooFit::Silence());

   Doc json{ws};
   const std::string onlyEdges = R"({"edges": [0.0, 1.0, 3.0], "min": "<absent>", "max": "<absent>"})";
   EXPECT_TRUE(matches(json, "data/dh/axes/0", onlyEdges));
   EXPECT_TRUE(matches(json, "domains/default_domain/axes/x", onlyEdges));

   Imported ws2{json};
   EXPECT_DOUBLE_EQ(ws2.var("x")->getMin(), 0.0);
   EXPECT_DOUBLE_EQ(ws2.var("x")->getMax(), 3.0);
   EXPECT_EQ(toJSON(ws2), json.json());

   for (std::string version : {"0.2", "0.3"}) {
      Imported ws3{document(version, {section("data", {R"({"name": "binned", "type": "binned", "contents": [1.0, 2.0],
         "axes": [{"name": "x", "min": 0.0, "max": 3.0, "edges": [0.0, 1.0, 3.0]}]})"})})};
      const bool warned =
         ws3.warnings.find("must not be given alongside \"edges\" from HS3 0.3 on") != std::string::npos;
      EXPECT_EQ(warned, version == "0.3") << version << ": " << ws3.warnings;
   }
}

// Errors of zero in binned data stay zero, also before the first positive error.
TEST(RooFitHS3, BinnedDataZeroErrors)
{
   Imported ws{document("0.3", {section("data", {R"({"name": "d", "type": "binned",
      "axes": [{"name": "x", "min": 0.0, "max": 3.0, "nbins": 3}],
      "contents": [10.0, 20.0, 30.0], "errors": [0.0, 2.0, 0.0]})"})})};
   auto *data = object<RooDataHist>(ws, "d");
   EXPECT_DOUBLE_EQ(data->weightSquared(0), 0.0);
   EXPECT_DOUBLE_EQ(data->weightSquared(1), 4.0);
   EXPECT_DOUBLE_EQ(data->weightSquared(2), 0.0);
}

// A HistFactory sample without "errors" contributes no statistical uncertainty to the staterror in HS3 >= 0.3.
// Before, RooFit assumed Poisson uncertainties for it.
TEST(RooFitHS3, HistFactorySampleWithoutErrors)
{
   auto makeJSON = [](std::string const &version) {
      const std::string staterror = R"({"type": "staterror", "parameters": ["g0", "g1"], "constraint": "Gauss"})";
      return document(
         version, {distributions({channel(
                     "model_channel0", {sample("a", R"({"contents": [10.0, 20.0], "errors": [4.0, 6.0]})", staterror),
                                        sample("b", R"({"contents": [10.0, 20.0]})", staterror)})})});
   };
   auto relativeErrors = [](RooWorkspace &ws) {
      std::vector<double> result;
      for (const char *name : {"g0_sigma", "g1_sigma"}) {
         auto *sigma = dynamic_cast<RooAbsReal *>(ws.arg(name));
         result.push_back(sigma ? sigma->getVal() : -1.0);
      }
      return result;
   };

   Imported ws{makeJSON("0.3")};
   const auto errors = relativeErrors(ws);
   EXPECT_DOUBLE_EQ(errors[0], 4.0 / 20.0);
   EXPECT_DOUBLE_EQ(errors[1], 6.0 / 40.0);

   Imported wsLegacy{makeJSON("0.2")};
   const auto legacyErrors = relativeErrors(wsLegacy);
   EXPECT_DOUBLE_EQ(legacyErrors[0], std::sqrt(16.0 + 10.0) / 20.0);
   EXPECT_DOUBLE_EQ(legacyErrors[1], std::sqrt(36.0 + 20.0) / 40.0);
}

namespace {

// The Gaussian constraint "<gamma>_constraint" of a gamma, with the relative uncertainty `sigma`.
std::string gammaConstraint(std::string const &gamma, std::string const &sigma = "0.1")
{
   return R"({"name": ")" + gamma + R"(_constraint", "type": "gaussian_dist", "x": ")" + gamma +
          R"(", "mean": 1.0, "sigma": )" + sigma + "}";
}

} // namespace

// HS3 has one per-bin modifier type, "shapefactor", whose parameters are constrained by the distributions named in
// "constraints", if given, and free otherwise. "shapesys" is a deprecated alias. HistFactory's ShapeSys (constrained)
// and ShapeFactor (free) are both written as "shapefactor".
TEST(RooFitHS3, HistFactoryShapeFactor)
{
   auto makeJSON = [](std::string const &version, std::string const &modifiers,
                      std::string const &constraintDist = "") {
      return HistFactoryDoc{version, modifiers}
         .distributions(constraintDist)
         .sections(defaultDomain({axes({"g_free_0", "g_free_1", "g_c_0", "g_c_1"}, 0, 5)}))
         .str();
   };
   auto countGammaConstraints = [](RooWorkspace &ws) {
      std::size_t n = 0;
      for (RooAbsArg *pdf : ws.allPdfs()) {
         if (dynamic_cast<RooGaussian *>(pdf) || dynamic_cast<RooPoisson *>(pdf)) {
            ++n;
         }
      }
      return n;
   };
   const std::string freeModifier =
      R"({"type": "shapefactor", "name": "free", "parameters": ["g_free_0", "g_free_1"]})";
   const std::string constrainedModifier = R"({"type": "shapefactor", "name": "constr",
      "parameters": ["g_c_0", "g_c_1"], "constraints": ["g_c_0_constraint", null]})";
   const std::string json = makeJSON("0.3", freeModifier + "," + constrainedModifier, gammaConstraint("g_c_0"));

   Imported ws{json};
   // The functions are named as in HistFactory: ShapeFactor for free, ShapeSys for constrained parameters.
   EXPECT_NE(ws.function("channel0_free_shapeFactor"), nullptr);
   EXPECT_NE(ws.function("channel0_constr_ShapeSys"), nullptr);
   EXPECT_EQ(countGammaConstraints(ws), 1u);
   ASSERT_NE(ws.var("g_free_0"), nullptr);
   EXPECT_FALSE(ws.var("g_free_0")->isConstant());

   // Both are written as "shapefactor", and free parameters don't prevent the round trip.
   Doc exported{ws};
   EXPECT_EQ(countOccurrences(exported.json(), R"("type":"shapefactor")"), 2u);
   EXPECT_THAT(exported.json(), Not(HasSubstr("shapesys")));
   const std::string modifiers = "distributions/model_channel0/samples/sig/modifiers/";
   EXPECT_TRUE(matches(exported, modifiers + "free", R"({"type": "shapefactor", "constraints": "<absent>"})"));
   EXPECT_TRUE(matches(exported, modifiers + "constr",
                       R"({"type": "shapefactor", "constraints": ["model_channel0_gaussian_constraints", null]})"));
   EXPECT_EQ(reexport(exported), exported.json());

   // "shapesys" is a deprecated alias.
   Imported ws3{replaced(json, R"("type": "shapefactor")", R"("type": "shapesys")")};
   EXPECT_THAT(ws3.warnings, HasSubstr(R"(the modifier type "shapesys" is deprecated)"));
   EXPECT_EQ(toJSON(ws3), exported.json());

   // In HS3 >= 0.3, "data.vals" without "constraints" don't create constraints.
   Imported ws4{makeJSON("0.3", R"({"type": "shapefactor", "name": "free", "parameters": ["g_free_0", "g_free_1"],
                                    "data": {"vals": [0.1, 0.2]}})")};
   EXPECT_THAT(ws4.warnings, HasSubstr(R"(the "data.vals" of modifier 'free' of sample 'sig' are ignored)"));
   EXPECT_NE(ws4.function("channel0_free_shapeFactor"), nullptr);
   EXPECT_EQ(countGammaConstraints(ws4), 0u);

   // Before HS3 0.3, a "shapesys" with "data.vals" is constrained as before, without deprecation warning.
   Imported ws5{makeJSON("0.2", R"({"type": "shapesys", "name": "constr", "data": {"vals": [0.1, 0.2]}})")};
   EXPECT_THAT(ws5.warnings, Not(HasSubstr("deprecated")));
   EXPECT_NE(ws5.function("channel0_constr_ShapeSys"), nullptr);
   EXPECT_EQ(countGammaConstraints(ws5), 2u);
}

// A shapefactor without "parameters" has one parameter per value in "data.vals", with the default names. This is also
// the case if the values are ignored because the modifier names no constraints.
TEST(RooFitHS3, HistFactoryShapeFactorWithoutParameters)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::ERROR);

   Imported ws{HistFactoryDoc{"0.3", R"({"type": "shapefactor", "name": "free", "data": {"vals": [0.1, 0.2]}})"}};
   auto *phf = object<ParamHistFunc>(ws, "channel0_free_shapeFactor");
   EXPECT_EQ(phf->paramList().size(), 2u);
   for (const char *name : {"gamma_free_bin_0", "gamma_free_bin_1"}) {
      auto *gamma = object<RooRealVar>(ws, name);
      EXPECT_NE(phf->paramList().find(*gamma), nullptr) << name;
      EXPECT_FALSE(gamma->isConstant()) << name;
   }
}

// The gammas of a shapefactor or staterror whose constraint distribution has a type that the HistFactory export
// doesn't recognize are written as free parameters. Their constraint is kept as a constraint term of the channel.
TEST(RooFitHS3, HistFactoryUnsupportedGammaConstraint)
{
   for (std::string const type : {"shapefactor", "staterror"}) {
      SCOPED_TRACE(type);
      Imported ws{
         HistFactoryDoc{"0.3", R"({"type": ")" + type + R"(", "name": "sys", "parameters": ["g0", "g1"],
                                   "constraints": ["g0_constraint", null]})"}
            .sampleData(R"({"contents": [10.0, 20.0], "errors": [1.0, 2.0]})")
            .distributions(
               R"({"name": "g0_constraint", "type": "gamma_dist", "x": "g0", "gamma": 100.0, "beta": 0.01, "mu": 0.0})")
            .sections(defaultDomain({axes({"g0", "g1"}, 0, 5)}))};
      Doc exported{ws};
      EXPECT_FALSE(exported.has("distributions/model_channel0/samples/0/modifiers/0/constraints"));

      Imported ws2{exported};
      auto *constraint = object<RooAbsPdf>(ws2, "g0_constraint");
      EXPECT_TRUE(ws2.pdf("model_channel0")->dependsOn(*constraint));
   }
}

namespace {

// A channel over "x" with two bins, whose sample is scaled by a ParamHistFunc with `nBins` bins over `var`: "x", or
// "y", which has a value that is not the default.
void buildChannelWithParamHistFunc(RooWorkspace &ws, std::string const &var, int nBins)
{
   RooRealVar x{"x", "x", 0.0, 2.0};
   RooRealVar y{"y", "y", 1.5, 0.0, 2.0};
   RooRealVar &phfVar = var == "x" ? x : y;
   phfVar.setBins(nBins);
   RooArgList gammas;
   for (int i = 0; i < nBins; ++i) {
      const std::string name = "g" + std::to_string(i);
      gammas.addOwned(std::make_unique<RooRealVar>(name.c_str(), name.c_str(), 1.0 + 0.1 * i, 0.0, 5.0));
   }
   ParamHistFunc phf{"myPHF", "myPHF", RooArgList{phfVar}, gammas};
   x.setBins(2);
   RooDataHist hist{"sig_hist", "sig_hist", x};
   hist.set(0, 10.0, -1.0);
   hist.set(1, 20.0, -1.0);
   RooHistFunc nominal{"sig_nominal", "sig_nominal", x, hist};
   RooProduct sig{"L_x_sig_channel0", "sig", RooArgList{nominal, phf}};
   RooRealSumPdf model{"model_channel0", "model", RooArgList{sig}, RooArgList{RooFit::RooConst(1.0)}, true};
   ws.import(model, RooFit::Silence());
}

} // namespace

// The ParamHistFunc as shapefactor and as custom modifier. The import names the functions of the channel differently,
// and the documents are stable from the second export on (the import adds the binned likelihood attributes).
INSTANTIATE_TEST_SUITE_P(
   HistFactoryParamHistFunc, RoundTrip,
   testing::Values(RoundTripModel{"AsShapeFactor", [](RooWorkspace &ws) { buildChannelWithParamHistFunc(ws, "x", 2); }}
                      .with(constraintStyles())
                      .withOtherComponents()
                      .stableFromSecondExport(),
                   RoundTripModel{"OverOtherVariable",
                                  [](RooWorkspace &ws) { buildChannelWithParamHistFunc(ws, "y", 3); }}
                      .with(constraintStyles())
                      .withOtherComponents()
                      .stableFromSecondExport()),
   roundTripName);

// A ParamHistFunc of a sample is written as a shapefactor modifier if it has the structure of the ones of HistFactory's
// ShapeSys and ShapeFactor, whatever its name: one parameter per bin of the channel, over its observables. The import
// recreates the function like that. Over other variables, a ParamHistFunc is written as a custom modifier, which refers
// to the function. Over the observables of the channel with another binning, it can't be written.
TEST(RooFitHS3, HistFactoryUnknownParamHistFunc)
{
   // The name of the ParamHistFunc doesn't matter.
   RooWorkspace ws;
   buildChannelWithParamHistFunc(ws, "x", 2);
   const std::string modifier = "distributions/model_channel0/samples/sig/modifiers/myPHF";
   EXPECT_EQ(Doc{ws}.dump(modifier), R"({"name":"myPHF","parameters":["g0","g1"],"type":"shapefactor"})");

   // With another binning of the observable of the channel, it can't be written.
   RooWorkspace wsOtherBinning;
   buildChannelWithParamHistFunc(wsOtherBinning, "x", 3);
   expectExportThrowsWithError(wsOtherBinning, "cannot write the ParamHistFunc 'myPHF'");

   // Over another variable, it is written as a custom modifier, and the import finds the function again.
   RooWorkspace wsOther;
   buildChannelWithParamHistFunc(wsOther, "y", 3);
   Doc exportedOther{wsOther};
   EXPECT_THAT(exportedOther.json(), Not(HasSubstr("shapefactor")));
   EXPECT_EQ(exportedOther.dump(modifier), R"({"name":"myPHF","type":"custom"})");

   Imported ws2{exportedOther};
   EXPECT_NE(dynamic_cast<ParamHistFunc *>(ws2.function("myPHF")), nullptr);
   ASSERT_NE(ws2.pdf("model_channel0"), nullptr);
   ASSERT_NE(ws2.var("y"), nullptr);
   EXPECT_DOUBLE_EQ(ws2.var("y")->getVal(), 1.5);
}

// A legacy "Gauss" constraint is centred at the nominal value of its parameter: 1 for a normfactor, and 0 for a
// normsys or histosys.
TEST(RooFitHS3, HistFactoryLegacyGaussConstraintNominal)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::ERROR);

   const std::string modifiers =
      R"({"type": "normfactor", "name": "mu", "parameter": "mu", "constraint_type": "Gauss"},
         {"type": "normsys", "name": "lumi", "constraint_type": "Gauss", "data": {"lo": 0.95, "hi": 1.05}})";
   Imported ws{HistFactoryDoc{"0.2", modifiers}};
   auto *muConstraint = object<RooGaussian>(ws, "muConstraint");
   EXPECT_DOUBLE_EQ(muConstraint->getMean().getVal(), 1.0);
   auto *alphaConstraint = object<RooGaussian>(ws, "alpha_lumiConstraint");
   EXPECT_DOUBLE_EQ(alphaConstraint->getMean().getVal(), 0.0);
}

// Only the gammas with a constraint distribution are configured from their relative uncertainty, like in HistFactory.
// The other gammas are free: for example, they are not set constant if they have no uncertainty.
TEST(RooFitHS3, HistFactoryUnconstrainedGammasStayFree)
{
   Imported ws{HistFactoryDoc{"0.3", R"({"type": "staterror", "parameters": ["stat_0", "stat_1"]},
                                        {"type": "shapefactor", "name": "sf", "parameters": ["sf_0", "sf_1"],
                                         "constraints": ["sf_0_constraint", null], "data": {"vals": [0.1, 0.0]}})"}
                  .distributions(gammaConstraint("sf_0"))
                  .sections(defaultDomain({axes({"stat_0", "stat_1", "sf_0", "sf_1"}, 0, 5)}))};
   // The sample has no "errors", so the staterror gammas have no uncertainty. But they have no constraints either.
   for (const char *name : {"stat_0", "stat_1", "sf_1"}) {
      auto *gamma = object<RooRealVar>(ws, name);
      EXPECT_FALSE(gamma->isConstant()) << name;
   }
   auto *constrained = object<RooRealVar>(ws, "sf_0");
   EXPECT_FALSE(constrained->isConstant());
   EXPECT_DOUBLE_EQ(constrained->getError(), 0.1);
}

// A channel-level "staterror" that only gives the "relThreshold" holds no constraint information: in HS3 >= 0.3, the
// gammas of a staterror without "constraints" are free also then.
TEST(RooFitHS3, HistFactoryStatErrorThresholdOnly)
{
   Imported ws{HistFactoryDoc{"0.3", R"({"type": "staterror", "parameters": ["stat_0", "stat_1"]})"}
                  .channelKeys(R"("staterror": {"relThreshold": 0.01})")
                  .sampleData(R"({"contents": [10.0, 20.0], "errors": [1.0, 2.0]})")};
   ASSERT_NE(ws.var("stat_0"), nullptr);
   for (RooAbsArg *pdf : ws.allPdfs()) {
      EXPECT_EQ(dynamic_cast<RooPoisson *>(pdf), nullptr) << pdf->GetName();
      EXPECT_EQ(dynamic_cast<RooGaussian *>(pdf), nullptr) << pdf->GetName();
   }
}

// Before HS3 0.3, RooFit created the global observable "nom_<parameter>" of the constraint of a normsys or histosys, so
// that the constraint distribution named in "constraint_name" could use it without defining it.
TEST(RooFitHS3, HistFactoryLegacyNamedConstraintWithImplicitNominal)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::ERROR);

   Imported ws{HistFactoryDoc{"0.2", R"({"type": "normsys", "name": "syst", "parameter": "alpha_syst",
                                         "constraint_name": "alpha_systConstraint", "data": {"lo": 0.9, "hi": 1.1}})"}
                  .distributions(R"({"name": "alpha_systConstraint", "type": "gaussian_dist", "x": "alpha_syst",
                                     "mean": "nom_alpha_syst", "sigma": 1.0})")};
   auto *nominal = object<RooRealVar>(ws, "nom_alpha_syst");
   EXPECT_TRUE(nominal->isConstant());
   EXPECT_DOUBLE_EQ(nominal->getVal(), 0.0);
}

namespace {

// The channels "model_A" and "model_B", whose sample "bkg" has a modifier with the given keys and the gammas g0 and
// g1: with constraints in channel A, without in channel B.
std::string gammasSharedBetweenChannels(std::string const &modifierKeys, std::string const &sampleData)
{
   auto sharedChannel = [&](std::string const &name, std::string const &constraints) {
      const std::string modifier = "{" + modifierKeys + R"(, "parameters": ["g0", "g1"])" + constraints + "}";
      return channel("model_" + name, {sample("bkg", sampleData, modifier)}, "",
                     R"({"name": "x_)" + name + R"(", "min": 0.0, "max": 2.0, "nbins": 2})");
   };
   return document("0.3", {distributions({sharedChannel("A", R"(, "constraints": ["g0_constraint", "g1_constraint"])"),
                                          sharedChannel("B", ""), gammaConstraint("g0"), gammaConstraint("g1")}),
                           defaultDomain({axes({"g0", "g1"}, 0, 5)})});
}

} // namespace

// The gammas of a shapefactor can be shared between channels. In each channel, only its own constraint terms constrain
// them, so that every channel is written as it is.
TEST(RooFitHS3, HistFactoryShapeFactorSharedBetweenChannels)
{
   Imported ws{
      gammasSharedBetweenChannels(R"("type": "shapefactor", "name": "shape")", R"({"contents": [10.0, 20.0]})")};
   EXPECT_NE(ws.function("A_shape_ShapeSys"), nullptr);
   EXPECT_NE(ws.function("B_shape_shapeFactor"), nullptr);

   Doc exported{ws};
   EXPECT_TRUE(exported.has("distributions/model_A/samples/0/modifiers/0/constraints"));
   EXPECT_FALSE(exported.has("distributions/model_B/samples/0/modifiers/0/constraints"));
}

// The constrained gammas of a shapefactor without "data.vals" get the relative uncertainty that their constraint
// distribution encodes as initial error, like in HistFactory. HistFactory's ShapeSys are written this way.
TEST(RooFitHS3, HistFactoryShapeFactorGammaErrorsFromConstraints)
{
   const std::string shapefactor = R"({"type": "shapefactor", "name": "sf", "parameters": ["g0", "g1", "g2"],
                                       "constraints": ["g0_constraint", "g1_constraint", null]})";
   Imported ws{document(
      "0.3", {distributions(
                 {channel("model_channel0", {sample("sig", R"({"contents": [10.0, 20.0, 30.0]})", shapefactor)}, "",
                          R"({"name": "x", "min": 0.0, "max": 3.0, "nbins": 3})"),
                  R"({"name": "g0_constraint", "type": "gaussian_dist", "x": "g0", "mean": 1.0, "sigma": 0.1})",
                  R"({"name": "g1_constraint", "type": "poisson_continuous_dist", "x": 400, "mean": "g1_expected"})"}),
              functions({R"({"name": "g1_expected", "type": "product", "factors": ["g1", 400]})"}),
              defaultDomain({axes({"g0", "g1", "g2"}, 0, 5)})})};
   ASSERT_NE(ws.var("g0"), nullptr);
   ASSERT_NE(ws.var("g1"), nullptr);
   ASSERT_NE(ws.var("g2"), nullptr);
   EXPECT_DOUBLE_EQ(ws.var("g0")->getError(), 0.1);
   EXPECT_DOUBLE_EQ(ws.var("g1")->getError(), 1.0 / std::sqrt(400.0));
   EXPECT_DOUBLE_EQ(ws.var("g2")->getError(), 0.0);
}

// If the samples have no "errors", the staterror gives no relative uncertainties. Gammas with constraint distributions
// are then configured from the uncertainties that these encode, instead of being set constant as without uncertainty.
TEST(RooFitHS3, HistFactoryStatErrorConstraintsWithoutSampleErrors)
{
   Imported ws{HistFactoryDoc{"0.3", R"({"type": "staterror", "parameters": ["g0", "g1"],
                                         "constraints": ["g0_constraint", "g1_constraint"]})"}
                  .distributions(gammaConstraint("g0", "0.1"))
                  .distributions(gammaConstraint("g1", "0.2"))
                  .sections(defaultDomain({axes({"g0", "g1"}, 0, 5)}))};
   for (auto const &[name, sigma] : {std::make_pair("g0", 0.1), std::make_pair("g1", 0.2)}) {
      auto *gamma = object<RooRealVar>(ws, name);
      EXPECT_FALSE(gamma->isConstant()) << name;
      EXPECT_DOUBLE_EQ(gamma->getError(), sigma) << name;
   }
}

// A shapefactor without "name" is named after its channel, its sample and its position among the modifiers, also if
// it names constraints.
TEST(RooFitHS3, HistFactoryShapeFactorWithoutName)
{
   Imported ws{HistFactoryDoc{"0.3", R"({"type": "shapefactor", "parameters": ["g0", "g1"],
                                         "constraints": ["g0_constraint", null]})"}
                  .distributions(gammaConstraint("g0"))
                  .sections(defaultDomain({axes({"g0", "g1"}, 0, 5)}))};
   EXPECT_NE(ws.function("channel0_channel0_sig_syst_0_ShapeSys"), nullptr);
}

namespace {

// A document before HS3 0.3 whose staterror has the gammas g0, the function "k0 * k1" with the value 1, and g1.
HistFactoryDoc functionGammaDoc(std::string const &staterror)
{
   return HistFactoryDoc{"0.2", staterror}
      .sampleData(R"({"contents": [10.0, 20.0], "errors": [1.0, 2.0]})")
      .sections(functions({R"({"name": "g0", "type": "generic_function", "expression": "k0 * k1"})"}))
      .sections(defaultDomain({axes({"k0", "k1", "g1"}, 0, 5)}))
      .sections(defaultValues({values({{"k0", 2.0}, {"k1", 0.5}})}));
}

} // namespace

// Before HS3 0.3, the gammas of a staterror could be functions defined elsewhere in the document. Unlike variables,
// they are not configured from their relative uncertainty, also if they have a constraint distribution.
TEST(RooFitHS3, HistFactoryFunctionGammaWithConstraint)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::ERROR);

   Imported ws{
      functionGammaDoc(R"({"type": "staterror", "parameters": ["g0", "g1"], "constraints": ["g0_constraint", null]})")
         .distributions(gammaConstraint("g0"))};
   auto *phf = object<ParamHistFunc>(ws, "mc_stat_channel0");
   RooAbsArg *g0 = phf->paramList().find("g0");
   ASSERT_NE(g0, nullptr);
   EXPECT_EQ(dynamic_cast<RooRealVar *>(g0), nullptr);
   EXPECT_DOUBLE_EQ(static_cast<RooAbsReal *>(g0)->getVal(), 1.0);
   EXPECT_DOUBLE_EQ(ws.var("k0")->getMax(), 5.0);
}

// Older documents gave the function of a custom modifier in "parameter", without "name".
TEST(RooFitHS3, HistFactoryLegacyCustomModifierParameter)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::ERROR);

   Imported ws{HistFactoryDoc{"0.2", R"({"type": "custom", "parameter": "scale"})"}
                  .sections(functions({R"({"name": "scale", "type": "product", "factors": ["k0", "k1"]})"}))
                  .sections(defaultDomain({axes({"k0", "k1"}, 0, 5)}))};
   // The "parameter" is used, so it must not be reported as ignored.
   EXPECT_THAT(ws.warnings, Not(HasSubstr("\"parameter\"")));
   auto *scale = object<RooAbsReal>(ws, "scale");
   EXPECT_TRUE(ws.pdf("model_channel0")->dependsOn(*scale));
}

// The staterror modifiers of all samples of a channel share the gammas, so the constraints that they name must agree.
TEST(RooFitHS3, HistFactoryStatErrorConstraintsMustAgree)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::FATAL);

   auto makeJSON = [](std::string const &constraintsA, std::string const &constraintsB) {
      auto withStatError = [](std::string const &name, std::string const &constraints) {
         return sample(name, R"({"contents": [10.0, 20.0], "errors": [1.0, 2.0]})",
                       R"({"type": "staterror", "parameters": ["g0", "g1"], "constraints": )" + constraints + "}");
      };
      return document("0.3", {distributions({channel("model_channel0", {withStatError("a", constraintsA),
                                                                        withStatError("b", constraintsB)}),
                                             gammaConstraint("g0"), gammaConstraint("g1")}),
                              defaultDomain({axes({"g0", "g1"}, 0, 5)})});
   };

   Imported ws{makeJSON(R"(["g0_constraint", null])", R"(["g0_constraint", null])")};

   EXPECT_THAT(importError(makeJSON(R"(["g0_constraint", null])", R"([null, "g1_constraint"])")),
               HasSubstr("different constraints"));
}

// The relative uncertainty that a Gaussian staterror constraint encodes is its sigma. It doesn't depend on the value of
// the gamma, which is the mean of the Gaussians that HistFactory creates, so the exported sample errors don't either.
TEST(RooFitHS3, HistFactoryStatErrorExportIndependentOfGammaValue)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::ERROR);

   const std::string staterror = R"({"type": "staterror", "parameters": ["g0", "g1"], "constraint": "Gauss"})";
   Imported ws{HistFactoryDoc{"0.2", staterror}.sampleData(R"({"contents": [10.0, 20.0], "errors": [1.0, 2.0]})")};
   const std::string errors = "distributions/model_channel0/samples/0/data/errors";
   const Doc nominal{ws};
   EXPECT_TRUE(matches(nominal, errors, "[1.0, 2.0]"));
   ws.var("g0")->setVal(1.2);
   EXPECT_EQ(Doc{ws}.dump(errors), nominal.dump(errors));
}

// Before HS3 0.3, the gammas of a staterror could be functions defined elsewhere in the document. If RooFit creates the
// constraints from the relative uncertainties, it constrains the functions too, but it only configures the gammas that
// are variables.
TEST(RooFitHS3, HistFactoryFunctionGammaWithLegacyConstraint)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::ERROR);

   Imported ws{functionGammaDoc(R"({"type": "staterror", "parameters": ["g0", "g1"], "constraint": "Poisson"})")};
   auto *phf = object<ParamHistFunc>(ws, "mc_stat_channel0");
   RooAbsArg *g0 = phf->paramList().find("g0");
   ASSERT_NE(g0, nullptr);
   EXPECT_EQ(dynamic_cast<RooRealVar *>(g0), nullptr);
   for (const char *name : {"g0_constraint", "g1_constraint"}) {
      auto *constraint = object<RooAbsPdf>(ws, name);
      EXPECT_TRUE(ws.pdf("model_channel0")->dependsOn(*constraint)) << name;
   }
   EXPECT_DOUBLE_EQ(ws.var("k0")->getMax(), 5.0);
   EXPECT_DOUBLE_EQ(ws.var("g1")->getMax(), 1.5);

   // The legacy constraint type "Const" sets the gammas that are variables constant.
   Imported wsConst{
      functionGammaDoc(R"({"type": "staterror", "parameters": ["g0", "g1"], "constraint_type": "Const"})")};
   EXPECT_TRUE(wsConst.var("g1")->isConstant());
   EXPECT_FALSE(wsConst.var("k0")->isConstant());
}

// The staterror gammas can be shared between channels. In each channel, only its own constraint terms constrain them.
TEST(RooFitHS3, HistFactoryStatErrorSharedBetweenChannels)
{
   Imported ws{
      gammasSharedBetweenChannels(R"("type": "staterror")", R"({"contents": [10.0, 20.0], "errors": [1.0, 2.0]})")};
   Doc exported{ws};
   EXPECT_TRUE(exported.has("distributions/model_A/samples/0/modifiers/0/constraints"));
   EXPECT_FALSE(exported.has("distributions/model_B/samples/0/modifiers/0/constraints"));
}

// The constraint terms of a channel can be grouped in a nested RooProdPdf: the constraints are found in it.
TEST(RooFitHS3, HistFactoryNestedConstraintTerms)
{
   Imported ws{HistFactoryDoc{"0.3", R"({"type": "shapefactor", "name": "sf", "parameters": ["g0", "g1"],
                                         "constraints": ["g0_constraint", "g1_constraint"]})"}
                  .distributions(gammaConstraint("g0", "0.1"))
                  .distributions(gammaConstraint("g1", "0.2"))
                  .sections(defaultDomain({axes({"g0", "g1"}, 0, 5)}))};
   auto *model = object<RooProdPdf>(ws, "model_channel0");
   RooAbsPdf *sum = nullptr;
   for (RooAbsArg *pdf : model->pdfList()) {
      if (auto *sumPdf = dynamic_cast<RooRealSumPdf *>(pdf)) {
         sum = sumPdf;
      }
   }
   ASSERT_NE(sum, nullptr);
   RooProdPdf constraints{"constraints_channel1", "", RooArgList{*ws.pdf("g0_constraint"), *ws.pdf("g1_constraint")}};
   RooProdPdf nestedModel{"model_channel1", "", RooArgList{*sum, constraints}};
   RooWorkspace wsNested;
   wsNested.import(nestedModel, RooFit::Silence());

   EXPECT_TRUE(matches(Doc{wsNested}, "distributions/model_channel1/samples/0/modifiers/0",
                       R"({"constraints": ["g0_constraint", "g1_constraint"]})"));

   // The likelihood lists the factors as constraint terms, like RooFit, which finds the nested product.
   RooDataHist data{"obsData", "", *wsNested.var("x")};
   wsNested.import(data, RooFit::Silence());
   RooFit::ModelConfig mc{"ModelConfig", &wsNested};
   mc.SetPdf("model_channel1");
   mc.SetObservables("x");
   wsNested.import(mc);
   EXPECT_TRUE(matches(Doc{wsNested}, "likelihoods/ModelConfig_obsData_likelihood",
                       R"({"aux_distributions": ["g0_constraint", "g1_constraint"]})"));
}

// Unnamed shapefactors are named after their channel, sample and position, so that the ones of different samples
// don't share their function.
TEST(RooFitHS3, HistFactoryUnnamedShapeFactorsOfSamples)
{
   const std::string contents = R"({"contents": [10.0, 20.0]})";
   Imported ws{document(
      "0.3", {distributions({channel(
                 "model_channel0", {sample("a", contents, R"({"type": "shapefactor", "parameters": ["a0", "a1"]})"),
                                    sample("b", contents, R"({"type": "shapefactor", "parameters": ["b0", "b1"]})")})}),
              defaultDomain({axes({"a0", "a1", "b0", "b1"}, 0, 5)})})};
   EXPECT_NE(ws.function("channel0_channel0_a_syst_0_shapeFactor"), nullptr);
   EXPECT_NE(ws.function("channel0_channel0_b_syst_0_shapeFactor"), nullptr);
   RooAbsPdf *model = ws.pdf("model_channel0");
   for (const char *name : {"a0", "a1", "b0", "b1"}) {
      ASSERT_NE(ws.var(name), nullptr) << name;
      EXPECT_TRUE(model->dependsOn(*ws.var(name))) << name;
   }
}

// The error messages about modifiers without the optional "name" refer to their position in the document, or to their
// parameter.
TEST(RooFitHS3, HistFactoryUnnamedModifierErrorMessages)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::FATAL);

   auto importError = [](std::string const &modifier) { return HS3Test::importError(HistFactoryDoc{"0.3", modifier}); };
   const std::string notAList =
      importError(R"({"type": "shapefactor", "parameters": ["g0", "g1"], "constraints": "g0_constraint"})");
   EXPECT_THAT(notAList, HasSubstr("samples[0](sig).modifiers[0](channel0_sig_syst_0).constraints: must be an array"));
   const std::string poisson =
      importError(R"({"type": "normsys", "parameters": ["alpha_a"], "constraint_type": "Poisson",
      "interpolation": {"type": "mult", "in": "poly6", "out": "exp"}, "data": {"lo": 0.9, "hi": 1.1}})");
   EXPECT_THAT(poisson,
               HasSubstr("legacy constraint value 'Poisson' for modifier 'alpha_a' is a known constraint type"));
}

// A RooRealVar that is constant in all parameter points must stay a RooRealVar, and a RooConstVar must stay a
// RooConstVar. The RooRealVar has a range in the domain and "const": true in the parameter points. The RooConstVar
// has a domain axis with a single value (HS3 #131), which RooFit reads back as a RooConstVar, and is not in the
// parameter points.
TEST(RooFitHS3, ConstRooRealVarStaysDistinctFromRooConstVar)
{
   RooRealVar x{"x", "x", 0.0, -10.0, 10.0};
   RooConstVar meanConst{"mean_const", "mean_const", 1.0};
   RooRealVar sigmaFixed{"sigma_fixed", "sigma_fixed", 1.0, 0.1, 10.0};
   sigmaFixed.setConstant(true);
   RooGaussian gauss{"gauss", "gauss", x, meanConst, sigmaFixed};

   RooWorkspace ws;
   ws.import(gauss, RooFit::Silence());
   // The RooRealVar is constant in every snapshot, with different values.
   RooRealVar &sigmaInWs = *ws.var("sigma_fixed");
   ws.saveSnapshot("snapshot_a", RooArgSet{sigmaInWs});
   sigmaInWs.setVal(2.0);
   ws.saveSnapshot("snapshot_b", RooArgSet{sigmaInWs});
   sigmaInWs.setVal(1.0);

   Doc doc{ws};

   // The RooRealVar is constant in every parameter point; the RooConstVar is in none.
   EXPECT_TRUE(matches(doc, "parameter_points", R"([
      {"name": "default_values",
       "parameters": [{"name": "sigma_fixed", "const": true, "min": "<absent>", "max": "<absent>"}, {"name": "x"}]},
      {"name": "snapshot_a",
       "parameters": [{"name": "sigma_fixed", "const": true, "min": "<absent>", "max": "<absent>"}]},
      {"name": "snapshot_b",
       "parameters": [{"name": "sigma_fixed", "const": true, "min": "<absent>", "max": "<absent>"}]}])"));

   // In the domain, the RooRealVar has a range and the RooConstVar a single value.
   EXPECT_TRUE(
      matches(doc, "domains/default_domain/axes/sigma_fixed", R"({"min": 0.1, "max": 10.0, "value": "<absent>"})"));
   EXPECT_EQ(doc.dump("domains/default_domain/axes/mean_const"), R"({"name":"mean_const","value":1.0})");

   // On import, the types and the constness are restored, also from the snapshots.
   Imported imported{doc};
   EXPECT_NE(dynamic_cast<RooConstVar *>(imported.obj("mean_const")), nullptr);
   EXPECT_EQ(imported.var("mean_const"), nullptr);
   auto *sigmaImported = object<RooRealVar>(imported, "sigma_fixed");
   EXPECT_TRUE(sigmaImported->isConstant());
   ASSERT_TRUE(imported.loadSnapshot("snapshot_b"));
   EXPECT_DOUBLE_EQ(sigmaImported->getVal(), 2.0);
   EXPECT_TRUE(sigmaImported->isConstant());
   ASSERT_TRUE(imported.loadSnapshot("snapshot_a")); // back to the value of the default parameter point

   // The round trip is stable.
   EXPECT_EQ(toJSON(imported), doc.json());
}

// Pdfs that share a parameter also share the helper functions that are exported for it.
TEST(RooFitHS3, SharedTransformedParameters)
{
   RooWorkspace wsExponential;
   wsExponential.factory("Exponential::exponential_1(x[0, 10], c[-0.1, -5, 0])");
   wsExponential.factory("Exponential::exponential_2(y[0, 10], c)");
   const std::string jsonExponential = toJSON(wsExponential);
   EXPECT_EQ(countOccurrences(jsonExponential, R"("name":"c_exponential_inverted")"), 1u) << jsonExponential;
   expectRoundTrip(wsExponential);

   RooWorkspace wsLognormal;
   wsLognormal.factory("Lognormal::lognormal_1(x[2, 1.1, 10], m0[2, 1.1, 10], k[2, 1.1, 5])");
   wsLognormal.factory("Lognormal::lognormal_2(y[2, 1.1, 10], m0, k)");
   const std::string jsonLognormal = toJSON(wsLognormal);
   EXPECT_EQ(countOccurrences(jsonLognormal, R"("name":"m0_lognormal_log")"), 1u) << jsonLognormal;
   EXPECT_EQ(countOccurrences(jsonLognormal, R"("name":"k_lognormal_log")"), 1u) << jsonLognormal;
   expectRoundTrip(wsLognormal);
}

// ModelConfigs that share the pdf and the dataset, like an S+B and B-only pair, are exported as separate analyses.
TEST(RooFitHS3, ModelConfigsSharingPdfAndData)
{
   auto roundTrip = [](RooWorkspace &ws, const char *pdf, const char *observables) {
      for (const char *name : {"ModelConfig", "ModelConfig_with_poi_0"}) {
         addModelConfig(ws, name, pdf, observables, "mu", "sigma");
      }
      Imported imported{toJSON(ws)};
      EXPECT_NE(imported.obj("ModelConfig"), nullptr);
      EXPECT_NE(imported.obj("ModelConfig_with_poi_0"), nullptr);
   };

   {
      RooWorkspace ws;
      ws.factory("Gaussian::pdf(x[0, -5, 5], mu[0, -5, 5], sigma[1, 0.1, 5])");
      RooDataSet data{"data", "data", *ws.var("x")};
      data.add(*ws.var("x"));
      ws.import(data, RooFit::Silence());
      roundTrip(ws, "pdf", "x");
   }
   {
      RooWorkspace ws;
      // Each component has its own observable, as HS3 requires.
      ws.factory("Gaussian::pdf_a(x_a[0, -5, 5], mu[0, -5, 5], sigma[1, 0.1, 5])");
      ws.factory("Gaussian::pdf_b(x_b[0, -5, 5], mu, sigma)");
      ws.factory("SIMUL::sim_pdf(channel[a=0, b=1], a=pdf_a, b=pdf_b)");
      RooArgSet observables{*ws.var("x_a"), *ws.var("x_b"), *ws.cat("channel")};
      RooDataSet data{"data", "data", observables};
      for (int index : {0, 1}) {
         ws.cat("channel")->setIndex(index);
         data.add(observables);
      }
      ws.import(data, RooFit::Silence());
      roundTrip(ws, "sim_pdf", "x_a,x_b,channel");
   }
}

// Before HS3 v0.3, name clashes only give a warning. RooFit still writes such documents.
TEST(RooFitHS3, NameClashesInOlderDocuments)
{
   auto expectImportWarns = [](std::string const &json, std::string const &name) {
      Imported ws{json};
      EXPECT_THAT(ws.warnings, HasSubstr("the name '" + name + "' is used")) << json;
   };

   const std::string point = R"({"name": "pp", "parameters": []})";
   expectImportWarns(document("0.2", {section("parameter_points", {point, point})}), "pp");

   // A named range and a pdf.
   {
      RooWorkspace ws;
      ws.factory("Gaussian::signal(x[0, -5, 5], mu[0, -5, 5], 1.0)");
      ws.var("x")->setRange("signal", -1, 1);
      expectImportWarns(toJSON(ws), "signal");
   }
   // A snapshot and a pdf.
   {
      RooWorkspace ws;
      ws.factory("Gaussian::model(x[0, -5, 5], mu[0, -5, 5], 1.0)");
      ws.saveSnapshot("model", *ws.var("mu"));
      expectImportWarns(toJSON(ws), "model");
   }
   // A dataset and a pdf.
   {
      RooWorkspace ws;
      ws.factory("Gaussian::model(x[0, -5, 5], mu[0, -5, 5], 1.0)");
      RooDataSet data{"model", "model", *ws.var("x")};
      data.add(*ws.var("x"));
      ws.import(data, RooFit::Silence());
      expectImportWarns(toJSON(ws), "model");
   }
}

// Infinite bounds of unbinned observables are written as null and have to be read back.
TEST(RooFitHS3, UnbinnedDataWithInfiniteBounds)
{
   RooRealVar x{"x", "x", 1.0, 0.0, RooNumber::infinity()};
   RooRealVar y{"y", "y", 1.0, -RooNumber::infinity(), RooNumber::infinity()};
   RooDataSet ds{"ds", "ds", RooArgSet{x, y}};
   ds.add(RooArgSet{x, y});

   RooWorkspace ws1;
   ws1.import(ds, RooFit::Silence());
   Imported ws2{toJSON(ws1)};
   EXPECT_EQ(ws2.data("ds")->numEntries(), 1);
   EXPECT_DOUBLE_EQ(ws2.var("x")->getMin(), 0.0);
   EXPECT_TRUE(RooNumber::isInfinite(ws2.var("x")->getMax()));
   EXPECT_TRUE(RooNumber::isInfinite(ws2.var("y")->getMin()));
   EXPECT_TRUE(RooNumber::isInfinite(ws2.var("y")->getMax()));
}

// A RooAddPdf of a single extended pdf has no coefficients, and it is still extended.
TEST(RooFitHS3, MixtureOfOneExtendedSummand)
{
   RooRealVar x{"x", "x", 0.0, 10.0};
   RooRealVar mean{"mean", "mean", 5.0, 0.0, 10.0};
   RooRealVar sigma{"sigma", "sigma", 1.0, 0.1, 5.0};
   RooRealVar nEvents{"nEvents", "nEvents", 100.0, 0.0, 1000.0};
   RooGaussian gauss{"gauss", "", x, mean, sigma};
   RooExtendPdf extended{"extended", "", gauss, nEvents};
   RooAddPdf mixture{"mixture", "", RooArgList{extended}};
   ASSERT_NE(mixture.extendMode(), RooAbsPdf::CanNotBeExtended);
   expectRoundTrip(mixture);
}

namespace {

// Imports a channel with a normsys that is encoded in an explicit formula and has a RooGaussian constraint.
void importFormulaNormSysChannel(RooWorkspace &ws)
{
   RooRealVar x{"x", "x", 0.0, 2.0};
   x.setBins(2);
   RooRealVar alpha{"alpha_sys", "alpha_sys", 0.0, -5.0, 5.0};
   RooRealVar nominal{"nom_alpha_sys", "nom_alpha_sys", 0.0};
   RooRealVar sigma{"sigma_alpha_sys", "sigma_alpha_sys", 1.0};
   RooRealVar mu{"mu", "mu", 1.0, 0.0, 5.0};
   RooHistFunc hist{"hist_sig", "hist_sig", RooArgSet{x}, makeDataHist("hist_sig_data", x, {100.0, 200.0})};
   RooBinWidthFunction binWidth{"sig_binWidth", "sig_binWidth", hist, true};
   RooProduct shapes{"model_ch_sig_shapes", "", RooArgList{hist, binWidth}};
   RooFormulaVar norm{"sig_norm", "(1+alpha_sys*0.1)*mu", RooArgList{alpha, mu}};
   RooRealSumPdf sum{"ch_model", "ch_model", RooArgList{shapes}, RooArgList{norm}, true};
   RooGaussian constraint{"alpha_sysConstraint", "", alpha, nominal, sigma};
   RooProdPdf prod{"model_ch", "model_ch", RooArgSet{constraint}, RooFit::Conditional(sum, x)};
   ws.import(prod, RooFit::Silence());
}

// The round trip of a channel with a formula among the factors of its sample, see expectRoundTrip(). The yield of the
// channel for the given values of a parameter stands in for the likelihood: the normalized pdfs don't depend on it.
// The import names the functions after the channel, and values agree up to rounding.
RoundTripModel formulaChannel(std::string const &pdf, std::string const &parameter, std::vector<double> const &values)
{
   return RoundTripModel{pdf, RoundTripModel::Builder{}}
      .with(constraintStyles())
      .withOtherComponents()
      .within(1e-12)
      .comparing([=](RooWorkspace &ws) {
         RooAbsPdf *channel = ws.pdf(pdf);
         if (!channel) {
            throw std::runtime_error("no pdf '" + pdf + "'");
         }
         RooArgSet nset{*ws.var("x")};
         std::vector<double> yields;
         for (double value : values) {
            ws.var(parameter)->setVal(value);
            yields.push_back(channel->expectedEvents(nset));
         }
         return yields;
      });
}

// Expects that the workspace has `n` RooRealSumPdfs, which use RooFit's binned likelihood and binned generation.
void expectBinnedSumPdfs(RooWorkspace &ws, std::size_t n)
{
   std::size_t nSums = 0;
   for (RooAbsArg *arg : ws.components()) {
      if (dynamic_cast<RooRealSumPdf *>(arg)) {
         ++nSums;
         EXPECT_TRUE(arg->getAttribute("BinnedLikelihood")) << arg->GetName();
         EXPECT_TRUE(arg->getAttribute("GenerateBinned")) << arg->GetName();
      }
   }
   EXPECT_EQ(nSums, n);
}

} // namespace

TEST(RooFitHS3, FormulaNormSysConstraint)
{
   RooWorkspace ws1;
   importFormulaNormSysChannel(ws1);
   Doc doc{ws1};
   EXPECT_TRUE(matches(doc, "distributions/model_ch/samples/sig/modifiers/sys",
                       R"({"parameters": ["alpha_sys"], "constraints": ["model_ch_gaussian_constraints"]})"));

   Imported ws2{doc};
   // The constraint survives the round trip; its name depends on the grouping of the constraints.
   bool constrained = false;
   for (RooAbsArg *pdf : ws2.allPdfs()) {
      constrained |= dynamic_cast<RooGaussian *>(pdf) && pdf->dependsOn(*ws2.var("alpha_sys"));
   }
   EXPECT_TRUE(constrained);
}

// A histfactory_dist is a product of Poisson distributions of the bin counts: the imported RooRealSumPdf uses RooFit's
// binned likelihood and binned generation, independent of its name and of the "misc" section.
TEST(RooFitHS3, HistFactoryImportEnablesBinnedLikelihood)
{
   Imported ws{HistFactoryDoc{"0.2", ""}};
   expectBinnedSumPdfs(ws, 1);
}

// A sample coefficient that is a sum at the top level, like "1 + 0.2 * k", is not a product of its terms: before, the
// HistFactory exporter split it at the "*" into the factors "1 + 0.2" and "k", which changes the model.
TEST(RooFitHS3, FormulaWithTopLevelSumIsNotSplit)
{
   for (std::string expression : {"1 + 0.2 * k", "1.5 - k * 0.2", "2e-1 * k + 1"}) {
      SCOPED_TRACE(expression);
      RooWorkspace ws;
      {
         RooRealVar x{"x", "x", 0.0, 2.0};
         x.setBins(2);
         RooRealVar k{"k", "k", 0.0, -1.0, 1.0};
         RooHistFunc hist{"hist_sig", "hist_sig", RooArgSet{x}, makeDataHist("hist_sig_data", x, {100.0, 200.0})};
         RooBinWidthFunction binWidth{"sig_binWidth", "sig_binWidth", hist, true};
         RooProduct shapes{"model_ch_sig_shapes", "", RooArgList{hist, binWidth}};
         RooFormulaVar norm{"sig_norm", expression.c_str(), RooArgList{k}};
         RooRealSumPdf sum{"model_ch", "model_ch", RooArgList{shapes}, RooArgList{norm}, true};
         ws.import(sum, RooFit::Silence());
      }
      // The import enables the binned likelihood, which the first export doesn't have.
      expectRoundTrip(ws, formulaChannel("model_ch", "k", {-1.0, 0.0, 0.5}).stableFromSecondExport());
   }
}

// A formula among the factors of a sample is written as one modifier per factor, unless that makes the channel
// invalid: then it stays one custom modifier.
TEST(RooFitHS3, FormulaIsNotSplitIfThatMakesTheChannelInvalid)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::ERROR);

   const std::string b2 = R"({"name": "b2", "type": "generic_function", "expression": "b*b"})";
   const std::string bc = R"({"name": "bc", "type": "generic_function", "expression": "b*c"})";
   struct Case {
      std::string modifiers; // of the sample in the document that is read
      std::string function;
      std::string exported; // the modifiers of the sample in the document that is written
   };
   const std::vector<Case> cases{
      // The parameter would become two normfactors of the same name.
      {R"({"name": "b2", "type": "custom"})", b2, R"([{"name": "b2", "type": "custom"}])"},
      // The sample already has a normfactor of the parameter.
      {R"({"name": "b", "parameter": "b", "type": "normfactor"}, {"name": "bc", "type": "custom"})", bc,
       R"([{"type": "normfactor", "parameters": ["b"]}, {"name": "bc", "type": "custom"}])"},
      // Without such a clash, the formula is split as before.
      {R"({"name": "bc", "type": "custom"})", bc,
       R"([{"type": "normfactor", "parameters": ["b"]}, {"type": "normfactor", "parameters": ["c"]}])"},
   };
   for (auto const &c : cases) {
      SCOPED_TRACE(c.modifiers);
      Imported ws{HistFactoryDoc{"0.3", c.modifiers}
                     .sections(functions({c.function}))
                     .sections(defaultDomain({axes({"b", "c"}, 0.1, 2.0)}))
                     .sections(defaultValues({values({{"b", 0.5}, {"c", 1.5}})}))};
      // The channel is the only distribution, and its only sample has exactly these modifiers.
      EXPECT_TRUE(matches(Doc{ws}, "distributions", R"([{"samples": [{"modifiers": )" + c.exported + "}]}]"));
      // The yield depends on the parameters in the same way.
      expectRoundTrip(ws, formulaChannel("model_channel0", "b", {0.7}));
   }
}

// The +1 and -1 sigma variations of a normsys that is encoded in a formula survive the round trip. In between, the
// formula is linear and the normsys that it is written as interpolates exponentially.
TEST(RooFitHS3, FormulaNormSysVariations)
{
   RooWorkspace ws;
   importFormulaNormSysChannel(ws);
   // The import enables the binned likelihood, which the first export doesn't have.
   expectRoundTrip(ws, formulaChannel("ch_model", "alpha_sys", {-1.0, 0.0, 1.0}).stableFromSecondExport());
}

// Exporting the snapshot of a ModelConfig doesn't change the variables in the workspace.
TEST(RooFitHS3, ExportKeepsVariableStateWithSnapshot)
{
   RooWorkspace ws;
   ws.factory("Gaussian::pdf(x[0, -5, 5], mu[0, -5, 5], 1.0)");
   RooFit::ModelConfig mc{"ModelConfig", &ws};
   mc.SetPdf("pdf");
   mc.SetParametersOfInterest("mu");
   mc.SetObservables("x");
   RooRealVar &mu = *ws.var("mu");
   mu.setVal(1.5);
   mc.SetSnapshot(RooArgSet{mu});
   ws.import(mc);

   mu.setVal(0.0);
   mu.setError(0.25);
   mu.setConstant(true);

   Doc doc{ws};
   EXPECT_DOUBLE_EQ(mu.getVal(), 0.0);
   EXPECT_DOUBLE_EQ(mu.getError(), 0.25);
   EXPECT_TRUE(mu.isConstant());
   EXPECT_TRUE(matches(doc, "misc/minimization/parameter_stepwidths/mu", R"({"step_width": 0.25})"));
   EXPECT_EQ(toJSON(ws), doc.json());
}

// The "init" of an analysis is the snapshot of its ModelConfig, whatever other snapshots exist.
TEST(RooFitHS3, AnalysisInitWithSeveralSnapshots)
{
   RooWorkspace ws;
   ws.factory("Gaussian::pdf(x[0, -5, 5], mu[0, -5, 5], 1.0)");
   RooRealVar &mu = *ws.var("mu");
   RooFit::ModelConfig mc{"ModelConfig", &ws};
   // The name of the second ModelConfig starts with the name of the first one.
   RooFit::ModelConfig mcBkgOnly{"ModelConfig_with_poi_0", &ws};
   for (RooFit::ModelConfig *config : {&mc, &mcBkgOnly}) {
      config->SetPdf("pdf");
      config->SetParametersOfInterest("mu");
      config->SetObservables("x");
   }

   // Only the last call of SetSnapshot() counts.
   mu.setVal(1.0);
   RooArgSet firstSet{mu};
   firstSet.setName("first");
   mc.SetSnapshot(firstSet);
   mu.setVal(1.5);
   RooArgSet secondSet{mu};
   secondSet.setName("second");
   mc.SetSnapshot(secondSet);
   mu.setVal(0.0);
   mcBkgOnly.SetSnapshot(RooArgSet{mu});
   ws.import(mc);
   ws.import(mcBkgOnly);

   Imported imported{toJSON(ws)};

   auto expectSnapshotValue = [&](const char *name, double expected) {
      std::unique_ptr<const RooArgSet> snapshot{static_cast<RooFit::ModelConfig *>(imported.obj(name))->GetSnapshot()};
      ASSERT_NE(snapshot, nullptr) << name;
      EXPECT_DOUBLE_EQ(static_cast<RooRealVar *>(snapshot->find("mu"))->getVal(), expected) << name;
   };
   expectSnapshotValue("ModelConfig", 1.5);
   expectSnapshotValue("ModelConfig_with_poi_0", 0.0);
}

// A histosys parameter that is a function can't be exported: it would come back as a free parameter. As a normsys
// parameter, it is written as a custom modifier, see HistFactoryGammaAndLogNormSyst.
TEST(RooFitHS3, HistoSysWithFunctionAsParameter)
{
   RooRealVar x{"x", "x", 0.0, 2.0};
   x.setBins(2);
   RooRealVar beta{"beta_sys", "beta_sys", 1.0, 0.0, 5.0};
   RooRealVar nominal{"nom_beta_sys", "nom_beta_sys", 1.0};
   RooRealVar sigma{"sigma_beta_sys", "sigma_beta_sys", 0.05};
   RooFormulaVar alphaOfBeta{"alphaOfBeta_sys", "(beta_sys-1)/0.05", RooArgList{beta}};
   RooStats::HistFactory::FlexibleInterpVar norm{"sig_norm", "", RooArgList{alphaOfBeta}, 1.0, {0.95}, {1.05}};
   RooHistFunc hist{"hist_sig", "hist_sig", RooArgSet{x}, makeDataHist("hist_sig_data", x, {100.0, 200.0})};
   PiecewiseInterpolation shape{"sig_shape", "", hist, RooArgList{hist}, RooArgList{hist}, RooArgList{alphaOfBeta}};
   RooBinWidthFunction binWidth{"sig_binWidth", "sig_binWidth", hist, true};
   RooProduct shapes{"model_ch_sig_shapes", "", RooArgList{shape, binWidth}};
   RooRealSumPdf sum{"ch_model", "ch_model", RooArgList{shapes}, RooArgList{norm}, true};
   RooGaussian constraint{"beta_sysConstraint", "", beta, nominal, sigma};
   RooProdPdf prod{"model_ch", "model_ch", RooArgSet{constraint}, RooFit::Conditional(sum, x)};

   RooWorkspace ws;
   ws.import(prod, RooFit::Silence());
   expectExportThrowsWithError(ws, "cannot find constraint for alphaOfBeta_sys");
}

// Documents that predate HS3 v0.2 imply a unit Gaussian constraint for normsys and histosys modifiers.
TEST(RooFitHS3, HistFactoryImplicitConstraintInOldDocuments)
{
   const std::string modifiers = R"(
      {"name": "norm", "parameter": "alpha_norm", "type": "normsys", "data": {"lo": 0.8, "hi": 1.2}},
      {"name": "shape", "parameter": "alpha_shape", "type": "histosys",
       "data": {"lo": {"contents": [8.0, 18.0]}, "hi": {"contents": [12.0, 22.0]}}})";

   Imported ws{HistFactoryDoc{"0.1.90", modifiers}.interpolation(R"({"type": "mult", "in": "poly6", "out": "exp"})")};
   EXPECT_NE(ws.pdf("alpha_normConstraint"), nullptr);
   EXPECT_NE(ws.pdf("alpha_shapeConstraint"), nullptr);
   EXPECT_NE(ws.var("nom_alpha_norm"), nullptr);
}

// Parameters without constraint are reported once per channel, because older RooFit versions constrained them.
TEST(RooFitHS3, HistFactoryUnconstrainedModifierWarning)
{
   Imported ws{HistFactoryDoc{"0.2", R"(
      {"name": "norm", "parameter": "alpha_norm", "type": "normsys", "data": {"lo": 0.8, "hi": 1.2}},
      {"name": "norm2", "parameter": "alpha_norm2", "type": "normsys", "data": {"lo": 0.9, "hi": 1.1}})"}
                  .interpolation(R"({"type": "mult", "in": "poly6", "out": "exp"})")};
   EXPECT_EQ(countOccurrences(ws.warnings, "has no constraint"), 1u) << ws.warnings;
}

// Gamma parameters can't have log-normal constraints. Requesting one must not silently give a Gaussian.
TEST(RooFitHS3, HistFactoryLogNormalGammaConstraint)
{
   for (std::string type : {"LogNormal", "Lognormal"}) {
      for (std::string key : {"constraint", "constraint_type"}) {
         expectImportThrowsWithError(HistFactoryDoc{"0.2", R"({"name": "shape", "type": "shapesys", ")" + key +
                                                              R"(": ")" + type + R"(", "data": {"vals": [0.1, 0.2]}})"},
                                     "unsupported constraint 'LogNormal'");
      }
   }

   expectImportThrowsWithError(HistFactoryDoc{"0.2", R"({"name": "mcstat", "type": "staterror"})"}.channelKeys(
                                  R"("staterror": {"constraint_type": "LogNormal"})"),
                               "unsupported constraint 'LogNormal'");
}

// ROOT 6.38 and 6.40.00 wrote the constraint type of the staterror modifier as "constraint_type".
TEST(RooFitHS3, HistFactoryStatErrorConstraintType)
{
   Imported ws{HistFactoryDoc{"0.2", R"({"name": "staterror", "type": "staterror", "constraint_type": "Gauss"})"}};
   EXPECT_NE(dynamic_cast<RooGaussian *>(ws.pdf("gamma_stat_channel0_bin_0_constraint")), nullptr);
}

// Errors of zero also stay zero if there is no positive error at all.
TEST(RooFitHS3, BinnedDataAllErrorsZero)
{
   Imported ws{document("0.2", {section("data", {R"({
      "name": "d", "type": "binned", "axes": [{"name": "x", "min": 0.0, "max": 2.0, "nbins": 2}],
      "contents": [10.0, 20.0], "errors": [0.0, 0.0]})"})})};
   auto *data = object<RooDataHist>(ws, "d");
   EXPECT_DOUBLE_EQ(data->weight(1), 20.0);
   EXPECT_DOUBLE_EQ(data->weightSquared(0), 0.0);
   EXPECT_DOUBLE_EQ(data->weightSquared(1), 0.0);
}

// Parameter points can set the index categories of combined distributions in the same document.
TEST(RooFitHS3, ParameterPointsWithDocumentCategories)
{
   for (std::string value : {R"("b")", "1"}) {
      Imported ws{R"({
         "metadata": {"hs3_version": "0.2"},
         "domains": [{"name": "default_domain", "type": "product_domain", "axes": [{"name": "x", "min": -5, "max": 5}]}],
         "distributions": [
            {"name": "pdf_a", "type": "gaussian_dist", "x": "x", "mean": 0.0, "sigma": 1.0},
            {"name": "pdf_b", "type": "gaussian_dist", "x": "x", "mean": 1.0, "sigma": 1.0}
         ],
         "parameter_points": [{"name": "point", "parameters": [{"name": "channel", "value": )" +
                  value + R"(}]}],
         "misc": {"ROOT_internal": {"combined_distributions": {"sim_pdf": {
            "distributions": ["pdf_a", "pdf_b"], "index_cat": "channel", "indices": [0, 1], "labels": ["a", "b"]}}}}
      })"};
      ASSERT_NE(ws.cat("channel"), nullptr) << value;
      auto *channel = dynamic_cast<RooCategory *>(ws.getSnapshot("point")->find("channel"));
      ASSERT_NE(channel, nullptr) << value;
      EXPECT_EQ(channel->getCurrentLabel(), std::string("b")) << value;
   }
}

// The warnings of the RooJSONFactoryWSTool are shown with the default message streams.
TEST(RooFitHS3, WarningsAreShownByDefault)
{
   EXPECT_TRUE(
      RooMsgService::instance().isActive(static_cast<RooAbsArg *>(nullptr), RooFit::ObjectHandling, RooFit::WARNING));
   const std::string warnings = warningsOf([] { RooJSONFactoryWSTool::warning("test message"); });
   EXPECT_THAT(warnings, HasSubstr("test message"));
}

namespace {

// Fills the bins of a histogram for a HistFactory measurement and writes it to the file. The error of a bin is given
// as a function of its content; without one, the bins keep their default errors.
void writeHistogram(TFile &file, TH1D hist, std::vector<double> const &contents,
                    std::function<double(double)> const &error = nullptr)
{
   for (std::size_t i = 0; i < contents.size(); ++i) {
      hist.SetBinContent(i + 1, contents[i]);
      if (error) {
         hist.SetBinError(i + 1, error(contents[i]));
      }
   }
   file.WriteTObject(&hist);
}

} // namespace

/// What kind of model is set up. Use this to instantiate
/// a test suite.
enum class MakeModelMode {
   OverallSyst,
   HistoSyst,
   StatSyst,
   ShapeSyst
};

// The model, and whether its bins are non-uniform.
using HFTestParam = std::tuple<MakeModelMode, bool>;

std::string getName(testing::TestParamInfo<HFTestParam> const &paramInfo)
{
   const std::string bins = std::get<1>(paramInfo.param) ? "CustomBins" : "EquidistantBins";
   switch (std::get<0>(paramInfo.param)) {
   case MakeModelMode::OverallSyst: return bins + "_OverallSyst";
   case MakeModelMode::HistoSyst: return bins + "_HistoSyst";
   case MakeModelMode::StatSyst: return bins + "_StatSyst";
   case MakeModelMode::ShapeSyst: return bins + "_ShapeSyst";
   }
   return bins;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// Fixture class to set up a toy hist factory model.
/// In the SetUp() phase
/// - A file with histograms is created. Depending on the value of MakeModelMode,
/// equidistant or custom bins are used, and shape systematics are added.
/// - A Measurement with the histograms in the file is created.
/// - The corresponding workspace is created.
class HFFixture : public testing::TestWithParam<HFTestParam> {
public:
   std::unique_ptr<RooWorkspace> ws;

   void SetUp() override
   {
      using namespace RooStats::HistFactory;

      const MakeModelMode makeModelMode = std::get<0>(GetParam());
      const bool customBins = std::get<1>(GetParam());
      const std::string inputFile = "TestMakeModel.root";

      {
         const double mu = 2.;
         const std::vector<double> sig{10., 20.};
         const std::vector<double> sigDown{8., 0.};
         const std::vector<double> bkg1{100., 0.};
         const std::vector<double> bkg2{0., 100.};
         std::vector<double> data(2);
         for (unsigned int bin = 0; bin < 2; ++bin) {
            if (makeModelMode == MakeModelMode::OverallSyst) {
               data[bin] = mu * sig[bin] + 100.;
            } else if (makeModelMode == MakeModelMode::HistoSyst) {
               // Set data such that alpha = -1., fit should pull parameter.
               data[bin] = mu * sigDown[bin] + 100.;
            } else if (makeModelMode == MakeModelMode::StatSyst) {
               // Kick bin 0, so the gammas have to adapt
               data[bin] = mu * sig[bin] + 100. + (bin == 0 ? 50. : 0.);
            } else if (makeModelMode == MakeModelMode::ShapeSyst) {
               // Distort data such that the shape systematics will pull gamma
               // down in one bin and up in the other.
               data[bin] = mu * sig[bin] + (bin == 0 ? 85. : 110);
            }
         }
         // Tighten the stat. errors of the model
         std::function<double(double)> statError;
         if (makeModelMode == MakeModelMode::StatSyst) {
            statError = [](double content) { return 0.1 * std::sqrt(content); };
         }

         TFile example(inputFile.c_str(), "RECREATE");
         const double customEdges[3] = {0., 1.8, 2.};
         auto write = [&](const char *name, std::vector<double> const &contents,
                          std::function<double(double)> const &error = nullptr) {
            writeHistogram(example, customBins ? TH1D{name, name, 2, customEdges} : TH1D{name, name, 2, 1, 2}, contents,
                           error);
         };
         write("data", data);
         write("signal", sig, statError);
         write("background1", bkg1, statError);
         write("background2", bkg2, statError);
         write("background1_statUncert", {.05, .05}); // A small statistical uncertainty: 5%
         write("histoUnc_sigUp", {12., 40.});
         write("histoUnc_sigDo", sigDown);
         write("background1_shapeSyst", {0.15, 0.0});
         write("background2_shapeSyst", {0.0, 0.10});
      }

      // Create the measurement
      Measurement meas("meas", "meas");
      meas.SetOutputFilePrefix("example_variableBins");
      meas.SetPOI("SigXsecOverSM");
      meas.AddConstantParam("Lumi");
      if (makeModelMode == MakeModelMode::HistoSyst || makeModelMode == MakeModelMode::ShapeSyst) {
         meas.AddConstantParam("gamma_stat_channel1_bin_0");
         meas.AddConstantParam("gamma_stat_channel1_bin_1");
      }
      meas.SetLumi(1.0);
      meas.SetLumiRelErr(0.10);

      Channel chan("channel1");
      chan.SetData("data", inputFile);
      chan.SetStatErrorConfig(0.005, "Poisson");

      Sample signal("signal", "signal", inputFile);
      signal.AddNormFactor("SigXsecOverSM", 1, 0, 3);
      if (makeModelMode == MakeModelMode::HistoSyst) {
         signal.AddHistoSys("SignalShape", "histoUnc_sigDo", inputFile, "", "histoUnc_sigUp", inputFile, "");
      }
      chan.AddSample(signal);

      Sample background1("background1", "background1", inputFile);
      background1.ActivateStatError("background1_statUncert", inputFile);
      if (makeModelMode == MakeModelMode::OverallSyst) {
         background1.AddOverallSys("syst2", 0.95, 1.05);
         background1.AddOverallSys("syst3", 0.99, 1.01);
      }
      if (makeModelMode == MakeModelMode::ShapeSyst) {
         background1.AddShapeSys("background1Shape", Constraint::Gaussian, "background1_shapeSyst", inputFile);
         meas.AddConstantParam("gamma_background1Shape_bin_1");
      }
      chan.AddSample(background1);

      Sample background2("background2", "background2", inputFile);
      background2.ActivateStatError();
      if (makeModelMode == MakeModelMode::OverallSyst) {
         background2.AddOverallSys("syst3", 0.99, 1.01);
         background2.AddOverallSys("syst4", 0.95, 1.05);
      }
      if (makeModelMode == MakeModelMode::ShapeSyst) {
         background2.AddShapeSys("background2Shape", Constraint::Poisson, "background2_shapeSyst", inputFile);
         meas.AddConstantParam("gamma_background2Shape_bin_0");
      }
      chan.AddSample(background2);
      meas.AddChannel(chan);

      // HistFactory must not warn about the measurement.
      RooHelpers::HijackMessageStream hijackW(RooFit::WARNING, RooFit::HistFactory);
      meas.CollectHistograms();
      ws = std::unique_ptr<RooWorkspace>(MakeModelAndMeasurementFast(meas));
      EXPECT_TRUE(hijackW.str().empty()) << "Warnings logged for HistFactory:\n" << hijackW.str();
   }
};

void setInitialFitParameters(RooWorkspace &ws, MakeModelMode makeModelMode)
{
   if (makeModelMode == MakeModelMode::OverallSyst) {
      // The final parameters of alpha_syst2 and alpha_syst4 are very close to the
      // pre-fit value zero. For the fit to converge reliably, the pre-fit values
      // are set away from the minimum.
      ws.var("alpha_syst2")->setVal(1.0);
      ws.var("alpha_syst4")->setVal(-1.0);
   }
   if (makeModelMode == MakeModelMode::ShapeSyst) {
      ws.var("gamma_background1Shape_bin_0")->setVal(0.7);
      ws.var("gamma_background2Shape_bin_1")->setVal(1.3);
   }
}

TEST_P(HFFixture, HS3ClosureLoop)
{
   const MakeModelMode makeModelMode = std::get<0>(GetParam());

   auto *mc = dynamic_cast<RooStats::ModelConfig *>(ws->obj("ModelConfig"));
   EXPECT_TRUE(mc != nullptr);

   RooAbsPdf *pdf = mc->GetPdf();
   EXPECT_TRUE(pdf != nullptr);

   Doc js{*ws};
   Imported wsFromJson{js};

   // Check that JSON > WS > JSON doesn't change the JSON
   EXPECT_EQ(toJSON(wsFromJson), js.json()) << "The JSON -> WS -> JSON roundtrip did not result in the original JSON!";

   // The RooSimultaneous and its data sets are the standard case of combined objects (HS3 #119): they are written
   // as a "combined_dist" and "combined" data sets, and read back into a RooSimultaneous and data sets with the
   // index category.
   EXPECT_TRUE(matches(js, "distributions/simPdf", R"({"type": "combined_dist"})"));
   EXPECT_TRUE(matches(js, "data/obsData", R"({"type": "combined"})"));
   EXPECT_TRUE(matches(js, "data/asimovData", R"({"type": "combined"})"));
   EXPECT_THAT(js.json(), Not(HasSubstr("combined_datasets")));
   EXPECT_THAT(js.json(), Not(HasSubstr("combined_distributions")));

   // The constraints of the modifiers are written as array distributions, one per type (HS3 #109/#124), and none
   // on its own.
   EXPECT_THAT(js.dump("distributions"), Not(HasSubstr(R"("type":"gaussian_dist")")));
   EXPECT_THAT(js.dump("distributions"), Not(HasSubstr(R"("type":"poisson_continuous_dist")")));
   for (std::string name : {"simPdf_gaussian_constraints", "simPdf_poisson_continuous_constraints"}) {
      EXPECT_TRUE(matches(js, "distributions/" + name, R"({"type": "array_dist"})"));
      EXPECT_GT(js["distributions/" + name + "/components"].num_children(), 0u) << name;
   }

   // HistFactory builds its Gaussian constraints with the parameter as x and the global observable as mean. They are
   // written the other way round, so that the observable is x, both in the array and on their own, and they are read
   // back with this orientation.
   RooArgSet const &globalObservables = *mc->GetGlobalObservables();
   auto expectObservableAsX = [&](RooFit::Detail::JSONNode const &gauss) {
      EXPECT_NE(globalObservables.find(gauss["x"].val().c_str()), nullptr) << gauss["x"].val();
      EXPECT_EQ(globalObservables.find(gauss["mean"].val().c_str()), nullptr) << gauss["mean"].val();
   };
   for (auto const &component : js["distributions/simPdf_gaussian_constraints/components"].children()) {
      expectObservableAsX(component);
   }
   Doc plain{*ws, {exportArrayDist(false)}};
   EXPECT_THAT(plain.json(), Not(HasSubstr("array_dist")));
   EXPECT_THAT(plain.json(), HasSubstr(R"("type":"gaussian_dist")"));
   for (auto const &dist : plain["distributions"].children()) {
      if (dist["type"].val() == "gaussian_dist") {
         expectObservableAsX(dist);
      }
   }
   for (RooAbsArg *arg : wsFromJson.allPdfs()) {
      if (auto *gauss = dynamic_cast<RooGaussian *>(arg)) {
         EXPECT_NE(globalObservables.find(gauss->getX().GetName()), nullptr) << gauss->GetName();
      }
   }

   auto *newmc = object<RooStats::ModelConfig>(wsFromJson, "ModelConfig");

   auto *newpdf = dynamic_cast<RooSimultaneous *>(newmc->GetPdf());
   ASSERT_NE(newpdf, nullptr);
   EXPECT_STREQ(newpdf->indexCat().GetName(), "channelCat");
   EXPECT_NE(newmc->GetObservables()->find("channelCat"), nullptr);
   EXPECT_EQ(wsFromJson.allData().size(), ws->allData().size());
   for (const char *dataName : {"obsData", "asimovData"}) {
      RooAbsData *origData = ws->data(dataName);
      RooAbsData *readData = wsFromJson.data(dataName);
      ASSERT_NE(origData, nullptr) << dataName;
      ASSERT_NE(readData, nullptr) << dataName;
      EXPECT_EQ(readData->numEntries(), origData->numEntries()) << dataName;
      EXPECT_DOUBLE_EQ(readData->sumEntries(), origData->sumEntries()) << dataName;
      EXPECT_NE(readData->get()->find("channelCat"), nullptr) << dataName;
   }

   RooAbsData *data = ws->data("obsData");
   EXPECT_TRUE(data != nullptr);

   RooAbsData *newdata = wsFromJson.data("obsData");
   EXPECT_TRUE(newdata != nullptr);

   RooArgSet const &globs = *mc->GetGlobalObservables();
   RooArgSet const &globsFromJson = *newmc->GetGlobalObservables();

   setInitialFitParameters(*ws, makeModelMode);
   setInitialFitParameters(wsFromJson, makeModelMode);

   using namespace RooFit;
   using Res = std::unique_ptr<RooFitResult>;

   Res result{pdf->fitTo(*data, Strategy(1), Minos(*mc->GetParametersOfInterest()), GlobalObservables(globs),
                         PrintLevel(-1), Save())};

   Res resultFromJson{newpdf->fitTo(*newdata, Strategy(1), Minos(*newmc->GetParametersOfInterest()),
                                    GlobalObservables(globsFromJson), PrintLevel(-1), Save())};

   // Do also the reverse comparison to check that the set of constant parameters matches
   EXPECT_TRUE(result->isIdentical(*resultFromJson));
   EXPECT_TRUE(resultFromJson->isIdentical(*result));
}

INSTANTIATE_TEST_SUITE_P(RooFitHS3, HFFixture,
                         testing::Combine(testing::Values(MakeModelMode::OverallSyst, MakeModelMode::HistoSyst,
                                                          MakeModelMode::StatSyst, MakeModelMode::ShapeSyst),
                                          testing::Values(false, true)), // non-uniform bins or not
                         getName);

// HS3 round trip of a free HistFactory ShapeFactor. It is written as a
// "shapefactor" modifier without "constraints", and must survive the
// JSON -> workspace -> JSON round trip with free gammas. The deprecated
// "shapesys" alias is read as well.
TEST(RooFitHS3, HistFactoryImportShapeFactorModifier)
{
   using namespace RooStats::HistFactory;
   const std::string inputFile = "TestHS3ShapeFactor.root";
   {
      TFile f(inputFile.c_str(), "RECREATE");
      writeHistogram(f, TH1D{"data", "data", 2, 1, 2}, {220, 230});
      writeHistogram(f, TH1D{"signal", "signal", 2, 1, 2}, {10, 20});
      writeHistogram(f, TH1D{"background", "background", 2, 1, 2}, {200, 200});
   }

   Measurement meas("meas", "meas");
   meas.SetOutputFilePrefix("HS3ShapeFactor");
   meas.SetPOI("SigXsecOverSM");
   meas.AddConstantParam("Lumi");
   meas.SetLumi(1.0);
   meas.SetLumiRelErr(0.10);

   Channel chan("channel1");
   chan.SetData("data", inputFile);

   Sample sig("signal", "signal", inputFile);
   sig.AddNormFactor("SigXsecOverSM", 1, 0, 3);
   chan.AddSample(sig);

   // ShapeFactor on the background: an unconstrained, bin-by-bin scaling.
   Sample bkg("background", "background", inputFile);
   ShapeFactor sf;
   sf.SetName("bkgShape");
   bkg.AddShapeFactor(sf);
   chan.AddSample(bkg);

   meas.AddChannel(chan);
   meas.CollectHistograms();

   std::unique_ptr<RooWorkspace> ws{MakeModelAndMeasurementFast(meas)};
   ASSERT_NE(ws, nullptr);

   Doc js{*ws};

   // The free ShapeFactor is written as "shapefactor", without constraints.
   EXPECT_EQ(
      js.dump("distributions/model_channel1/samples/background/modifiers/bkgShape"),
      R"({"name":"bkgShape","parameters":["gamma_bkgShape_bin_0","gamma_bkgShape_bin_1"],"type":"shapefactor"})");
   EXPECT_THAT(js.json(), Not(HasSubstr("shapesys")));

   Imported wsFromJson{js};

   // The imported workspace should expose the same free ParamHistFunc gammas.
   ASSERT_NE(wsFromJson.var("gamma_bkgShape_bin_0"), nullptr);
   ASSERT_NE(wsFromJson.var("gamma_bkgShape_bin_1"), nullptr);
   EXPECT_FALSE(wsFromJson.var("gamma_bkgShape_bin_0")->isConstant());

   // Re-exporting should give back the original JSON.
   EXPECT_EQ(toJSON(wsFromJson), js.json()) << "JSON -> WS -> JSON roundtrip changed the JSON";

   // The deprecated "shapesys" alias gives the same model.
   EXPECT_EQ(reexport(replaced(js, R"("type":"shapefactor")", R"("type":"shapesys")")), js.json());
}

namespace {

// The modifiers of the samples of all histfactory_dist channels in a document, as "<type>:<parameters>".
std::multiset<std::string> histFactoryModifiers(std::string const &json, int &nChannels)
{
   std::multiset<std::string> out;
   nChannels = 0;
   auto tree = RooFit::Detail::JSONTree::create(json);
   for (auto const &dist : tree->rootnode()["distributions"].children()) {
      if (dist["type"].val() != "histfactory_dist") {
         continue;
      }
      ++nChannels;
      for (auto const &sample : dist["samples"].children()) {
         for (auto const &mod : sample["modifiers"].children()) {
            std::string entry = mod["type"].val() + ":";
            if (auto *params = mod.find("parameters")) {
               for (auto const &par : params->children()) {
                  entry += par.val() + ",";
               }
            } else if (mod["type"].val() == "custom") {
               entry += mod["name"].val();
            }
            out.insert(entry);
         }
      }
   }
   return out;
}

} // namespace

// The HS3 export of HistFactory models must not depend on the names of the objects: many workspaces were made with
// old or custom versions of HistFactory, or modified after their creation, and still have the structure of a
// HistFactory model. This test builds a HistFactory model, gives every object of the model (except the observables
// and the channel category, which the data refer to) a name that has nothing to do with the HistFactory naming
// scheme, changes its modifiers (adds a custom one, removes one, swaps a constraint, adds a second
// FlexibleInterpVar), uses bins of width pi/10 and, in one channel, the constant inverse bin width of HistFactory
// before ROOT 6.26, and checks that the channels are still written as histfactory_dist, with the expected modifiers,
// and that the likelihood is the same after the round trip.
TEST(RooFitHS3, HistFactoryExportDoesNotDependOnNames)
{
   using namespace RooStats::HistFactory;
   // A bin width that is not exactly representable, as for angular quantities.
   const double binWidth = TMath::Pi() / 10.;

   const std::string inputFile = "TestHS3Names.root";
   {
      TFile f(inputFile.c_str(), "RECREATE");
      auto hist = [&](const char *name, std::vector<double> const &contents) {
         writeHistogram(f, TH1D(name, name, contents.size(), 0, binWidth * contents.size()), contents,
                        [](double content) { return 0.1 * content; });
      };
      hist("data_sr", {130, 120, 110});
      hist("sig_sr", {20, 15, 10});
      hist("bkg_sr", {100, 100, 100});
      hist("bkg_sr_lo", {95, 98, 101});
      hist("bkg_sr_hi", {105, 102, 99});
      hist("bkg_unc_sr", {0.05, 0.1, 0.2});
      hist("data_cr", {210, 190});
      hist("sig_cr", {5, 5});
      hist("bkg_cr", {200, 200});
   }

   Measurement meas("meas", "meas");
   meas.SetOutputFilePrefix("HS3Names");
   meas.SetPOI("mu");
   meas.SetLumi(1.0);
   meas.SetLumiRelErr(0.05);

   Channel sr("SR");
   sr.SetData("data_sr", inputFile);
   sr.SetStatErrorConfig(0.0, "Poisson");
   Sample sigSR("signal", "sig_sr", inputFile);
   sigSR.AddNormFactor("mu", 1, 0, 5);
   sigSR.AddOverallSys("sigSys", 0.9, 1.1);
   sr.AddSample(sigSR);
   Sample bkgSR("background", "bkg_sr", inputFile);
   bkgSR.AddHistoSys("bkgShape", "bkg_sr_lo", inputFile, "", "bkg_sr_hi", inputFile, "");
   bkgSR.AddShapeSys("bkgUnc", Constraint::Poisson, "bkg_unc_sr", inputFile);
   bkgSR.ActivateStatError();
   sr.AddSample(bkgSR);
   meas.AddChannel(sr);

   Channel cr("CR");
   cr.SetData("data_cr", inputFile);
   Sample sigCR("signal", "sig_cr", inputFile);
   sigCR.AddNormFactor("mu", 1, 0, 5);
   cr.AddSample(sigCR);
   Sample bkgCR("background", "bkg_cr", inputFile);
   bkgCR.AddOverallSys("bkgNorm", 0.8, 1.2);
   ShapeFactor sf;
   sf.SetName("crFree");
   bkgCR.AddShapeFactor(sf);
   cr.AddSample(bkgCR);
   meas.AddChannel(cr);

   meas.CollectHistograms();
   std::unique_ptr<RooWorkspace> ws{MakeModelAndMeasurementFast(meas)};
   ASSERT_NE(ws, nullptr);
   auto *mc = static_cast<RooStats::ModelConfig *>(ws->obj("ModelConfig"));
   RooAbsData *data = ws->data("obsData");
   ASSERT_NE(mc, nullptr);
   ASSERT_NE(data, nullptr);

   // A copy of the model that we can modify.
   std::unique_ptr<RooAbsPdf> model{static_cast<RooAbsPdf *>(mc->GetPdf()->cloneTree())};
   RooArgSet nodes;
   model->treeNodeServerList(&nodes);
   auto node = [&](std::string const &name) -> RooAbsArg & {
      RooAbsArg *arg = nodes.find(name.c_str());
      if (!arg) {
         throw std::runtime_error("no node " + name);
      }
      return *arg;
   };
   auto sampleProduct = [&](std::string const &channel, std::string const &sample) -> RooProduct & {
      auto &sum = static_cast<RooRealSumPdf &>(node(channel + "_model"));
      for (RooAbsArg *func : sum.funcList()) {
         if (std::string{func->GetName()}.find(sample + "_" + channel) != std::string::npos) {
            return dynamic_cast<RooProduct &>(*func);
         }
      }
      throw std::runtime_error("no sample " + sample + " in channel " + channel);
   };
   auto channelPdf = [&](std::string const &channel) -> RooProdPdf & {
      return dynamic_cast<RooProdPdf &>(node("model_" + channel));
   };

   // Add a custom modifier to the background of the CR.
   auto &k = *new RooRealVar("k", "k", 0., -1., 1.);
   auto &custom = *new RooFormulaVar("customScale", "1 + 0.2 * @0", RooArgList{k});
   sampleProduct("CR", "background").addTerm(&custom);
   model->addOwnedComponents(RooArgSet{custom, k});

   // Remove the normsys of the signal of the SR, by replacing its FlexibleInterpVar with 1. Its constraint stays in
   // the model, attached to no modifier.
   RooAbsArg *sigSysFiv = nullptr;
   for (RooAbsArg *arg : nodes) {
      if (dynamic_cast<FlexibleInterpVar *>(arg) && arg->dependsOn(node("alpha_sigSys"))) {
         sigSysFiv = arg;
      }
   }
   ASSERT_NE(sigSysFiv, nullptr);
   auto &one = *new RooConstVar("removedSigSys", "", 1.0);
   one.setAttribute((std::string{"ORIGNAME:"} + sigSysFiv->GetName()).c_str());
   model->addOwnedComponents(RooArgSet{one});
   std::vector<RooAbsArg *> fivClients(sigSysFiv->clients().begin(), sigSysFiv->clients().end());
   for (RooAbsArg *client : fivClients) {
      client->redirectServers(RooArgSet{one}, false, true);
   }
   // A constraint term without modifier is written, but not part of the likelihood after the import; with a constant
   // parameter, it only adds a constant.
   static_cast<RooRealVar &>(node("alpha_sigSys")).setConstant(true);

   // Swap the constraint of the normsys of the CR background for a narrower Gaussian.
   auto &alphaBkgNorm = static_cast<RooRealVar &>(node("alpha_bkgNorm"));
   auto &nomBkgNorm = static_cast<RooRealVar &>(node("nom_alpha_bkgNorm"));
   auto &narrow = *new RooGaussian("narrowConstraint", "", alphaBkgNorm, nomBkgNorm, RooFit::RooConst(0.5));
   narrow.setAttribute("ORIGNAME:alpha_bkgNormConstraint");
   model->addOwnedComponents(RooArgSet{narrow});
   channelPdf("CR").redirectServers(RooArgSet{narrow}, false, true);

   // In the CR, the division by the bin width takes the form of HistFactory before ROOT 6.26: a constant inverse bin
   // width, with 7 significant digits as HistFactory wrote it.
   RooAbsArg *crBinWidth = nullptr;
   for (RooAbsArg *arg : nodes) {
      if (dynamic_cast<RooBinWidthFunction *>(arg) && arg->dependsOn(node("obs_x_CR"))) {
         crBinWidth = arg;
      }
   }
   ASSERT_NE(crBinWidth, nullptr);
   char inverseWidth[32];
   std::snprintf(inverseWidth, sizeof(inverseWidth), "%e", 1. / binWidth);
   auto &legacyWidth = *new RooRealVar("legacyInverseWidth", "", std::stod(inverseWidth));
   legacyWidth.setConstant(true);
   legacyWidth.setAttribute((std::string{"ORIGNAME:"} + crBinWidth->GetName()).c_str());
   model->addOwnedComponents(RooArgSet{legacyWidth});
   std::vector<RooAbsArg *> binWidthClients(crBinWidth->clients().begin(), crBinWidth->clients().end());
   for (RooAbsArg *client : binWidthClients) {
      client->redirectServers(RooArgSet{legacyWidth}, false, true);
   }

   // Add a second FlexibleInterpVar with a new constrained parameter to the background of the SR.
   auto &extra = *new RooRealVar("extra", "extra", 0., -5., 5.);
   auto &nomExtra = *new RooRealVar("nom_extra", "nom_extra", 0., -10., 10.);
   nomExtra.setConstant(true);
   auto &extraConstraint = *new RooGaussian("extraConstraint", "", extra, nomExtra, RooFit::RooConst(1.));
   auto &extraFiv = *new FlexibleInterpVar("extraFiv", "", RooArgList{extra}, 1., {0.9}, {1.15}, {4});
   sampleProduct("SR", "background").addTerm(&extraFiv);
   channelPdf("SR").addPdfs(RooArgSet{extraConstraint});
   model->addOwnedComponents(RooArgSet{extraFiv, extraConstraint, extra, nomExtra});

   // Rename everything except the observables, the channel category and the weight variable. The new names are the
   // old ones reversed, so no prefix or suffix of the HistFactory naming scheme survives. The POI gets a name that
   // contains "binWidth".
   RooArgSet keep{*mc->GetObservables()};
   nodes.removeAll();
   model->treeNodeServerList(&nodes);
   std::map<std::string, std::string> newNames;
   for (RooAbsArg *arg : nodes) {
      std::string name = arg->GetName();
      // Constants are named after their value.
      if (keep.find(name.c_str()) || dynamic_cast<RooAbsCategory *>(arg) || dynamic_cast<RooConstVar *>(arg)) {
         continue;
      }
      std::string reversed{name.rbegin(), name.rend()};
      newNames[name] = name == "mu" ? "signal_binWidth_scale" : "x" + reversed;
   }
   for (RooAbsArg *arg : nodes) {
      auto found = newNames.find(arg->GetName());
      if (found != newNames.end()) {
         arg->SetName(found->second.c_str());
      }
   }
   auto renamed = [&](std::string const &name) { return newNames.at(name); };

   RooWorkspace wsMod{"wsMod"};
   wsMod.import(*model, RooFit::Silence());
   wsMod.import(*data, RooFit::Silence());
   auto collect = [&](RooArgSet const &original, std::vector<std::string> const &added) {
      RooArgSet out;
      for (RooAbsArg *arg : original) {
         out.add(*wsMod.arg(renamed(arg->GetName())));
      }
      for (auto const &name : added) {
         out.add(*wsMod.arg(renamed(name)));
      }
      return out;
   };
   RooStats::ModelConfig mcMod{"ModelConfig", &wsMod};
   mcMod.SetPdf(model->GetName());
   mcMod.SetObservables(*mc->GetObservables());
   mcMod.SetParametersOfInterest(collect(*mc->GetParametersOfInterest(), {}));
   mcMod.SetNuisanceParameters(collect(*mc->GetNuisanceParameters(), {"k", "extra"}));
   mcMod.SetGlobalObservables(collect(*mc->GetGlobalObservables(), {"nom_extra"}));
   wsMod.import(mcMod);

   std::string json;
   const std::string warnings = warningsOf([&] { json = toJSON(wsMod); });
   // The renamed legacy inverse bin width is identified by its value, with a warning.
   EXPECT_THAT(warnings, HasSubstr("'" + renamed("legacyInverseWidth") + "'"));

   // Both channels are written by the HistFactory exporter, with the expected modifiers.
   int nChannels = 0;
   const auto modifiers = histFactoryModifiers(json, nChannels);
   EXPECT_EQ(nChannels, 2) << json;
   auto count = [&](std::string const &type, std::string const &param) {
      return modifiers.count(type + ":" + param + ",");
   };
   EXPECT_EQ(count("normfactor", renamed("mu")), 2u) << json;
   EXPECT_EQ(count("normsys", renamed("alpha_sigSys")), 0u) << json;
   EXPECT_EQ(count("normsys", renamed("alpha_bkgNorm")), 1u) << json;
   EXPECT_EQ(count("normsys", renamed("extra")), 1u) << json;
   EXPECT_EQ(count("histosys", renamed("alpha_bkgShape")), 1u) << json;
   EXPECT_EQ(modifiers.count("custom:" + renamed("customScale")), 1u) << json;
   // The only custom modifier is the added one: the bin widths are not written.
   std::size_t nCustom = 0;
   for (auto const &mod : modifiers) {
      nCustom += mod.rfind("custom:", 0) == 0;
   }
   EXPECT_EQ(nCustom, 1u) << json;
   std::size_t nPerBin = 0;
   for (auto const &mod : modifiers) {
      nPerBin += mod.rfind("shapefactor:", 0) == 0 || mod.rfind("staterror:", 0) == 0;
   }
   // SR: shapesys and staterror of the background; CR: the free shapefactor.
   EXPECT_EQ(nPerBin, 3u) << json;

   // The likelihood is the same after the round trip, at several parameter points.
   Imported wsBack{json};
   auto *mcBack = static_cast<RooStats::ModelConfig *>(wsBack.obj("ModelConfig"));
   ASSERT_NE(mcBack, nullptr);
   auto *dataBack = object<RooAbsData>(wsBack, "obsData");

   // The importer enables the binned likelihood for every channel, whatever the names.
   expectBinnedSumPdfs(wsBack, 2);

   // The first round trip loses what the HistFactory model of the import doesn't have: the "misc" attributes of the
   // renamed RooRealSumPdfs (the importer creates its own, with the binned likelihood enabled), and the constraint term
   // without modifier. From the second export on, the document is stable.
   const std::string json2 = toJSON(wsBack);
   EXPECT_EQ(reexport(json2), json2);

   using RooFit::GlobalObservables;
   std::unique_ptr<RooAbsReal> nll{
      wsMod.pdf(model->GetName())->createNLL(*wsMod.data("obsData"), GlobalObservables(*mcMod.GetGlobalObservables()))};
   std::unique_ptr<RooAbsReal> nllBack{
      mcBack->GetPdf()->createNLL(*dataBack, GlobalObservables(*mcBack->GetGlobalObservables()))};
   RooArgSet params;
   nll->getParameters(nullptr, params);
   // Up to a constant: the constraint terms may be written in a different but equivalent form. The tolerance allows for
   // the 7 significant digits of the legacy inverse bin width.
   const double nll0 = nll->getVal();
   const double nllBack0 = nllBack->getVal();
   const std::vector<double> shifts{0.3, -0.5, 0.2};
   for (double shift : shifts) {
      for (RooAbsArg *arg : params) {
         auto *var = dynamic_cast<RooRealVar *>(arg);
         if (!var || var->isConstant()) {
            continue;
         }
         const double value = var->getVal() + shift * 0.1 * std::max(std::abs(var->getVal()), 1.);
         var->setVal(std::clamp(value, var->getMin(), var->getMax()));
         auto *varBack = wsBack.var(var->GetName());
         ASSERT_NE(varBack, nullptr) << var->GetName();
         varBack->setVal(var->getVal());
      }
      EXPECT_NEAR(nllBack->getVal() - nllBack0, nll->getVal() - nll0, 1e-5) << "shift " << shift;
   }
}

namespace {

// A channel with the structure of HistFactory before ROOT 6.26, named as in its workspaces: the samples have a constant
// RooRealVar "binWidth_<observables>_<i>" with the number of bins over the range as coefficient in the RooRealSumPdf.
std::unique_ptr<RooWorkspace> makeLegacyHistFactoryWorkspace()
{
   RooRealVar obs{"obs_x_channel1", "obs_x_channel1", 1.0, 2.0};
   obs.setBins(2);
   RooRealVar mu{"SigXsecOverSM", "SigXsecOverSM", 1.0, 0.0, 3.0};
   RooRealVar alpha{"alpha_syst", "alpha_syst", 0.0, -5.0, 5.0};
   RooRealVar nomAlpha{"nom_alpha_syst", "nom_alpha_syst", 0.0, -10.0, 10.0};
   nomAlpha.setConstant(true);
   RooGaussian constraint{"alpha_systConstraint", "", alpha, nomAlpha, RooFit::RooConst(1.0)};
   RooStats::HistFactory::FlexibleInterpVar epsilon{
      "background_channel1_epsilon", "", RooArgList{alpha}, 1.0, {0.95}, {1.05}, {4}};
   RooHistFunc sigHist{"signal_channel1_nominal", "", RooArgSet{obs},
                       makeDataHist("signal_channel1nominalDHist", obs, {20.0, 10.0})};
   RooHistFunc bkgHist{"background_channel1_nominal", "", RooArgSet{obs},
                       makeDataHist("background_channel1nominalDHist", obs, {100.0, 100.0})};
   RooProduct sig{"signal_channel1_overallSyst_x_Exp", "", RooArgList{sigHist, mu}};
   RooProduct bkg{"background_channel1_overallSyst_x_Exp", "", RooArgList{bkgHist, epsilon}};
   RooRealVar binWidth0{"binWidth_obs_x_channel1_0", "", 2.0};
   RooRealVar binWidth1{"binWidth_obs_x_channel1_1", "", 2.0};
   RooRealSumPdf sum{"channel1_model", "", RooArgList{sig, bkg}, RooArgList{binWidth0, binWidth1}, true};
   RooProdPdf model{"model_channel1", "", RooArgList{constraint, sum}};

   auto ws = std::make_unique<RooWorkspace>("channel1");
   ws->import(model, RooFit::Silence());
   ws->import(*makeDataHist("obsData", obs, {122.0, 112.0}), RooFit::Silence());
   RooStats::ModelConfig mc{"ModelConfig", ws.get()};
   mc.SetPdf(model.GetName());
   mc.SetObservables(RooArgSet{obs});
   mc.SetParametersOfInterest(RooArgSet{mu});
   mc.SetNuisanceParameters(RooArgSet{alpha});
   mc.SetGlobalObservables(RooArgSet{nomAlpha});
   ws->import(mc);
   ws->saveSnapshot("NominalParamValues", ws->allVars());
   return ws;
}

} // namespace

// Before ROOT 6.26, HistFactory multiplied every sample with a constant inverse bin width, a RooRealVar
// "binWidth_<observables>_<i>" in the coefficients of the RooRealSumPdf. histfactory_dist implies it, so it is not
// written. It is identified by its structure, not by its name; a factor that is identified only by its structure is
// reported with a warning, and the identification can be switched off.
TEST(RooFitHS3, HistFactoryLegacyBinWidthFactor)
{
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::NumericIntegration, true};

   std::unique_ptr<RooWorkspace> ws = makeLegacyHistFactoryWorkspace();
   auto *mc = static_cast<RooStats::ModelConfig *>(ws->obj("ModelConfig"));
   ASSERT_NE(mc, nullptr);

   // A copy of the workspace in which the bin width factors have names without "binWidth".
   std::unique_ptr<RooAbsPdf> model{static_cast<RooAbsPdf *>(mc->GetPdf()->cloneTree())};
   RooArgSet nodes;
   model->treeNodeServerList(&nodes);
   std::vector<std::string> renamedFactors;
   for (RooAbsArg *arg : nodes) {
      if (std::string{arg->GetName()}.find("binWidth") != std::string::npos) {
         renamedFactors.push_back("w_" + std::to_string(renamedFactors.size()));
         arg->SetName(renamedFactors.back().c_str());
      }
   }
   ASSERT_FALSE(renamedFactors.empty());
   auto makeWorkspace = [&](RooWorkspace &out) {
      out.import(*model, RooFit::Silence());
      out.import(*ws->data("obsData"), RooFit::Silence());
      RooStats::ModelConfig mcOut{"ModelConfig", &out};
      mcOut.SetPdf(model->GetName());
      mcOut.SetObservables(*mc->GetObservables());
      mcOut.SetParametersOfInterest(*mc->GetParametersOfInterest());
      mcOut.SetNuisanceParameters(*mc->GetNuisanceParameters());
      mcOut.SetGlobalObservables(*mc->GetGlobalObservables());
      out.import(mcOut);
   };
   auto normfactorsOf = [&](std::string const &json) {
      int nChannels = 0;
      const auto modifiers = histFactoryModifiers(json, nChannels);
      EXPECT_EQ(nChannels, 1) << json;
      std::size_t n = 0;
      for (auto const &name : renamedFactors) {
         n += modifiers.count("normfactor:" + name + ",");
      }
      return n;
   };

   // With the original names, the factors are not written as modifiers, and there is no warning. (They are still
   // written as part of the snapshot of the workspace, as parameter point.)
   {
      Doc doc{*ws};
      int nChannels = 0;
      for (auto const &modifier : histFactoryModifiers(doc, nChannels)) {
         EXPECT_THAT(modifier, Not(HasSubstr("binWidth")));
      }
      EXPECT_THAT(doc.warnings(), Not(HasSubstr("inverse bin width")));
   }

   // With other names, they are identified by their structure, with a warning, and the likelihood is unchanged.
   RooWorkspace wsRenamed;
   makeWorkspace(wsRenamed);
   {
      Doc doc{wsRenamed};
      EXPECT_EQ(normfactorsOf(doc), 0u) << doc.json();
      // They are not model parameters either: no domain has them.
      for (auto const &factor : renamedFactors) {
         EXPECT_THAT(doc.dump("domains"), Not(HasSubstr("\"name\":\"" + factor + "\"")));
      }
      EXPECT_THAT(doc.warnings(), HasSubstr("'" + renamedFactors.front() + "'"));
      EXPECT_THAT(doc.warnings(), HasSubstr("inverse bin width"));

      Imported wsBack{doc};
      auto *mcBack = static_cast<RooStats::ModelConfig *>(wsBack.obj("ModelConfig"));
      ASSERT_NE(mcBack, nullptr);
      using RooFit::GlobalObservables;
      std::unique_ptr<RooAbsReal> nll{
         mc->GetPdf()->createNLL(*ws->data("obsData"), GlobalObservables(*mc->GetGlobalObservables()))};
      std::unique_ptr<RooAbsReal> nllBack{
         mcBack->GetPdf()->createNLL(*wsBack.data("obsData"), GlobalObservables(*mcBack->GetGlobalObservables()))};
      const double nll0 = nll->getVal();
      const double nllBack0 = nllBack->getVal();
      for (double mu : {0.5, 2.}) {
         ws->var("SigXsecOverSM")->setVal(mu);
         wsBack.var("SigXsecOverSM")->setVal(mu);
         EXPECT_NEAR(nllBack->getVal() - nllBack0, nll->getVal() - nll0, 1e-6) << "mu = " << mu;
      }
      ws->var("SigXsecOverSM")->setVal(1.);
   }

   // A factor that is floating in a snapshot is not a bin width.
   {
      RooWorkspace wsSnapshot;
      makeWorkspace(wsSnapshot);
      RooRealVar &factor = *wsSnapshot.var(renamedFactors.front());
      factor.setConstant(false);
      wsSnapshot.saveSnapshot("floating", RooArgSet{factor});
      factor.setConstant(true);
      const std::string json = toJSON(wsSnapshot);
      EXPECT_EQ(normfactorsOf(json), 1u) << json;
   }

   // The identification can be switched off: the factors are then written as normfactors.
   const std::string jsonWithFactors = toJSON(wsRenamed, {identifyLegacyBinWidthFactors(false)});
   EXPECT_EQ(normfactorsOf(jsonWithFactors), renamedFactors.size()) << jsonWithFactors;
}

// If the exported document is not valid, this is reported, but everything that can be exported is still written. Here
// the pdf of the ModelConfig has no exporter, so the likelihood refers to a distribution that doesn't exist.
TEST(RooFitHS3, InvalidExportIsStillWritten)
{
   RooWorkspace ws{"ws"};
   ws.factory("Voigtian::voigt(x[0, -10, 10], mu[1, -5, 5], width[1, 0.1, 5], sigma[1, 0.1, 5])");
   ws.factory("Gaussian::gauss(x, mu, sigma)");
   RooDataSet data{"data", "data", *ws.var("x")};
   data.add(*ws.var("x"));
   ws.import(data);
   ws.saveSnapshot("snap", *ws.var("mu"));
   addModelConfig(ws, "mc", "voigt", "x", "mu");

   Doc doc{ws};
   EXPECT_THAT(doc.warnings(),
               HasSubstr("the exported document is not valid: the distribution 'voigt' does not exist"));

   EXPECT_TRUE(matches(doc, "metadata", R"({"hs3_version": "0.3"})"));
   for (std::string section : {"domains", "parameter_points", "data", "likelihoods", "analyses"}) {
      EXPECT_GT(doc[section].num_children(), 0u) << section << ": " << doc.json();
   }
   EXPECT_TRUE(doc.has("parameter_points/snap"));
   EXPECT_TRUE(doc.has("distributions/gauss"));
}

// A named RooConstVar is exported as a fixed domain axis. JSON can't represent a value that is not finite, which is
// reported on export, and such an axis is not read back as a parameter without range.
TEST(RooFitHS3, NonFiniteRooConstVar)
{
   RooWorkspace ws{"ws"};
   RooRealVar x{"x", "x", 0.0, -10.0, 10.0};
   RooConstVar cinf{"cinf", "cinf", std::numeric_limits<double>::infinity()};
   RooFormulaVar formula{"formula", "x[0] + x[1]", {x, cinf}};
   ws.import(formula, RooFit::Silence());

   Doc doc{ws};
   EXPECT_THAT(doc.warnings(), HasSubstr("the single value of the parameter 'cinf' is not finite"));

   expectImportThrowsWithError(doc, "must be a number");
}

// A fixed domain axis (HS3 #131): the parameter only takes the given "value", and RooFit reads it as a RooConstVar.
// It needs no entry in a parameter point. A parameter point must not give it another value, the axis has no range,
// binning or categories, and all domains that list the parameter give it the same value.
TEST(RooFitHS3, FixedDomainAxis)
{
   // A Gaussian with the mean "mZ" and the width "wZ", which has a fixed axis. Given are the axis of "mZ" in the
   // default domain, and optionally its axis in a second domain and an entry of the parameter point.
   auto makeJSON = [](std::string const &axis, std::string const &otherAxis = "", std::string const &point = "") {
      const std::string domain = R"({"name": "default_domain", "type": "product_domain", "axes": [)" +
                                 join({axes({"x"}, 80, 100), R"({"name": "wZ", "value": 2.4955})", axis}) + "]}";
      const std::string other =
         otherAxis.empty() ? "" : R"({"name": "other", "type": "product_domain", "axes": [)" + otherAxis + "]}";
      return document(
         "0.3",
         {distributions({R"({"name": "model", "type": "gaussian_dist", "x": "x", "mean": "mZ", "sigma": "wZ"})"}),
          section("domains", {domain, other}), defaultValues({values({{"x", 90.0}}), point})});
   };
   const std::string mZ = R"({"name": "mZ", "value": 91.188})";

   Imported ws{makeJSON(mZ)};
   for (const char *name : {"mZ", "wZ"}) {
      object<RooConstVar>(ws, name);
      EXPECT_EQ(ws.var(name), nullptr) << name;
   }
   EXPECT_DOUBLE_EQ(static_cast<RooConstVar *>(ws.obj("mZ"))->getVal(), 91.188);
   EXPECT_DOUBLE_EQ(static_cast<RooConstVar *>(ws.obj("wZ"))->getVal(), 2.4955);
   // Written back the same way: as fixed axes, not in the parameter points.
   Doc exported{ws};
   EXPECT_EQ(exported.dump("domains/default_domain/axes/mZ"), R"({"name":"mZ","value":91.188})");
   EXPECT_EQ(exported.dump("domains/default_domain/axes/wZ"), R"({"name":"wZ","value":2.4955})");
   EXPECT_THAT(exported.dump("parameter_points"), Not(testing::AnyOf(HasSubstr("mZ"), HasSubstr("wZ"))));
   EXPECT_EQ(reexport(exported), exported.json());

   // The same value in a parameter point is fine, also with "const"; another value is an error.
   {
      Imported wsSame{makeJSON(mZ, "", R"({"name": "mZ", "value": 91.188, "const": true})")};
   }
   expectImportThrowsWithError(makeJSON(mZ, "", R"({"name": "mZ", "value": 90.0})"),
                               "the parameter point 'default_values' gives the parameter 'mZ' the value 90");

   // The single value excludes a range, a binning and categories, and must be a number.
   for (const char *key : {R"("min": 0.0)", R"("max": 100.0)", R"("nbins": 10)", R"("edges": [0.0, 1.0])"}) {
      expectImportThrowsWithError(makeJSON(std::string(R"({"name": "mZ", "value": 91.188, )") + key + "}"),
                                  "has \"value\" and");
   }
   expectImportThrowsWithError(makeJSON(R"({"name": "mZ", "value": "heavy"})"), "must be a number");
   expectImportThrowsWithError(makeJSON(R"({"name": "mZ", "value": null})"), "must be a number");

   // All domains that list the parameter give it the same single value.
   expectImportThrowsWithError(makeJSON(mZ, R"({"name": "mZ", "value": 90.0})"),
                               "the parameter 'mZ' has different single values in the domains");
   expectImportThrowsWithError(makeJSON(mZ, R"({"name": "mZ", "min": 80.0, "max": 100.0})"),
                               "but a range in the domain 'other'");
   expectImportThrowsWithError(makeJSON(mZ, R"({"name": "mZ", "categories": ["a", "b"]})"),
                               "but categories in the domain 'other'");

   // A parameter without domain is a RooRealVar: constant with "const": true, otherwise free, with a warning.
   const std::string unused = R"({"name": "unused", "value": 0.0})";
   {
      Imported wsNoDomain{makeJSON(unused, "", R"({"name": "mZ", "value": 91.188, "const": true})")};
      ASSERT_NE(wsNoDomain.var("mZ"), nullptr);
      EXPECT_TRUE(wsNoDomain.var("mZ")->isConstant());
      EXPECT_THAT(wsNoDomain.warnings, Not(HasSubstr("'mZ' is not listed in any domain")));
   }
   {
      Imported wsFree{makeJSON(unused, "", mZ)};
      ASSERT_NE(wsFree.var("mZ"), nullptr);
      EXPECT_FALSE(wsFree.var("mZ")->isConstant());
      EXPECT_THAT(wsFree.warnings, HasSubstr("the parameter 'mZ' is not listed in any domain and is not constant"));
   }

   // Files of earlier ROOT versions marked named RooConstVars with the attribute "is_const_var".
   Imported wsLegacy{document(
      "0.2", {distributions({R"({"name": "model", "type": "gaussian_dist", "x": "x", "mean": "mZ", "sigma": 2.5})"}),
              defaultDomain({axes({"x"}, 80, 100)}),
              defaultValues({values({{"x", 90.0}}), R"({"name": "mZ", "value": 91.188, "const": true})"}),
              R"("misc": {"ROOT_internal": {"attributes": {"mZ": {"is_const_var": 1}}}})"})};
   ASSERT_NE(dynamic_cast<RooConstVar *>(wsLegacy.obj("mZ")), nullptr);
   EXPECT_DOUBLE_EQ(static_cast<RooConstVar *>(wsLegacy.obj("mZ"))->getVal(), 91.188);
}

// Binned contents are flattened with the last axis varying fastest (HS3 "Flattening of Binned Contents"). RooFit
// writes and reads them in that order.
TEST(RooFitHS3, FlatteningOrderOfBinnedContents)
{
   Imported ws{document("0.3", {section("data", {R"({"name": "d", "type": "binned",
      "axes": [{"name": "x", "min": 0.0, "max": 2.0, "nbins": 2}, {"name": "y", "min": 0.0, "max": 3.0, "nbins": 3}],
      "contents": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]})"})})};
   auto *data = object<RooDataHist>(ws, "d");
   RooRealVar &x = *ws.var("x");
   RooRealVar &y = *ws.var("y");
   for (int ix = 0; ix < 2; ++ix) {
      for (int iy = 0; iy < 3; ++iy) {
         x.setBin(ix);
         y.setBin(iy);
         EXPECT_DOUBLE_EQ(data->weight(RooArgSet{x, y}, 0), 3.0 * ix + iy) << "bin (" << ix << ", " << iy << ")";
      }
   }

   // The export writes the same order back, and whole numbers without decimals.
   Doc doc{ws};
   EXPECT_TRUE(matches(doc, "data/d", R"({"axes": [{"name": "x"}, {"name": "y"}], "contents": [0, 1, 2, 3, 4, 5]})"));
   EXPECT_THAT(doc.json(), HasSubstr(R"("contents":[0,1,2,3,4,5])"));
}

// Categorical axes in domains create RooCategory objects, with one state per category. Binned and unbinned data can
// have categorical axes. Categories are identified by name: the data may list them in a different order than the
// domain, and the bins of binned data follow the order of the data axis.
TEST(RooFitHS3, CategoricalAxesInDomainsAndData)
{
   auto makeJSON = [](std::string const &data, std::string const &channelAxis = R"(["SR", "CR"])") {
      return document("0.3",
                      {defaultDomain({R"({"name": "channel", "categories": )" + channelAxis + "}", axes({"x"}, 0, 2)}),
                       R"("parameter_points": [{"name": "point", "parameters": [{"name": "channel", "value": "CR"}]}])",
                       section("data", {data})});
   };
   const std::string binned = R"({"name": "binned", "type": "binned",
      "axes": [{"name": "channel", "categories": ["CR", "SR"]}, {"name": "x", "min": 0.0, "max": 2.0, "nbins": 2}],
      "contents": [1.0, 2.0, 3.0, 4.0]})";
   const std::string unbinned = R"({"name": "unbinned", "type": "unbinned",
      "axes": [{"name": "channel", "categories": ["SR", "CR"]}, {"name": "x", "min": 0.0, "max": 2.0}],
      "entries": [["SR", 0.5], ["CR", 1.5], ["CR", 0.25]]})";

   Imported ws{makeJSON(binned + "," + unbinned)};

   // The domain defines the category, with the position in "categories" as index.
   auto *channel = object<RooCategory>(ws, "channel");
   EXPECT_EQ(channel->size(), 2u);
   EXPECT_EQ(channel->lookupIndex("SR"), 0);
   EXPECT_EQ(channel->lookupIndex("CR"), 1);

   // A parameter point sets the category by its label.
   const RooArgSet *point = ws.getSnapshot("point");
   ASSERT_NE(point, nullptr);
   auto *channelInPoint = dynamic_cast<RooCategory *>(point->find("channel"));
   ASSERT_NE(channelInPoint, nullptr);
   EXPECT_EQ(channelInPoint->getCurrentLabel(), std::string("CR"));

   // Binned data: the bins follow the order of the data axis ("CR" first), and are found by label.
   auto *binnedData = object<RooDataHist>(ws, "binned");
   RooRealVar &x = *ws.var("x");
   x.setBins(2);
   const std::map<std::pair<std::string, int>, double> expected{
      {{"CR", 0}, 1.0}, {{"CR", 1}, 2.0}, {{"SR", 0}, 3.0}, {{"SR", 1}, 4.0}};
   for (auto const &item : expected) {
      channel->setLabel(item.first.first);
      x.setBin(item.first.second);
      EXPECT_DOUBLE_EQ(binnedData->weight(RooArgSet{*channel, x}, 0), item.second)
         << item.first.first << ", bin " << item.first.second;
   }

   // Unbinned data: the entries of a categorical axis are the labels.
   auto *unbinnedData = object<RooAbsData>(ws, "unbinned");
   ASSERT_EQ(unbinnedData->numEntries(), 3);
   const std::vector<std::string> labels{"SR", "CR", "CR"};
   for (int i = 0; i < 3; ++i) {
      auto const &row = *unbinnedData->get(i);
      EXPECT_EQ(static_cast<RooCategory const &>(row["channel"]).getCurrentLabel(), labels[i]) << "entry " << i;
   }

   // Invalid documents.
   expectImportThrowsWithError(
      makeJSON(R"({"name": "u", "type": "unbinned", "axes": [{"name": "channel", "categories": ["SR", "CR"]}],
                   "entries": [["VR"]]})"),
      "is not a category of the axis");
   expectImportThrowsWithError(
      makeJSON(R"({"name": "u", "type": "unbinned", "axes": [{"name": "channel", "categories": ["SR", "VR"]}],
                   "entries": [["SR"]]})"),
      "differ from the states of the category");
   expectImportThrowsWithError(
      makeJSON(R"({"name": "b", "type": "binned", "axes": [{"name": "channel", "categories": ["SR", "VR"]}],
                   "contents": [1.0, 2.0]})"),
      "differ from those of the category");
   expectImportThrowsWithError(makeJSON("", R"(["SR", "SR"])"), "is listed twice");
   expectImportThrowsWithError(makeJSON("", R"([])"), "non-empty list of strings");
   expectImportThrowsWithError(
      makeJSON(R"({"name": "b", "type": "binned", "axes": [{"name": "channel", "categories": ["SR", "CR"], "nbins": 2}],
                   "contents": [1.0, 2.0]})"),
      "is categorical and must not have \"nbins\"");
}

// RooFit writes per-bin arrays in the order HS3 prescribes for the axes it writes (last axis fastest): the contents of
// binned data and of RooHistFunc ("step"), and the parameters of ParamHistFunc ("step"). Each is built with its
// variables in both orders, and with a known value (or parameter name) per bin.
TEST(RooFitHS3, ExportedBinOrderFollowsAxes)
{
   RooRealVar x{"x", "x", 0.0, 2.0};
   x.setBins(2);
   RooRealVar y{"y", "y", 0.0, 3.0};
   y.setBins(3);
   auto valueOf = [](int ix, int iy) { return 10.0 * ix + iy + 1.0; };
   auto paramName = [](int ix, int iy) { return "g_" + std::to_string(ix) + "_" + std::to_string(iy); };

   // Call `check(entry, ix, iy)` for the entry of the list `array` of the node at `path` that HS3 assigns to the bin
   // (ix, iy), given the axes of the node.
   auto forEachBin = [&](Doc const &doc, std::string const &path, const char *array, auto &&check) {
      const std::vector<std::string> axes = doc.names(path + "/axes");
      ASSERT_EQ(axes.size(), 2u);
      const bool xFirst = axes[0] == "x";
      EXPECT_EQ(axes[xFirst ? 1 : 0], "y");
      auto const &entries = doc[path + "/" + array];
      ASSERT_EQ(entries.num_children(), 6u);
      for (int ix = 0; ix < 2; ++ix) {
         for (int iy = 0; iy < 3; ++iy) {
            const std::size_t k = xFirst ? ix * 3 + iy : iy * 2 + ix;
            check(entries.child(k), ix, iy);
         }
      }
   };
   // Evaluate `name` in `ws` at the center of each bin.
   auto expectValues = [&](RooWorkspace &ws, const char *name) {
      auto *func = object<RooAbsReal>(ws, name);
      for (int ix = 0; ix < 2; ++ix) {
         for (int iy = 0; iy < 3; ++iy) {
            ws.var("x")->setBin(ix);
            ws.var("y")->setBin(iy);
            EXPECT_DOUBLE_EQ(func->getVal(), valueOf(ix, iy)) << name << " bin (" << ix << ", " << iy << ")";
         }
      }
   };

   for (bool xFirst : {true, false}) {
      SCOPED_TRACE(xFirst ? "variables (x, y)" : "variables (y, x)");
      RooArgList vars = xFirst ? RooArgList{x, y} : RooArgList{y, x};

      // Binned data and RooHistFunc.
      RooDataHist dh{"dh", "dh", vars};
      for (int ix = 0; ix < 2; ++ix) {
         for (int iy = 0; iy < 3; ++iy) {
            x.setBin(ix);
            y.setBin(iy);
            dh.set(RooArgSet{x, y}, valueOf(ix, iy));
         }
      }
      RooHistFunc histFunc{"hist_func", "hist_func", vars, dh};

      // ParamHistFunc: its parameters are ordered like the bins of a TH1, with the first variable varying fastest.
      std::vector<std::unique_ptr<RooRealVar>> params;
      RooArgList paramList;
      auto &first = static_cast<RooRealVar &>(vars[0]);
      auto &second = static_cast<RooRealVar &>(vars[1]);
      for (int i1 = 0; i1 < second.getBins(); ++i1) {
         for (int i0 = 0; i0 < first.getBins(); ++i0) {
            const int ix = xFirst ? i0 : i1;
            const int iy = xFirst ? i1 : i0;
            params.push_back(std::make_unique<RooRealVar>(paramName(ix, iy).c_str(), "", valueOf(ix, iy), 0.0, 100.0));
            paramList.add(*params.back());
         }
      }
      ParamHistFunc paramHistFunc{"param_hist_func", "param_hist_func", vars, paramList};
      // Check the assumption about ParamHistFunc that this test relies on.
      for (int ix = 0; ix < 2; ++ix) {
         for (int iy = 0; iy < 3; ++iy) {
            x.setBin(ix);
            y.setBin(iy);
            ASSERT_DOUBLE_EQ(paramHistFunc.getVal(), valueOf(ix, iy)) << "bin (" << ix << ", " << iy << ")";
         }
      }

      RooWorkspace ws;
      ws.import(dh, RooFit::Silence());
      ws.import(histFunc, RooFit::RecycleConflictNodes(), RooFit::Silence());
      ws.import(paramHistFunc, RooFit::RecycleConflictNodes(), RooFit::Silence());
      Doc doc{ws};
      forEachBin(doc, "data/dh", "contents", [&](auto const &entry, int ix, int iy) {
         EXPECT_DOUBLE_EQ(entry.val_double(), valueOf(ix, iy)) << "data bin (" << ix << ", " << iy << ")";
      });
      forEachBin(doc, "functions/hist_func/data", "contents", [&](auto const &entry, int ix, int iy) {
         EXPECT_DOUBLE_EQ(entry.val_double(), valueOf(ix, iy)) << "RooHistFunc bin (" << ix << ", " << iy << ")";
      });
      forEachBin(doc, "functions/param_hist_func", "parameters", [&](auto const &entry, int ix, int iy) {
         EXPECT_EQ(entry.val(), paramName(ix, iy)) << "ParamHistFunc bin (" << ix << ", " << iy << ")";
      });

      // Read back, the functions have the same value in every bin.
      Imported imported{doc};
      expectValues(imported, "hist_func");
      expectValues(imported, "param_hist_func");
   }
}

// The per-bin parameters of a multi-dimensional HistFactory modifier follow the order of HS3 (last axis fastest), like
// the bin contents. Each gamma gets the relative uncertainty of its own bin, and the export keeps the order.
TEST(RooFitHS3, HistFactoryMultiDimensionalBinOrder)
{
   auto gammaName = [](int ix, int iy) { return "g_" + std::to_string(ix) + "_" + std::to_string(iy); };
   auto content = [](int ix, int iy) { return 100.0 * (1 + 3 * ix + iy); };
   auto relError = [](int ix, int iy) { return 0.01 * (1 + 3 * ix + iy); };
   auto makeJSON = [&](std::string const &version) {
      std::string params;
      std::string contents;
      std::string errors;
      for (int ix = 0; ix < 2; ++ix) {
         for (int iy = 0; iy < 3; ++iy) {
            const std::string sep = params.empty() ? "" : ", ";
            params += sep + "\"" + gammaName(ix, iy) + "\"";
            contents += sep + std::to_string(content(ix, iy));
            errors += sep + std::to_string(content(ix, iy) * relError(ix, iy));
         }
      }
      const std::string data = R"({"contents": [)" + contents + R"(], "errors": [)" + errors + "]}";
      const std::string staterror =
         R"({"type": "staterror", "parameters": [)" + params + R"(], "constraint": "Gauss"})";
      const std::string xy = R"({"name": "x", "min": 0.0, "max": 2.0, "nbins": 2},
                                {"name": "y", "min": 0.0, "max": 3.0, "nbins": 3})";
      return document(version, {distributions({channel("model_channel0", {sample("sig", data, staterror)}, "", xy)})});
   };

   Imported ws{makeJSON("0.3")};
   auto *phf = object<ParamHistFunc>(ws, "mc_stat_channel0");
   for (int ix = 0; ix < 2; ++ix) {
      for (int iy = 0; iy < 3; ++iy) {
         // The constraint of each gamma has the relative uncertainty of its bin.
         auto *sigma = dynamic_cast<RooAbsReal *>(ws.arg((gammaName(ix, iy) + "_sigma").c_str()));
         ASSERT_NE(sigma, nullptr) << gammaName(ix, iy);
         EXPECT_NEAR(sigma->getVal(), relError(ix, iy), 1e-12) << gammaName(ix, iy);
         // In its bin, the ParamHistFunc takes the value of that gamma.
         ws.var(gammaName(ix, iy))->setVal(1.0 + 0.001 * (1 + 3 * ix + iy));
      }
   }
   for (int ix = 0; ix < 2; ++ix) {
      for (int iy = 0; iy < 3; ++iy) {
         ws.var("x")->setBin(ix);
         ws.var("y")->setBin(iy);
         EXPECT_DOUBLE_EQ(phf->getVal(), ws.var(gammaName(ix, iy))->getVal()) << "bin (" << ix << ", " << iy << ")";
      }
   }
   for (int ix = 0; ix < 2; ++ix) {
      for (int iy = 0; iy < 3; ++iy) {
         ws.var(gammaName(ix, iy))->setVal(1.0);
      }
   }

   // The export writes the gammas in the same order.
   Doc exported{ws};
   EXPECT_TRUE(matches(exported, "distributions/model_channel0", R"({"axes": [{"name": "x"}, {"name": "y"}]})"));
   EXPECT_TRUE(matches(exported, "distributions/model_channel0/samples/sig/modifiers/staterror",
                       R"({"parameters": ["g_0_0", "g_0_1", "g_0_2", "g_1_0", "g_1_1", "g_1_2"]})"));

   // Documents before HS3 0.3 don't define the order; they are rejected.
   expectImportThrowsWithError(makeJSON("0.2"), "Documents before HS3 0.3 don't define the order");
   expectImportThrowsWithError(
      document("0.2", {defaultDomain({axes({"x"}, 0, 2), axes({"y"}, 0, 3), axes({"a", "b"}, 0, 5)}),
                       defaultValues({values({{"a", 1.0}, {"b", 2.0}})}),
                       functions({R"({"name": "phf", "type": "step", "variables": ["x", "y"], "parameters": ["a", "b"],
                                      "axes": [{"name": "x", "min": 0.0, "max": 2.0, "nbins": 2},
                                               {"name": "y", "min": 0.0, "max": 3.0, "nbins": 1}]})"})}),
      "Documents before HS3 0.3 don't define the order");
}

// A RooSimultaneous is a "combined_dist" (HS3 #119): the components are selected by a key with one label per column,
// and the columns are declared as categorical axes of the default domain. The ROOT-internal form of older files is
// still read.
TEST(RooFitHS3, CombinedDist)
{
   RooWorkspace ws;
   ws.factory("Gaussian::model(x[-8, 8], mean[0, -8, 8], sigma[0.3, 0.1, 10])");
   // The components have their own observables: HS3 requires distinct variates of the components.
   ws.factory("Gaussian::model_ctl(x_ctl[-8, 8], mean, 1.0)");
   // The states are defined in this order, with non-contiguous indices.
   ws.factory("SIMUL::simPdf(sample[physics=3, control=7], physics=model, control=model_ctl)");

   // Structure of the written document.
   const std::string combinedDist = R"({"type": "combined_dist", "columns": ["sample"], "components": [
      {"key": ["physics"], "dist": "model"}, {"key": ["control"], "dist": "model_ctl"}]})";
   Doc doc{ws};
   EXPECT_TRUE(matches(doc, "distributions/simPdf", combinedDist));
   EXPECT_THAT(doc.json(), Not(HasSubstr("combined_distributions")));
   // The column is a categorical axis of the default domain, with the states in their order.
   EXPECT_TRUE(matches(doc, "domains/default_domain/axes/sample", R"({"categories": ["physics", "control"]})"));

   // Round trip: the same RooSimultaneous, with the index category from the domain (indices = positions).
   expectRoundTrip(ws);
   Imported ws2{doc};
   auto *simPdf = object<RooSimultaneous>(ws2, "simPdf");
   EXPECT_STREQ(simPdf->indexCat().GetName(), "sample");
   EXPECT_EQ(simPdf->indexCat().lookupIndex("physics"), 0);
   EXPECT_EQ(simPdf->indexCat().lookupIndex("control"), 1);
   EXPECT_STREQ(simPdf->getPdf("physics")->GetName(), "model");
   EXPECT_STREQ(simPdf->getPdf("control")->GetName(), "model_ctl");

   // Files written by earlier ROOT versions describe the RooSimultaneous in misc.ROOT_internal.
   Imported wsLegacy{document(
      "0.2",
      {defaultDomain({axes({"x", "x_ctl", "mean"}, -8, 8)}),
       distributions({R"({"name": "model", "type": "gaussian_dist", "x": "x", "mean": "mean", "sigma": 0.3})",
                      R"({"name": "model_ctl", "type": "gaussian_dist", "x": "x_ctl", "mean": "mean", "sigma": 1.0})"}),
       R"("misc": {"ROOT_internal": {"combined_distributions": {"simPdf": {
            "index_cat": "sample", "labels": ["physics", "control"], "indices": [3, 7],
            "distributions": ["model", "model_ctl"]}}}})"})};
   auto *legacySimPdf = object<RooSimultaneous>(wsLegacy, "simPdf");
   EXPECT_EQ(legacySimPdf->indexCat().lookupIndex("control"), 7);
   EXPECT_STREQ(legacySimPdf->getPdf("control")->GetName(), "model_ctl");
   // Its export is in the standard form.
   Doc fromLegacy{wsLegacy};
   EXPECT_TRUE(matches(fromLegacy, "distributions/simPdf", combinedDist));
   EXPECT_THAT(fromLegacy.json(), Not(HasSubstr("combined_distributions")));
}

namespace {

// A RooSimultaneous over the RooSuperCategory of (region, kind).
void buildSuperCategorySimultaneous(RooWorkspace &ws)
{
   // The components have their own observables: HS3 requires distinct variates of the components.
   RooRealVar xSR{"x_sr", "x_sr", 0.0, 10.0};
   RooRealVar xCR{"x_cr", "x_cr", 0.0, 10.0};
   RooRealVar xGlobal{"x_sr_global", "x_sr_global", 0.0, 10.0};
   RooGaussian sr{"sr", "sr", xSR, 3.0, 1.0};
   RooGaussian cr{"cr", "cr", xCR, 6.0, 1.0};
   RooGaussian srGlobal{"sr_global", "sr_global", xGlobal, 5.0, 2.0};
   // The super category is built over (region, kind); the columns are written sorted by name: (kind, region).
   RooCategory region{"region", "region", {{"SR", 0}, {"CR", 1}}};
   RooCategory kind{"kind", "kind", {{"physics", 0}, {"global", 1}}};
   RooSuperCategory index{"index", "index", RooArgSet{region, kind}};
   // Three of the four keys have a component.
   RooSimultaneous simPdf{
      "simPdf", "simPdf", {{"{SR;physics}", &sr}, {"{CR;physics}", &cr}, {"{SR;global}", &srGlobal}}, index};
   ws.import(simPdf, RooFit::Silence());
   // The round trip goes through the states of one category at a time: start from the key that the other two differ
   // from in one column.
   ws.cat("region")->setLabel("SR");
   ws.cat("kind")->setLabel("physics");
}

} // namespace

// The round trip compares each component across the range of its observable, and the RooSimultaneous for the three
// keys.
// The name of the super category is not written.
INSTANTIATE_TEST_SUITE_P(CombinedDistSeveralColumns, RoundTrip,
                         testing::Values(RoundTripModel{"SuperCategory", buildSuperCategorySimultaneous}
                                            .over({"x_sr", "x_cr", "x_sr_global"})
                                            .withOtherComponents()),
                         roundTripName);

// A RooSimultaneous over a RooSuperCategory has several columns, sorted by name, and keys with one label per column.
TEST(RooFitHS3, CombinedDistSeveralColumns)
{
   RooWorkspace ws;
   buildSuperCategorySimultaneous(ws);
   Doc doc{ws};
   EXPECT_TRUE(matches(doc, "distributions/simPdf", R"({"columns": ["kind", "region"], "components": [
      {"key": ["global", "SR"], "dist": "sr_global"}, {"key": ["physics", "CR"], "dist": "cr"},
      {"key": ["physics", "SR"], "dist": "sr"}]})"));

   // The super category is rebuilt over the columns, and every key selects its component.
   Imported ws2{doc};
   auto *simPdf2 = object<RooSimultaneous>(ws2, "simPdf");
   ASSERT_NE(dynamic_cast<RooSuperCategory const *>(&simPdf2->indexCat()), nullptr);
   for (auto const &component : doc["distributions/simPdf/components"].children()) {
      ws2.cat("kind")->setLabel(component["key"].child(0).val());
      ws2.cat("region")->setLabel(component["key"].child(1).val());
      EXPECT_EQ(simPdf2->getPdf(simPdf2->indexCat().getCurrentLabel())->GetName(), component["dist"].val());
   }
}

TEST(RooFitHS3, CombinedDistImportErrors)
{
   auto makeJSON = [](std::string const &columns, std::string const &components,
                      std::string const &sampleAxis = R"({"name": "sample", "categories": ["a", "b"]})") {
      return document(
         "0.3", {defaultDomain({axes({"x", "x_b"}, 0, 10), sampleAxis}),
                 distributions({R"({"name": "ga", "type": "gaussian_dist", "x": "x", "mean": 3.0, "sigma": 1.0})",
                                R"({"name": "gb", "type": "gaussian_dist", "x": "x_b", "mean": 6.0, "sigma": 1.0})",
                                R"({"name": "simPdf", "type": "combined_dist", "columns": )" + columns +
                                   R"(, "components": )" + components + "}"})});
   };
   const std::string good = R"([{"key": ["a"], "dist": "ga"}, {"key": ["b"], "dist": "gb"}])";
   {
      Imported ws{makeJSON(R"(["sample"])", good)};
      EXPECT_NE(dynamic_cast<RooSimultaneous *>(ws.pdf("simPdf")), nullptr);
   }
   expectImportThrowsWithError(
      makeJSON(R"(["channel"])", good),
      "the column 'channel' of the combined_dist 'simPdf' is not declared by a categorical axis");
   expectImportThrowsWithError(makeJSON(R"([])", good), "must be a non-empty list of names of categorical axes");
   expectImportThrowsWithError(makeJSON(R"(["sample", "sample"])", R"([{"key": ["a", "a"], "dist": "ga"}])"),
                               "the column 'sample' of the combined_dist 'simPdf' is listed twice");
   expectImportThrowsWithError(makeJSON(R"(["sample"])", R"([{"key": ["a", "b"], "dist": "ga"}])"),
                               "must have a \"key\" with 1 labels");
   expectImportThrowsWithError(makeJSON(R"(["sample"])", R"([{"key": ["c"], "dist": "ga"}])"),
                               "the label 'c' in position 0 of a key of the combined_dist 'simPdf' is not one of the "
                               "categories of 'sample'");
   expectImportThrowsWithError(
      makeJSON(R"(["sample"])", R"([{"key": ["a"], "dist": "ga"}, {"key": ["a"], "dist": "gb"}])"),
      "the key [\"a\"] is used by two components");
   expectImportThrowsWithError(makeJSON(R"(["sample"])", R"([])"),
                               "\"components\" of the combined_dist 'simPdf' must be a non-empty list");
   expectImportThrowsWithError(makeJSON(R"(["sample"])", R"([{"key": ["a"]}])"), "must have a \"key\" and a \"dist\"");
   // Whether a combined distribution is extended follows from its components: declaring it is not allowed, whatever
   // the value.
   expectImportThrowsWithError(makeJSON(R"(["sample"])", good + R"(, "extended": false)"),
                               "the combined_dist 'simPdf' must not have \"extended\"");
   // The column must be categorical.
   expectImportThrowsWithError(
      makeJSON(R"(["sample"])", good, R"({"name": "sample", "min": 0.0, "max": 1.0})"),
      "the column 'sample' of the combined_dist 'simPdf' is not declared by a categorical axis");
}

namespace {

// Unbinned data on one axis in [0, 10], with entries like "[1.0], [2.0]".
std::string unbinnedData(std::string const &name, std::string const &axis, std::string const &entries)
{
   return R"({"name": ")" + name + R"(", "type": "unbinned", "axes": [)" + axes({axis}, 0, 10) + R"(], "entries": [)" +
          entries + "]}";
}

} // namespace

// A data set with a category observable is a "combined" data set (HS3 #119): one sub-data set per state, selected
// by a key, with the category declared as a categorical axis of the default domain. A likelihood pairs the combined
// data set with the combined_dist. The ROOT-internal form of older files is still read.
TEST(RooFitHS3, CombinedData)
{
   RooWorkspace ws{"ws"};
   ws.factory("Gaussian::model_1(x_1[0, 10], mean[5, 0, 10], sigma_1[0.5, 0.1, 10])");
   ws.factory("Gaussian::model_2(x_2[0, 10], mean, sigma_2[0.8, 0.1, 10])");
   ws.factory("SIMUL::simPdf(channelCat[channel_1=0, channel_2=1], channel_1=model_1, channel_2=model_2)");
   std::unique_ptr<RooDataSet> d1{ws.pdf("model_1")->generate(*ws.var("x_1"), 20)};
   std::unique_ptr<RooDataSet> d2{ws.pdf("model_2")->generate(*ws.var("x_2"), 30)};
   std::map<std::string, RooDataSet *> parts{{"channel_1", d1.get()}, {"channel_2", d2.get()}};
   RooDataSet obsData{"obsData",
                      "obsData",
                      {*ws.var("x_1"), *ws.var("x_2")},
                      RooFit::Index(*ws.cat("channelCat")),
                      RooFit::Import(parts)};
   ws.import(obsData);
   addModelConfig(ws, "ModelConfig", "simPdf", "x_1,x_2,channelCat", "mean");

   // Structure: the combined entry, one sub-data set per state with only the observables of its component, the
   // domain axis, the likelihood pairing the combined objects.
   Doc doc{ws};
   EXPECT_TRUE(matches(doc, "data/obsData", R"({"type": "combined", "columns": ["channelCat"], "axes": "<absent>",
      "components": [{"key": ["channel_1"], "dataset": "obsData_channel_1"},
                     {"key": ["channel_2"], "dataset": "obsData_channel_2"}]})"));
   EXPECT_TRUE(matches(doc, "data/obsData_channel_1", R"({"type": "unbinned", "axes": [{"name": "x_1"}]})"));
   EXPECT_TRUE(matches(doc, "data/obsData_channel_2", R"({"type": "unbinned", "axes": [{"name": "x_2"}]})"));
   EXPECT_THAT(doc.json(), Not(HasSubstr("combined_datasets")));
   EXPECT_TRUE(doc.has("domains/default_domain/axes/channelCat"));
   EXPECT_TRUE(matches(doc, "likelihoods/ModelConfig_obsData_likelihood",
                       R"({"distributions": ["simPdf"], "data": ["obsData"]})"));
   EXPECT_TRUE(doc.has("analyses/ModelConfig_obsData"));

   // The data set of a workspace has the category column, and the entries of the two states.
   auto expectEntriesPerState = [](RooWorkspace &w) {
      EXPECT_EQ(w.allData().size(), 1u);
      auto *data = object<RooAbsData>(w, "obsData");
      EXPECT_EQ(data->numEntries(), 50);
      auto *channelCat = dynamic_cast<RooCategory *>(data->get()->find("channelCat"));
      ASSERT_NE(channelCat, nullptr);
      auto split = data->split(*channelCat);
      ASSERT_EQ(split.size(), 2u);
      for (auto const &part : split) {
         EXPECT_EQ(part->numEntries(), std::string(part->GetName()) == "channel_1" ? 20 : 30) << part->GetName();
      }
   };

   // Round trip: one data set with the category column, the same entries per state, the ModelConfig observes the
   // category, and the document is stable.
   Imported ws2{doc};
   expectEntriesPerState(ws2);
   auto *mc2 = object<RooFit::ModelConfig>(ws2, "ModelConfig");
   EXPECT_EQ(mc2->GetPdf(), ws2.pdf("simPdf"));
   EXPECT_NE(mc2->GetObservables()->find("channelCat"), nullptr);
   EXPECT_EQ(toJSON(ws2), doc.json());

   // Without a RooSimultaneous over the category, the category is an ordinary observable: the data set is written
   // with a categorical axis (HS3 combined data sets require a combined distribution to pair with), and read back
   // with the same entries per state.
   {
      RooWorkspace wsDataOnly;
      wsDataOnly.import(obsData);
      Doc dataOnly{wsDataOnly};
      EXPECT_FALSE(dataOnly.has("data/obsData_channel_1"));
      EXPECT_TRUE(matches(dataOnly, "data/obsData", R"({"type": "unbinned"})"));
      EXPECT_TRUE(matches(dataOnly, "data/obsData/axes/channelCat", R"({"categories": ["channel_1", "channel_2"]})"));

      Imported wsDataOnly2{dataOnly};
      expectEntriesPerState(wsDataOnly2);
      EXPECT_EQ(toJSON(wsDataOnly2), dataOnly.json());
   }

   // Files written by earlier ROOT versions: components named after the data set and the state, the combination
   // described in misc.ROOT_internal, and the likelihood listing the components.
   Imported wsLegacy{document(
      "0.2",
      {defaultDomain({axes({"x_1", "x_2", "mean"}, 0, 10)}),
       distributions({R"({"name": "model_1", "type": "gaussian_dist", "x": "x_1", "mean": "mean", "sigma": 0.5})",
                      R"({"name": "model_2", "type": "gaussian_dist", "x": "x_2", "mean": "mean", "sigma": 0.8})"}),
       section("data", {unbinnedData("obsData_channel_1", "x_1", "[1.0], [2.0]"),
                        unbinnedData("obsData_channel_2", "x_2", "[3.0]")}),
       R"("likelihoods": [{"name": "simPdf_obsData_likelihood", "distributions": ["model_1", "model_2"],
                           "data": ["obsData_channel_1", "obsData_channel_2"]}],
          "analyses": [{"name": "simPdf_obsData", "likelihood": "simPdf_obsData_likelihood",
                        "parameters_of_interest": ["mean"]}],
          "misc": {"ROOT_internal": {
             "combined_datasets": {"obsData": {"index_cat": "channelCat", "labels": ["channel_1", "channel_2"],
                                               "indices": [0, 1]}},
             "combined_distributions": {"simPdf": {"index_cat": "channelCat", "labels": ["channel_1", "channel_2"],
                                                   "indices": [0, 1], "distributions": ["model_1", "model_2"]}},
             "ModelConfigs": {"simPdf_obsData": {"mcName": "ModelConfig", "pdfName": "simPdf"}}}})"})};
   EXPECT_EQ(wsLegacy.allData().size(), 1u);
   ASSERT_NE(wsLegacy.data("obsData"), nullptr);
   EXPECT_EQ(wsLegacy.data("obsData")->numEntries(), 3);
   auto *mcLegacy = object<RooFit::ModelConfig>(wsLegacy, "ModelConfig");
   EXPECT_EQ(mcLegacy->GetPdf(), wsLegacy.pdf("simPdf"));
   Doc fromLegacy{wsLegacy};
   EXPECT_TRUE(matches(fromLegacy, "data/obsData", R"({"type": "combined"})"));
   EXPECT_THAT(fromLegacy.json(), Not(HasSubstr("combined_datasets")));
}

// A data set with several category observables has several columns, sorted by name. Keys without entries are
// written if the matching RooSimultaneous has a component for them, and left out otherwise.
TEST(RooFitHS3, CombinedDataSeveralColumns)
{
   RooWorkspace ws;
   buildSuperCategorySimultaneous(ws);
   // Entries per key; none for (physics, CR), and (global, CR) has no component.
   const std::map<std::vector<std::string>, int> entries{{{"physics", "SR"}, 3}, {{"global", "SR"}, 1}};
   RooArgSet columns{*ws.var("x_sr"), *ws.var("x_cr"), *ws.var("x_sr_global"), *ws.cat("region"), *ws.cat("kind")};
   RooDataSet obs{"obs", "obs", columns};
   for (auto const &item : entries) {
      ws.cat("kind")->setLabel(item.first[0].c_str());
      ws.cat("region")->setLabel(item.first[1].c_str());
      RooRealVar &x = *ws.var(item.first[0] == "global" ? "x_sr_global" : "x_sr");
      for (int i = 0; i < item.second; ++i) {
         x.setVal(1.0 + i);
         obs.add(columns);
      }
   }
   ws.import(obs);
   Doc doc{ws};
   EXPECT_TRUE(matches(doc, "data/obs", R"({"columns": ["kind", "region"], "components": [
      {"key": ["global", "SR"], "dataset": "obs_global_SR"}, {"key": ["physics", "CR"], "dataset": "obs_physics_CR"},
      {"key": ["physics", "SR"], "dataset": "obs_physics_SR"}]})"));
   EXPECT_TRUE(matches(doc, "data/obs_physics_CR", R"({"entries": []})"));

   // Round trip: one data set with both category columns and the same entries per key; stable document.
   Imported ws2{doc};
   EXPECT_EQ(ws2.allData().size(), 1u);
   auto *obs2 = object<RooAbsData>(ws2, "obs");
   EXPECT_EQ(toJSON(ws2), doc.json());
   std::map<std::vector<std::string>, int> entries2;
   for (int i = 0; i < obs2->numEntries(); ++i) {
      RooArgSet const &row = *obs2->get(i);
      ++entries2[{static_cast<RooCategory const &>(row["kind"]).getCurrentLabel(),
                  static_cast<RooCategory const &>(row["region"]).getCurrentLabel()}];
   }
   EXPECT_EQ(entries2, entries);
   // The RooSimultaneous splits it by its RooSuperCategory, which is rebuilt over the columns.
   auto parts = obs2->split(*dynamic_cast<RooSimultaneous *>(ws2.pdf("simPdf")), /*createEmptyDataSets=*/true);
   std::map<std::string, int> partEntries;
   for (auto const &part : parts) {
      partEntries[part->GetName()] = part->numEntries();
   }
   const std::map<std::string, int> expectedParts{
      {"{physics;SR}", 3}, {"{physics;CR}", 0}, {"{global;SR}", 1}, {"{global;CR}", 0}};
   EXPECT_EQ(partEntries, expectedParts);
}

TEST(RooFitHS3, CombinedDataImportErrors)
{
   // The combined data set "obs" with the given keys, and the data sets "da" and "db".
   auto makeJSON = [](std::string const &combined) {
      return document("0.3",
                      {defaultDomain({axes({"x", "x_b"}, 0, 10), R"({"name": "sample", "categories": ["a", "b"]})"}),
                       section("data", {unbinnedData("da", "x", "[1.0]"), unbinnedData("db", "x_b", "[2.0], [3.0]"),
                                        R"({"name": "obs", "type": "combined", )" + combined + "}"})});
   };
   const std::string good =
      R"("columns": ["sample"], "components": [{"key": ["a"], "dataset": "da"}, {"key": ["b"], "dataset": "db"}])";
   {
      Imported ws{makeJSON(good)};
      EXPECT_EQ(ws.allData().size(), 1u);
      ASSERT_NE(ws.data("obs"), nullptr);
      EXPECT_EQ(ws.data("obs")->numEntries(), 3);
   }
   expectImportThrowsWithError(makeJSON(R"("axes": [{"name": "x", "min": 0.0, "max": 10.0}], )" + good),
                               "the combined data set 'obs' must not have \"axes\"");
   expectImportThrowsWithError(
      makeJSON(R"("columns": ["channel"], "components": [{"key": ["a"], "dataset": "da"}])"),
      "the column 'channel' of the combined data set 'obs' is not declared by a categorical axis");
   expectImportThrowsWithError(makeJSON(R"("columns": ["sample"], "components": [])"),
                               "\"components\" of the combined data set 'obs' must be a non-empty list");
   expectImportThrowsWithError(makeJSON(R"("columns": ["sample"], "components": [{"key": ["a"]}])"),
                               "must have a \"key\" and a \"dataset\"");
   expectImportThrowsWithError(makeJSON(R"("columns": ["sample"], "components": [{"key": ["c"], "dataset": "da"}])"),
                               "the label 'c' in position 0 of a key of the combined data set 'obs'");
   expectImportThrowsWithError(makeJSON(R"("columns": ["sample"], "components": [{"key": ["a"], "dataset": "dc"}])"),
                               "the component 'dc' of the combined data set 'obs' is not a data set of the document");
   expectImportThrowsWithError(
      makeJSON(
         R"("columns": ["sample"], "components": [{"key": ["a"], "dataset": "da"}, {"key": ["b"], "dataset": "da"}])"),
      "the component 'da' of the combined data set 'obs' is not a data set of the document, or it is already a "
      "component");
   expectImportThrowsWithError(
      makeJSON(
         R"("columns": ["sample"], "components": [{"key": ["a"], "dataset": "da"}, {"key": ["a"], "dataset": "db"}])"),
      "the key [\"a\"] is used by two components of the combined data set 'obs'");
}

// A likelihood that pairs a combined_dist with a combined data set: HS3 treats a key without sub-data set as an
// unobserved component, which RooFit cannot represent (it would be a channel without entries, adding the expected
// events of an extended component to the likelihood). This is an error for extended components and a warning
// otherwise; a sub-data set without component is ignored with a warning.
TEST(RooFitHS3, CombinedDataPairing)
{
   // A combined_dist with a component for each of `pdfLabels`, extended or not, paired with a combined data set that
   // has a sub-data set for each of `dataLabels`.
   auto makeJSON = [](bool extended, std::vector<std::string> const &pdfLabels,
                      std::vector<std::string> const &dataLabels) {
      // The components <prefix><label> of the distribution ("dist") or of the data ("dataset").
      auto components = [](std::string const &what, std::string const &prefix, std::vector<std::string> const &labels) {
         std::vector<std::string> out;
         for (auto const &label : labels) {
            out.push_back(R"({"key": [")" + label + R"("], ")" + what + R"(": ")" + prefix + label + "\"}");
         }
         return "[" + join(out) + "]";
      };
      std::vector<std::string> dists;
      std::vector<std::string> data;
      for (auto const &[label, mean] :
           std::vector<std::pair<std::string, std::string>>{{"a", "3.0"}, {"b", "4.0"}, {"c", "5.0"}}) {
         dists.push_back(R"({"name": "g_)" + label + R"(", "type": "gaussian_dist", "x": "x_)" + label +
                         R"(", "mean": )" + mean + R"(, "sigma": 1.0})");
         if (extended) {
            dists.push_back(R"({"name": "e_)" + label + R"(", "type": "rate_extended_dist", "dist": "g_)" + label +
                            R"(", "rate": "n_)" + label + "\"}");
         }
         data.push_back(unbinnedData("d_" + label, "x_" + label, "[5.0]"));
      }
      dists.push_back(R"({"name": "simPdf", "type": "combined_dist", "columns": ["sample"], "components": )" +
                      components("dist", extended ? "e_" : "g_", pdfLabels) + "}");
      data.push_back(R"({"name": "obs", "type": "combined", "columns": ["sample"], "components": )" +
                     components("dataset", "d_", dataLabels) + "}");
      return document("0.3", {defaultDomain({axes({"x_a", "x_b", "x_c"}, 0, 10), axes({"n_a", "n_b", "n_c"}, 0, 100),
                                             R"({"name": "sample", "categories": ["a", "b", "c"]})"}),
                              distributions(dists), section("data", data),
                              R"("likelihoods": [{"name": "nll", "distributions": ["simPdf"], "data": ["obs"]}])",
                              R"("analyses": [{"name": "fit", "likelihood": "nll", "domains": ["default_domain"]}])"});
   };

   // Every component has a sub-data set: no messages.
   {
      Imported ws{makeJSON(true, {"a", "b", "c"}, {"a", "b", "c"})};
      EXPECT_EQ(ws.warnings, "");
      auto *mc = object<RooFit::ModelConfig>(ws, "fit");
      EXPECT_EQ(mc->GetPdf(), ws.pdf("simPdf"));
   }
   // An extended component without sub-data set is an error.
   expectImportThrowsWithError(makeJSON(true, {"a", "b", "c"}, {"a", "b"}),
                               "the combined data set 'obs' has no sub-data set for the key [\"c\"] of the extended "
                               "component 'e_c' of 'simPdf'");
   // A non-extended component without sub-data set is a warning, and so is a sub-data set without component.
   EXPECT_THAT(Imported{makeJSON(false, {"a", "b", "c"}, {"a", "b"})}.warnings,
               HasSubstr("the combined data set 'obs' has no sub-data set for the key [\"c\"] of the component "
                         "'g_c' of 'simPdf', which RooFit treats as present and empty"));
   EXPECT_THAT(Imported{makeJSON(true, {"a", "b"}, {"a", "b", "c"})}.warnings,
               HasSubstr("the sub-data set for the key [\"c\"] of the combined data set 'obs' is ignored, "
                         "because 'simPdf' has no component for it"));
}

// The names of a RooSimultaneous and of a combined data set are part of the standard document now, not of
// misc.ROOT_internal: a document without "misc" imports with the same names, and the ModelConfig finds its pdf
// through the likelihood.
TEST(RooFitHS3, CombinedObjectsKeepNamesWithoutMisc)
{
   RooWorkspace ws{"ws"};
   ws.factory("Gaussian::model_1(x_1[0, 10], mean[5, 0, 10], sigma_1[0.5, 0.1, 10])");
   ws.factory("Gaussian::model_2(x_2[0, 10], mean, sigma_2[0.8, 0.1, 10])");
   // Not named "simPdf", which older documents relied on as the default name.
   ws.factory("SIMUL::channels(region[SR=0, CR=1], SR=model_1, CR=model_2)");
   std::unique_ptr<RooDataSet> d1{ws.pdf("model_1")->generate(*ws.var("x_1"), 20)};
   std::unique_ptr<RooDataSet> d2{ws.pdf("model_2")->generate(*ws.var("x_2"), 30)};
   std::map<std::string, RooDataSet *> parts{{"SR", d1.get()}, {"CR", d2.get()}};
   RooDataSet observed{"observed",
                       "observed",
                       {*ws.var("x_1"), *ws.var("x_2")},
                       RooFit::Index(*ws.cat("region")),
                       RooFit::Import(parts)};
   ws.import(observed);
   addModelConfig(ws, "ModelConfig", "channels", "x_1,x_2,region", "mean");

   auto tree = RooFit::Detail::JSONTree::create(toJSON(ws));
   ASSERT_TRUE(tree->rootnode().has_child("misc"));
   tree->rootnode().erase("misc");
   const std::string withoutMisc = jsonString(*tree);
   EXPECT_THAT(withoutMisc, Not(HasSubstr("\"misc\"")));

   Imported ws2{withoutMisc};
   auto *channels = object<RooSimultaneous>(ws2, "channels");
   EXPECT_STREQ(channels->indexCat().GetName(), "region");
   EXPECT_STREQ(channels->getPdf("SR")->GetName(), "model_1");
   EXPECT_STREQ(channels->getPdf("CR")->GetName(), "model_2");
   EXPECT_EQ(ws2.allData().size(), 1u);
   auto *observed2 = object<RooAbsData>(ws2, "observed");
   EXPECT_EQ(observed2->numEntries(), 50);
   EXPECT_NE(observed2->get()->find("region"), nullptr);
   // Without misc.ROOT_internal.ModelConfigs, the ModelConfig is named after the analysis.
   auto *mc2 = object<RooFit::ModelConfig>(ws2, "ModelConfig_observed");
   EXPECT_EQ(mc2->GetPdf(), channels);
   EXPECT_NE(mc2->GetObservables()->find("region"), nullptr);
   EXPECT_NE(mc2->GetParametersOfInterest()->find("mean"), nullptr);

   // The analysis is named after the ModelConfig, which got the name of the analysis: the names of the analysis and
   // the likelihood change once on the re-export. From then on, the document is stable.
   const std::string json2 = toJSON(ws2);
   EXPECT_EQ(reexport(json2), json2);
}

namespace {

// The domain and the data of a document that is not valid HS3: the sub-data sets "obs_SR" and "obs_CR" of the
// combined data set "obs" share the axis "mass".
std::string combinedDataWithSharedAxis()
{
   return join({defaultDomain({R"({"name": "channel", "categories": ["SR", "CR"]})", axes({"mass"}, 0, 10)}),
                section("data", {unbinnedData("obs_SR", "mass", "[4.0]"), unbinnedData("obs_CR", "mass", "[5.0]"),
                                 R"({"name": "obs", "type": "combined", "columns": ["channel"],
                                     "components": [{"key": ["SR"], "dataset": "obs_SR"},
                                                    {"key": ["CR"], "dataset": "obs_CR"}]})"})});
}

} // namespace

// In HS3, the components of a combined distribution share no variate: a quantity measured in several components is a
// separate variate in each of them ("mass_SR", "mass_CR"). A RooSimultaneous whose components share an observable can
// therefore not be written, unless explicitly allowed: then the document is written with a warning. A document whose
// sub-data sets share an axis is rejected, or read with a warning if it declares a version before 0.3.
TEST(RooFitHS3, CombinedComponentsMustNotShareObservables)
{
   RooRealVar mass{"mass", "mass", 0.0, 10.0};
   RooCategory channel{"channel", "channel", {{"SR", 0}, {"CR", 1}}};
   RooGaussian sr{"sr", "sr", mass, 3.0, 1.0};
   RooGaussian cr{"cr", "cr", mass, 6.0, 1.0};
   RooSimultaneous simPdf{"simPdf", "simPdf", {{"SR", &sr}, {"CR", &cr}}, channel};
   RooDataSet obs{"obs", "obs", {mass, channel}};
   for (const char *label : {"SR", "CR"}) {
      channel.setLabel(label);
      mass.setVal(4.0);
      obs.add({mass, channel});
   }
   RooWorkspace ws;
   ws.import(simPdf, RooFit::Silence());
   ws.import(obs);
   const std::string errors =
      captureMessages(RooFit::ERROR, RooFit::IO, [&] { EXPECT_THROW(toJSON(ws), std::runtime_error); });
   EXPECT_THAT(errors,
               HasSubstr("the variate 'mass' is shared by the components 'cr' and 'sr' of the combined_dist 'simPdf'"));
   EXPECT_THAT(errors, HasSubstr("'mass_SR' and 'mass_CR'"));

   // Written anyway if allowed, with a warning; the document is rejected on import.
   Doc invalid{ws, {allowExportSharedVariates(true)}};
   EXPECT_THAT(invalid.warnings(), HasSubstr("the variate 'mass' is shared by the components 'cr' and 'sr'"));
   EXPECT_THAT(invalid.warnings(), HasSubstr("not valid HS3"));
   expectImportThrowsWithError(invalid, "the variate 'mass' is shared by the components 'cr' and 'sr'");

   const std::string sharedAxis =
      "the variate 'mass' is shared by the components 'obs_SR' and 'obs_CR' of the combined data set 'obs'";
   expectImportThrowsWithError(document("0.3", {combinedDataWithSharedAxis()}), sharedAxis);
   Imported imported{document("0.2", {combinedDataWithSharedAxis()})};
   EXPECT_THAT(imported.warnings, HasSubstr(sharedAxis));
   auto *data = object<RooAbsData>(imported, "obs");
   EXPECT_EQ(data->numEntries(), 2);
}

// A combined data set may have a column that no distribution models, e.g. an event number. RooAbsData::split() keeps
// such a column in every sub-data set, so all components of the combined data set have it as axis, which HS3 doesn't
// allow. If its variable is used by nothing else in the workspace, it is declared as a bookkeeping column in
// misc.ROOT_internal.bookkeeping_columns, whether or not a likelihood lists the data set, and the document is read with
// a warning. A column in a set of the ModelConfig is no bookkeeping column, and neither is a declared column that is a
// variate of a distribution.
TEST(RooFitHS3, CombinedDataColumnWithoutDistribution)
{
   RooRealVar xSR{"x_SR", "x_SR", 0.0, 10.0};
   RooRealVar xCR{"x_CR", "x_CR", 0.0, 10.0};
   RooRealVar eventNumber{"eventNumber", "eventNumber", 0.0, 1e6};
   RooGaussian sr{"sr", "sr", xSR, 5.0, 1.0};
   RooGaussian cr{"cr", "cr", xCR, 4.0, 2.0};
   RooCategory channel{"channel", "channel", {{"SR", 0}, {"CR", 1}}};
   RooSimultaneous simPdf{"simPdf", "simPdf", {{"SR", &sr}, {"CR", &cr}}, channel};
   RooDataSet obsData{"obsData", "obsData", {xSR, xCR, eventNumber, channel}};
   int event = 0;
   for (auto const &[label, value] :
        std::vector<std::pair<const char *, double>>{{"SR", 5.5}, {"CR", 3.0}, {"CR", 6.0}}) {
      channel.setLabel(label);
      xSR.setVal(value);
      xCR.setVal(value);
      eventNumber.setVal(event++);
      obsData.add({xSR, xCR, eventNumber, channel});
   }

   for (bool withLikelihood : {false, true}) {
      SCOPED_TRACE(withLikelihood ? "with likelihood" : "without likelihood");
      RooWorkspace ws;
      ws.import(simPdf, RooFit::Silence());
      ws.import(obsData, RooFit::Silence());
      if (withLikelihood) {
         RooStats::ModelConfig mc{"ModelConfig", &ws};
         mc.SetPdf("simPdf");
         mc.SetObservables(RooArgSet{xSR, xCR, channel});
         ws.import(mc);
      }
      Doc doc{ws};
      EXPECT_THAT(doc.warnings(), Not(HasSubstr("is shared by")));
      EXPECT_EQ(doc.json().find(R"("likelihoods")") != std::string::npos, withLikelihood);
      EXPECT_TRUE(doc.has("data/obsData_SR/axes/eventNumber"));
      EXPECT_TRUE(doc.has("data/obsData_CR/axes/eventNumber"));
      EXPECT_EQ(doc.dump("misc/ROOT_internal/bookkeeping_columns"), R"({"obsData":["eventNumber"]})");

      Imported imported{doc};
      EXPECT_THAT(imported.warnings,
                  HasSubstr("the axis 'eventNumber' of the components of the combined data set 'obsData' is "
                            "declared as a bookkeeping column"));
      auto *data = object<RooAbsData>(imported, "obsData");
      EXPECT_EQ(data->numEntries(), 3);
      EXPECT_NE(data->get()->find("eventNumber"), nullptr);
      // Stable.
      EXPECT_EQ(Doc{imported}.json(), doc.json());

      // Without the declaration, the document is not valid HS3.
      expectImportThrowsWithError(replaced(doc, "bookkeeping_columns", "other_columns"),
                                  "the variate 'eventNumber' is shared by the components");
   }

   // A column in the observables of the ModelConfig is no bookkeeping column.
   {
      RooWorkspace ws;
      ws.import(simPdf, RooFit::Silence());
      ws.import(obsData, RooFit::Silence());
      RooStats::ModelConfig mc{"ModelConfig", &ws};
      mc.SetPdf("simPdf");
      mc.SetObservables(RooArgSet{xSR, xCR, eventNumber, channel});
      ws.import(mc);
      const std::string errors =
         captureMessages(RooFit::ERROR, RooFit::IO, [&] { EXPECT_THROW(toJSON(ws), std::runtime_error); });
      EXPECT_THAT(errors, HasSubstr("the variate 'eventNumber' is shared by the components"));
   }

   // A declared column that is a variate of a distribution is no bookkeeping column.
   expectImportThrowsWithError(
      document("0.3",
               {distributions({R"({"name": "g", "type": "gaussian_dist", "x": "mass", "mean": 1.0, "sigma": 1.0})"}),
                combinedDataWithSharedAxis(),
                R"("misc": {"ROOT_internal": {"bookkeeping_columns": {"obs": ["mass"]}}})"}),
      "the variate 'mass' is shared by the components 'obs_SR' and 'obs_CR'");
}

// With RooJSONFactoryWSTool::config().renameCombinedObservables, the observables of combined data sets are renamed
// per key on export ("obs_SR", or "obs_SR_x" with several observables), and the components of the matching
// RooSimultaneous are written with them; nodes that depend on a renamed observable get one copy per component. This
// allows exporting a RooSimultaneous whose components share an observable. The likelihood stays the same, and a
// second export gives the same document.
TEST(RooFitHS3, RenameCombinedObservables)
{
   ScopedConfig config{{renameCombinedObservables(true)}};

   RooRealVar mass{"mass", "mass", 0.0, 10.0};
   RooCategory channel{"channel", "channel", {{"SR", 0}, {"CR", 1}}};
   // A function of the observable, shared by both components.
   RooFormulaVar shift{"shift", "mass - 1.0", {mass}};
   RooGaussian sr{"sr", "sr", shift, 3.0, 1.0};
   RooGaussian cr{"cr", "cr", shift, 6.0, 1.5};
   RooSimultaneous simPdf{"simPdf", "simPdf", {{"SR", &sr}, {"CR", &cr}}, channel};
   RooDataSet obs{"data", "data", {mass, channel}};
   for (auto const &[label, value] :
        std::vector<std::pair<const char *, double>>{{"SR", 3.5}, {"SR", 4.5}, {"CR", 6.0}, {"CR", 7.5}, {"CR", 8.0}}) {
      channel.setLabel(label);
      mass.setVal(value);
      obs.add({mass, channel});
   }
   std::unique_ptr<RooAbsReal> nll{simPdf.createNLL(obs)};
   const double expectedNll = nll->getVal();

   RooWorkspace ws;
   ws.import(simPdf, RooFit::Silence());
   ws.import(obs);
   // The components and the functions of the observable, one per key, with the renamed observables.
   Doc doc{ws};
   EXPECT_TRUE(matches(doc, "distributions/simPdf/components",
                       R"([{"key": ["CR"], "dist": "cr_CR"}, {"key": ["SR"], "dist": "sr_SR"}])"));
   for (std::string key : {"SR", "CR"}) {
      EXPECT_THAT(doc["functions/shift_" + key + "/expression"].val(), HasSubstr("obs_" + key));
      EXPECT_TRUE(matches(doc, "data/data_" + key + "/axes", R"([{"name": "obs_)" + key + R"("}])"));
   }
   EXPECT_FALSE(doc.has("functions/shift"));

   // Read back, the likelihood is the same.
   Imported imported{doc};
   auto *simPdf2 = object<RooSimultaneous>(imported, "simPdf");
   auto *obs2 = object<RooAbsData>(imported, "data");
   std::unique_ptr<RooAbsReal> nll2{simPdf2->createNLL(*obs2)};
   EXPECT_NEAR(nll2->getVal(), expectedNll, 1e-10);

   // The observables already have their new names, so a second export gives the same document.
   EXPECT_EQ(toJSON(imported), doc.json());

   // Several observables per key: "obs_<key>_<observable>". Without a RooSimultaneous, all observables of the data
   // set are renamed.
   RooRealVar x{"x", "x", 0.0, 1.0};
   RooRealVar y{"y", "y", 0.0, 1.0};
   RooDataSet xy{"xy", "xy", {x, y, channel}};
   for (const char *label : {"SR", "CR"}) {
      channel.setLabel(label);
      xy.add({x, y, channel});
   }
   RooWorkspace wsXY;
   wsXY.import(xy);
   EXPECT_THAT(Doc{wsXY}.names("data/xy_SR/axes"), testing::UnorderedElementsAre("obs_SR_x", "obs_SR_y"));
}

// Histograms are written with axes named after the observables of the RooHistFunc or RooHistPdf, which are mapped
// onto the variables of the histogram and need not have the same names. This matters when the observables are
// renamed per key (reported in the review): the "histfactory_dist" of a channel and a "histogram_dist" have the
// renamed observables as axes, like the data, and the renamed observables are not written as parameters.
TEST(RooFitHS3, HistogramAxesNamedAfterObservables)
{
   // A RooHistFunc with observables of other names than the variables of its histogram is written with the names of
   // its observables.
   {
      RooRealVar u{"u", "u", 0.0, 2.0};
      u.setBins(2);
      RooRealVar v{"v", "v", 0.0, 2.0};
      v.setBins(2);
      std::unique_ptr<RooDataHist> dh = makeDataHist("dh_v", v, {1.0, 3.0});
      RooHistFunc func{"func", "func", RooArgList{u}, RooArgList{v}, *dh};
      RooWorkspace ws;
      ws.import(func, RooFit::Silence());
      EXPECT_TRUE(matches(Doc{ws}, "functions/func/data/axes", R"([{"name": "u"}])"));
   }

   ScopedConfig config{{renameCombinedObservables(true)}};

   // A HistFactory channel (SR) and a RooHistPdf (CR) on the same observable.
   RooRealVar x{"x", "x", 0.0, 2.0};
   x.setBins(2);
   std::unique_ptr<RooDataHist> template_ = makeDataHist("template", x, {10.0, 20.0});
   RooHistFunc shape{"shape", "shape", RooArgSet{x}, *template_};
   RooBinWidthFunction width{"width", "width", shape, true};
   RooProduct sample{"sample", "sample", {shape, width}};
   RooRealVar mu{"mu", "mu", 1.0, 0.1, 3.0};
   RooRealSumPdf model{"model_SR", "model_SR", {sample}, {mu}, true};
   std::unique_ptr<RooDataHist> crHist = makeDataHist("crHist", x, {5.0, 1.0});
   RooHistPdf cr{"cr", "cr", RooArgSet{x}, *crHist};
   RooCategory channel{"channel", "channel", {{"SR", 0}, {"CR", 1}}};
   RooSimultaneous simPdf{"simPdf", "simPdf", {{"SR", &model}, {"CR", &cr}}, channel};
   // Binned data in the usual HistFactory form, one weighted entry per bin.
   RooRealVar weight{"weight", "weight", 1.0};
   RooDataSet obsData{"obsData", "obsData", {x, channel, weight}, RooFit::WeightVar(weight)};
   for (auto const &[label, value, w] :
        std::vector<std::tuple<const char *, double, double>>{{"SR", 0.5, 1.0}, {"SR", 1.5, 2.0}, {"CR", 0.2, 1.0}}) {
      channel.setLabel(label);
      x.setVal(value);
      obsData.add({x, channel}, w);
   }

   RooWorkspace ws;
   ws.import(simPdf, RooFit::Silence());
   ws.import(obsData, RooFit::Silence());
   // The channel and the RooHistPdf are the only distributions next to the combined one.
   Doc doc{ws};
   EXPECT_THAT(doc.names("distributions"), testing::UnorderedElementsAre("simPdf", "model_SR_SR", "cr_CR"));
   EXPECT_TRUE(
      matches(doc, "distributions/model_SR_SR", R"({"type": "histfactory_dist", "axes": [{"name": "obs_SR"}]})"));
   EXPECT_TRUE(
      matches(doc, "distributions/cr_CR", R"({"type": "histogram_dist", "data": {"axes": [{"name": "obs_CR"}]}})"));
   EXPECT_TRUE(matches(doc, "data/obsData_SR/axes", R"([{"name": "obs_SR"}])"));
   EXPECT_TRUE(matches(doc, "data/obsData_CR/axes", R"([{"name": "obs_CR"}])"));
   // The original observable is gone from the document.
   EXPECT_EQ(doc["domains"].num_children(), 1u);
   EXPECT_EQ(doc.names("domains/default_domain/axes"), (std::vector<std::string>{"channel", "mu", "obs_CR", "obs_SR"}));
   EXPECT_THAT(doc.dump("parameter_points"), Not(HasSubstr(R"("name":"x")")));

   // Read back, the likelihood changes the same way with the parameter, and the document is stable from the second
   // export on (the HistFactory importer adds its attributes and step widths).
   Imported imported{doc};
   auto nllDifference = [](RooAbsPdf &pdf, RooAbsData &data, RooRealVar &param) {
      std::unique_ptr<RooAbsReal> nll{pdf.createNLL(data)};
      const double nominal = nll->getVal();
      const double old = param.getVal();
      param.setVal(1.5);
      const double difference = nll->getVal() - nominal;
      param.setVal(old);
      return difference;
   };
   ASSERT_NE(imported.pdf("simPdf"), nullptr);
   ASSERT_NE(imported.data("obsData"), nullptr);
   ASSERT_NE(imported.var("mu"), nullptr);
   const double expected = nllDifference(simPdf, obsData, mu);
   EXPECT_NE(expected, 0.0);
   EXPECT_NEAR(nllDifference(*imported.pdf("simPdf"), *imported.data("obsData"), *imported.var("mu")), expected, 1e-8);
   const std::string json2 = toJSON(imported);
   EXPECT_EQ(reexport(json2), json2);
}

// A weighted RooDataSet with one entry per bin, in the order of the bins, is written as binned data. With further
// entries it is unbinned data: a second entry in the last bin was mistaken for binned data, and a value after the
// last bin content was written (reported in the review).
TEST(RooFitHS3, WeightedDataSetAsBinnedData)
{
   auto exportOne = [](std::vector<std::pair<double, double>> const &entries) {
      RooRealVar x{"x", "x", 0.0, 2.0};
      x.setBins(2);
      RooRealVar weight{"weight", "weight", 1.0};
      RooDataSet data{"data", "data", {x, weight}, RooFit::WeightVar(weight)};
      for (auto const &[value, w] : entries) {
         x.setVal(value);
         data.add(x, w);
      }
      RooWorkspace ws;
      ws.import(data, RooFit::Silence());
      return Doc{ws};
   };

   const Doc binned = exportOne({{0.5, 1.0}, {1.5, 2.0}});
   EXPECT_TRUE(matches(binned, "data/0", R"({"type": "binned", "contents": [1, 2]})"));
   EXPECT_THAT(binned.json(), HasSubstr(R"("contents":[1,2])"));

   // Entries in the bins 0, 1, 1.
   const Doc unbinned = exportOne({{0.5, 1.0}, {1.5, 1.0}, {1.7, 1.0}});
   EXPECT_TRUE(matches(unbinned, "data/0", R"({"type": "unbinned", "entries": [[0.5], [1.5], [1.7]]})"));
   EXPECT_THAT(unbinned.json(), HasSubstr(R"("entries":[[0.5],[1.5],[1.7]])"));
}

// HS3 requires the terms of a likelihood to have distinct variates, like the components of a combined distribution.
// Two channels with a common observable can't be exported, unless explicitly allowed: then the document is written
// with a warning, and it is rejected on import.
TEST(RooFitHS3, SharedVariatesInLikelihood)
{
   RooWorkspace ws;
   ws.factory("Gaussian::pdf_a(x[0, -5, 5], mu[0, -5, 5], sigma[1, 0.1, 5])");
   ws.factory("Gaussian::pdf_b(x, mu, sigma)");
   ws.factory("SIMUL::sim_pdf(channel[a=0, b=1], a=pdf_a, b=pdf_b)");
   RooArgSet observables{*ws.var("x"), *ws.cat("channel")};
   RooDataSet data{"data", "data", observables};
   for (int index : {0, 1}) {
      ws.cat("channel")->setIndex(index);
      data.add(observables);
   }
   ws.import(data, RooFit::Silence());
   addModelConfig(ws, "ModelConfig", "sim_pdf", "x,channel", "mu");

   const std::string message = "the variate 'x' is shared by the terms 'pdf_a' and 'pdf_b' of the likelihood";
   expectExportThrowsWithError(ws, message);

   Doc exported{ws, {allowExportSharedVariates(true)}};
   EXPECT_THAT(exported.warnings(), HasSubstr(message));
   EXPECT_THAT(exported.warnings(), HasSubstr("not valid HS3"));
   EXPECT_FALSE(exported.json().empty());

   expectImportThrowsWithError(exported, message);
}

namespace {

void buildConstraintInModelAndExternal(RooWorkspace &ws)
{
   ws.factory("Gaussian::shape(x[0, -5, 5], theta[0, -5, 5], 1.0)");
   ws.factory("Gaussian::constraint(theta_obs[0, -5, 5], theta, 1.0)");
   ws.factory("PROD::model(shape, constraint)");
   ws.var("theta_obs")->setConstant(true);
   RooDataSet data{"data", "data", *ws.var("x")};
   ws.var("x")->setVal(0.7);
   data.add(*ws.var("x"));
   ws.import(data, RooFit::Silence());
   RooFit::ModelConfig mc{"ModelConfig", &ws};
   mc.SetPdf("model");
   mc.SetParametersOfInterest("theta");
   mc.SetObservables("x");
   mc.SetGlobalObservables("theta_obs");
   mc.SetExternalConstraints(RooArgSet{*ws.pdf("constraint")});
   ws.import(mc);
}

} // namespace

// The document reads back with the same NLL, and is stable.
INSTANTIATE_TEST_SUITE_P(ConstraintInModelAndExternal, RoundTrip,
                         testing::Values(RoundTripModel{"Model", buildConstraintInModelAndExternal}
                                            .with(constraintStyles())
                                            .likelihoodOf("ModelConfig", "data")
                                            .within(1e-10)),
                         roundTripName);

// A constraint that is both a factor of the model and an external constraint of the ModelConfig is one term of the
// likelihood in RooFit. It is written once, as a term of the likelihood: the product keeps its other factors. The
// document has distinct variates.
TEST(RooFitHS3, ConstraintInModelAndExternal)
{
   RooWorkspace ws;
   buildConstraintInModelAndExternal(ws);

   Doc exported{ws};
   EXPECT_TRUE(matches(exported, "distributions/model", R"({"factors": ["shape"]})"));
   EXPECT_TRUE(
      matches(exported, "likelihoods/ModelConfig_data_likelihood", R"({"aux_distributions": ["constraint"]})"));

   EXPECT_THAT(Imported{exported}.warnings, Not(HasSubstr("is shared by the")));
}

// The parameters that only an external constraint of the ModelConfig depends on keep their roles through a round
// trip: its observable stays a global observable, and its parameter a nuisance parameter.
TEST(RooFitHS3, ExternalConstraintKeepsRoles)
{
   RooWorkspace ws;
   ws.factory("Gaussian::physics(x[0, -5, 5], mu[0, -2, 2], 1.0)");
   ws.factory("Gaussian::constraint(gobs[0, -5, 5], theta[0, -5, 5], 1.0)");
   ws.var("gobs")->setConstant(true);
   RooDataSet data{"data", "data", *ws.var("x")};
   data.add(*ws.var("x"));
   ws.import(data, RooFit::Silence());
   RooFit::ModelConfig mc{"ModelConfig", &ws};
   mc.SetPdf("physics");
   mc.SetObservables("x");
   mc.SetParametersOfInterest("mu");
   mc.SetNuisanceParameters("theta");
   mc.SetGlobalObservables("gobs");
   mc.SetExternalConstraints(RooArgSet{*ws.pdf("constraint")});
   ws.import(mc);

   const std::string exported = toJSON(ws);
   Imported ws2{exported};
   auto *mc2 = object<RooFit::ModelConfig>(ws2, "ModelConfig");
   ASSERT_NE(mc2->GetGlobalObservables(), nullptr) << exported;
   ASSERT_NE(mc2->GetNuisanceParameters(), nullptr) << exported;
   EXPECT_EQ(mc2->GetGlobalObservables()->size(), 1u) << exported;
   EXPECT_NE(mc2->GetGlobalObservables()->find("gobs"), nullptr) << exported;
   EXPECT_EQ(mc2->GetNuisanceParameters()->size(), 1u) << exported;
   EXPECT_NE(mc2->GetNuisanceParameters()->find("theta"), nullptr) << exported;
   ASSERT_NE(mc2->GetExternalConstraints(), nullptr);
   EXPECT_NE(mc2->GetExternalConstraints()->find("constraint"), nullptr);
}

namespace {

// Two channels with constraints as factors: one shared by the channels, one of the first channel only.
void buildConstraintFactorsOfChannels(RooWorkspace &ws)
{
   ws.factory("Gaussian::a(xa[0, -5, 5], mu[0, -2, 2], 1.0)");
   ws.factory("Gaussian::b(xb[0, -5, 5], expr::mu_theta('mu + theta', mu, theta[0, -5, 5]), 1.0)");
   ws.factory("Gaussian::shared(gobs[0, -5, 5], theta, 1.0)");
   ws.factory("Gaussian::own(gobs_eta[0, -5, 5], eta[0, -5, 5], 1.0)");
   ws.factory("Gaussian::a_shape(xa, eta, 2.0)");
   ws.var("gobs")->setConstant(true);
   ws.var("gobs_eta")->setConstant(true);
   ws.factory("PROD::channel_a(SUM::a_sum(f[0.5, 0, 1] * a, a_shape), shared, own)");
   ws.factory("PROD::channel_b(b, shared)");
   ws.factory("SIMUL::model(channel[A=0, B=1], A=channel_a, B=channel_b)");
   RooArgSet observables{*ws.var("xa"), *ws.var("xb"), *ws.cat("channel")};
   RooDataSet data{"data", "data", observables};
   for (int index : {0, 1}) {
      ws.cat("channel")->setIndex(index);
      ws.var("xa")->setVal(0.3);
      ws.var("xb")->setVal(-0.6);
      data.add(observables);
   }
   ws.import(data, RooFit::Silence());
   RooFit::ModelConfig mc{"ModelConfig", &ws};
   mc.SetPdf("model");
   mc.SetParametersOfInterest("mu");
   mc.SetNuisanceParameters("theta,eta,f");
   mc.SetObservables(observables);
   mc.SetGlobalObservables("gobs,gobs_eta");
   ws.import(mc);
}

} // namespace

// The document reads back with the same NLL. The import makes a constraint a factor of the channels that depend on
// its parameter: "shared" is no factor of "channel_a" any more, so the second export lists the terms in another
// order. From then on, the document is stable.
INSTANTIATE_TEST_SUITE_P(ConstraintFactorsOfChannels, RoundTrip,
                         testing::Values(RoundTripModel{"Model", buildConstraintFactorsOfChannels}
                                            .with(constraintStyles())
                                            .over({"xa", "xb"})
                                            .likelihoodOf("ModelConfig", "data")
                                            .within(1e-10)
                                            .stableFromSecondExport()),
                         roundTripName);

// Constraints that RooFit finds in the channels of a model (RooAbsPdf::getAllConstraints()) are written as terms of
// the likelihood, not as factors of the products, also if only one channel has them. A constraint shared by two
// channels is one term of the likelihood in RooFit, and it is written once. A product that another pdf uses keeps its
// constraints.
TEST(RooFitHS3, ConstraintFactorsOfChannelsAreLikelihoodTerms)
{
   RooWorkspace ws;
   buildConstraintFactorsOfChannels(ws);

   Doc exported{ws};
   EXPECT_TRUE(matches(exported, "distributions/channel_a", R"({"factors": ["a_sum"]})"));
   EXPECT_TRUE(matches(exported, "distributions/channel_b", R"({"factors": ["b"]})"));
   EXPECT_THAT(exported.strings("likelihoods/ModelConfig_data_likelihood/aux_distributions"),
               testing::UnorderedElementsAre("shared", "own"));

   EXPECT_THAT(Imported{exported}.warnings, Not(HasSubstr("is shared by the")));

   // A product that another pdf uses keeps its constraint.
   RooWorkspace wsUsed;
   wsUsed.factory("Gaussian::shape(x[0, -5, 5], theta[0, -5, 5], 1.0)");
   wsUsed.factory("Gaussian::constraint(gobs[0, -5, 5], theta, 1.0)");
   wsUsed.var("gobs")->setConstant(true);
   wsUsed.factory("PROD::model(shape, constraint)");
   wsUsed.factory("SUM::other(f[0.5, 0, 1] * model, Uniform::flat(x))");
   RooDataSet dataUsed{"data", "data", *wsUsed.var("x")};
   dataUsed.add(*wsUsed.var("x"));
   wsUsed.import(dataUsed, RooFit::Silence());
   addModelConfig(wsUsed, "ModelConfig", "model", "x", "theta", /*nuisance=*/nullptr, /*globals=*/"gobs");
   Doc exportedUsed{wsUsed};
   EXPECT_TRUE(matches(exportedUsed, "distributions/model", R"({"factors": ["shape", "constraint"]})"));
   EXPECT_THAT(exportedUsed.json(), Not(HasSubstr("aux_distributions")));
}

namespace {

// The "domains" with one product domain of the given name, see defaultDomain(). RooFit reads a domain with another
// name than "default_domain" also as a named range of its variables.
std::string domain(std::string const &name, std::vector<std::string> const &axes)
{
   return R"("domains": [{"name": ")" + name + R"(", "type": "product_domain", "axes": [)" + join(axes) + "]}]";
}

// The values of constant parameters for a parameter point, see values().
std::string constValues(std::vector<std::pair<std::string, double>> const &parameters)
{
   std::vector<std::string> out;
   for (auto const &parameter : parameters) {
      out.push_back(replaced(values({parameter}), "}", R"(, "const": true})"));
   }
   return join(out);
}

// The names of the observables of the ModelConfig `name` of `ws`, sorted.
std::vector<std::string> observableNames(RooWorkspace &ws, const char *name)
{
   std::vector<std::string> names;
   auto *mc = dynamic_cast<RooFit::ModelConfig *>(ws.obj(name));
   if (mc && mc->GetObservables()) {
      for (RooAbsArg const *arg : *mc->GetObservables()) {
         names.push_back(arg->GetName());
      }
   }
   std::sort(names.begin(), names.end());
   return names;
}

// Remove "observables" from misc.ROOT_internal.ModelConfigs of the document `json`.
std::string withoutObservablesList(std::string const &json)
{
   auto tree = RooFit::Detail::JSONTree::create(json);
   auto &modelConfigs = tree->rootnode()["misc"]["ROOT_internal"]["ModelConfigs"];
   for (auto &item : modelConfigs.children()) {
      if (item.has_child("observables")) {
         item.erase("observables");
      }
   }
   std::stringstream ss;
   tree->rootnode().writeJSON(ss);
   return ss.str();
}

} // namespace

// The observables of a ModelConfig without data set are not given by the data of its likelihood. RooFit writes them
// in misc.ROOT_internal.ModelConfigs. Without that list, they are the variates of the likelihood that are neither in
// a domain of the analysis nor constant, which HS3 makes observables, and the columns of a combined distribution.
TEST(RooFitHS3, ObservablesWithoutData)
{
   RooWorkspace ws;
   ws.factory("Gaussian::shape(x[0, -5, 5], mu[0, -2, 2], 1.0)");
   ws.factory("Gaussian::shift(x, theta[0, -5, 5], 2.0)");
   ws.factory("SUM::phys(f[0.5, 0, 1] * shape, shift)");
   ws.factory("Gaussian::constraint(gobs[0, -5, 5], theta, 1.0)");
   ws.var("gobs")->setConstant(true);
   ws.factory("PROD::model(phys, constraint)");
   addModelConfig(ws, "ModelConfig", "model", "x", "mu", /*nuisance=*/"theta,f", /*globals=*/"gobs");

   Doc exported{ws};
   EXPECT_EQ(exported.dump("misc/ROOT_internal/ModelConfigs/ModelConfig/observables"), R"(["x"])");
   Imported ws2{exported};
   EXPECT_EQ(observableNames(ws2, "ModelConfig"), std::vector<std::string>{"x"});
   // The constraint is a factor of the model again, so the document is stable.
   EXPECT_EQ(toJSON(ws2), exported.json());

   // Without the list: inferred, with a message.
   const std::string withoutList = withoutObservablesList(exported);
   ASSERT_THAT(withoutList, Not(HasSubstr(R"("observables")")));
   RooWorkspace ws3;
   const std::string info = captureMessages(RooFit::INFO, RooFit::IO, [&] { importInto(ws3, withoutList); });
   EXPECT_EQ(observableNames(ws3, "ModelConfig"), std::vector<std::string>{"x"});
   EXPECT_THAT(info, HasSubstr("neither in a domain of the analysis nor constant: 'x'"));

   // A RooSimultaneous: the index category is an observable as well.
   RooWorkspace wsSim;
   wsSim.factory("Gaussian::a(xa[0, -5, 5], mu[0, -2, 2], 1.0)");
   wsSim.factory("Gaussian::b(xb[0, -5, 5], mu, 1.0)");
   wsSim.factory("SIMUL::sim(channel[A=0, B=1], A=a, B=b)");
   addModelConfig(wsSim, "ModelConfig", "sim", "xa,xb,channel", "mu");
   const std::vector<std::string> simObservables{"channel", "xa", "xb"};
   const std::string exportedSim = toJSON(wsSim);
   for (std::string const &json : {exportedSim, withoutObservablesList(exportedSim)}) {
      Imported wsSim2{json};
      EXPECT_EQ(observableNames(wsSim2, "ModelConfig"), simObservables) << json;
   }

   // A document written by hand, with one domain for all parameters of the analysis. Like RooFit, it has no data set:
   // the likelihood is a pure model with "data": null.
   const std::string handWritten = R"({
  "metadata": {"hs3_version": "0.3"},
  "distributions": [
    {"name": "shape", "type": "gaussian_dist", "x": "x", "mean": "mu", "sigma": 1.0},
    {"name": "constraint", "type": "gaussian_dist", "x": "gobs", "mean": "theta", "sigma": 1.0},
    {"name": "model", "type": "product_dist", "factors": ["shape"]}
  ],
  "domains": [
    {"name": "parameters", "type": "product_domain",
     "axes": [{"name": "mu", "min": -2.0, "max": 2.0}, {"name": "theta", "min": -5.0, "max": 5.0}]},
    {"name": "default_domain", "type": "product_domain", "axes": [{"name": "x", "min": -5.0, "max": 5.0}]}
  ],
  "parameter_points": [{"name": "default_values", "parameters": [
    {"name": "mu", "value": 0.0}, {"name": "theta", "value": 0.0}, {"name": "gobs", "value": 0.0, "const": true}]}],
  "likelihoods": [{"name": "lh", "distributions": ["model"], "data": null, "aux_distributions": ["constraint"]}],
  "analyses": [{"name": "analysis", "likelihood": "lh", "parameters_of_interest": ["mu"], "domains": ["parameters"]}]
})";
   Imported ws4{handWritten};
   EXPECT_EQ(observableNames(ws4, "analysis"), std::vector<std::string>{"x"});
}

namespace {

// A key for channel(): the interpolation of HistFactory.
const std::string polyExpInterpolation = R"("default_interpolation": {"type": "mult", "in": "poly6", "out": "exp"})";

// A document with the HistFactory channel `HF` of one bin of `x`: its sample has a normfactor `mu` and a normsys of
// `theta` with the constraint `constraint`, whose global observable is `gobs`. With further top-level sections.
std::string constrainedChannelJSON(std::vector<std::string> const &sections = {})
{
   const std::string xAxis = R"({"name": "x", "min": 0.0, "max": 1.0, "nbins": 1})";
   const std::string modifiers = R"({"name": "mu", "type": "normfactor", "parameters": ["mu"]},
      {"name": "syst", "type": "normsys", "parameters": ["theta"], "data": {"lo": 0.9, "hi": 1.1},
       "constraints": ["constraint"]})";
   std::vector<std::string> all{
      distributions(
         {channel("HF", {sample("signal", R"({"contents": [10.0]})", modifiers)}, polyExpInterpolation, xAxis),
          R"({"name": "constraint", "type": "gaussian_dist", "x": "gobs", "mean": "theta", "sigma": 1.0})"}),
      defaultDomain({xAxis, axes({"mu"}, 0, 5), axes({"theta", "gobs"}, -5, 5)}),
      defaultValues({values({{"mu", 1.0}, {"theta", 0.0}}), constValues({{"gobs", 0.0}})})};
   all.insert(all.end(), sections.begin(), sections.end());
   return document("0.3", all);
}

// A workspace with the HistFactory channel of constrainedChannelJSON() as the component `A` of the RooSimultaneous
// `model`, with a data set and a ModelConfig.
void makeRenamedHistFactoryWorkspace(RooWorkspace &ws)
{
   importInto(ws, constrainedChannelJSON());
   RooCategory channel{"channel", "channel", {{"A", 0}}};
   RooSimultaneous model{"model", "model", {{"A", ws.pdf("HF")}}, channel};
   ws.import(model, RooFit::Silence());
   RooRealVar &x = *ws.var("x");
   RooCategory &index = *ws.cat("channel");
   RooRealVar weight{"weight", "weight", 1.0};
   RooDataSet data{"data", "data", {x, index, weight}, RooFit::WeightVar(weight)};
   x.setVal(0.5);
   index.setLabel("A");
   data.add({x, index}, 10.0);
   ws.import(data, RooFit::Silence());
   addModelConfig(ws, "mc", "model", "x,channel", "mu", /*nuisance=*/"theta", /*globals=*/"gobs");
}

// The styles with renamed observables of the combined data sets.
std::vector<RoundTripStyle> withRenamedObservables(std::vector<RoundTripStyle> styles)
{
   for (auto &style : styles) {
      style.name += "Renamed";
      style.flags.push_back(renameCombinedObservables(true));
   }
   return styles;
}

} // namespace

// The constraint terms of the likelihood are looked for in the channels as they are written: with
// renameCombinedObservables, the components of the RooSimultaneous are written as renamed clones, and a HistFactory
// channel is still recognised as one. The likelihood has the same constraint terms with and without renaming, with
// and without arrays: it lists them as they are written (the array that holds a constraint), so the document has no
// reference to a distribution that is not written (reported in the review of #24013). The document reads back with
// the same NLL, see the round trips below.
TEST(RooFitHS3, ConstraintTermsOfRenamedChannels)
{
   RooWorkspace ws;
   makeRenamedHistFactoryWorkspace(ws);

   for (bool arrays : {true, false}) {
      SCOPED_TRACE(arrays ? "arrays" : "no arrays");
      const std::string constraints = arrays ? R"(["model_gaussian_constraints"])" : R"(["constraint"])";
      Doc plain{ws, {exportArrayDist(arrays)}};
      Doc renamed{ws, {exportArrayDist(arrays), renameCombinedObservables(true)}};
      ASSERT_THAT(renamed.json(), HasSubstr("obs_A"));
      EXPECT_EQ(plain.dump("likelihoods/0/aux_distributions"), constraints);
      EXPECT_EQ(renamed.dump("likelihoods/0/aux_distributions"), constraints);
   }
}

// With and without renamed observables, for the four layouts of the constraint terms. With renamed observables, the
// first document differs from the following ones: it has the attributes of "HF_model", a node of the original
// channel that is not in the document, where the channel is "HF_A". The following ones have them for "HF_A_model".
INSTANTIATE_TEST_SUITE_P(RenamedHistFactoryChannel, RoundTrip,
                         testing::Values(RoundTripModel{"Plain", makeRenamedHistFactoryWorkspace}
                                            .with(constraintStyles())
                                            .likelihoodOf("mc", "data")
                                            .within(1e-10)
                                            .withOtherComponents(),
                                         RoundTripModel{"Renamed", makeRenamedHistFactoryWorkspace}
                                            .with(withRenamedObservables(constraintStyles()))
                                            .likelihoodOf("mc", "data")
                                            .within(1e-10)
                                            .stableFromSecondExport()),
                         roundTripName);

// A channel with the constant inverse bin width of HistFactory before ROOT 6.26, as a component of a RooSimultaneous
// whose observables are renamed. The renamed channel is a clone, but the constant is not cloned, so the original
// channel still uses it: it is still identified as the inverse bin width, not written as a normfactor, and the
// likelihood is the same after the round trip.
TEST(RooFitHS3, LegacyBinWidthFactorOfRenamedChannel)
{
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::NumericIntegration, true};

   std::unique_ptr<RooWorkspace> legacy = makeLegacyHistFactoryWorkspace();
   RooWorkspace ws;
   RooCategory channelCat{"channelCat", "channelCat", {{"channel1", 0}}};
   RooSimultaneous simPdf{"simPdf", "simPdf", {{"channel1", legacy->pdf("model_channel1")}}, channelCat};
   ws.import(simPdf, RooFit::Silence());
   RooRealVar &obs = *ws.var("obs_x_channel1");
   RooCategory &index = *ws.cat("channelCat");
   RooRealVar weight{"weight", "weight", 1.0};
   RooDataSet data{"obsData", "obsData", {obs, index, weight}, RooFit::WeightVar(weight)};
   index.setLabel("channel1");
   for (auto const &[x, w] : std::vector<std::pair<double, double>>{{1.25, 122.0}, {1.75, 112.0}}) {
      obs.setVal(x);
      data.add({obs, index}, w);
   }
   ws.import(data, RooFit::Silence());
   RooStats::ModelConfig mc{"ModelConfig", &ws};
   mc.SetPdf("simPdf");
   mc.SetObservables("obs_x_channel1,channelCat");
   mc.SetParametersOfInterest("SigXsecOverSM");
   mc.SetNuisanceParameters("alpha_syst");
   mc.SetGlobalObservables("nom_alpha_syst");
   ws.import(mc);

   const std::string json = toJSON(ws, {renameCombinedObservables(true)});
   ASSERT_THAT(json, HasSubstr("obs_channel1"));
   int nChannels = 0;
   const auto modifiers = histFactoryModifiers(json, nChannels);
   EXPECT_EQ(nChannels, 1) << json;
   for (auto const &modifier : modifiers) {
      EXPECT_THAT(modifier, Not(HasSubstr("binWidth"))) << ": " << json;
   }

   Imported ws2{json};
   auto *mc2 = static_cast<RooStats::ModelConfig *>(ws2.obj("ModelConfig"));
   ASSERT_NE(mc2, nullptr);
   ASSERT_NE(ws2.data("obsData"), nullptr);
   using RooFit::GlobalObservables;
   std::unique_ptr<RooAbsReal> nll{
      ws.pdf("simPdf")->createNLL(*ws.data("obsData"), GlobalObservables(*mc.GetGlobalObservables()))};
   std::unique_ptr<RooAbsReal> nll2{
      mc2->GetPdf()->createNLL(*ws2.data("obsData"), GlobalObservables(*mc2->GetGlobalObservables()))};
   // Up to a constant.
   const double ref = nll->getVal();
   const double ref2 = nll2->getVal();
   for (auto const &[mu, alpha] : std::vector<std::pair<double, double>>{{1.5, 0.3}, {0.7, -0.4}}) {
      for (RooWorkspace *w : std::initializer_list<RooWorkspace *>{&ws, &ws2}) {
         w->var("SigXsecOverSM")->setVal(mu);
         w->var("alpha_syst")->setVal(alpha);
      }
      EXPECT_NEAR(nll2->getVal() - ref2, nll->getVal() - ref, 1e-8) << mu << ", " << alpha;
   }
}

// The constant inverse bin width of HistFactory before ROOT 6.26 is identified by its structure, among other things by
// being used only by its channel. Objects outside the workspace don't count: e.g. a likelihood created from the model
// before the export (as for a fit) refers to the constant from its computation graph.
TEST(RooFitHS3, LegacyBinWidthFactorWithLikelihood)
{
   RooHelpers::LocalChangeMsgLevel chmsglvl{RooFit::WARNING, 0u, RooFit::NumericIntegration, true};

   std::unique_ptr<RooWorkspace> legacy = makeLegacyHistFactoryWorkspace();
   auto *mcLegacy = static_cast<RooStats::ModelConfig *>(legacy->obj("ModelConfig"));
   ASSERT_NE(mcLegacy, nullptr);
   // Without "binWidth" in their names, the factors are identified only by their structure.
   std::unique_ptr<RooAbsPdf> model{static_cast<RooAbsPdf *>(mcLegacy->GetPdf()->cloneTree())};
   RooArgSet nodes;
   model->treeNodeServerList(&nodes);
   std::vector<std::string> factors;
   for (RooAbsArg *arg : nodes) {
      if (std::string{arg->GetName()}.find("binWidth") != std::string::npos) {
         factors.push_back("w_" + std::to_string(factors.size()));
         arg->SetName(factors.back().c_str());
      }
   }
   ASSERT_FALSE(factors.empty());
   RooWorkspace ws;
   ws.import(*model, RooFit::Silence());
   ws.import(*legacy->data("obsData"), RooFit::Silence());
   RooStats::ModelConfig mc{"ModelConfig", &ws};
   mc.SetPdf(model->GetName());
   mc.SetObservables(*mcLegacy->GetObservables());
   mc.SetParametersOfInterest(*mcLegacy->GetParametersOfInterest());
   mc.SetNuisanceParameters(*mcLegacy->GetNuisanceParameters());
   mc.SetGlobalObservables(*mcLegacy->GetGlobalObservables());
   ws.import(mc);

   auto normfactors = [&](std::string const &json) {
      int nChannels = 0;
      const auto modifiers = histFactoryModifiers(json, nChannels);
      EXPECT_EQ(nChannels, 1) << json;
      std::size_t n = 0;
      for (auto const &name : factors) {
         n += modifiers.count("normfactor:" + name + ",");
      }
      return n;
   };
   const std::string before = toJSON(ws);
   EXPECT_EQ(normfactors(before), 0u) << before;

   std::unique_ptr<RooAbsReal> nll{ws.pdf(model->GetName())->createNLL(*ws.data("obsData"))};
   ASSERT_NE(nll, nullptr);
   nll->getVal();
   const std::string withLikelihood = toJSON(ws);
   EXPECT_EQ(normfactors(withLikelihood), 0u) << withLikelihood;
   EXPECT_EQ(withLikelihood, before);
}

namespace {

// A HistFactory channel and an ordinary product sharing the constraint of the HistFactory parameter `theta`. The
// HistFactory parameter `eta` has a constraint of its own. The channels are named `hfName` and `otherName`, to
// control the order in which the export visits them. The constraint enters the likelihood twice, through the
// modifier and as factor of the product, so the document is not valid HS3 0.3: its variate `theta_obs` is shared by
// two terms. As a document of version 0.2 it is read with a warning, like a RooFit model with the constraint in
// both channels.
std::string
sharedConstraintJSON(std::string const &hfName, std::string const &otherName, std::string const &version = "0.3")
{
   const std::string xHF = R"({"name": "x_hf", "min": 0.0, "max": 1.0, "nbins": 1})";
   const std::string modifiers = R"({"name": "mu", "type": "normfactor", "parameters": ["mu"]},
      {"name": "shared", "type": "normsys", "parameters": ["theta"], "data": {"lo": 0.9, "hi": 1.1},
       "constraints": ["shared_constraint"]},
      {"name": "own", "type": "normsys", "parameters": ["eta"], "data": {"lo": 0.8, "hi": 1.2},
       "constraints": ["own_constraint"]})";
   return document(
      version,
      {distributions(
          {channel(hfName, {sample("signal", R"({"contents": [10.0]})", modifiers)}, polyExpInterpolation, xHF),
           R"({"name": "shape", "type": "gaussian_dist", "x": "x_other", "mean": "theta", "sigma": 1.0})",
           R"({"name": ")" + otherName + R"(", "type": "product_dist", "factors": ["shape", "shared_constraint"]})",
           R"({"name": "shared_constraint", "type": "gaussian_dist", "x": "theta_obs", "mean": "theta", "sigma": 1.0})",
           R"({"name": "own_constraint", "type": "gaussian_dist", "x": "eta_obs", "mean": "eta", "sigma": 1.0})"}),
       defaultDomain({xHF, axes({"x_other", "theta", "eta"}, -5, 5), axes({"mu"}, 0, 5)}),
       defaultValues({values({{"theta", 0.0}}), constValues({{"theta_obs", 0.0}}), values({{"eta", 0.0}}),
                      constValues({{"eta_obs", 0.0}}), values({{"mu", 1.0}})}),
       section("data", {R"({"name": "data_hf", "type": "binned", "axes": [)" + xHF + R"(], "contents": [10.0]})",
                        R"({"name": "data_other", "type": "unbinned", "axes": [)" + axes({"x_other"}, -5, 5) +
                           R"(], "entries": [[0.2]]})"}),
       R"("likelihoods": [{"name": "lh", "distributions": [")" + hfName + R"(", ")" + otherName +
          R"("], "data": ["data_hf", "data_other"]}])",
       R"("analyses": [{"name": "analysis", "likelihood": "lh", "domains": ["default_domain"],
                        "parameters_of_interest": ["mu"]}])"});
}

// The NLL of the model of sharedConstraintJSON() at a parameter point that moves both constrained parameters.
double sharedConstraintNLL(RooWorkspace &ws, std::string const &hfName, std::string const &otherName)
{
   auto *mc = dynamic_cast<RooStats::ModelConfig *>(ws.obj("analysis"));
   auto *simPdf = mc ? dynamic_cast<RooSimultaneous *>(mc->GetPdf()) : nullptr;
   if (!simPdf) {
      return std::numeric_limits<double>::quiet_NaN();
   }
   ws.var("mu")->setVal(1.2);
   ws.var("theta")->setVal(0.4);
   ws.var("eta")->setVal(-0.3);
   RooRealVar &xHF = *ws.var("x_hf");
   RooRealVar &xOther = *ws.var("x_other");
   RooRealVar weight{"weight", "weight", 1.0};
   auto &indexCat = dynamic_cast<RooCategory &>(const_cast<RooAbsCategoryLValue &>(simPdf->indexCat()));
   RooDataSet combined{"combined", "combined", {xHF, xOther, indexCat, weight}, RooFit::WeightVar(weight)};
   xOther.setVal(0.0);
   xHF.setVal(0.5);
   indexCat.setLabel(hfName.c_str());
   combined.add({xHF, xOther, indexCat}, 10.0);
   xOther.setVal(0.2);
   indexCat.setLabel(otherName.c_str());
   combined.add({xHF, xOther, indexCat}, 1.0);
   std::unique_ptr<RooAbsReal> nll{simPdf->createNLL(combined)};
   return nll->getVal();
}

// The model of sharedConstraintJSON() as RooFit reads it from a document of version 0.2.
RoundTripModel sharedConstraintModel(std::string const &name, std::string const &hfName, std::string const &otherName)
{
   return RoundTripModel::fromDocument(name, sharedConstraintJSON(hfName, otherName, "0.2"))
      .with({{"Arrays", {}}, {"NoArrays", {exportArrayDist(false)}}})
      .over({"x_hf", "x_other"})
      .comparing([=](RooWorkspace &ws) { return std::vector<double>{sharedConstraintNLL(ws, hfName, otherName)}; })
      .within(1e-10)
      .withOtherComponents()
      .stableFromSecondExport();
}

} // namespace

// The document reads back with the same likelihood, and is stable from the second export on: see the test.
INSTANTIATE_TEST_SUITE_P(HistFactoryConstraintSharedWithOtherPdf, RoundTrip,
                         testing::Values(sharedConstraintModel("HistFactoryFirst", "model_hf", "model_other"),
                                         sharedConstraintModel("HistFactoryLast", "z_model_hf", "a_model_other")),
                         roundTripName);

// A constraint shared between a HistFactory channel and another pdf is written once, as a term of the likelihood,
// whatever the order in which the export visits them, also when the constraints of the HistFactory channels are
// collected into arrays.
TEST(RooFitHS3, HistFactoryConstraintSharedWithOtherPdf)
{
   // Not valid HS3 0.3.
   expectImportThrowsWithError(sharedConstraintJSON("model_hf", "model_other"),
                               "the variate 'theta_obs' is shared by the terms 'model_other' and 'shared_constraint'");

   // The HistFactory channel is visited first, and last.
   for (auto const &names : std::vector<std::pair<std::string, std::string>>{{"model_hf", "model_other"},
                                                                             {"z_model_hf", "a_model_other"}}) {
      const auto &hfName = names.first;
      const auto &otherName = names.second;
      Imported ws{sharedConstraintJSON(hfName, otherName, "0.2")};
      EXPECT_THAT(ws.warnings, HasSubstr("the variate 'theta_obs' is shared by the terms '" + otherName +
                                         "' and 'shared_constraint'"));
      const double nll = sharedConstraintNLL(ws, hfName, otherName);
      ASSERT_FALSE(std::isnan(nll));
      for (bool arrays : {true, false}) {
         SCOPED_TRACE(hfName + (arrays ? ", arrays" : ", no arrays"));
         Doc exported{ws, {exportArrayDist(arrays)}};
         const std::string modifiers = "distributions/" + hfName + "/samples/signal/modifiers/";
         EXPECT_TRUE(matches(exported, "distributions/shared_constraint", R"({"type": "gaussian_dist"})"));
         EXPECT_TRUE(matches(exported, modifiers + "theta", R"({"constraints": ["shared_constraint"]})"));
         // The product only keeps the factor that is not a term of the likelihood.
         EXPECT_TRUE(matches(exported, "distributions/" + otherName, R"({"factors": ["shape"]})"));
         // The constraint that only the HistFactory channel uses is still collected.
         const std::string array = "analysis_simPdf_gaussian_constraints";
         EXPECT_EQ(exported.has("distributions/own_constraint"), !arrays);
         EXPECT_EQ(exported.has("distributions/" + array), arrays);
         if (arrays) {
            EXPECT_TRUE(matches(exported, "distributions/" + array, R"({"components": [{"x": "eta_obs"}]})"));
            EXPECT_TRUE(matches(exported, modifiers + "eta", R"({"constraints": [")" + array + R"("]})"));
         }

         // Valid HS3 0.3: read back without warnings.
         Imported ws2{exported};
         EXPECT_THAT(ws2.warnings, Not(HasSubstr("is shared by the terms")));
         // The re-imported model has the constraint only in the HistFactory channel, so with arrays it is collected
         // now. The model that is read from this second document still has the likelihood of the original one.
         const std::string exported2 = toJSON(ws2, {exportArrayDist(arrays)});
         EXPECT_EQ(exported2.find(R"("constraints":["shared_constraint"])") == std::string::npos, arrays) << exported2;
         Imported ws3{exported2};
         EXPECT_NEAR(sharedConstraintNLL(ws3, hfName, otherName), nll, 1e-10 * std::abs(nll));
      }
   }

   // A component of a tagged array product that another product also uses cannot be written: the export fails
   // instead of leaving a dangling reference, whichever of the two is visited first.
   RooRealVar nomA{"nom_a", "nom_a", 0.0, -5.0, 5.0};
   RooRealVar a{"a", "a", 0.0, -5.0, 5.0};
   RooRealVar x{"x", "x", 1.0, 0.0, 10.0};
   RooGaussian gaussA{"gauss_a", "gauss_a", nomA, a, RooFit::RooConst(1.0)};
   RooGaussian shape{"shape", "shape", x, a, RooFit::RooConst(1.0)};
   for (auto const &[arrayName, error] : std::vector<std::pair<std::string, std::string>>{
           {"a_array", "is written as a component of the array"}, {"z_array", "is already written as 'gauss_a'"}}) {
      RooProdPdf array{arrayName.c_str(), arrayName.c_str(), {gaussA}};
      array.setAttribute("hs3_array_dist");
      RooProdPdf other{"m_other", "m_other", {shape, gaussA}};
      RooWorkspace wsTagged;
      wsTagged.import(array, RooFit::Silence());
      wsTagged.import(other, RooFit::RecycleConflictNodes(), RooFit::Silence());
      expectExportThrowsWithError(wsTagged, error);
   }
}

// The variates of an array distribution are the observables of its components: a likelihood term that shares one of
// them with an array is rejected like any other shared variate.
TEST(RooFitHS3, ArrayDistVariatesInLikelihood)
{
   const std::string array = R"({"name": "constraints", "type": "array_dist", "component_type": "gaussian_dist",
      "default_parameters": {"sigma": 1.0}, "components": [{"x": "nom_a", "mean": "a"}, {"x": "nom_b", "mean": "b"}]})";
   const std::string likelihood =
      R"({"name": "lh", "distributions": ["model"], "data": ["data"], "aux_distributions": ["constraints"]})";
   const std::string json = document(
      "0.3",
      {distributions({R"({"name": "model", "type": "gaussian_dist", "x": "obs", "mean": "a", "sigma": 1.0})", array,
                      R"({"name": "other", "type": "gaussian_dist", "x": "nom_b", "mean": "b", "sigma": 2.0})"}),
       section("data", {R"({"name": "data", "type": "unbinned", "axes": [)" + axes({"obs"}, -5, 5) +
                        R"(], "entries": [[0.3]]})"}),
       defaultDomain({axes({"obs", "a", "b"}, -5, 5)}), defaultValues({constValues({{"nom_a", 0.0}, {"nom_b", 0.0}})}),
       section("likelihoods", {likelihood}),
       R"("analyses": [{"name": "an", "likelihood": "lh", "domains": ["default_domain"]}])"});
   {
      Imported ws{json};
   }
   expectImportThrowsWithError(replaced(json, R"(["constraints"])", R"(["constraints", "other"])"),
                               "the variate 'nom_b' is shared by the terms 'constraints[nom_b]' and 'other' of the "
                               "likelihood 'lh'");
}

namespace {

// The HistFactory channel of constrainedChannelJSON() with data, whose constraint is a term of the likelihood. With
// `external`, the constraint is also an external constraint of the ModelConfig, which is redundant: it is part of the
// pdf already.
void importConstrainedChannel(RooWorkspace &ws, bool external)
{
   importInto(ws, constrainedChannelJSON({R"(
      "data": [{"name": "data", "type": "binned", "contents": [10.0],
                "axes": [{"name": "x", "min": 0.0, "max": 1.0, "nbins": 1}]}],
      "likelihoods": [{"name": "lh", "distributions": ["HF"], "data": ["data"], "aux_distributions": ["constraint"]}],
      "analyses": [{"name": "mc", "likelihood": "lh", "parameters_of_interest": ["mu"],
                    "domains": ["default_domain"]}])"}));
   if (external) {
      modelConfig(ws, "mc").SetExternalConstraints(RooArgSet{*ws.pdf("constraint")});
   }
}

} // namespace

// A constraint of a HistFactory modifier that is also an external constraint of the ModelConfig is referred to by
// its own name, so it is not collected into an array: the export must not leave the likelihood with a reference to
// a distribution that is not written.
TEST(RooFitHS3, HistFactoryConstraintAlsoExternal)
{
   RooWorkspace ws;
   importConstrainedChannel(ws, /*external=*/true);
   // RooFit counts the constraint once.
   RooWorkspace withoutExternal;
   importConstrainedChannel(withoutExternal, /*external=*/false);
   ASSERT_TRUE(sameNumbers(likelihoodScan(ws, "mc", "data"), likelihoodScan(withoutExternal, "mc", "data"), 1e-10));

   Doc exported{ws};
   EXPECT_TRUE(matches(exported, "distributions/constraint", R"({"type": "gaussian_dist"})"));
   EXPECT_TRUE(
      matches(exported, "distributions/HF/samples/signal/modifiers/theta", R"({"constraints": ["constraint"]})"));
   EXPECT_TRUE(matches(exported, "likelihoods/0", R"({"aux_distributions": ["constraint"]})"));
}

// The document reads back with the same likelihood. The constraint is no external constraint then, so the following
// documents differ from the first one: with arrays, it is collected.
INSTANTIATE_TEST_SUITE_P(HistFactoryConstraintAlsoExternal, RoundTrip,
                         testing::Values(RoundTripModel{"Model",
                                                        [](RooWorkspace &ws) { importConstrainedChannel(ws, true); }}
                                            .with(constraintStyles())
                                            .likelihoodOf("mc", "data")
                                            .within(1e-10)
                                            .stableFromSecondExport()),
                         roundTripName);

namespace {

// A product of a model for the observable `x` and a constraint, with a data set and a ModelConfig.
void buildProductWithConstraint(RooWorkspace &ws)
{
   ws.factory("Gaussian::shape(x[0, -5, 5], mu[0, -2, 2], 1.0)");
   ws.factory("Gaussian::shift(x, theta[0, -5, 5], 2.0)");
   ws.factory("SUM::phys(f[0.5, 0, 1] * shape, shift)");
   ws.factory("Gaussian::constraint(gobs[0.3, -5, 5], theta, 1.0)");
   ws.var("gobs")->setConstant(true);
   ws.factory("PROD::model(phys, constraint)");
   RooDataSet data{"data", "data", *ws.var("x")};
   ws.var("x")->setVal(0.3);
   data.add(*ws.var("x"));
   ws.import(data, RooFit::Silence());
   addModelConfig(ws, "ModelConfig", "model", "x", "mu", /*nuisance=*/"theta,f", /*globals=*/"gobs");
}

} // namespace

// With config().exportConstraintData, a constraint of an ordinary product is written as a term of the likelihood
// paired with a data set of one entry, like the constraints of HistFactory channels. On import, it is a factor of
// the model again, the data set sets the observed value of its global observable, and it is not imported as a data
// set. The likelihood is the same with and without the option, see the round trips below.
TEST(RooFitHS3, ConstraintDataOfProductConstraints)
{
   RooWorkspace ws;
   buildProductWithConstraint(ws);

   for (bool constraintData : {false, true}) {
      SCOPED_TRACE(constraintData ? "with constraint data" : "without constraint data");
      Doc exported{ws, {exportConstraintData(constraintData)}};
      EXPECT_TRUE(matches(exported, "distributions/model", R"({"factors": ["phys"]})"));
      if (constraintData) {
         EXPECT_TRUE(matches(exported, "likelihoods/0",
                             R"({"distributions": ["model", "constraint"], "data": ["data", "constraint_data"]})"));
         EXPECT_THAT(exported.json(), Not(HasSubstr("aux_distributions")));
      } else {
         EXPECT_TRUE(matches(exported, "likelihoods/0", R"({"aux_distributions": ["constraint"]})"));
      }

      Imported ws2{exported};
      auto *mc2 = object<RooFit::ModelConfig>(ws2, "ModelConfig");
      EXPECT_STREQ(mc2->GetPdf()->GetName(), "model");
      // The constraint is a factor of the model again, as in the original, so that the model is the sampling
      // distribution of the auxiliary measurement as well. It is not an external constraint.
      EXPECT_TRUE(!mc2->GetExternalConstraints() || mc2->GetExternalConstraints()->empty()) << exported.json();
      EXPECT_TRUE(mc2->GetPdf()->dependsOn(*ws2.var("gobs"))) << exported.json();
      ASSERT_NE(mc2->GetGlobalObservables(), nullptr) << exported.json();
      EXPECT_NE(mc2->GetGlobalObservables()->find("gobs"), nullptr) << exported.json();
      EXPECT_EQ(ws2.data("constraint_data"), nullptr);
      EXPECT_DOUBLE_EQ(ws2.var("gobs")->getVal(), 0.3);
      EXPECT_TRUE(ws2.var("gobs")->isConstant());
   }
}

// The document is stable, and reads back with the same likelihood.
INSTANTIATE_TEST_SUITE_P(ConstraintDataOfProductConstraints, RoundTrip,
                         testing::Values(RoundTripModel{"Model", buildProductWithConstraint}
                                            .with(constraintStyles())
                                            .likelihoodOf("ModelConfig", "data")
                                            .within(1e-10)),
                         roundTripName);

// A continuous data axis that is unbounded on one side leaves that bound out (HS3 0.3: null is not defined for bounds),
// and an absent bound is read as unbounded.
TEST(RooFitHS3, UnboundedDataAxes)
{
   RooRealVar x{"x", "x", 1.0, 0.0, RooNumber::infinity()};
   RooDataSet ds{"ds", "ds", RooArgSet{x}};
   for (double val : {1.0, 2.0, 30.0}) {
      x.setVal(val);
      ds.add(RooArgSet{x});
   }
   RooWorkspace ws;
   ws.import(ds, RooFit::Silence());

   EXPECT_TRUE(matches(Doc{ws}, "data/ds/axes/0", R"({"min": 0.0, "max": "<absent>"})"));

   RooWorkspace ws2;
   ws2.factory("y[0]");
   importInto(ws2, document("0.3", {R"("data": [{"name": "d", "type": "unbinned", "axes": [{"name": "y", "max": 5.0}],
                                                 "entries": [[-100.0], [1.0]]}])"}));
   auto *y = object<RooRealVar>(ws2, "y");
   EXPECT_TRUE(RooNumber::isInfinite(y->getMin()));
   EXPECT_DOUBLE_EQ(y->getMax(), 5.0);
   EXPECT_DOUBLE_EQ(static_cast<RooRealVar *>(ws2.data("d")->get(0)->find("y"))->getVal(), -100.0);
}

// HS3 0.3 calls the location parameter of landau_dist "mu". Older documents with "mean" are still read.
TEST(RooFitHS3, LandauLocationKey)
{
   RooWorkspace ws;
   ws.factory("Landau::landau(x[0, 10], loc[5, 0, 10], sigma[1.0, 0.1, 10])");
   const Doc doc{ws};
   EXPECT_TRUE(matches(doc, "distributions/landau", R"({"mu": "loc", "mean": "<absent>"})"));
   EXPECT_THAT(doc.json(), Not(HasSubstr(R"("mean":"loc")")));

   for (std::string const version : {"0.2", "0.3"}) {
      Imported ws2{document(
         version,
         {domain("d", {axes({"x"}, 0, 10)}),
          distributions({R"({"name": "landau", "type": "landau_dist", "x": "x", "mean": 5.0, "sigma": 1.0})"})})};
      EXPECT_NE(dynamic_cast<RooLandau *>(ws2.pdf("landau")), nullptr);
      EXPECT_EQ(ws2.warnings.find("renamed to \"mu\"") != std::string::npos, version == "0.3") << ws2.warnings;
   }
}

// The boolean literals TRUE and FALSE and the exact comparisons === and !== of the HS3 expression language are
// translated on import.
TEST(RooFitHS3, ExpressionBuiltins)
{
   const std::string function = R"json({"name": "f", "type": "generic",
      "expression": "ifelse(x === 1 && TRUE, 2, ifelse(x !== 2 || FALSE, 3, 4))"})json";
   Imported ws{document("0.3", {domain("d", {axes({"x"}, -5, 5)}), functions({function})})};
   auto *f = object<RooAbsReal>(ws, "f");
   EXPECT_EQ(ws.arg("TRUE"), nullptr);
   EXPECT_EQ(ws.arg("FALSE"), nullptr);
   for (auto [x, expected] : {std::pair{1.0, 2.0}, std::pair{2.0, 4.0}, std::pair{3.0, 3.0}}) {
      ws.var("x")->setVal(x);
      EXPECT_DOUBLE_EQ(f->getVal(), expected) << "x=" << x;
   }
}

// The names of the HS3 expression language are reserved, and expressions must not refer to distributions (HS3 0.3).
TEST(RooFitHS3, ReservedNamesAndDistributionReferences)
{
   expectImportThrowsWithError(
      document("0.3", {domain("d", {axes({"x"}, 0, 1)}),
                       functions({R"({"name": "exp", "type": "generic", "expression": "2*x"})"})}),
      "the function 'exp' has a name that the HS3 expression language reserves");
   expectImportThrowsWithError(
      document("0.3",
               {domain("d", {axes({"x"}, -5, 5)}),
                distributions({R"({"name": "g", "type": "gaussian_dist", "x": "x", "mean": 0.0, "sigma": 1.0})"}),
                functions({R"({"name": "f", "type": "generic", "expression": "2*g"})"})}),
      "refers to the distribution 'g'");

   // The export refuses expressions that refer to a pdf.
   RooWorkspace ws;
   ws.factory("Gaussian::g(x[-5, 5], 0.0, 1.0)");
   ws.factory("expr::f('2*g', g)");
   expectExportThrowsWithError(ws, "refers to the pdf 'g'");

   // Also for a TF1 with a pdf as parameter.
   RooRealVar y{"y", "y", 0.5, -1.0, 1.0};
   RooGaussian gy{"gy", "gy", y, RooFit::RooConst(0.0), RooFit::RooConst(1.0)};
   TF1 tf1{"tf1", "[0]*x", -1.0, 1.0};
   RooTFnBinding binding{"binding", "binding", &tf1, RooArgList{y}, RooArgList{gy}};
   RooWorkspace ws2;
   ws2.import(binding, RooFit::Silence());
   expectExportThrowsWithError(ws2, "refers to the pdf 'gy'");
}

// Point data is read as a RooDataSet with one entry of one variable, and such a data set is written as point data.
TEST(RooFitHS3, PointData)
{
   const std::string yAxis = axes({"y"}, -10, 10);
   const std::string point = R"({"name": "obs", "type": "point", "value": 1.5, "axes": [)" + yAxis + "]}";
   Imported ws{document("0.3", {domain("d", {yAxis}), section("data", {point})})};
   auto *data = object<RooDataSet>(ws, "obs");
   ASSERT_EQ(data->numEntries(), 1);
   ASSERT_EQ(data->get()->size(), 1u);
   EXPECT_DOUBLE_EQ(static_cast<RooRealVar *>(data->get(0)->find("y"))->getVal(), 1.5);

   Doc exported{ws};
   EXPECT_TRUE(matches(exported, "data/obs", R"({"type": "point", "value": 1.5, "axes": [{"name": "y"}]})"));
   EXPECT_EQ(reexport(exported), exported.json());

   // A data set with more than one entry stays unbinned data.
   RooRealVar x{"x", "x", 0.0, 1.0};
   RooDataSet two{"two", "two", RooArgSet{x}};
   two.add(RooArgSet{x});
   two.add(RooArgSet{x});
   RooWorkspace ws3;
   ws3.import(two, RooFit::Silence());
   EXPECT_TRUE(matches(Doc{ws3}, "data/two", R"({"type": "unbinned"})"));
}

// A number given in-line in the "data" of a likelihood, or point data without axes, is the value of the single
// variate of the paired distribution (HS3 0.3).
TEST(RooFitHS3, DataWithoutAxes)
{
   auto makeJSON = [](std::string const &distribution, std::string const &data, std::string const &dataSets) {
      return document(
         "0.3",
         {domain("d", {axes({"x", "y", "mu"}, -5, 5)}),
          distributions({R"({"name": "gx", "type": "gaussian_dist", "x": "x", "mean": "mu", "sigma": 1.0})",
                         R"({"name": "gy", "type": "gaussian_dist", "x": "y", "mean": "mu", "sigma": 1.0})",
                         R"({"name": "gxy", "type": "product_dist", "factors": ["gx", "gy"]})"}),
          section("data", {dataSets}),
          R"("likelihoods": [{"name": "lh", "distributions": [")" + distribution + R"("], "data": [)" + data + "]}]",
          R"("analyses": [{"name": "ana", "likelihood": "lh", "domains": ["d"], "parameters_of_interest": ["mu"]}])"});
   };

   {
      // A number in-line.
      Imported ws{makeJSON("gx", "1.5", "")};
      auto *data = object<RooAbsData>(ws, "gx_data");
      ASSERT_EQ(data->numEntries(), 1);
      EXPECT_DOUBLE_EQ(static_cast<RooRealVar *>(data->get(0)->find("x"))->getVal(), 1.5);
   }
   {
      // Point data without axes.
      Imported ws{makeJSON("gy", R"("obs")", R"({"name": "obs", "type": "point", "value": -2.0})")};
      auto *data = object<RooAbsData>(ws, "obs");
      ASSERT_EQ(data->numEntries(), 1);
      EXPECT_DOUBLE_EQ(static_cast<RooRealVar *>(data->get(0)->find("y"))->getVal(), -2.0);
   }
   // A distribution with two variates has no single variate to take the value.
   expectImportThrowsWithError(makeJSON("gxy", "1.5", ""), "can't be determined");
}

// HS3 0.3: fixed axes only in domains, point data named after the paired variate, and one normalization range for a
// variate that has both an axis in the paired data set and one in the domains of the analysis.
TEST(RooFitHS3, DataAxesAgainstDomains)
{
   auto fixedAxis = [](std::string const &entry) {
      return R"("data": [{"name": "d", "type": "unbinned", "axes": [{"name": "x", "value": 1.0}], "entries": [[)" +
             entry + "]]}]";
   };
   expectImportThrowsWithError(document("0.3", {fixedAxis("1.0")}), "must not have a fixed \"value\"");
   {
      // Older versions of RooFit wrote the constant observables of data sets with a fixed "value": the data set is
      // read as before, without a range from the axis.
      Imported ws{document("0.2", {defaultDomain({axes({"x"}, 0, 5)}), fixedAxis("2.5")})};
      auto *data = object<RooAbsData>(ws, "d");
      ASSERT_EQ(data->numEntries(), 1);
      EXPECT_DOUBLE_EQ(static_cast<RooRealVar *>(data->get(0)->find("x"))->getVal(), 2.5);
   }

   auto makeJSON = [](std::string const &version, std::string const &domainAxis, std::string const &data) {
      return document(
         version,
         {domain("ana_domain", {domainAxis, axes({"mu"}, -5, 5)}),
          distributions({R"({"name": "g", "type": "gaussian_dist", "x": "x", "mean": "mu", "sigma": 1.0})"}),
          section("data", {data}), R"("likelihoods": [{"name": "lh", "distributions": ["g"], "data": ["obs"]}])",
          R"("analyses": [{"name": "ana", "likelihood": "lh", "domains": ["ana_domain"],
                           "parameters_of_interest": ["mu"]}])"});
   };
   const std::string unbinned = R"({"name": "obs", "type": "unbinned", "axes": [{"name": "x", "min": 0, "max": 10}],
                                    "entries": [[1.0], [2.0]]})";

   {
      Imported ws{makeJSON("0.3", R"({"name": "x", "min": 0, "max": 10})", unbinned)};
   }
   expectImportThrowsWithError(makeJSON("0.3", R"({"name": "x", "min": 0, "max": 5})", unbinned),
                               "the axis 'x' of the data set 'obs' has the range");
   expectImportThrowsWithError(makeJSON("0.3", R"({"name": "x", "min": 0})", unbinned),
                               "the axis 'x' of the data set 'obs' has the range");
   {
      // Older documents are read with a warning.
      Imported ws{makeJSON("0.2", R"({"name": "x", "min": 0, "max": 5})", unbinned)};
      EXPECT_THAT(ws.warnings, HasSubstr("the axis 'x' of the data set 'obs' has the range"));
   }

   expectImportThrowsWithError(
      makeJSON("0.3", R"({"name": "x", "min": 0, "max": 10})",
               R"({"name": "obs", "type": "point", "value": 1.0, "axes": [{"name": "y", "min": 0, "max": 10}]})"),
      "is not the variate 'x' of the distribution 'g'");
}

// HS3 0.3: the "constraints" of a HistFactory modifier add no term; every likelihood with the channel must list each
// named distribution exactly once.
TEST(RooFitHS3, HistFactoryConstraintsListedOnce)
{
   auto makeJSON = [](std::string const &version, std::string const &aux) {
      const std::string obsAxis = R"({"name": "obs_c", "min": 0.0, "max": 2.0, "nbins": 2})";
      const std::string normsys =
         R"({"type": "normsys", "parameters": ["alpha_a"], "constraints": ["c_a"], "data": {"lo": 0.9, "hi": 1.1}})";
      return document(
         version,
         {domain("d", {axes({"alpha_a"}, -5, 5), R"({"name": "nom_a", "value": 0.0})"}),
          distributions({channel("model_c", {sample("s", R"({"contents": [10.0, 20.0]})", normsys)},
                                 polyExpInterpolation, obsAxis),
                         R"({"name": "c_a", "type": "gaussian_dist", "x": "nom_a", "mean": "alpha_a", "sigma": 1.0})"}),
          R"("data": [{"name": "obs", "type": "binned", "axes": [)" + obsAxis + R"(], "contents": [11, 19]}])",
          R"("likelihoods": [{"name": "lh", "distributions": ["model_c"], "data": ["obs"], "aux_distributions": [)" +
             aux + "]}]",
          R"("analyses": [{"name": "ana", "likelihood": "lh", "domains": ["d"]}])"});
   };

   {
      Imported ws{makeJSON("0.3", R"("c_a")")};
   }
   expectImportThrowsWithError(makeJSON("0.3", ""), "the constraint 'c_a' named by a modifier of a HistFactory "
                                                    "channel contributes 0 times to the likelihood 'lh'");
   expectImportThrowsWithError(makeJSON("0.3", R"("c_a", "c_a")"), "contributes 2 times");
   {
      // Older documents get the named constraints added, as before.
      Imported ws{makeJSON("0.2", "")};
      EXPECT_NE(ws.pdf("c_a"), nullptr);
   }
}

namespace {

// Parts of the documents with auxiliary measurements of the parameters `theta` and `eta`, whose observed values 0.2
// and -0.3 are the global observables `g1` and `g2`.

// The constraints as the array distribution "aux".
const std::string auxArray = R"({"name": "aux", "type": "array_dist", "component_type": "gaussian_dist",
   "default_parameters": {"sigma": 1.0}, "components": [{"x": "g1", "mean": "theta"}, {"x": "g2", "mean": "eta"}]})";

// The combined distribution "model" of the regions "physics" and "auxiliary", which is "aux".
const std::string auxRegionsModel = R"({"name": "model", "type": "combined_dist", "columns": ["channel"],
   "components": [{"key": ["physics"], "dist": "physics"}, {"key": ["auxiliary"], "dist": "aux"}]})";

// A data set with the observed values, and one without entries for a region without data.
std::string auxMeasurement(std::string const &name)
{
   return R"({"name": ")" + name + R"(", "type": "unbinned", "axes": [)" + axes({"g1", "g2"}, -8, 8) +
          R"(], "entries": [[0.2, -0.3]]})";
}
const std::string auxNoData = R"({"name": "empty", "type": "unbinned", "axes": [], "entries": []})";

// The analysis "ModelConfig" of the likelihood "likelihood" with the domain "domain".
const std::string modelConfigAnalysis = R"("analyses": [{"name": "ModelConfig", "likelihood": "likelihood",
   "domains": ["domain"], "parameters_of_interest": ["mu"]}])";

// The domain "domain" with the given axes of `x` and `mu`, and the default values of the parameters.
std::string auxDomainAndValues(std::string const &x, std::string const &mu)
{
   const std::string channels = R"({"name": "channel", "categories": ["physics", "auxiliary"]})";
   return join(
      {domain("domain", {channels, x, mu, axes({"theta", "eta"}, -3, 3), axes({"g1", "g2"}, -8, 8)}),
       defaultValues({values({{"mu", 1.0}, {"theta", 0.0}, {"eta", 0.0}}), constValues({{"g1", 0.2}, {"g2", -0.3}})})});
}

// A physics region that is no HistFactory channel: a Gaussian of `x` around the function "mean", mu+theta+eta.
const std::string gaussianPhysics =
   R"({"name": "physics", "type": "gaussian_dist", "x": "x", "mean": "mean", "sigma": 1.0})";
const std::string gaussianPhysicsMean =
   R"("functions": [{"name": "mean", "type": "generic_function", "expression": "mu+theta+eta"}])";

// The points (mu, theta, eta) at which the likelihoods of the models with a Gaussian physics region are compared.
const std::vector<std::array<double, 3>> auxPoints{{1.0, 0.0, 0.0}, {0.5, 0.4, -0.5}, {1.2, -0.4, 0.5}};

void auxSetPoint(RooWorkspace &ws, std::array<double, 3> const &point)
{
   ws.var("mu")->setVal(point[0]);
   ws.var("theta")->setVal(point[1]);
   ws.var("eta")->setVal(point[2]);
}

std::vector<double> relativeToFirst(std::vector<double> values)
{
   const double first = values.front();
   for (double &value : values) {
      value -= first;
   }
   return values;
}

// The likelihood at the points, relative to the first one. The parameters stay at the last point.
std::vector<double> auxScan(RooWorkspace &ws, RooAbsReal &nll)
{
   std::vector<double> out;
   for (auto const &point : auxPoints) {
      auxSetPoint(ws, point);
      out.push_back(nll.getVal());
   }
   return relativeToFirst(out);
}

// The same from the definition of the model, for the data -0.4, 0.2 and 0.7 of `x`: -log of the three Gaussians of
// `x`, normalized over [-5, 5], and of the two constraints. The constraints are normalized over [-8, 8], which is all
// of the Gaussian within the precision.
std::vector<double> auxExpectedScan()
{
   std::vector<double> out;
   for (auto const &point : auxPoints) {
      const double mean = point[0] + point[1] + point[2];
      double nll = 0.5 * (0.2 - point[1]) * (0.2 - point[1]) + 0.5 * (-0.3 - point[2]) * (-0.3 - point[2]);
      for (double x : {-0.4, 0.2, 0.7}) {
         const double norm = 0.5 * (std::erf((5.0 - mean) / std::sqrt(2.0)) - std::erf((-5.0 - mean) / std::sqrt(2.0)));
         nll += 0.5 * (x - mean) * (x - mean) + std::log(norm);
      }
      out.push_back(nll);
   }
   return relativeToFirst(out);
}

} // namespace

// A constraint contributes to a likelihood through the data that it is paired with: a component of a combined
// distribution without data is no contribution. Reported for auxiliary measurements that are a region of the model,
// so that they are part of the sampling distribution, and also a term of their own (ROOT PR #24201).
TEST(RooFitHS3, HistFactoryConstraintContributionsCountData)
{
   // The constraints of the channel "physics" are the array "aux", which is also the component "auxiliary" of the
   // combined distribution "model".
   auto makeJSON = [](std::string const &auxiliaryData, std::string const &measurement) {
      const std::string xAxis = R"({"name": "x", "min": 0.0, "max": 2.0, "nbins": 2})";
      const std::string modifiers = R"({"type": "normfactor", "parameters": ["mu"]},
         {"name": "s1", "type": "normsys", "parameters": ["theta"], "data": {"lo": 0.9, "hi": 1.1},
          "constraints": ["aux"]},
         {"name": "s2", "type": "normsys", "parameters": ["eta"], "data": {"lo": 0.8, "hi": 1.2},
          "constraints": ["aux"]})";
      const std::string physics =
         channel("physics", {sample("signal", R"({"contents": [10.0, 15.0]})", modifiers)}, polyExpInterpolation);
      const std::string physicsData =
         R"({"name": "physics_data", "type": "binned", "axes": [)" + xAxis + R"(], "contents": [10.0, 15.0]})";
      const std::string combData = R"({"name": "combData", "type": "combined", "columns": ["channel"],
         "components": [{"key": ["physics"], "dataset": "physics_data"})" +
                                   auxiliaryData + "]}";
      const std::string likelihood = std::string{R"({"name": "likelihood", "distributions": ["model")"} +
                                     (measurement.empty() ? "" : R"(, "aux")") + R"(], "data": ["combData")" +
                                     measurement + "]}";
      return document(
         "0.3",
         {distributions({physics, auxArray, auxRegionsModel}),
          section("data", {physicsData, auxNoData, auxMeasurement("one"), combData, auxMeasurement("measurement")}),
          section("likelihoods", {likelihood}), modelConfigAnalysis, auxDomainAndValues(xAxis, axes({"mu"}, 0, 3))});
   };
   const std::string emptyRegion = R"(, {"key": ["auxiliary"], "dataset": "empty"})";
   const std::string observedRegion = R"(, {"key": ["auxiliary"], "dataset": "one"})";
   const std::string measurement = R"(, "measurement")";

   // A region with a data set that has no entries, and the measurement as a term of its own: one contribution.
   // Listing the constraint twice is probably not intended, so there is a warning.
   EXPECT_THAT(Imported{makeJSON(emptyRegion, measurement)}.warnings,
               HasSubstr("the constraint 'aux' is a term of the likelihood 'likelihood' and also a component of "
                         "the combined distribution 'model' without data"));
   // The same without any data component for the region, which is also reported.
   EXPECT_THAT(Imported{makeJSON("", measurement)}.warnings,
               HasSubstr("the component 'aux' of the combined distribution 'model' has no component in the data "
                         "set 'combData'"));
   // The measurement twice: as the data of the region and as a term of its own.
   expectImportThrowsWithError(makeJSON(observedRegion, measurement), "the constraint 'aux' named by a modifier of a "
                                                                      "HistFactory channel contributes 2 times");
   // No measurement at all: a region without data is no observation.
   expectImportThrowsWithError(makeJSON(emptyRegion, ""), "contributes 0 times to the likelihood 'likelihood', but it "
                                                          "must contribute exactly once (it is a component of the "
                                                          "combined distribution 'model' without data");
}

// An auxiliary measurement that is a region of the model, so that it is part of the sampling distribution: a
// component of the combined distribution that consists of constraint terms, with the observed values in the global
// observables. It is written once, as that component with a data set of the observed values, and never through its
// factors or the components of its array (reported on ROOT PR #24201).
TEST(RooFitHS3, AuxiliaryMeasurementAsRegionOfTheModel)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::ERROR);

   const std::string product = R"({"name": "aux", "type": "product_dist", "factors": ["c1", "c2"]},
      {"name": "c1", "type": "gaussian_dist", "x": "g1", "mean": "theta", "sigma": 1.0},
      {"name": "c2", "type": "gaussian_dist", "x": "g2", "mean": "eta", "sigma": 1.0})";
   auto makeJSON = [](std::string const &aux) {
      const std::string xAxis = axes({"x"}, -5, 5);
      const std::string physicsData = R"({"name": "physics_data", "type": "unbinned", "axes": [)" + xAxis +
                                      R"(], "entries": [[-0.4], [0.2], [0.7]]})";
      const std::string combData = R"({"name": "combData", "type": "combined", "columns": ["channel"], "components": [
         {"key": ["physics"], "dataset": "physics_data"}, {"key": ["auxiliary"], "dataset": "empty"}]})";
      const std::string likelihood =
         R"({"name": "likelihood", "distributions": ["model", "aux"], "data": ["combData", "measurement"]})";
      return document("0.3", {distributions({gaussianPhysics, aux, auxRegionsModel}), gaussianPhysicsMean,
                              section("data", {physicsData, auxNoData, combData, auxMeasurement("measurement")}),
                              section("likelihoods", {likelihood}), modelConfigAnalysis,
                              auxDomainAndValues(xAxis, axes({"mu"}, -3, 3))});
   };

   // The likelihood at three points, relative to the first one, with the constraint terms wherever they are.
   auto scan = [](RooWorkspace &ws) {
      auto *mc = dynamic_cast<RooStats::ModelConfig *>(ws.obj("ModelConfig"));
      EXPECT_NE(mc, nullptr);
      std::vector<double> out;
      if (!mc || !ws.data("combData")) {
         return out;
      }
      RooArgSet none;
      std::unique_ptr<RooAbsReal> nll{mc->GetPdf()->createNLL(
         *ws.data("combData"),
         RooFit::GlobalObservables(mc->GetGlobalObservables() ? *mc->GetGlobalObservables() : none),
         RooFit::ExternalConstraints(mc->GetExternalConstraints() ? *mc->GetExternalConstraints() : none),
         RooFit::Offset(false))};
      out = auxScan(ws, *nll);
      auxSetPoint(ws, auxPoints.front());
      return out;
   };
   // The same from the document itself.
   const std::vector<double> expected = auxExpectedScan();

   for (std::string const &aux : {auxArray, product}) {
      for (bool withData : {false, true}) {
         for (bool asChannels : {false, true}) {
            SCOPED_TRACE(std::string(aux == auxArray ? "array" : "product") + (withData ? ", data" : ", aux") +
                         (asChannels ? ", read as channels" : ", read as constraints"));
            ScopedConfig config{{exportConstraintData(withData), importConstraintsAsChannels(asChannels)}};
            std::string previous;
            try {
               auto ws = std::make_unique<Imported>(makeJSON(aux));
               for (int generation = 0; generation < 3; ++generation) {
                  const std::vector<double> values = scan(*ws);
                  ASSERT_EQ(values.size(), expected.size());
                  for (std::size_t i = 0; i < values.size(); ++i) {
                     EXPECT_NEAR(values[i], expected[i], 1e-9) << "generation " << generation << ", point " << i;
                  }
                  const std::string exported = toJSON(*ws);
                  // Read as components, the group stays one: the only term of the likelihood is the model.
                  if (asChannels) {
                     EXPECT_TRUE(matches(Doc{exported}, "likelihoods",
                                         R"([{"distributions": ["model"], "aux_distributions": "<absent>"}])"));
                  }
                  if (!previous.empty()) {
                     EXPECT_EQ(exported, previous) << "the document changes after the first export";
                  }
                  previous = exported;
                  ws = std::make_unique<Imported>(exported);
               }
            } catch (std::exception const &e) {
               ADD_FAILURE() << e.what() << "\n" << previous;
            }
         }
      }
   }
}

// The pdf of a model stays the sampling distribution of its auxiliary measurements over a round trip, also if its
// channels are no products: such a channel is wrapped in a product with its constraint terms, from which RooFit
// takes them when it builds a likelihood (reported on ROOT PR #24201).
TEST(RooFitHS3, ConstraintsStayInTheSamplingDistribution)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::ERROR);

   RooWorkspace ws{"original"};
   for (const char *definition :
        {"sum::mean(mu[1,0,3],theta[0,-3,3],eta[0,-3,3])", "Gaussian::physics(x[0,-5,5],mean,1)",
         "Gaussian::c1(g1[0.2,-8,8],theta,1)", "Gaussian::c2(g2[-0.3,-8,8],eta,1)", "PROD::aux(c1,c2)",
         "SIMUL::model(channel[physics,auxiliary],physics=physics,auxiliary=aux)"}) {
      ASSERT_NE(ws.factory(definition), nullptr) << definition;
   }
   ws.var("g1")->setConstant(true);
   ws.var("g2")->setConstant(true);
   RooArgSet columns{*ws.var("x"), *ws.cat("channel")};
   RooDataSet data{"data", "data", columns};
   ws.cat("channel")->setLabel("physics");
   for (double value : {-0.4, 0.2, 0.7}) {
      ws.var("x")->setVal(value);
      data.add(columns);
   }
   ws.import(data, RooFit::Silence());
   RooStats::ModelConfig mc{"ModelConfig", &ws};
   mc.SetPdf(*ws.pdf("model"));
   mc.SetObservables(columns);
   mc.SetParametersOfInterest(RooArgSet{*ws.var("mu")});
   mc.SetNuisanceParameters(RooArgSet{*ws.var("theta"), *ws.var("eta")});
   mc.SetGlobalObservables(RooArgSet{*ws.var("g1"), *ws.var("g2")});
   ws.import(mc);

   // The likelihood as RooFit builds it from the pdf and the global observables alone, relative to the first point.
   auto scan = [](RooWorkspace &w) {
      auto *config = dynamic_cast<RooStats::ModelConfig *>(w.obj("ModelConfig"));
      std::unique_ptr<RooAbsReal> nll{config->GetPdf()->createNLL(
         *w.data("data"), RooFit::GlobalObservables(*config->GetGlobalObservables()), RooFit::Offset(false))};
      return auxScan(w, *nll);
   };
   // The same from the definition of the model, with the constraints.
   const std::vector<double> expected = auxExpectedScan();

   for (bool constraintData : {false, true}) {
      SCOPED_TRACE(constraintData ? "with constraint data" : "without constraint data");
      ScopedConfig config{{exportConstraintData(constraintData)}};
      std::string previous;
      std::string exported = toJSON(ws);
      for (int generation = 1; generation <= 3; ++generation) {
         Imported imported{exported};
         auto *mc2 = object<RooStats::ModelConfig>(imported, "ModelConfig");
         // The global observables are observables of the pdf, and nothing is left outside of it.
         ASSERT_NE(mc2->GetGlobalObservables(), nullptr);
         EXPECT_EQ(mc2->GetGlobalObservables()->size(), 2u);
         EXPECT_TRUE(mc2->GetPdf()->dependsOn(*imported.var("g1"))) << exported;
         EXPECT_TRUE(mc2->GetPdf()->dependsOn(*imported.var("g2"))) << exported;
         EXPECT_TRUE(!mc2->GetExternalConstraints() || mc2->GetExternalConstraints()->empty()) << exported;
         // RooFit finds the constraints in the pdf.
         const std::vector<double> values = scan(imported);
         for (std::size_t i = 0; i < expected.size(); ++i) {
            EXPECT_NEAR(values[i], expected[i], 1e-9) << "generation " << generation << ", point " << i;
         }
         previous = exported;
         exported = toJSON(imported);
         if (generation > 1) {
            EXPECT_EQ(exported, previous) << "the document changes after the second export";
         }
      }
   }
}

// A Gaussian constraint is written with its global observable as "x". Whether that is the x or the mean of the
// RooGaussian must not depend on the order in which the pdfs are written: here, the constraint comes before the
// channel that names it, and a product keeps it from becoming a component of an array (reported on ROOT PR #24201).
TEST(RooFitHS3, GaussianConstraintObservableBeforeItsChannel)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::ERROR);

   const std::string xAxis = R"({"name": "x", "min": 0.0, "max": 2.0, "nbins": 2})";
   const std::string modifiers = R"({"type": "normfactor", "parameters": ["mu"]},
      {"name": "s", "type": "normsys", "parameters": ["theta"], "data": {"lo": 0.9, "hi": 1.1},
       "constraints": ["constraint"]})";
   const std::string likelihood =
      R"({"name": "likelihood", "distributions": ["physics"], "data": ["data"], "aux_distributions": ["constraint"]})";
   Imported ws{document(
      "0.3",
      {distributions(
          {channel("physics", {sample("signal", R"({"contents": [10.0, 15.0]})", modifiers)}, polyExpInterpolation),
           R"({"name": "constraint", "type": "gaussian_dist", "x": "theta", "mean": "g", "sigma": 1.0})"}),
       section("data", {R"({"name": "data", "type": "binned", "axes": [)" + xAxis + R"(], "contents": [10.0, 15.0]})"}),
       section("likelihoods", {likelihood}), modelConfigAnalysis,
       domain("domain", {xAxis, axes({"mu"}, 0, 3), axes({"theta"}, -3, 3), axes({"g"}, -5, 5)}),
       defaultValues({values({{"mu", 1.0}, {"theta", 0.0}}), constValues({{"g", 0.0}})})})};
   ASSERT_NE(ws.factory("PROD::keep_scalar(constraint)"), nullptr);
   auto *mc = object<RooStats::ModelConfig>(ws, "ModelConfig");
   mc->SetGlobalObservables(RooArgSet{*ws.var("g")});

   std::string exported;
   EXPECT_NO_THROW(exported = toJSON(ws, {exportConstraintData(true)}));
   ASSERT_FALSE(exported.empty());

   EXPECT_TRUE(matches(Doc{exported}, "distributions/constraint", R"({"x": "g", "mean": "theta"})"));

   Imported ws2{exported};
}

// EXPERIMENTAL import of constraints as channels: an auxiliary observation that is a term of two likelihoods becomes
// a component of both of their combined data sets, each of which needs a data set of its own.
TEST(RooFitHS3, AuxiliaryObservationSharedByTwoLikelihoods)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::ERROR);

   const std::string physicsData = R"(
      {"name": "observed_physics", "type": "unbinned", "axes": [{"name": "x", "min": -5.0, "max": 5.0}],
       "entries": [[-0.4], [0.2], [0.7]]},
      {"name": "asimov_physics", "type": "unbinned", "axes": [{"name": "x", "min": -5.0, "max": 5.0}],
       "entries": [[1.0]]},
      {"name": "observed", "type": "combined", "columns": ["channel"],
       "components": [{"key": ["physics"], "dataset": "observed_physics"}]},
      {"name": "asimov", "type": "combined", "columns": ["channel"],
       "components": [{"key": ["physics"], "dataset": "asimov_physics"}]})";
   const std::string likelihoods = R"(
      {"name": "likelihood_observed", "distributions": ["model", "aux"], "data": ["observed", "measurement"]},
      {"name": "likelihood_asimov", "distributions": ["model", "aux"], "data": ["asimov", "measurement"]})";
   const std::string analyses = R"(
      {"name": "ModelConfig_observed", "likelihood": "likelihood_observed", "domains": ["domain"],
       "parameters_of_interest": ["mu"]},
      {"name": "ModelConfig_asimov", "likelihood": "likelihood_asimov", "domains": ["domain"],
       "parameters_of_interest": ["mu"]})";
   const std::string json = document(
      "0.3", {distributions({gaussianPhysics, auxArray, auxRegionsModel}), gaussianPhysicsMean,
              section("data", {physicsData, auxMeasurement("measurement")}), section("likelihoods", {likelihoods}),
              section("analyses", {analyses}), auxDomainAndValues(axes({"x"}, -5, 5), axes({"mu"}, -3, 3))});

   Imported ws{json, {importConstraintsAsChannels(true)}};
   // Both combined data sets have the observation as the entry of the auxiliary channel.
   for (const char *name : {"observed", "asimov"}) {
      auto *data = object<RooAbsData>(ws, name);
      std::unique_ptr<RooAbsData> slice{data->reduce("channel==channel::auxiliary")};
      EXPECT_EQ(slice->numEntries(), 1) << name;
   }
}

// HS3 0.3: the named constraints of a staterror are authoritative; nothing is derived from the errors of the samples.
TEST(RooFitHS3, StatErrorFromDocument)
{
   // A sample error of 10% would give gamma_0 the range [0, 1.5] if it were derived from the errors.
   const std::string obsAxis = R"({"name": "obs_c", "min": 0.0, "max": 2.0, "nbins": 2})";
   const std::string staterror =
      R"({"type": "staterror", "parameters": ["gamma_0", "gamma_1"], "constraints": ["c_0", null]})";
   const std::vector<std::string> sections{
      domain("d", {axes({"gamma_0", "gamma_1"}, 0, 5), R"({"name": "nom_gamma_0", "value": 1.0})"}),
      distributions(
         {channel("model_c", {sample("s", R"({"contents": [10.0, 20.0], "errors": [1.0, 0.0]})", staterror)}, "",
                  obsAxis),
          R"({"name": "c_0", "type": "gaussian_dist", "x": "nom_gamma_0", "mean": "gamma_0", "sigma": 0.1})"}),
      R"("data": [{"name": "obs", "type": "binned", "axes": [)" + obsAxis + R"(], "contents": [11, 19]}])",
      R"("likelihoods": [{"name": "lh", "distributions": ["model_c"], "data": ["obs"], "aux_distributions": ["c_0"]}])",
      R"("analyses": [{"name": "ana", "likelihood": "lh", "domains": ["d"]}])"};
   Imported ws{document("0.3", sections)};
   // The ranges come from the domain, and the bin without uncertainty is not fixed, since the document doesn't say so.
   EXPECT_DOUBLE_EQ(ws.var("gamma_0")->getMax(), 5.0);
   EXPECT_FALSE(ws.var("gamma_0")->isConstant());
   EXPECT_FALSE(ws.var("gamma_1")->isConstant());

   // Older documents get the constrained gammas configured from the errors, as before. The unconstrained gamma of the
   // bin without uncertainty is left free in any version, see HistFactoryUnconstrainedGammasStayFree.
   Imported ws2{document("0.2", sections)};
   EXPECT_DOUBLE_EQ(ws2.var("gamma_0")->getMax(), 1.5);
   EXPECT_FALSE(ws2.var("gamma_1")->isConstant());

   // RooFit writes a fixed gamma as fixed, with a null constraint, so the configuration survives a round trip.
   ws2.var("gamma_1")->setConstant(true);
   const std::string exported = toJSON(ws2);
   Imported ws3{exported};
   EXPECT_TRUE(ws3.var("gamma_1")->isConstant()) << exported;
   EXPECT_FALSE(ws3.var("gamma_0")->isConstant()) << exported;
   EXPECT_DOUBLE_EQ(ws3.var("gamma_0")->getMax(), 1.5) << exported;
}

// A ModelConfig without data is written as a pure model, a likelihood with "data": null, and read back without a data
// set. Documents before 0.3 used the number 0 for this.
TEST(RooFitHS3, PureModelLikelihood)
{
   RooWorkspace ws;
   ws.factory("Gaussian::pdf(x[0, -5, 5], mu[0, -5, 5], 1.0)");
   addModelConfig(ws, "ModelConfig", "pdf", "x", "mu");

   Doc json{ws};
   EXPECT_TRUE(matches(json, "likelihoods/0", R"({"data": null, "distributions": ["pdf"]})"));

   Imported ws2{json};
   auto *mc2 = object<RooFit::ModelConfig>(ws2, "ModelConfig");
   EXPECT_EQ(mc2->GetPdf(), ws2.pdf("pdf"));
   EXPECT_TRUE(ws2.allData().empty());
   EXPECT_EQ(toJSON(ws2), json.json());

   // The placeholder of older documents.
   const std::string legacy =
      replaced(replaced(json, R"("data":null)", R"("data":[0])"), R"("hs3_version":"0.3")", R"("hs3_version":"0.2")");
   EXPECT_EQ(reexport(legacy), json.json());

   // Only "data": [0] is the placeholder. A 0 in a longer array is an observation, also before 0.3.
   Imported ws4{document(
      "0.2", {domain("d", {axes({"x", "y"}, -5, 5)}),
              distributions({R"({"name": "gx", "type": "gaussian_dist", "x": "x", "mean": 0.0, "sigma": 1.0})",
                             R"({"name": "gy", "type": "gaussian_dist", "x": "y", "mean": 0.0, "sigma": 1.0})"}),
              R"("likelihoods": [{"name": "lh", "distributions": ["gx", "gy"], "data": [0, 0]}])",
              R"("analyses": [{"name": "ana", "likelihood": "lh", "domains": ["d"]}])"})};
   // The observations of the two terms, merged into the data set of the RooSimultaneous of the terms.
   ASSERT_NE(ws4.data("ana_data"), nullptr);
   EXPECT_EQ(ws4.data("ana_data")->numEntries(), 2);
}

namespace {

// An array distribution of two RooLognormal in the parametrization of RooFit, which share the shape parameter.
void buildArrayOfLognormals(RooWorkspace &ws)
{
   RooRealVar nomA{"nom_a", "nom_a", 1.0, 0.1, 10.0};
   RooRealVar nomB{"nom_b", "nom_b", 1.0, 0.1, 10.0};
   RooRealVar a{"a", "a", 1.0, 0.1, 10.0};
   RooRealVar b{"b", "b", 1.0, 0.1, 10.0};
   RooRealVar k{"k", "k", 1.2, 1.01, 3.0};
   RooLognormal lognormalA{"lognormal_a", "lognormal_a", nomA, a, k};
   RooLognormal lognormalB{"lognormal_b", "lognormal_b", nomB, b, k};
   ASSERT_FALSE(lognormalA.useStandardParametrization());
   RooProdPdf array{"constraints", "constraints", {lognormalA, lognormalB}};
   array.setAttribute("hs3_array_dist");
   ws.import(array, RooFit::Silence());
}

} // namespace

// A RooLognormal is by default parametrized by its median and shape parameter, and written as a lognormal_dist of the
// functions log(median) and log(shape). It can be a component of an array distribution like one in the standard
// parametrization, and is read back in the parametrization of RooFit, by the same rule as a distribution on its own.
TEST(RooFitHS3, ArrayDistOfLognormalsInRooFitParametrization)
{
   RooWorkspace ws;
   buildArrayOfLognormals(ws);

   Doc exported{ws};
   EXPECT_EQ(exported["distributions"].num_children(), 1u);
   EXPECT_TRUE(matches(exported, "distributions/constraints", R"({
      "type": "array_dist", "component_type": "lognormal_dist", "default_parameters": {"sigma": "k_lognormal_log"},
      "components": [{"x": "nom_a", "mu": "a_lognormal_log"}, {"mu": "b_lognormal_log"}]})"));
   // The functions are written once each, also the one that both components share.
   EXPECT_EQ(exported["functions"].num_children(), 3u);
   EXPECT_TRUE(matches(exported, "functions/a_lognormal_log", R"json({"expression": "log(a)"})json"));
   EXPECT_TRUE(exported.has("functions/k_lognormal_log"));

   Imported ws2{exported};
   auto *imported = object<RooLognormal>(ws2, "nom_a_constraint");
   EXPECT_FALSE(imported->useStandardParametrization());
   EXPECT_STREQ(imported->getMedian().GetName(), "a");
   EXPECT_STREQ(imported->getShapeK().GetName(), "k");
}

// The components are named after their observables when they are read back. Their product has the same values.
INSTANTIATE_TEST_SUITE_P(
   ArrayDistOfLognormalsInRooFitParametrization, RoundTrip,
   testing::Values(RoundTripModel{"Model", buildArrayOfLognormals}.over({"nom_a", "nom_b"}).withOtherComponents()),
   roundTripName);

// A parameter of a RooFit class that is parametrized differently than its HS3 distribution is written as a function
// that is named after the parameter. A literal constant has no name that could be part of an identifier, so its
// value is spelled as one, with all digits that it takes to read it back unchanged.
TEST(RooFitHS3, FunctionsOfLiteralConstantsHaveValidNames)
{
   const double shape = 1.23456789012;
   RooRealVar x{"x", "x", 1.0, 0.1, 10.0};
   RooRealVar m0{"m0", "m0", 1.0, 0.1, 10.0};
   RooLognormal lognormal{"lognormal", "lognormal", x, m0, RooFit::RooConst(shape)};
   RooRealVar y{"y", "y", 1.0, 0.0, 10.0};
   RooExponential exponential{"exponential", "exponential", y, RooFit::RooConst(-0.1)};

   RooWorkspace ws;
   ws.import(lognormal, RooFit::Silence());
   ws.import(exponential, RooFit::Silence());

   Doc exported{ws};
   ASSERT_EQ(exported["functions"].num_children(), 3u);
   for (auto const &name : exported.names("functions")) {
      EXPECT_TRUE(RooJSONFactoryWSTool::isValidName(name)) << name;
   }
   EXPECT_TRUE(matches(exported, "functions/const_1p23456789012_lognormal_log",
                       R"json({"expression": "log(1.23456789012)"})json"));
   EXPECT_TRUE(matches(exported, "functions/const_m0p1_exponential_inverted", R"json({"expression": "-(-0.1)"})json"));
   EXPECT_TRUE(exported.has("functions/m0_lognormal_log"));

   Imported ws2{exported};
   auto *lognormal2 = object<RooLognormal>(ws2, "lognormal");
   EXPECT_FALSE(lognormal2->useStandardParametrization());
   EXPECT_STREQ(lognormal2->getMedian().GetName(), "m0");
   EXPECT_EQ(lognormal2->getShapeK().getVal(), shape);
   auto *exponential2 = object<RooExponential>(ws2, "exponential");
   EXPECT_FALSE(exponential2->negateCoefficient());
   EXPECT_EQ(exponential2->coefficient().getVal(), -0.1);
   EXPECT_EQ(toJSON(ws2), exported.json());

   // Older versions named the function after the number itself, which is still read.
   Imported ws3{document(
      "0.2", {distributions({R"({"name": "lognormal", "type": "lognormal_dist", "x": "x", "mu": "m0_lognormal_log",
                                 "sigma": "1.2_lognormal_log"})"}),
              functions({R"json({"name": "m0_lognormal_log", "type": "generic", "expression": "log(m0)"})json",
                         R"json({"name": "1.2_lognormal_log", "type": "generic", "expression": "log(1.2)"})json"}),
              defaultDomain({axes({"x", "m0"}, 0.1, 10)}), defaultValues({values({{"x", 1.0}, {"m0", 1.0}})}),
              R"("misc": {"ROOT_internal": {"attributes": {"m0_lognormal_log": {"tags": ["roofit_skip"]},
                                                           "1.2_lognormal_log": {"tags": ["roofit_skip"]}}}})"})};
   auto *lognormal3 = object<RooLognormal>(ws3, "lognormal");
   EXPECT_FALSE(lognormal3->useStandardParametrization());
   EXPECT_EQ(lognormal3->getShapeK().getVal(), 1.2);

   // A parameter can't have a name that would be read as a literal constant.
   RooRealVar reserved{"const_1p5", "const_1p5", 1.5, 1.01, 3.0};
   RooLognormal clash{"clash", "clash", x, m0, reserved};
   RooWorkspace ws4;
   ws4.import(clash, RooFit::Silence());
   expectExportThrowsWithError(ws4, "names of the form 'const_<number>' are reserved");
}

// RooFit has no class for an array distribution: the RooProdPdf that holds its components together is a helper, which
// the importer marks with the attribute "hs3_helper". A helper is no statement of the user about the model. It is
// only written if another object refers to it or if the HS3 object that it stands for is written, and a helper that
// is not written doesn't influence how other objects are written.
TEST(RooFitHS3, HelperObjects)
{
   const std::string arrayJSON = document(
      "0.3", {distributions({R"({"name": "constraints", "type": "array_dist", "component_type": "gaussian_dist",
                                 "default_parameters": {"sigma": 1.0},
                                 "components": [{"x": "nom_a", "mean": "a"}, {"x": "nom_b", "mean": "b"}]})"}),
              defaultDomain({axes({"a", "b", "nom_a", "nom_b"}, -5, 5)}),
              defaultValues({values({{"a", 0.0}, {"b", 0.0}, {"nom_a", 0.0}, {"nom_b", 0.0}})})});

   using Types = std::map<std::string, std::string>;
   auto exportTypes = [](RooWorkspace &ws, bool arrays, std::string *json = nullptr) {
      Doc exported{ws, {exportArrayDist(arrays)}};
      if (json) {
         *json = exported;
      }
      Types out;
      if (auto const *distributions = exported.find("distributions")) {
         for (auto const &dist : distributions->children()) {
            out[dist["name"].val()] = dist["type"].val();
         }
      }
      return out;
   };

   // The helper is marked, and the mark is not written.
   {
      Imported ws{arrayJSON};
      ASSERT_NE(ws.pdf("constraints"), nullptr);
      EXPECT_TRUE(ws.pdf("constraints")->getAttribute("hs3_helper"));
      EXPECT_FALSE(ws.pdf("nom_a_constraint")->getAttribute("hs3_helper"));

      // With arrays, the HS3 object that the helper stands for is written.
      std::string withArrays;
      EXPECT_EQ(exportTypes(ws, true, &withArrays), (Types{{"constraints", "array_dist"}})) << withArrays;
      EXPECT_THAT(withArrays, Not(HasSubstr("hs3_helper")));

      // Without arrays, the helper is not written, but its components are: nothing of the model is lost.
      std::string plain;
      EXPECT_EQ(exportTypes(ws, false, &plain),
                (Types{{"nom_a_constraint", "gaussian_dist"}, {"nom_b_constraint", "gaussian_dist"}}))
         << plain;
      EXPECT_THAT(plain, Not(HasSubstr("hs3_helper")));
      Imported ws2{plain};
      EXPECT_EQ(ws2.pdf("constraints"), nullptr);
   }

   // A helper that another object refers to by name is written.
   {
      Imported ws{arrayJSON};
      RooRealVar x{"x", "x", 0.0, -10.0, 10.0};
      RooRealVar m{"m", "m", 0.0, -10.0, 10.0};
      RooGaussian model{"model", "model", x, m, RooFit::RooConst(2.0)};
      RooProdPdf product{"user_product", "user_product", RooArgList{model, *ws.pdf("constraints")}};
      ws.import(product, RooFit::RecycleConflictNodes(), RooFit::Silence());
      const Types plain = exportTypes(ws, false);
      EXPECT_EQ(plain.count("user_product"), 1u);
      ASSERT_EQ(plain.count("constraints"), 1u);
      EXPECT_EQ(plain.at("constraints"), "product_dist");
      EXPECT_EQ(plain.count("nom_a_constraint"), 1u);
      const Types withArrays = exportTypes(ws, true);
      ASSERT_EQ(withArrays.count("constraints"), 1u);
      EXPECT_EQ(withArrays.at("constraints"), "array_dist");
   }

   // So is a helper that a ModelConfig names.
   {
      Imported ws{arrayJSON};
      RooStats::ModelConfig mc{"ModelConfig", &ws};
      mc.SetPdf(*ws.pdf("constraints"));
      mc.SetObservables(ws.argSet("nom_a,nom_b"));
      mc.SetParametersOfInterest(ws.argSet("a"));
      ws.import(mc);
      const Types plain = exportTypes(ws, false);
      ASSERT_EQ(plain.count("constraints"), 1u);
      EXPECT_EQ(plain.at("constraints"), "product_dist");
   }

   // A product that a user marked as array distribution is no helper, and is always written.
   {
      RooRealVar nomA{"nom_a", "nom_a", 0.0, -5.0, 5.0};
      RooRealVar a{"a", "a", 0.0, -5.0, 5.0};
      RooGaussian gaussA{"gauss_a", "gauss_a", nomA, a, RooFit::RooConst(1.0)};
      RooProdPdf array{"constraints", "constraints", RooArgList{gaussA}};
      array.setAttribute("hs3_array_dist");
      RooWorkspace ws;
      ws.import(array, RooFit::Silence());
      EXPECT_EQ(exportTypes(ws, false), (Types{{"constraints", "product_dist"}, {"gauss_a", "gaussian_dist"}}));
   }

   // The constraints of HistFactory channels can be written individually and collected into arrays again: the helpers
   // of the arrays leave no product behind that would refer to the constraints by name.
   {
      Imported ws{twoChannelHistFactoryJSON(true)};
      const Types withArrays = exportTypes(ws, true);
      std::string plain;
      for (auto const &[name, type] : exportTypes(ws, false, &plain)) {
         EXPECT_NE(type, "product_dist") << name;
         EXPECT_NE(type, "array_dist") << name;
      }
      // The arrays that are created for the export are named after the model, so only the types are the same.
      auto types = [](Types const &distributions) {
         std::multiset<std::string> out;
         for (auto const &item : distributions) {
            out.insert(item.second);
         }
         return out;
      };
      Imported ws2{plain};
      EXPECT_EQ(types(exportTypes(ws2, true)), types(withArrays));
      EXPECT_EQ(types(withArrays).count("array_dist"), 2u);
   }
}

namespace {

// A document with a Gaussian model of the observable `obs` and the array distributions `arrays` (name and list of
// indices i of components over gobs_i with mean theta_i). The component over gobs_`odd` of all but the first array
// has another width, unless `wrongValue` is given. With `combined`, the model and the arrays are the components of the
// combined distribution "combined", each with its data set: the arrays with one entry of the observed values 0.1*i of
// their gobs_i, which is `wrongValue` for gobs_`odd` in all but the first array if given. Otherwise the arrays are
// auxiliary distributions of the likelihood, with the gobs_i as constant parameters.
std::string identicalComponentsJSON(std::vector<std::pair<std::string, std::vector<int>>> const &arrays, int odd = -1,
                                    bool combined = false, double wrongValue = 0.0)
{
   std::string distributions = R"({"name": "model", "type": "gaussian_dist", "x": "obs", "mean": "mu", "sigma": 1.0})";
   std::string data = R"({"name": "data", "type": "unbinned", "axes": [{"name": "obs", "min": -5.0, "max": 5.0}],
                          "entries": [[0.3]]})";
   std::string aux;
   std::string components = R"({"key": ["physics"], "dist": "model"})";
   std::string dataComponents = R"({"key": ["physics"], "dataset": "data"})";
   std::string labels = R"("physics")";
   std::set<int> indices;
   for (auto const &[name, members] : arrays) {
      const bool first = name == arrays.front().first;
      std::string arrayComponents;
      std::string axes;
      std::string values;
      for (int i : members) {
         indices.insert(i);
         const std::string n = std::to_string(i);
         arrayComponents += std::string(arrayComponents.empty() ? "" : ", ") + R"({"x": "gobs_)" + n +
                            R"(", "mean": "theta_)" + n + R"(")" +
                            (i == odd && !first && wrongValue == 0.0 ? R"(, "sigma": 2.0)" : "") + "}";
         axes += std::string(axes.empty() ? "" : ", ") + R"({"name": "gobs_)" + n + R"(", "min": -5.0, "max": 5.0})";
         values += std::string(values.empty() ? "" : ", ") +
                   std::to_string(i == odd && !first && wrongValue != 0.0 ? wrongValue : 0.1 * i);
      }
      distributions +=
         R"(, {"name": ")" + name +
         R"(", "type": "array_dist", "component_type": "gaussian_dist", "default_parameters": {"sigma": 1.0}, "components": [)" +
         arrayComponents + "]}";
      aux += std::string(aux.empty() ? "" : ", ") + "\"" + name + "\"";
      components += R"(, {"key": [")" + name + R"("], "dist": ")" + name + R"("})";
      dataComponents += R"(, {"key": [")" + name + R"("], "dataset": ")" + name + R"(_data"})";
      labels += R"(, ")" + name + R"(")";
      data += R"(, {"name": ")" + name + R"(_data", "type": "unbinned", "axes": [)" + axes + R"(], "entries": [[)" +
              values + "]]}";
   }
   std::string domainAxes = R"({"name": "mu", "min": -5.0, "max": 5.0})";
   std::string observables = R"({"name": "obs", "min": -5.0, "max": 5.0})";
   std::string values = R"({"name": "mu", "value": 0.0})";
   for (int i : indices) {
      const std::string n = std::to_string(i);
      domainAxes += R"(, {"name": "theta_)" + n + R"(", "min": -5.0, "max": 5.0})";
      values += R"(, {"name": "theta_)" + n + R"(", "value": 0.0})";
      if (combined) {
         observables += R"(, {"name": "gobs_)" + n + R"(", "min": -5.0, "max": 5.0})";
      } else {
         // The constant gobs_i are in the domain of the analysis, which makes them global observables: without
         // them, RooFit's normalization of the constraint terms would be ambiguous.
         domainAxes += R"(, {"name": "gobs_)" + n + R"(", "min": -5.0, "max": 5.0})";
         values += R"(, {"name": "gobs_)" + n + R"(", "value": )" + std::to_string(0.1 * i) + R"(, "const": true})";
      }
   }
   if (combined) {
      distributions += R"(, {"name": "combined", "type": "combined_dist", "columns": ["region"], "components": [)" +
                       components + "]}";
      data +=
         R"(, {"name": "combData", "type": "combined", "columns": ["region"], "components": [)" + dataComponents + "]}";
      observables += R"(, {"name": "region", "categories": [)" + labels + "]}";
   }
   const std::string likelihood =
      combined ? R"({"name": "lh", "distributions": ["combined"], "data": ["combData"]})"
               : R"({"name": "lh", "distributions": ["model"], "data": ["data"], "aux_distributions": [)" + aux + "]}";
   return R"({"metadata": {"hs3_version": "0.3"}, "distributions": [)" + distributions + R"(], "data": [)" + data +
          R"(], "domains": [{"name": "parameters", "type": "product_domain", "axes": [)" + domainAxes +
          R"(]}, {"name": "default_domain", "type": "product_domain", "axes": [)" + observables +
          R"(]}], "parameter_points": [{"name": "default_values", "parameters": [)" + values +
          R"(]}], "likelihoods": [)" + likelihood +
          R"(], "analyses": [{"name": "analysis", "likelihood": "lh", "parameters_of_interest": ["mu"], "domains": ["parameters"]}]})";
}

// The NLL of the ModelConfig "analysis" for the data set that the import made of the data of the likelihood.
double identicalComponentsNLL(RooWorkspace &ws)
{
   auto *mc = dynamic_cast<RooFit::ModelConfig *>(ws.obj("analysis"));
   RooAbsData *data = nullptr;
   for (const char *name : {"analysis_data", "combData", "data"}) {
      if (!data) {
         data = ws.data(name);
      }
   }
   if (!mc || !data) {
      return std::numeric_limits<double>::quiet_NaN();
   }
   for (int i = 1; i <= 6; ++i) {
      if (RooRealVar *theta = ws.var(("theta_" + std::to_string(i)).c_str())) {
         theta->setVal(0.2 * i - 0.5);
      }
   }
   RooArgSet external;
   if (mc->GetExternalConstraints()) {
      external.add(*mc->GetExternalConstraints());
   }
   RooArgSet globs;
   if (mc->GetGlobalObservables()) {
      globs.add(*mc->GetGlobalObservables());
   }
   std::unique_ptr<RooAbsReal> nll{
      mc->GetPdf()->createNLL(*data, RooFit::ExternalConstraints(external), RooFit::GlobalObservables(globs))};
   return nll->getVal();
}

} // namespace

// HS3 composes a combined distribution and an array distribution by union (PR #161): identical factors enter once.
// The model and two arrays over gobs_1..gobs_4 and gobs_3..gobs_6 with identical components for gobs_3 and gobs_4,
// as components of one combined distribution, give the likelihood of the model and one array over gobs_1..gobs_6.
// RooFit has the shared components as one object, and merges the two array channels into one array with one entry of
// data. The terms of a likelihood are a plain product, so the two arrays as auxiliary distributions of the likelihood
// are two factors with shared variates, which is rejected, as are components over the same observable that are not
// identical, and identical components with different observed values.
TEST(RooFitHS3, ArrayDistIdenticalComponents)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace wsAll;
   ASSERT_TRUE(RooJSONFactoryWSTool{wsAll}.importJSONfromString(
      identicalComponentsJSON({{"Z", {1, 2, 3, 4, 5, 6}}}, -1, /*combined=*/true)));
   const double nllAll = identicalComponentsNLL(wsAll);
   ASSERT_FALSE(std::isnan(nllAll));

   const std::string json = identicalComponentsJSON({{"Za", {1, 2, 3, 4}}, {"Zb", {3, 4, 5, 6}}}, -1, true);
   RooWorkspace ws;
   ASSERT_TRUE(RooJSONFactoryWSTool{ws}.importJSONfromString(json)) << json;
   EXPECT_NEAR(identicalComponentsNLL(ws), nllAll, 1e-10 * std::abs(nllAll));
   auto *za = dynamic_cast<RooProdPdf *>(ws.pdf("Za"));
   auto *zb = dynamic_cast<RooProdPdf *>(ws.pdf("Zb"));
   ASSERT_NE(za, nullptr);
   ASSERT_NE(zb, nullptr);
   EXPECT_EQ(za->pdfList().find("gobs_3_constraint"), zb->pdfList().find("gobs_3_constraint"));
   auto *mc = dynamic_cast<RooFit::ModelConfig *>(ws.obj("analysis"));
   ASSERT_NE(mc, nullptr);
   auto *simPdf = dynamic_cast<RooSimultaneous *>(mc->GetPdf());
   ASSERT_NE(simPdf, nullptr);
   EXPECT_EQ(simPdf->indexCat().size(), 3); // physics, Za, Zb; Zb has no pdf any more
   auto *merged = dynamic_cast<RooProdPdf *>(simPdf->getPdf("Za"));
   ASSERT_NE(merged, nullptr);
   EXPECT_EQ(merged->pdfList().size(), 6u);
   EXPECT_EQ(simPdf->getPdf("Zb"), nullptr);
   ASSERT_NE(ws.data("analysis_data"), nullptr);
   EXPECT_EQ(ws.data("analysis_data")->numEntries(), 2);

   // Written back as one combined distribution with the model and one array; the document is stable from there on.
   const std::string exported = RooJSONFactoryWSTool{ws}.exportJSONtoString();
   auto tree = RooFit::Detail::JSONTree::create(exported);
   ASSERT_EQ(tree->rootnode()["likelihoods"].num_children(), 1u) << exported;
   const auto *combined = RooJSONFactoryWSTool::findNamedChild(tree->rootnode()["distributions"], "analysis_simPdf");
   ASSERT_NE(combined, nullptr) << exported;
   EXPECT_EQ((*combined)["components"].num_children(), 2u) << exported;
   // What only the replaced combined distribution uses is not written: neither it nor the arrays of its channels.
   std::set<std::string> distributionNames;
   for (auto const &dist : tree->rootnode()["distributions"].children()) {
      distributionNames.insert(dist["name"].val());
   }
   EXPECT_EQ(distributionNames, (std::set<std::string>{"analysis_Za_merged", "analysis_simPdf", "model"})) << exported;
   RooWorkspace ws2;
   ASSERT_TRUE(RooJSONFactoryWSTool{ws2}.importJSONfromString(exported)) << exported;
   EXPECT_NEAR(identicalComponentsNLL(ws2), nllAll, 1e-10 * std::abs(nllAll));
   const std::string exported2 = RooJSONFactoryWSTool{ws2}.exportJSONtoString();
   RooWorkspace ws3;
   ASSERT_TRUE(RooJSONFactoryWSTool{ws3}.importJSONfromString(exported2)) << exported2;
   EXPECT_EQ(RooJSONFactoryWSTool{ws3}.exportJSONtoString(), exported2);

   // The terms of a likelihood are a plain product: the arrays as auxiliary distributions share variates.
   expectImportThrowsWithError(identicalComponentsJSON({{"Za", {1, 2, 3, 4}}, {"Zb", {3, 4, 5, 6}}}),
                               "the variate 'gobs_3' is shared by the terms");
   // A component over gobs_3 with another width is not identical: two factors of the combined distribution share the
   // variate, which the checks of the document report before anything is imported.
   expectImportThrowsWithError(identicalComponentsJSON({{"Za", {1, 2, 3, 4}}, {"Zb", {3, 4, 5, 6}}}, 3, true),
                               "the variate 'gobs_3' is shared by the terms 'Za[gobs_3]' and 'Zb[gobs_3]'");
   // Identical components with different observed values.
   expectImportThrowsWithError(identicalComponentsJSON({{"Za", {1, 2, 3, 4}}, {"Zb", {3, 4, 5, 6}}}, 3, true, 0.7),
                               "the observed value of 'gobs_3' differs");
}

namespace {

// A HistFactory workspace with the given channels, each with a signal scaled by `mu` and a background with the
// systematics `shared` and the one of the channel, and statistical uncertainties for both. The luminosity has the
// relative uncertainty `lumiRelErr`. The function `configure` can change the measurement.
std::unique_ptr<RooWorkspace>
makeMergeableHistFactoryWorkspace(std::string const &name,
                                  std::vector<std::pair<std::string, std::string>> const &channels,
                                  double lumiRelErr = 0.1,
                                  std::function<void(RooStats::HistFactory::Measurement &)> const &configure = {})
{
   using namespace RooStats::HistFactory;
   Measurement meas(name.c_str(), name.c_str());
   meas.SetOutputFilePrefix("HS3Merge_" + name);
   meas.SetPOI("mu");
   meas.AddConstantParam("Lumi");
   meas.SetLumi(1.0);
   meas.SetLumiRelErr(lumiRelErr);
   // The channels and samples own their histograms.
   auto hist = [&](std::string const &histName, double first, double second) {
      auto *h = new TH1D(histName.c_str(), histName.c_str(), 2, 0.0, 2.0);
      h->SetDirectory(nullptr);
      h->SetBinContent(1, first);
      h->SetBinContent(2, second);
      h->SetBinError(1, 0.1 * first);
      h->SetBinError(2, 0.1 * second);
      return h;
   };
   for (auto const &[channelName, own] : channels) {
      const double shift = channelName.size() % 3;
      Channel channel(channelName);
      channel.SetData(hist("data_" + channelName, 22.0 + shift, 31.0 - shift));
      Sample signal("signal_" + channelName);
      signal.SetHisto(hist("signal_" + channelName, 5.0, 8.0 + shift));
      signal.AddNormFactor("mu", 1.0, 0.0, 5.0);
      signal.ActivateStatError();
      Sample background("background_" + channelName);
      background.SetHisto(hist("background_" + channelName, 20.0 + shift, 21.0));
      background.ActivateStatError();
      background.AddOverallSys("shared", 0.9, 1.1);
      background.AddOverallSys(own, 0.95, 1.05);
      channel.AddSample(signal);
      channel.AddSample(background);
      meas.AddChannel(channel);
   }
   if (configure) {
      configure(meas);
   }
   RooHelpers::LocalChangeMsgLevel quiet(RooFit::WARNING, 0u, RooFit::ObjectHandling, false);
   return std::unique_ptr<RooWorkspace>{RooStats::HistFactory::MakeModelAndMeasurementFast(meas)};
}

// The serialized children of `node`.
std::vector<std::string> childrenJSON(RooFit::Detail::JSONNode const &node)
{
   std::vector<std::string> out;
   for (auto const &child : node.children()) {
      std::stringstream ss;
      child.writeJSON(ss);
      out.push_back(ss.str());
   }
   return out;
}

std::string joinJSON(std::vector<std::string> const &items)
{
   std::string out;
   for (std::string const &item : items) {
      out += (out.empty() ? "" : ", ") + item;
   }
   return out;
}

// Merges the documents `a` and `b` written by RooFit into one document, as a user combining two analyses would do: the
// distributions and data sets of `b` whose names also occur in `a` (its "simPdf", its arrays of constraints, its
// combined data) get the suffix "_B"; parameters keep their names, such that parameters with the same name are the
// same parameter. The index category of both is the same column with the labels of both, or, with `renameColumn`, the
// column of `b` is "channelCat_B". With `nested`, the two combined distributions are the components of the combined
// distribution "joint" over the column "analysis", which the likelihood lists as its only term, with the
// combined data set "combData" (the way HS3 recommends for models that share factors); otherwise, the likelihood
// lists the two combined distributions as two terms. The RooFit metadata and the Asimov data sets are dropped.
std::string mergeDocuments(std::string const &a, std::string b, bool renameColumn = false, bool nested = false)
{
   if (renameColumn) {
      b = std::regex_replace(b, std::regex{"\"channelCat\""}, "\"channelCat_B\"");
   }
   using RooFit::Detail::JSONTree;
   auto treeA = JSONTree::create(a);
   auto const &rootA = treeA->rootnode();
   {
      auto treeB = JSONTree::create(b);
      for (const char *key : {"distributions", "data"}) {
         for (auto const &child : treeB->rootnode()[key].children()) {
            const std::string name = child["name"].val();
            if (RooJSONFactoryWSTool::findNamedChild(rootA[key], name)) {
               b = std::regex_replace(b, std::regex{"\"" + name + "\""}, "\"" + name + "_B\"");
            }
         }
      }
   }
   auto treeB = JSONTree::create(b);
   auto const &rootB = treeB->rootnode();

   // Named entries of both, the first of each name.
   auto unite = [](std::vector<RooFit::Detail::JSONNode const *> const &lists) {
      std::vector<std::string> out;
      std::set<std::string> names;
      for (auto const *list : lists) {
         for (auto const &child : list->children()) {
            if (!names.insert(child["name"].val()).second) {
               continue;
            }
            std::stringstream ss;
            if (child.has_child("categories")) {
               // The labels of the column are those of both documents.
               std::set<std::string> categories;
               for (auto const *other : lists) {
                  for (auto const &axis : other->children()) {
                     if (axis["name"].val() == child["name"].val()) {
                        for (auto const &label : axis["categories"].children()) {
                           categories.insert(label.val());
                        }
                     }
                  }
               }
               std::string labels;
               for (std::string const &label : categories) {
                  labels += (labels.empty() ? "\"" : ", \"") + label + "\"";
               }
               ss << R"({"name": ")" << child["name"].val() << R"(", "categories": [)" << labels << "]}";
            } else {
               child.writeJSON(ss);
            }
            out.push_back(ss.str());
         }
      }
      return joinJSON(out);
   };
   auto domainAxes = [](RooFit::Detail::JSONNode const &root, std::string const &name) {
      return &(*RooJSONFactoryWSTool::findNamedChild(root["domains"], name))["axes"];
   };
   auto pointParameters = [](RooFit::Detail::JSONNode const &root) {
      return &(*RooJSONFactoryWSTool::findNamedChild(root["parameter_points"], "default_values"))["parameters"];
   };
   auto likelihoodList = [](RooFit::Detail::JSONNode const &root, const char *key) {
      auto const &likelihood =
         *RooJSONFactoryWSTool::findNamedChild(root["likelihoods"], "ModelConfig_obsData_likelihood");
      return likelihood.has_child(key) ? childrenJSON(likelihood[key]) : std::vector<std::string>{};
   };

   std::vector<std::string> distributions = childrenJSON(rootA["distributions"]);
   for (std::string const &item : childrenJSON(rootB["distributions"])) {
      distributions.push_back(item);
   }
   std::vector<std::string> data;
   for (auto const *root : {&rootA, &rootB}) {
      for (std::string const &item : childrenJSON((*root)["data"])) {
         if (item.find("asimovData") == std::string::npos) {
            data.push_back(item);
         }
      }
   }
   if (nested) {
      distributions.push_back(R"({"name": "joint", "type": "combined_dist", "columns": ["analysis"],
         "components": [{"key": ["A"], "dist": "simPdf"}, {"key": ["B"], "dist": "simPdf_B"}]})");
      data.push_back(R"({"name": "combData", "type": "combined", "columns": ["analysis"],
         "components": [{"key": ["A"], "dataset": "obsData"}, {"key": ["B"], "dataset": "obsData_B"}]})");
   }
   std::string domains;
   for (const char *domain : {"default_domain", "ModelConfig_obsData_nuisance_parameters",
                              "ModelConfig_obsData_global_observables", "ModelConfig_obsData_parameters_of_interest"}) {
      std::string axes = unite({domainAxes(rootA, domain), domainAxes(rootB, domain)});
      if (nested && domain == std::string{"default_domain"}) {
         axes += R"(, {"name": "analysis", "categories": ["A", "B"]})";
      }
      domains += std::string(domains.empty() ? "" : ", ") + R"({"name": ")" + domain +
                 R"(", "type": "product_domain", "axes": [)" + axes + "]}";
   }
   std::vector<std::string> aux = likelihoodList(rootA, "aux_distributions");
   for (std::string const &item : likelihoodList(rootB, "aux_distributions")) {
      aux.push_back(item);
   }
   return R"({"metadata": {"hs3_version": "0.3"}, "distributions": [)" + joinJSON(distributions) +
          R"(], "functions": [)" + unite({&rootA["functions"], &rootB["functions"]}) + R"(], "data": [)" +
          joinJSON(data) + R"(], "domains": [)" + domains +
          R"(], "parameter_points": [{"name": "default_values", "parameters": [)" +
          unite({pointParameters(rootA), pointParameters(rootB)}) + R"(]}], "likelihoods": [{"name": "lh", )" +
          (nested ? R"("distributions": ["joint"], "data": ["combData"])"
                  : R"("distributions": ["simPdf", "simPdf_B"], "data": ["obsData", "obsData_B"])") +
          (aux.empty() ? std::string{} : R"(, "aux_distributions": [)" + joinJSON(aux) + "]") +
          R"(}], "analyses": [{"name": "combination", "likelihood": "lh", "parameters_of_interest": ["mu"], "domains": ["ModelConfig_obsData_nuisance_parameters", "ModelConfig_obsData_global_observables", "ModelConfig_obsData_parameters_of_interest"]}]})";
}

// The NLL of the ModelConfig `mcName` of `ws` for the union of the data sets `dataNames`, at a point in which every
// parameter is shifted away from its nominal value by `scale` times a step, such that every constraint term matters.
// The parameters are the ones of `parameters` (the nuisance parameters of the ModelConfig by default) and "mu".
double mergedHistFactoryNLL(RooWorkspace &ws, std::string const &mcName, std::vector<std::string> const &dataNames,
                            double scale = 1.0, std::set<std::string> parameters = {})
{
   auto *mc = dynamic_cast<RooFit::ModelConfig *>(ws.obj(mcName));
   if (!mc || !mc->GetPdf() || !mc->GetNuisanceParameters()) {
      return std::numeric_limits<double>::quiet_NaN();
   }
   if (parameters.empty()) {
      for (RooAbsArg const *arg : *mc->GetNuisanceParameters()) {
         parameters.insert(arg->GetName());
      }
   }
   ws.var("mu")->setVal(1.0 + 0.3 * scale);
   // The value depends on the name only, to give the same point in workspaces with another order of the parameters.
   for (std::string const &name : parameters) {
      RooRealVar *var = ws.var(name);
      if (!var) {
         return std::numeric_limits<double>::quiet_NaN();
      }
      const int step = static_cast<int>(std::hash<std::string>{}(name) % 5) - 2;
      // "+ 0.0" makes -0.0 (a negative step at scale 0) 0.0, which the export would write as it is.
      var->setVal(name.rfind("gamma", 0) == 0 ? 1.0 + 0.01 * step * scale : 0.1 * step * scale + 0.0);
   }
   RooArgSet vars;
   for (std::string const &name : dataNames) {
      vars.add(*ws.data(name)->get(), /*silent=*/true);
   }
   RooRealVar weight{"merged_weight", "merged_weight", 1.0};
   vars.add(weight);
   RooDataSet merged{"merged", "merged", vars, RooFit::WeightVar(weight)};
   for (std::string const &name : dataNames) {
      RooAbsData &data = *ws.data(name);
      for (int entry = 0; entry < data.numEntries(); ++entry) {
         RooArgSet const &values = *data.get(entry);
         merged.add(values, data.weight());
      }
   }
   RooArgSet external;
   if (mc->GetExternalConstraints()) {
      external.add(*mc->GetExternalConstraints());
   }
   RooArgSet globs;
   if (mc->GetGlobalObservables()) {
      globs.add(*mc->GetGlobalObservables());
   }
   std::unique_ptr<RooAbsReal> nll{
      mc->GetPdf()->createNLL(merged, RooFit::ExternalConstraints(external), RooFit::GlobalObservables(globs))};
   return nll->getVal();
}

std::set<std::string> namesOf(RooArgSet const *set)
{
   std::set<std::string> out;
   if (set) {
      for (RooAbsArg const *arg : *set) {
         out.insert(arg->GetName());
      }
   }
   return out;
}

} // namespace

// Measurement::AddGammaSyst() and AddLogNormSyst() make the input of a normsys a function of the constrained parameter
// (#24304). Its variation is written as a custom modifier: the FlexibleInterpVar if it has no other inputs (channel
// "B"), else one with only this input. The FlexibleInterpVar of channel A is split, so its name stands for less after
// the import, and only the likelihood is compared.
TEST(RooFitHS3, HistFactoryGammaAndLogNormSyst)
{
   auto ws = makeMergeableHistFactoryWorkspace("syst", {{"A", "ownA"}, {"B", "ownB"}}, 0.1, [](auto &meas) {
      meas.AddGammaSyst("shared", 0.1);
      meas.AddLogNormSyst("ownB", 0.05);
   });
   Doc doc{*ws};
   EXPECT_TRUE(matches(doc, "distributions/model_A/samples/background_A/modifiers",
                       R"([{}, {"type": "normsys", "parameters": ["alpha_ownA"]},
                           {"type": "custom", "name": "background_A_A_epsilon_alphaOfBeta_shared"}, {}])"));
   EXPECT_TRUE(matches(doc, "distributions/model_B/samples/background_B/modifiers",
                       R"([{}, {"type": "custom", "name": "background_B_B_epsilon"}, {}])"));
   EXPECT_TRUE(matches(doc, "distributions/beta_sharedConstraint", R"({"type": "gamma_dist", "x": "beta_shared"})"));
   Imported imported{doc};
   EXPECT_TRUE(sameNumbers(likelihoodScan(*ws, "ModelConfig", "obsData"),
                           likelihoodScan(imported, "ModelConfig", "obsData"), 0.));
   const std::string json = toJSON(imported);
   EXPECT_EQ(reexport(json), json);
}

// Two documents written by RooFit, each with a HistFactory model whose constraint terms are collected in array
// distributions, are combined. The systematic `shared` and the luminosity are in both: the arrays of both documents
// have identical components for them. HS3 composes a combined distribution by union, but the terms of a likelihood by
// a plain product (PR #161), so the two models are joined in a combined distribution "joint", whose components
// are their combined distributions, with the constraint terms as components paired with their data (as RooFit writes
// them with exportConstraintData). RooFit flattens it into one RooSimultaneous over a RooSuperCategory, with the label
// "" in a column that doesn't apply to a channel, and merges the arrays of each type into one, in which the identical
// components are one object. The likelihood is that of a single HistFactory model with the channels of both
// documents. Written back, the likelihood has one combined distribution and one data set. The two models as two terms
// of the likelihood share variates, which is rejected, as is a combination in which the documents disagree on the
// uncertainty of the luminosity.
TEST(RooFitHS3, MergedHistFactoryDocumentsWithArrayConstraints)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   auto wsA = makeMergeableHistFactoryWorkspace("A", {{"chA", "onlyA"}});
   auto wsB = makeMergeableHistFactoryWorkspace("B", {{"chB", "onlyB"}});
   auto wsAB = makeMergeableHistFactoryWorkspace("AB", {{"chA", "onlyA"}, {"chB", "onlyB"}});
   auto wsB2 = makeMergeableHistFactoryWorkspace("B2", {{"chB", "onlyB"}}, 0.2);
   ASSERT_NE(wsA, nullptr);
   ASSERT_NE(wsB, nullptr);
   ASSERT_NE(wsAB, nullptr);
   ASSERT_NE(wsB2, nullptr);
   auto withConstraintData = [](RooWorkspace &ws) {
      RooJSONFactoryWSTool::config().exportConstraintData = true;
      std::string json = RooJSONFactoryWSTool{ws}.exportJSONtoString();
      RooJSONFactoryWSTool::config().exportConstraintData = false;
      return json;
   };
   const std::string jsonA = withConstraintData(*wsA);
   const std::string jsonB = withConstraintData(*wsB);

   // The NLL differences between a point with shifted parameters and the nominal one: the constraint terms are
   // channels with data in the combination and constraint terms in the single model, which differs by a constant.
   auto *mcAB = static_cast<RooFit::ModelConfig *>(wsAB->obj("ModelConfig"));
   std::set<std::string> const parameters = namesOf(mcAB->GetNuisanceParameters());
   auto nllDifference = [&](RooWorkspace &ws, std::string const &mcName, std::string const &dataName) {
      return mergedHistFactoryNLL(ws, mcName, {dataName}, 1.0, parameters) -
             mergedHistFactoryNLL(ws, mcName, {dataName}, 0.0, parameters);
   };
   const double expected = nllDifference(*wsAB, "ModelConfig", "obsData");
   ASSERT_FALSE(std::isnan(expected));

   for (bool renameColumn : {false, true}) {
      SCOPED_TRACE(renameColumn ? "a column per document" : "the same column");
      const std::string json = mergeDocuments(jsonA, jsonB, renameColumn, /*nested=*/true);
      RooWorkspace ws;
      ASSERT_TRUE(RooJSONFactoryWSTool{ws}.importJSONfromString(json)) << json;

      // The identical components of the two arrays are one object, which the merged array has once.
      auto *arrayA = dynamic_cast<RooProdPdf *>(ws.pdf("simPdf_gaussian_constraints"));
      auto *arrayB = dynamic_cast<RooProdPdf *>(ws.pdf("simPdf_gaussian_constraints_B"));
      ASSERT_NE(arrayA, nullptr);
      ASSERT_NE(arrayB, nullptr);
      for (const char *component : {"nom_alpha_shared_constraint", "nominalLumi_constraint"}) {
         ASSERT_NE(arrayA->pdfList().find(component), nullptr) << component;
         EXPECT_EQ(arrayA->pdfList().find(component), arrayB->pdfList().find(component)) << component;
      }
      auto *mc = dynamic_cast<RooFit::ModelConfig *>(ws.obj("combination"));
      ASSERT_NE(mc, nullptr);
      auto *simPdf = dynamic_cast<RooSimultaneous *>(mc->GetPdf());
      ASSERT_NE(simPdf, nullptr);
      EXPECT_STREQ(simPdf->GetName(), "combination_simPdf");
      std::map<std::string, std::size_t> arrays; // component type -> number of components
      std::size_t channels = 0;
      for (auto const &state : simPdf->indexCat()) {
         RooAbsPdf *channel = simPdf->getPdf(state.first);
         if (!channel) {
            continue;
         }
         ++channels;
         if (auto *array = dynamic_cast<RooProdPdf *>(channel); array && channel->getAttribute("hs3_array_dist")) {
            arrays[array->pdfList()[0].ClassName()] += array->pdfList().size();
         }
      }
      EXPECT_EQ(channels, 4u); // chA, chB and one array per type
      // Lumi, onlyA, onlyB and shared; the statistical uncertainties of both channels.
      EXPECT_EQ(arrays, (std::map<std::string, std::size_t>{{"RooGaussian", 4}, {"RooPoisson", 4}}));
      ASSERT_NE(ws.data("combination_data"), nullptr);

      // The likelihood is that of the single model: the shared constraint terms count once.
      EXPECT_NEAR(nllDifference(ws, "combination", "combination_data"), expected, 1e-8 * std::abs(expected) + 1e-9);

      // Written back, the likelihood has one combined distribution and one data set, and the document reads back
      // with the same likelihood; from there on, it is stable.
      const std::string exported = RooJSONFactoryWSTool{ws}.exportJSONtoString();
      auto tree = RooFit::Detail::JSONTree::create(exported);
      auto const &likelihoods = tree->rootnode()["likelihoods"];
      ASSERT_EQ(likelihoods.num_children(), 1u) << exported;
      EXPECT_EQ(childrenJSON(likelihoods.child(0)["distributions"]), std::vector<std::string>{"\"combination_simPdf\""})
         << exported;
      EXPECT_EQ(childrenJSON(likelihoods.child(0)["data"]), std::vector<std::string>{"\"combination_data\""})
         << exported;
      // What only the replaced combined distributions use is not written: neither they nor the arrays that were merged.
      std::map<std::string, int> types;
      for (auto const &dist : tree->rootnode()["distributions"].children()) {
         ++types[dist["type"].val()];
      }
      EXPECT_EQ(types, (std::map<std::string, int>{{"combined_dist", 1}, {"array_dist", 2}, {"histfactory_dist", 2}}))
         << exported;
      RooWorkspace ws2;
      ASSERT_TRUE(RooJSONFactoryWSTool{ws2}.importJSONfromString(exported)) << exported;
      EXPECT_NEAR(nllDifference(ws2, "combination", "combination_data"), expected, 1e-8 * std::abs(expected) + 1e-9);
      const std::string exported2 = RooJSONFactoryWSTool{ws2}.exportJSONtoString();
      RooWorkspace ws3;
      ASSERT_TRUE(RooJSONFactoryWSTool{ws3}.importJSONfromString(exported2)) << exported2;
      EXPECT_EQ(RooJSONFactoryWSTool{ws3}.exportJSONtoString(), exported2);
   }

   // The two models as two terms of the likelihood: the shared constraints would enter twice.
   expectImportThrowsWithError(
      mergeDocuments(RooJSONFactoryWSTool{*wsA}.exportJSONtoString(), RooJSONFactoryWSTool{*wsB}.exportJSONtoString()),
      "is shared by the terms");

   // The single model is written as one combined distribution with one array per type of constraint.
   auto tree = RooFit::Detail::JSONTree::create(RooJSONFactoryWSTool{*wsAB}.exportJSONtoString());
   std::map<std::string, int> types;
   for (auto const &dist : tree->rootnode()["distributions"].children()) {
      ++types[dist["type"].val()];
   }
   EXPECT_EQ(types, (std::map<std::string, int>{{"combined_dist", 1}, {"array_dist", 2}, {"histfactory_dist", 2}}));

   // If the documents disagree on the uncertainty of the luminosity, the combination is rejected rather than having
   // one of the two constraints or both: here already because the range of the observable 'nominalLumi' differs (its
   // components would not be identical either).
   expectImportThrowsWithError(mergeDocuments(jsonA, withConstraintData(*wsB2), false, true),
                               "the axis 'nominalLumi' of the data set");
}

// A likelihood with a combined distribution and a distribution that is not combined, each with its data. RooFit has
// one pdf per ModelConfig: a RooSimultaneous whose index has the column of the combined distribution and the column
// `<analysis>_categoryIndex` of the terms, with the label "" in the former for the term that is not combined, and one
// data set with both columns. The likelihood is that of both terms. Written back, it has one combined distribution.
TEST(RooFitHS3, LikelihoodOfCombinedAndPlainTerms)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   const std::string json = R"({
  "metadata": {"hs3_version": "0.3"},
  "distributions": [
    {"name": "g1", "type": "gaussian_dist", "x": "x1", "mean": "mu", "sigma": 1.0},
    {"name": "g2", "type": "gaussian_dist", "x": "x2", "mean": "mu", "sigma": 2.0},
    {"name": "g3", "type": "gaussian_dist", "x": "x3", "mean": "mu", "sigma": 0.5},
    {"name": "sim", "type": "combined_dist", "columns": ["cat"],
     "components": [{"key": ["c1"], "dist": "g1"}, {"key": ["c2"], "dist": "g2"}]}
  ],
  "data": [
    {"name": "d1", "type": "unbinned", "axes": [{"name": "x1", "min": -10.0, "max": 10.0}], "entries": [[0.5], [1.5]]},
    {"name": "d2", "type": "unbinned", "axes": [{"name": "x2", "min": -10.0, "max": 10.0}], "entries": [[-1.0]]},
    {"name": "d3", "type": "unbinned", "axes": [{"name": "x3", "min": -10.0, "max": 10.0}], "entries": [[0.2], [0.4], [0.1]]},
    {"name": "dsim", "type": "combined", "columns": ["cat"],
     "components": [{"key": ["c1"], "dataset": "d1"}, {"key": ["c2"], "dataset": "d2"}]}
  ],
  "domains": [{"name": "default_domain", "type": "product_domain", "axes": [
    {"name": "x1", "min": -10.0, "max": 10.0}, {"name": "x2", "min": -10.0, "max": 10.0},
    {"name": "x3", "min": -10.0, "max": 10.0}, {"name": "mu", "min": -5.0, "max": 5.0},
    {"name": "cat", "categories": ["c1", "c2"]}]}],
  "parameter_points": [{"name": "default_values", "parameters": [{"name": "mu", "value": 0.3}]}],
  "likelihoods": [{"name": "lh", "distributions": ["sim", "g3"], "data": ["dsim", "d3"]}],
  "analyses": [{"name": "an", "likelihood": "lh", "parameters_of_interest": ["mu"], "domains": ["default_domain"]}]
})";
   RooWorkspace ws;
   ASSERT_TRUE(RooJSONFactoryWSTool{ws}.importJSONfromString(json));
   auto *mc = dynamic_cast<RooFit::ModelConfig *>(ws.obj("an"));
   ASSERT_NE(mc, nullptr);
   auto *simPdf = dynamic_cast<RooSimultaneous *>(mc->GetPdf());
   ASSERT_NE(simPdf, nullptr);
   EXPECT_STREQ(simPdf->GetName(), "an_simPdf");
   // The columns sorted by name: "an_categoryIndex", "cat".
   EXPECT_EQ(simPdf->getPdf("{sim;c1}"), ws.pdf("g1"));
   EXPECT_EQ(simPdf->getPdf("{sim;c2}"), ws.pdf("g2"));
   EXPECT_EQ(simPdf->getPdf("{g3;}"), ws.pdf("g3"));
   EXPECT_NE(mc->GetObservables()->find("an_categoryIndex"), nullptr);
   EXPECT_NE(mc->GetObservables()->find("cat"), nullptr);
   RooAbsData *data = ws.data("an_data");
   ASSERT_NE(data, nullptr);
   EXPECT_EQ(data->numEntries(), 6);
   EXPECT_EQ(ws.data("dsim"), nullptr);
   EXPECT_EQ(ws.data("d3"), nullptr);

   // The NLL of the terms, up to the constant that RooFit adds for a RooSimultaneous of non-extended pdfs: the
   // difference between two values of mu is the same.
   auto nllOfTerms = [](double mu) {
      double out = 0.0;
      for (auto const &[x, sigma] : std::vector<std::pair<double, double>>{
              {0.5, 1.0}, {1.5, 1.0}, {-1.0, 2.0}, {0.2, 0.5}, {0.4, 0.5}, {0.1, 0.5}}) {
         const double norm = ROOT::Math::normal_cdf(10.0, sigma, mu) - ROOT::Math::normal_cdf(-10.0, sigma, mu);
         out -= std::log(ROOT::Math::normal_pdf(x, sigma, mu) / norm);
      }
      return out;
   };
   std::unique_ptr<RooAbsReal> nll{simPdf->createNLL(*data)};
   ws.var("mu")->setVal(0.3);
   const double atFirst = nll->getVal();
   ws.var("mu")->setVal(-0.4);
   EXPECT_NEAR(nll->getVal() - atFirst, nllOfTerms(-0.4) - nllOfTerms(0.3), 1e-9);

   // Written back: one likelihood with one combined distribution and one data set, with "" for the column that doesn't
   // apply, which reads back to the same model.
   const std::string exported = RooJSONFactoryWSTool{ws}.exportJSONtoString();
   auto tree = RooFit::Detail::JSONTree::create(exported);
   ASSERT_EQ(tree->rootnode()["likelihoods"].num_children(), 1u) << exported;
   auto const &lh = tree->rootnode()["likelihoods"].child(0);
   EXPECT_EQ(childrenJSON(lh["distributions"]), std::vector<std::string>{"\"an_simPdf\""}) << exported;
   EXPECT_EQ(childrenJSON(lh["data"]), std::vector<std::string>{"\"an_data\""}) << exported;
   EXPECT_NE(exported.find(R"("key":["g3",""])"), std::string::npos) << exported;
   RooWorkspace ws2;
   ASSERT_TRUE(RooJSONFactoryWSTool{ws2}.importJSONfromString(exported)) << exported;
   EXPECT_EQ(RooJSONFactoryWSTool{ws2}.exportJSONtoString(), exported);
}

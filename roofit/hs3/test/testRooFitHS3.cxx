// Tests for the RooJSONFactoryWSTool
// Authors: Carsten D. Burgard, DESY/ATLAS, 12/2021
//          Jonas Rembser, CERN 12/2022

#include <RooFitHS3/JSONIO.h>
#include <RooFitHS3/RooJSONFactoryWSTool.h>

#include <RooAddPdf.h>
#include <RooAddition.h>
#include <RooBinWidthFunction.h>
#include <RooBinning.h>
#include <RooCategory.h>
#include <RooConstVar.h>
#include <RooDataSet.h>
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
#include <RooLognormal.h>
#include <RooMultiVarGaussian.h>
#include <RooPoisson.h>
#include <RooProdPdf.h>
#include <RooProduct.h>
#include <RooRealIntegral.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooSpline.h>
#include <RooUniformBinning.h>
#include <RooStats/HistFactory/ParamHistFunc.h>
#include <RooStats/HistFactory/FlexibleInterpVar.h>
#include <RooStats/HistFactory/PiecewiseInterpolation.h>
#include <RooWorkspace.h>

#include <cmath>
#include <memory>
#include <sstream>
#include <string_view>
#include <vector>

#include <TROOT.h>

#include <gtest/gtest.h>

namespace {

// If the JSON files should be written out for debugging purpose.
const bool writeJsonFiles = false;

RooFit::Detail::JSONNode *findMutableNamedChild(RooFit::Detail::JSONNode &node, std::string const &name)
{
   for (RooFit::Detail::JSONNode &child : node.children()) {
      if (child.has_child("name") && child["name"].val() == name) {
         return &child;
      }
   }
   return nullptr;
}

std::string jsonString(RooFit::Detail::JSONTree &tree)
{
   std::ostringstream stream;
   tree.rootnode().writeJSON(stream);
   return stream.str();
}

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
         const double val2 = r2->getVal(nset2);
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

std::string parameterStepWidthsNode(std::string const &json)
{
   const std::string key = "\"parameter_stepwidths\":[";
   const auto begin = json.find(key);
   if (begin == std::string::npos) {
      return "";
   }
   const auto end = json.find("]", begin);
   if (end == std::string::npos) {
      return "";
   }
   return json.substr(begin, end - begin + 1);
}

std::string defaultDomainAxesNode(std::string const &json)
{
   const std::string key = "\"domains\":[";
   const auto domainsBegin = json.find(key);
   if (domainsBegin == std::string::npos) {
      return "";
   }
   const auto axesBegin = json.find("\"axes\":[", domainsBegin);
   if (axesBegin == std::string::npos) {
      return "";
   }
   const auto axesEnd = json.find("}]", axesBegin);
   if (axesEnd == std::string::npos) {
      return "";
   }
   return json.substr(axesBegin, axesEnd - axesBegin + 2);
}

class ScopedNoDomainConstVarImportFlag {
public:
   explicit ScopedNoDomainConstVarImportFlag(bool value)
      : _oldValue{RooJSONFactoryWSTool::config().importNoDomainParametersAsRooConstVars}
   {
      RooJSONFactoryWSTool::config().importNoDomainParametersAsRooConstVars = value;
   }

   ~ScopedNoDomainConstVarImportFlag()
   {
      RooJSONFactoryWSTool::config().importNoDomainParametersAsRooConstVars = _oldValue;
   }

private:
   bool _oldValue;
};

// Runs `action` and returns everything that was logged at the given message level and topic while it ran.
template <class Action>
std::string captureMessages(RooFit::MsgLevel level, RooFit::MsgTopic topic, Action &&action)
{
   RooHelpers::HijackMessageStream stream{level, topic};
   action();
   return stream.str();
}

// Counts the number of (non-overlapping) occurrences of `needle` in `haystack`.
std::size_t countOccurrences(std::string_view haystack, std::string_view needle)
{
   std::size_t result = 0;
   for (std::size_t pos = 0; (pos = haystack.find(needle, pos)) != std::string_view::npos; pos += needle.size()) {
      ++result;
   }
   return result;
}

// Asserts that exporting `ws` to HS3 throws and logs an error message containing `expectedReason`.
void expectExportThrowsWithError(RooWorkspace &ws, std::string const &expectedReason)
{
   bool threw = false;
   std::string exported;
   const std::string errors = captureMessages(RooFit::ERROR, RooFit::IO, [&] {
      try {
         exported = RooJSONFactoryWSTool{ws}.exportJSONtoString();
      } catch (const std::runtime_error &) {
         threw = true;
      }
   });

   EXPECT_TRUE(threw);
   EXPECT_TRUE(exported.empty()) << "A HistFactory object was returned despite incompatible duplicate modifiers";
   EXPECT_NE(errors.find(expectedReason), std::string::npos) << errors;
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

TEST(RooFitHS3, ParameterPointsDoNotExportRanges)
{
   RooWorkspace ws{"workspace"};
   ws.factory("Gaussian::pdf(x[0, 10], mu[1, -5, 5], sigma[2, 0.1, 10])");

   const std::string json = RooJSONFactoryWSTool{ws}.exportJSONtoString();
   auto tree = RooFit::Detail::JSONTree::create(json);

   for (auto const &point : tree->rootnode()["parameter_points"].children()) {
      for (auto const &parameter : point["parameters"].children()) {
         EXPECT_FALSE(parameter.has_child("min")) << parameter["name"].val();
         EXPECT_FALSE(parameter.has_child("max")) << parameter["name"].val();
      }
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

   const std::string json = RooJSONFactoryWSTool{ws}.exportJSONtoString();
   auto tree = RooFit::Detail::JSONTree::create(json);
   auto const *defaultDomain = RooJSONFactoryWSTool::findNamedChild(tree->rootnode()["domains"], "default_domain");
   ASSERT_NE(defaultDomain, nullptr);

   auto const *xAxis = RooJSONFactoryWSTool::findNamedChild((*defaultDomain)["axes"], "x");
   ASSERT_NE(xAxis, nullptr);
   ASSERT_TRUE(xAxis->has_child("min"));
   ASSERT_TRUE(xAxis->has_child("max"));
   EXPECT_FALSE((*xAxis)["min"].is_null());
   EXPECT_FALSE((*xAxis)["max"].is_null());
   EXPECT_DOUBLE_EQ((*xAxis)["min"].val_double(), -10.0);
   EXPECT_DOUBLE_EQ((*xAxis)["max"].val_double(), 10.0);

   auto const *meanAxis = RooJSONFactoryWSTool::findNamedChild((*defaultDomain)["axes"], "mean");
   ASSERT_NE(meanAxis, nullptr);
   ASSERT_TRUE(meanAxis->has_child("min"));
   ASSERT_TRUE(meanAxis->has_child("max"));
   EXPECT_TRUE((*meanAxis)["min"].is_null());
   EXPECT_TRUE((*meanAxis)["max"].is_null());

   RooWorkspace imported;
   ASSERT_TRUE(RooJSONFactoryWSTool{imported}.importJSONfromString(json));
   auto *importedMean = imported.var("mean");
   ASSERT_NE(importedMean, nullptr);
   EXPECT_TRUE(std::isinf(importedMean->getMin()));
   EXPECT_LT(importedMean->getMin(), 0.0);
   EXPECT_TRUE(std::isinf(importedMean->getMax()));
   EXPECT_GT(importedMean->getMax(), 0.0);
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

   const std::string json = RooJSONFactoryWSTool{ws}.exportJSONtoString();
   auto tree = RooFit::Detail::JSONTree::create(json);
   auto const *defaultDomain = RooJSONFactoryWSTool::findNamedChild(tree->rootnode()["domains"], "default_domain");
   ASSERT_NE(defaultDomain, nullptr);

   auto const *uniformAxis = RooJSONFactoryWSTool::findNamedChild((*defaultDomain)["axes"], "uniform");
   ASSERT_NE(uniformAxis, nullptr);
   ASSERT_TRUE(uniformAxis->has_child("nbins"));
   EXPECT_EQ((*uniformAxis)["nbins"].val_int(), 7);
   EXPECT_FALSE(uniformAxis->has_child("edges"));

   auto const *nonuniformAxis = RooJSONFactoryWSTool::findNamedChild((*defaultDomain)["axes"], "nonuniform");
   ASSERT_NE(nonuniformAxis, nullptr);
   ASSERT_TRUE(nonuniformAxis->has_child("edges"));
   EXPECT_FALSE(nonuniformAxis->has_child("nbins"));
   auto const &edges = (*nonuniformAxis)["edges"];
   ASSERT_EQ(edges.num_children(), 4u);
   EXPECT_DOUBLE_EQ(edges.child(0).val_double(), 0.0);
   EXPECT_DOUBLE_EQ(edges.child(1).val_double(), 1.0);
   EXPECT_DOUBLE_EQ(edges.child(2).val_double(), 1.5);
   EXPECT_DOUBLE_EQ(edges.child(3).val_double(), 3.0);

   RooWorkspace imported;
   ASSERT_TRUE(RooJSONFactoryWSTool{imported}.importJSONfromString(json));
   auto *importedUniform = imported.var("uniform");
   ASSERT_NE(importedUniform, nullptr);
   EXPECT_EQ(importedUniform->getBins(), 7);

   auto *importedNonuniform = imported.var("nonuniform");
   ASSERT_NE(importedNonuniform, nullptr);
   auto const &importedBinning = importedNonuniform->getBinning();
   EXPECT_FALSE(importedBinning.isUniform());
   ASSERT_EQ(importedBinning.numBins(), 3);
   EXPECT_DOUBLE_EQ(importedBinning.binLow(0), 0.0);
   EXPECT_DOUBLE_EQ(importedBinning.binHigh(0), 1.0);
   EXPECT_DOUBLE_EQ(importedBinning.binHigh(1), 1.5);
   EXPECT_DOUBLE_EQ(importedBinning.binHigh(2), 3.0);
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

   RooFit::ModelConfig mc{"mc", &ws1};
   mc.SetPdf(*ws1.pdf("model"));
   mc.SetObservables("x");
   mc.SetParametersOfInterest("mu");
   mc.SetNuisanceParameters("sigma");
   ws1.import(mc);

   const std::string json = RooJSONFactoryWSTool{ws1}.exportJSONtoString();
   const std::string parameterStepWidths = parameterStepWidthsNode(json);
   ASSERT_FALSE(parameterStepWidths.empty()) << json;
   EXPECT_NE(parameterStepWidths.find("\"name\":\"mu\""), std::string::npos) << parameterStepWidths;
   EXPECT_NE(parameterStepWidths.find("\"name\":\"sigma\""), std::string::npos) << parameterStepWidths;
   EXPECT_NE(parameterStepWidths.find("\"name\":\"theta\""), std::string::npos) << parameterStepWidths;
   EXPECT_NE(parameterStepWidths.find("\"step_width\":0.12"), std::string::npos) << parameterStepWidths;
   EXPECT_NE(parameterStepWidths.find("\"step_width\":0.2"), std::string::npos) << parameterStepWidths;
   EXPECT_EQ(parameterStepWidths.find("\"error_lo\""), std::string::npos) << parameterStepWidths;
   EXPECT_EQ(parameterStepWidths.find("\"error_hi\""), std::string::npos) << parameterStepWidths;
   EXPECT_EQ(parameterStepWidths.find("\"name\":\"x\""), std::string::npos) << parameterStepWidths;

   RooWorkspace ws2{"workspace2"};
   ASSERT_TRUE(RooJSONFactoryWSTool{ws2}.importJSONfromString(json));

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

   const std::string json = RooJSONFactoryWSTool{ws1}.exportJSONtoString();
   const std::string parameterStepWidths = parameterStepWidthsNode(json);
   ASSERT_FALSE(parameterStepWidths.empty()) << json;
   EXPECT_NE(parameterStepWidths.find("\"name\":\"mu\""), std::string::npos) << parameterStepWidths;
   EXPECT_NE(parameterStepWidths.find("\"name\":\"sigma\""), std::string::npos) << parameterStepWidths;
   EXPECT_EQ(parameterStepWidths.find("\"name\":\"x\""), std::string::npos) << parameterStepWidths;

   RooWorkspace ws2{"workspace2"};
   ASSERT_TRUE(RooJSONFactoryWSTool{ws2}.importJSONfromString(json));

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

   const std::string json = RooJSONFactoryWSTool{ws}.exportJSONtoString();
   const auto fixedPos = json.find("\"const\":true,\"name\":\"fixed\"");
   ASSERT_NE(fixedPos, std::string::npos) << json;
   const auto fixedBegin = json.rfind("{", fixedPos);
   ASSERT_NE(fixedBegin, std::string::npos) << json;
   const auto fixedEnd = json.find("}", fixedPos);
   ASSERT_NE(fixedEnd, std::string::npos) << json;
   const std::string fixedNode = json.substr(fixedBegin, fixedEnd - fixedBegin);

   EXPECT_NE(fixedNode.find("\"const\":true"), std::string::npos) << fixedNode;
   EXPECT_EQ(fixedNode.find("\"min\""), std::string::npos) << fixedNode;
   EXPECT_EQ(fixedNode.find("\"max\""), std::string::npos) << fixedNode;
}

TEST(RooFitHS3, ParameterStepWidthsImportAfterDefaultSnapshot)
{
   const std::string json = R"({
      "metadata": {"hs3_version": "0.1.90"},
      "parameter_points": [
         {
            "name": "default_values",
            "parameters": [
               {"name": "mu", "value": 0.0, "err": 0.01}
            ]
         }
      ],
      "misc": {
         "minimization": {
            "parameter_stepwidths": [
               {"name": "mu", "step_width": 0.42},
               {"name": "missing", "step_width": 1.0}
            ]
         }
      }
   })";

   RooWorkspace ws{"workspace"};
   ScopedNoDomainConstVarImportFlag flagGuard{false};
   ASSERT_TRUE(RooJSONFactoryWSTool{ws}.importJSONfromString(json));

   ASSERT_NE(ws.var("mu"), nullptr);
   EXPECT_TRUE(ws.var("mu")->hasError());
   EXPECT_DOUBLE_EQ(ws.var("mu")->getError(), 0.42);
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

TEST(RooFitHS3, RooGaussianConstVarSigmaExport)
{
   ScopedNoDomainConstVarImportFlag flagGuard{true};

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

   const std::string json = RooJSONFactoryWSTool{ws}.exportJSONtoString();
   const std::string domainAxes = defaultDomainAxesNode(json);
   ASSERT_FALSE(domainAxes.empty()) << json;

   EXPECT_NE(json.find("\"sigma\":\"sigma_const\""), std::string::npos);
   EXPECT_NE(json.find("\"name\":\"sigma_const\""), std::string::npos);
   EXPECT_EQ(json.find("is_const_var"), std::string::npos);
   EXPECT_EQ(json.find("\"sigma\":1.0"), std::string::npos);
   EXPECT_NE(json.find("\"sigma\":2.0"), std::string::npos);
   EXPECT_EQ(domainAxes.find("\"name\":\"sigma_const\""), std::string::npos) << domainAxes;

   EXPECT_NE(json.find("\"sigma\":\"sigma_real\""), std::string::npos);
   EXPECT_NE(json.find("\"name\":\"sigma_real\""), std::string::npos);
   EXPECT_NE(domainAxes.find("\"name\":\"sigma_real\""), std::string::npos) << domainAxes;

   // The unbounded constant RooRealVar is still a RooRealVar, so it gets a
   // domain axis with explicit null bounds that distinguishes it from a RooConstVar.
   auto tree = RooFit::Detail::JSONTree::create(json);
   auto const *defaultDomain = RooJSONFactoryWSTool::findNamedChild(tree->rootnode()["domains"], "default_domain");
   ASSERT_NE(defaultDomain, nullptr);
   auto const *meanAxis = RooJSONFactoryWSTool::findNamedChild((*defaultDomain)["axes"], "mean");
   ASSERT_NE(meanAxis, nullptr);
   EXPECT_TRUE((*meanAxis)["min"].is_null());
   EXPECT_TRUE((*meanAxis)["max"].is_null());

   RooWorkspace imported;
   ASSERT_TRUE(RooJSONFactoryWSTool{imported}.importJSONfromString(json));
   EXPECT_NE(dynamic_cast<RooConstVar *>(imported.obj("sigma_const")), nullptr);
   EXPECT_EQ(imported.var("sigma_const"), nullptr);
   EXPECT_NE(dynamic_cast<RooRealVar *>(imported.obj("sigma_real")), nullptr);
   EXPECT_NE(dynamic_cast<RooRealVar *>(imported.obj("mean")), nullptr);

   const std::string roundTripJson = RooJSONFactoryWSTool{imported}.exportJSONtoString();
   const std::string roundTripDomainAxes = defaultDomainAxesNode(roundTripJson);
   EXPECT_NE(roundTripJson.find("\"sigma\":\"sigma_const\""), std::string::npos);
   EXPECT_EQ(roundTripJson.find("is_const_var"), std::string::npos);
   EXPECT_EQ(roundTripDomainAxes.find("\"name\":\"sigma_const\""), std::string::npos) << roundTripDomainAxes;

   const std::string legacyJson = R"({
      "metadata":{"hs3_version":"0.2"},
      "parameter_points":[{"name":"default_values","parameters":[
         {"name":"x","value":0.0},
         {"name":"mean","value":0.0},
         {"name":"sigma_const","value":1.0,"const":true}
      ]}],
      "distributions":[{"name":"gauss","type":"gaussian_dist","x":"x","mean":"mean","sigma":"sigma_const"}]
   })";
   RooWorkspace legacyImport;
   {
      ScopedNoDomainConstVarImportFlag legacyFlagGuard{false};
      ASSERT_TRUE(RooJSONFactoryWSTool{legacyImport}.importJSONfromString(legacyJson));
   }
   EXPECT_NE(dynamic_cast<RooRealVar *>(legacyImport.obj("sigma_const")), nullptr);
   EXPECT_EQ(dynamic_cast<RooConstVar *>(legacyImport.obj("sigma_const")), nullptr);
}

TEST(RooFitHS3, RooConstVarCollectionProxyExport)
{
   ScopedNoDomainConstVarImportFlag flagGuard{true};

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

   const std::string json = RooJSONFactoryWSTool{ws}.exportJSONtoString();
   const std::string domainAxes = defaultDomainAxesNode(json);
   ASSERT_FALSE(domainAxes.empty()) << json;
   EXPECT_NE(json.find("\"coefficients\":[\"frac_const\"]"), std::string::npos);
   EXPECT_NE(json.find("\"name\":\"frac_const\""), std::string::npos);
   EXPECT_EQ(json.find("is_const_var"), std::string::npos);
   EXPECT_EQ(domainAxes.find("\"name\":\"frac_const\""), std::string::npos) << domainAxes;

   RooWorkspace imported;
   ASSERT_TRUE(RooJSONFactoryWSTool{imported}.importJSONfromString(json));
   EXPECT_NE(dynamic_cast<RooConstVar *>(imported.obj("frac_const")), nullptr);
   EXPECT_EQ(imported.var("frac_const"), nullptr);
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

TEST(RooFitHS3, BinnedGenericPdfUniformRoundTrip)
{
   RooRealVar x{"x", "x", 0.0, 10.0};
   RooGenericPdf pdf{"binnedPdf", "binnedPdf", "floor(x / 2.0) + 1.0", RooArgList{x}};
   pdf.setBinning(x, RooUniformBinning{0.0, 10.0, 5}, /*checkFlatness=*/false);

   RooWorkspace source{"source"};
   source.import(pdf, RooFit::Silence());
   const std::string json = RooJSONFactoryWSTool{source}.exportJSONtoString();

   auto tree = RooFit::Detail::JSONTree::create(json);
   auto const *pdfNode = RooJSONFactoryWSTool::findNamedChild(tree->rootnode()["distributions"], "binnedPdf");
   ASSERT_NE(pdfNode, nullptr);
   ASSERT_TRUE(pdfNode->has_child("axes"));
   auto const *axis = RooJSONFactoryWSTool::findNamedChild((*pdfNode)["axes"], "x");
   ASSERT_NE(axis, nullptr);
   EXPECT_DOUBLE_EQ((*axis)["min"].val_double(), 0.0);
   EXPECT_DOUBLE_EQ((*axis)["max"].val_double(), 10.0);
   EXPECT_EQ((*axis)["nbins"].val_int(), 5);
   EXPECT_FALSE(axis->has_child("edges"));

   RooWorkspace imported{"imported"};
   ASSERT_TRUE(RooJSONFactoryWSTool{imported}.importJSONfromString(json));
   auto *importedPdf = dynamic_cast<RooGenericPdf *>(imported.pdf("binnedPdf"));
   ASSERT_NE(importedPdf, nullptr);
   auto *importedX = imported.var("x");
   ASSERT_NE(importedX, nullptr);
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
   const std::string json = RooJSONFactoryWSTool{source}.exportJSONtoString();

   auto tree = RooFit::Detail::JSONTree::create(json);
   auto *formulaNode = findMutableNamedChild(tree->rootnode()["functions"], "binnedFunction");
   ASSERT_NE(formulaNode, nullptr);
   EXPECT_EQ((*formulaNode)["type"].val(), "generic");
   ASSERT_TRUE(formulaNode->has_child("axes"));

   RooWorkspace legacyImport{"legacyImport"};
   ASSERT_TRUE(RooJSONFactoryWSTool{legacyImport}.importJSONfromString(json));
   auto *legacyFormula = dynamic_cast<RooFormulaVar *>(legacyImport.function("binnedFunction"));
   ASSERT_NE(legacyFormula, nullptr);
   ASSERT_NE(legacyFormula->getBinning(*legacyImport.var("x")), nullptr);

   (*formulaNode)["type"].clear();
   (*formulaNode)["type"] << "generic_function";
   RooWorkspace standardImport{"standardImport"};
   ASSERT_TRUE(RooJSONFactoryWSTool{standardImport}.importJSONfromString(jsonString(*tree)));
   auto *standardFormula = dynamic_cast<RooFormulaVar *>(standardImport.function("binnedFunction"));
   ASSERT_NE(standardFormula, nullptr);
   auto *standardX = standardImport.var("x");
   ASSERT_NE(standardX, nullptr);
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
   const std::string json = RooJSONFactoryWSTool{source}.exportJSONtoString();

   auto tree = RooFit::Detail::JSONTree::create(json);
   auto const *pdfNode = RooJSONFactoryWSTool::findNamedChild(tree->rootnode()["distributions"], "binnedPdf");
   ASSERT_NE(pdfNode, nullptr);
   ASSERT_TRUE(pdfNode->has_child("axes"));
   ASSERT_EQ((*pdfNode)["axes"].num_children(), 2u);
   auto const *xAxis = RooJSONFactoryWSTool::findNamedChild((*pdfNode)["axes"], "x");
   ASSERT_NE(xAxis, nullptr);
   ASSERT_TRUE(xAxis->has_child("edges"));
   ASSERT_EQ((*xAxis)["edges"].num_children(), 4u);
   for (std::size_t i = 0; i < 4; ++i) {
      EXPECT_DOUBLE_EQ((*xAxis)["edges"].child(i).val_double(), xEdges[i]);
   }

   RooWorkspace imported{"imported"};
   ASSERT_TRUE(RooJSONFactoryWSTool{imported}.importJSONfromString(json));
   auto *importedPdf = dynamic_cast<RooGenericPdf *>(imported.pdf("binnedPdf"));
   ASSERT_NE(importedPdf, nullptr);
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
   auto tree = RooFit::Detail::JSONTree::create(RooJSONFactoryWSTool{source}.exportJSONtoString());

   auto const *pdfNode = RooJSONFactoryWSTool::findNamedChild(tree->rootnode()["distributions"], "ordinaryPdf");
   auto const *functionNode = RooJSONFactoryWSTool::findNamedChild(tree->rootnode()["functions"], "ordinaryFunction");
   ASSERT_NE(pdfNode, nullptr);
   ASSERT_NE(functionNode, nullptr);
   EXPECT_FALSE(pdfNode->has_child("axes"));
   EXPECT_FALSE(functionNode->has_child("axes"));
}

TEST(RooFitHS3, BinnedGenericRejectsMalformedAxes)
{
   RooRealVar x{"x", "x", 0.0, 10.0};
   RooGenericPdf pdf{"binnedPdf", "binnedPdf", "floor(x)", RooArgList{x}};
   pdf.setBinning(x, RooUniformBinning{0.0, 10.0, 5}, /*checkFlatness=*/false);

   RooWorkspace source{"source"};
   source.import(pdf, RooFit::Silence());
   const std::string validJson = RooJSONFactoryWSTool{source}.exportJSONtoString();

   auto expectRejected = [&](auto configure) {
      auto tree = RooFit::Detail::JSONTree::create(validJson);
      auto *pdfNode = findMutableNamedChild(tree->rootnode()["distributions"], "binnedPdf");
      ASSERT_NE(pdfNode, nullptr);
      configure(*pdfNode);

      RooWorkspace imported{"imported"};
      RooJSONFactoryWSTool tool{imported};
      EXPECT_THROW(tool.importJSONfromString(jsonString(*tree)), std::runtime_error);
   };

   expectRejected([](RooFit::Detail::JSONNode &pdfNode) {
      pdfNode["axes"].clear();
      pdfNode["axes"] << "not-a-sequence";
   });

   expectRejected([](RooFit::Detail::JSONNode &pdfNode) {
      auto &duplicate = pdfNode["axes"].append_child().set_map();
      duplicate["name"] << "x";
      duplicate["min"] << 0.0;
      duplicate["max"] << 10.0;
      duplicate["nbins"] << 5;
   });

   expectRejected([](RooFit::Detail::JSONNode &pdfNode) {
      auto &axis = pdfNode["axes"].child(0);
      axis["name"].clear();
      axis["name"] << "notADependency";
   });

   expectRejected([](RooFit::Detail::JSONNode &pdfNode) {
      auto &axis = pdfNode["axes"].child(0);
      axis["nbins"].clear();
      axis["nbins"] << 2.5;
   });

   expectRejected([](RooFit::Detail::JSONNode &pdfNode) {
      auto &axis = pdfNode["axes"].child(0);
      axis["nbins"].clear();
      axis["nbins"] << 0;
   });

   expectRejected([](RooFit::Detail::JSONNode &pdfNode) {
      auto &axis = pdfNode["axes"].child(0);
      axis["max"].clear();
      axis["max"] << 0.0;
   });

   expectRejected([](RooFit::Detail::JSONNode &pdfNode) {
      auto &axis = pdfNode["axes"].child(0);
      axis.clear();
      axis.set_map();
      axis["name"] << "x";
      auto &edges = axis["edges"].set_seq();
      edges.append_child() << 0.0;
   });

   expectRejected([](RooFit::Detail::JSONNode &pdfNode) {
      auto &axis = pdfNode["axes"].child(0);
      axis.clear();
      axis.set_map();
      axis["name"] << "x";
      auto &edges = axis["edges"].set_seq();
      edges.append_child() << 0.0;
      edges.append_child() << 2.0;
      edges.append_child() << 1.0;
   });
}

TEST(RooFitHS3, GenericExpressionCleanup)
{
   RooRealVar x{"x", "x", 0.5, -1.0, 1.0};
   RooFormulaVar formula{"formula", "formula",
                         "TMath::Floor(x) + TMath::Ceil(x) + TMath::Abs(x) + TMath::Tan(x) + "
                         "TMath::ASin(x / 2.) + TMath::ACos(x / 2.) + TMath::ATan(x) + TMath::Pi() + TMath::E()",
                         RooArgList{x}};

   RooWorkspace ws1{"ws_expr_cleanup"};
   ws1.import(formula, RooFit::Silence());
   const std::string json = RooJSONFactoryWSTool{ws1}.exportJSONtoString();

   for (const char *token : {"floor", "ceil", "abs", "tan", "asin", "acos", "atan", "PI", "EULER"}) {
      EXPECT_NE(json.find(token), std::string::npos) << json;
   }
   EXPECT_EQ(json.find("TMath::Pi"), std::string::npos) << json;
   EXPECT_EQ(json.find("TMath::E"), std::string::npos) << json;

   RooWorkspace ws2{"ws_expr_cleanup_2"};
   ASSERT_TRUE(RooJSONFactoryWSTool{ws2}.importJSONfromString(json));
   auto *imported = ws2.function("formula");
   ASSERT_NE(imported, nullptr);
   ws2.var("x")->setVal(0.5);
   EXPECT_DOUBLE_EQ(imported->getVal(), formula.getVal());
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
   const std::string json = RooJSONFactoryWSTool{ws1}.exportJSONtoString();

   for (const char *token : {"PIOver2", "tanH", "sinH"}) {
      EXPECT_EQ(json.find(token), std::string::npos) << json;
   }

   RooWorkspace ws2{"ws_expr_cleanup_prefixes_2"};
   ASSERT_TRUE(RooJSONFactoryWSTool{ws2}.importJSONfromString(json));
   auto *imported = ws2.function("formula");
   ASSERT_NE(imported, nullptr);
   ws2.var("x")->setVal(0.5);
   EXPECT_DOUBLE_EQ(imported->getVal(), formula.getVal());
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

   const std::string json = RooJSONFactoryWSTool{ws1}.exportJSONtoString();
   EXPECT_NE(json.find("\"type\":\"binvolume\""), std::string::npos) << json;
   EXPECT_NE(json.find("\"type\":\"inverse_binvolume\""), std::string::npos) << json;
   EXPECT_EQ(json.find("divideByBinWidth"), std::string::npos) << json;
   EXPECT_EQ(json.find("\"type\":\"binwidth\""), std::string::npos) << json;

   RooWorkspace ws2{"ws_binvolume_2"};
   ASSERT_TRUE(RooJSONFactoryWSTool{ws2}.importJSONfromString(json));
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

   const std::string json = RooJSONFactoryWSTool{ws1}.exportJSONtoString();
   EXPECT_NE(json.find("\"name\":\"histFunc\",\"type\":\"step\""), std::string::npos) << json;
   EXPECT_EQ(json.find("\"name\":\"histFunc\",\"type\":\"histogram\""), std::string::npos) << json;
   const auto paramHistFuncPos = json.find("\"name\":\"paramHistFunc\"");
   ASSERT_NE(paramHistFuncPos, std::string::npos) << json;
   const auto paramHistFuncEnd = json.find("}", paramHistFuncPos);
   ASSERT_NE(paramHistFuncEnd, std::string::npos) << json;
   const std::string paramHistFuncNode = json.substr(paramHistFuncPos, paramHistFuncEnd - paramHistFuncPos);
   EXPECT_NE(paramHistFuncNode.find("\"type\":\"step\""), std::string::npos) << paramHistFuncNode;

   RooWorkspace ws2{"ws_step_2"};
   ASSERT_TRUE(RooJSONFactoryWSTool{ws2}.importJSONfromString(json));
   EXPECT_NE(dynamic_cast<RooHistFunc *>(ws2.function("histFunc")), nullptr);
   EXPECT_NE(dynamic_cast<ParamHistFunc *>(ws2.function("paramHistFunc")), nullptr);
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

   RooWorkspace ws;
   ws.factory("Polynomial::poly1(x[0, 10], {a_2[0.003, -10, 10]}, 2)");
   const std::string json = RooJSONFactoryWSTool{ws}.exportJSONtoString();
   EXPECT_NE(json.find("\"coefficients\":[1.0,0.0,\"a_2\"]"), std::string::npos) << json;
   EXPECT_EQ(json.find("\"coefficients\":[\"1.0\""), std::string::npos) << json;
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
   registerExporter<TestExporter<1>>(klass, /*topPriority=*/true);
   RooJSONFactoryWSTool{ws}.exportJSONtoString();
   EXPECT_EQ(TestExporter<1>::callCounter()--, 1);

   // 2. Add new exporter by class pointer with bottom priority.
   //    We expect the previous TestExporter<1> to still be used.
   registerExporter<TestExporter<2>>(klass, /*topPriority=*/false);
   RooJSONFactoryWSTool{ws}.exportJSONtoString();
   EXPECT_EQ(TestExporter<1>::callCounter()--, 1);

   // 3. Add new exporter by name with top priority.
   //    We expect this to get used.
   registerExporter<TestExporter<3>>(std::string{className}, /*topPriority=*/true);
   RooJSONFactoryWSTool{ws}.exportJSONtoString();
   EXPECT_EQ(TestExporter<3>::callCounter()--, 1);

   // 4. Add new exporter by name with bottom priority.
   //    We expect the previous TestExporter<3> to still be used.
   registerExporter<TestExporter<4>>(std::string{className}, /*topPriority=*/false);
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
   EXPECT_EQ(axesNode.find("\"value\""), std::string::npos) << axesNode;
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

TEST(RooFitHS3, HistFactoryDuplicateModifiersAreCombined)
{
   const std::string jsonStr = R"({
      "metadata": {"hs3_version": "0.1.90"},
      "distributions": [
         {
            "name": "model_channel0",
            "type": "histfactory_dist",
            "axes": [
               {"name": "x", "min": 0.0, "max": 2.0, "nbins": 2}
            ],
            "samples": [
               {
                  "name": "sig",
                  "data": {"contents": [10.0, 20.0]},
                  "modifiers": [
                     {
                        "name": "norm",
                        "parameter": "alpha_norm",
                        "type": "normsys",
                        "interpolation": 1,
                        "data": {"lo": 0.8, "hi": 1.2}
                     },
                     {
                        "name": "norm",
                        "parameter": "alpha_norm",
                        "type": "normsys",
                        "interpolation": 1,
                        "data": {"lo": 0.9, "hi": 1.1}
                     },
                     {
                        "name": "poly",
                        "parameter": "alpha_poly",
                        "type": "normsys",
                        "data": {"lo": 0.8, "hi": 1.2}
                     },
                     {
                        "name": "poly",
                        "parameter": "alpha_poly",
                        "type": "normsys",
                        "data": {"lo": 0.75, "hi": 1.25}
                     },
                     {
                        "name": "shape_input_1",
                        "parameter": "alpha_shape",
                        "type": "histosys",
                        "data": {
                           "lo": {"contents": [8.0, 18.0]},
                           "hi": {"contents": [12.0, 22.0]}
                        }
                     },
                     {
                        "name": "shape_input_2",
                        "parameter": "alpha_shape",
                        "type": "histosys",
                        "data": {
                           "lo": {"contents": [9.0, 16.0]},
                           "hi": {"contents": [11.0, 24.0]}
                        }
                     }
                  ]
               }
            ]
         }
      ]
   })";

   RooWorkspace ws1{"ws_duplicate_modifiers"};
   ASSERT_TRUE(RooJSONFactoryWSTool{ws1}.importJSONfromString(jsonStr));

   std::string exported;
   const std::string warnings =
      captureMessages(RooFit::WARNING, RooFit::IO, [&] { exported = RooJSONFactoryWSTool{ws1}.exportJSONtoString(); });

   EXPECT_NE(warnings.find("combined 2 duplicate modifiers named 'norm' of type 'normsys'"), std::string::npos)
      << warnings;
   EXPECT_NE(warnings.find("combined 2 duplicate modifiers named 'poly' of type 'normsys'"), std::string::npos)
      << warnings;
   EXPECT_NE(warnings.find("combined 2 duplicate modifiers named 'shape' of type 'histosys'"), std::string::npos)
      << warnings;

   EXPECT_EQ(countOccurrences(exported, "\"name\":\"norm\""), 1u) << exported;
   EXPECT_EQ(countOccurrences(exported, "\"name\":\"poly\""), 1u) << exported;
   EXPECT_EQ(countOccurrences(exported, "\"name\":\"shape\""), 1u) << exported;

   RooWorkspace ws2{"ws_duplicate_modifiers_roundtrip"};
   ASSERT_TRUE(RooJSONFactoryWSTool{ws2}.importJSONfromString(exported));

   auto *normInterp = dynamic_cast<RooStats::HistFactory::FlexibleInterpVar *>(ws2.function("sig_channel0_epsilon"));
   ASSERT_NE(normInterp, nullptr);
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

   auto *shapeInterp = dynamic_cast<PiecewiseInterpolation *>(ws2.function("histoSys_model_channel0_sig"));
   ASSERT_NE(shapeInterp, nullptr);
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
   // also tested away from their anchors.
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
   const std::string sentinelErrors = captureMessages(
      RooFit::ERROR, RooFit::IO, [] { RooJSONFactoryWSTool::warning("duplicate-modifier warning-level sentinel"); });
   EXPECT_TRUE(sentinelErrors.empty()) << sentinelErrors;
}

struct IncompatibleModifierCase {
   std::string name;
   std::string modifiers;
   std::string extraDistributions;
   std::string expectedReason;
};

std::string makeIncompatibleModifierWorkspace(const IncompatibleModifierCase &testCase)
{
   return R"({
      "metadata": {"hs3_version": "0.1.90"},
      "domains": [
         {
            "name": "default_domain",
            "type": "product_domain",
            "axes": [
               {"name": "alpha_dup", "min": -5.0, "max": 5.0},
               {"name": "dup", "min": -5.0, "max": 5.0}
            ]
         }
      ],
      "parameter_points": [
         {
            "name": "default_values",
            "parameters": [
               {"name": "alpha_dup", "value": 0.0},
               {"name": "dup", "value": 0.0}
            ]
         }
      ],
      "distributions": [)" +
          testCase.extraDistributions + R"(
         {
            "name": "model_channel0",
            "type": "histfactory_dist",
            "axes": [
               {"name": "x", "min": 0.0, "max": 2.0, "nbins": 2}
            ],
            "samples": [
               {
                  "name": "sig",
                  "data": {"contents": [10.0, 20.0]},
                  "modifiers": [)" +
          testCase.modifiers + R"(
                  ]
               }
            ]
         }
      ]
   })";
}

class HistFactoryIncompatibleDuplicateModifiers : public testing::TestWithParam<IncompatibleModifierCase> {};

TEST_P(HistFactoryIncompatibleDuplicateModifiers, ExportFails)
{
   const auto &testCase = GetParam();
   RooWorkspace ws{"ws_incompatible_duplicate_modifiers"};
   ASSERT_TRUE(RooJSONFactoryWSTool{ws}.importJSONfromString(makeIncompatibleModifierWorkspace(testCase)));

   expectExportThrowsWithError(ws, testCase.expectedReason);
}

INSTANTIATE_TEST_SUITE_P(
   HistFactory, HistFactoryIncompatibleDuplicateModifiers,
   testing::Values(IncompatibleModifierCase{"Parameter",
                                            R"(
            {
               "name": "input_1", "parameter": "alpha_dup", "type": "normsys", "constraint": "sharedConstraint",
               "interpolation": 1, "data": {"lo": 0.8, "hi": 1.2}
            },
            {
               "name": "input_2", "parameter": "dup", "type": "normsys", "constraint": "sharedConstraint",
               "interpolation": 1, "data": {"lo": 0.9, "hi": 1.1}
            })",
                                            R"(
            {
               "name": "sharedConstraint", "type": "gaussian_dist", "x": "alpha_dup", "mean": "dup", "sigma": 1.0
            },)",
                                            "parameter metadata differs"},
                   IncompatibleModifierCase{"Interpolation",
                                            R"(
            {
               "name": "input_1", "parameter": "alpha_dup", "type": "normsys",
               "interpolation": 1, "data": {"lo": 0.8, "hi": 1.2}
            },
            {
               "name": "input_2", "parameter": "alpha_dup", "type": "normsys",
               "interpolation": 4, "data": {"lo": 0.9, "hi": 1.1}
            })",
                                            "", "interpolation codes differ"},
                   IncompatibleModifierCase{"Constraint",
                                            R"(
            {
               "name": "input_1", "parameter": "alpha_dup", "type": "normsys",
               "interpolation": 1, "data": {"lo": 0.8, "hi": 1.2}
            },
            {
               "name": "input_2", "parameter": "dup", "type": "normsys",
               "interpolation": 1, "data": {"lo": 0.9, "hi": 1.1}
            })",
                                            "", "constraint metadata differs"},
                   IncompatibleModifierCase{"UnrepresentableNormFactor",
                                            R"(
            {"name": "mu", "type": "normfactor"},
            {"name": "mu", "type": "normfactor"})",
                                            "", "cannot be combined without changing its meaning"},
                   IncompatibleModifierCase{"LinearSpaceNormSys",
                                            R"(
            {
               "name": "input_1", "parameter": "alpha_dup", "type": "normsys",
               "interpolation": 0, "data": {"lo": 0.8, "hi": 1.2}
            },
            {
               "name": "input_2", "parameter": "alpha_dup", "type": "normsys",
               "interpolation": 0, "data": {"lo": 0.9, "hi": 1.1}
            })",
                                            "", "multiplicative combination is only valid for log-space"}),
   [](const testing::TestParamInfo<IncompatibleModifierCase> &paramInfo) { return paramInfo.param.name; });

TEST(RooFitHS3, HistFactoryDuplicateHistoSysWithDifferentBinningFails)
{
   RooRealVar y{"y", "y", 0.0, 3.0};
   y.setBins(3);

   RooWorkspace ws{"ws_incompatible_histosys_binning"};
   importDuplicateHistoSysModel(ws, /*interpCode=*/4, /*low2=*/{7.0, 8.0, 9.0}, /*high2=*/{13.0, 14.0, 15.0}, &y);

   expectExportThrowsWithError(ws, "histogram binning differs");
}

TEST(RooFitHS3, HistFactoryDuplicateHistoSysWithNonDefaultInterpolationFails)
{
   RooWorkspace ws{"ws_incompatible_histosys_interpolation"};
   importDuplicateHistoSysModel(ws, /*interpCode=*/2, /*low2=*/{9.0, 16.0}, /*high2=*/{11.0, 24.0});

   expectExportThrowsWithError(ws, "non-default interpolation cannot currently be represented by the HS3 exporter");
}

TEST(RooFitHS3, HistFactoryConstraintKeyMigration)
{
   const std::string jsonStr = R"({
      "metadata": {"hs3_version": "0.1.90"},
      "domains": [
         {
            "name": "default_domain",
            "type": "product_domain",
            "axes": [
               {"name": "obs_channel0", "min": 0.0, "max": 2.0, "nbins": 2},
               {"name": "nom_mu", "min": 1.0, "max": 1.0},
               {"name": "sigma_mu", "min": 1.0, "max": 1.0}
            ]
         }
      ],
      "parameter_points": [
         {
            "name": "default_values",
            "parameters": [
               {"name": "nom_mu", "value": 1.0, "const": true},
               {"name": "sigma_mu", "value": 1.0, "const": true}
            ]
         }
      ],
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
                  "data": {"contents": [10.0, 20.0]},
                  "modifiers": [
                     {
                        "name": "mu",
                        "parameter": "mu",
                        "type": "normfactor",
                        "constraint_name": "muConstraint"
                     }
                  ]
               }
            ]
         },
         {
            "name": "muConstraint",
            "type": "gaussian_dist",
            "x": "mu",
            "mean": "nom_mu",
            "sigma": "sigma_mu"
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

   RooWorkspace ws{"ws_hf_constraint"};
   ASSERT_TRUE(RooJSONFactoryWSTool{ws}.importJSONfromString(jsonStr));

   const std::string exported = RooJSONFactoryWSTool{ws}.exportJSONtoString();
   EXPECT_NE(exported.find("\"constraint\":\"muConstraint\""), std::string::npos) << exported;
   EXPECT_EQ(exported.find("constraint_name"), std::string::npos) << exported;
   EXPECT_EQ(exported.find("constraint_type"), std::string::npos) << exported;
}

TEST(RooFitHS3, HistFactoryLegacyConstraintTypeInConstraintKey)
{
   const std::string jsonStr = R"({
      "metadata": {"hs3_version": "0.1.90"},
      "domains": [
         {
            "name": "default_domain",
            "type": "product_domain",
            "axes": [
               {"name": "obs_channel0", "min": 0.0, "max": 2.0, "nbins": 2}
            ]
         }
      ],
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
                  "data": {"contents": [10.0, 20.0]},
                  "modifiers": [
                     {
                        "name": "lumi",
                        "type": "normsys",
                        "constraint": "Gauss",
                        "data": {"lo": 0.95, "hi": 1.05}
                     }
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

   RooWorkspace ws{"ws_hf_legacy_constraint_type"};
   ASSERT_TRUE(RooJSONFactoryWSTool{ws}.importJSONfromString(jsonStr));

   auto *constraint = dynamic_cast<RooGaussian *>(ws.pdf("alpha_lumiConstraint"));
   ASSERT_NE(constraint, nullptr);
   auto *alpha = ws.var("alpha_lumi");
   ASSERT_NE(alpha, nullptr);
   EXPECT_DOUBLE_EQ(alpha->getError(), 1.0);

   const std::string exported = RooJSONFactoryWSTool{ws}.exportJSONtoString();
   EXPECT_NE(exported.find("\"constraint\":\"alpha_lumiConstraint\""), std::string::npos) << exported;
   EXPECT_EQ(exported.find("\"constraint\":\"Gauss\""), std::string::npos) << exported;
   EXPECT_EQ(exported.find("constraint_name"), std::string::npos) << exported;
   EXPECT_EQ(exported.find("constraint_type"), std::string::npos) << exported;
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

namespace {

std::unique_ptr<RooFitResult> writeJSONAndFitModel(std::string &jsonStr)
{
   using namespace RooFit;

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

   RooFit::ModelConfig modelConfig{"ModelConfig"};

   modelConfig.SetWS(ws);
   modelConfig.SetPdf("simPdf");
   modelConfig.SetParametersOfInterest("mean");
   modelConfig.SetObservables("x_1,x_2");

   ws.import(modelConfig);

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

   auto &pdf = *ws.pdf("simPdf");
   auto &data = *ws.data("obsData");

   // Export before fitting to keep the prefit values
   jsonStr = RooJSONFactoryWSTool{ws}.exportJSONtoString();

   return std::unique_ptr<RooFitResult>{
      pdf.fitTo(data, Save(), PrintLevel(-1), PrintEvalErrors(-1), Minimizer("Minuit2"))};
}

std::unique_ptr<RooFitResult> readJSONAndFitModel(std::string const &jsonStr)
{
   using namespace RooFit;

   RooWorkspace ws{"workspace"};
   RooJSONFactoryWSTool tool{ws};

   tool.importJSONfromString(jsonStr);

   // Make sure that there is exactly one dataset in the new workspace, and
   // that there are no spurious datasets left over from first importing the
   // channel datasets that later get merged to the combined dataset
   EXPECT_EQ(ws.allData().size(), 1) << "Unexpected number of datasets in the new workspace";

   auto &pdf = *ws.pdf("simPdf");
   auto &data = *ws.data("obsData");

   return std::unique_ptr<RooFitResult>{
      pdf.fitTo(data, Save(), PrintLevel(-1), PrintEvalErrors(-1), Minimizer("Minuit2"))};
}

} // namespace

TEST(RooFitHS3, SimultaneousFit)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   using namespace RooFit;

   std::string jsonStr;

   std::unique_ptr<RooFitResult> res1 = writeJSONAndFitModel(jsonStr);
   std::unique_ptr<RooFitResult> res2 = readJSONAndFitModel(jsonStr);

   // todo: also check the modelconfig for equality

   EXPECT_TRUE(res2->isIdentical(*res1));
}

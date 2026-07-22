/*
 * Project: RooFit
 * Authors:
 *   Carsten D. Burgard, DESY/ATLAS, Dec 2021
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFitHS3/RooJSONFactoryWSTool.h>

#include <RooAbsCachedPdf.h>
#include <RooAddPdf.h>
#include <RooAddModel.h>
#include <RooBinning.h>
#include <RooBinSamplingPdf.h>
#include <RooBinWidthFunction.h>
#include <RooCategory.h>
#include <RooDataHist.h>
#include <RooDecay.h>
#include <RooDerivative.h>
#include <RooExponential.h>
#include <RooExtendPdf.h>
#include <RooFFTConvPdf.h>
#include <RooFit/Detail/JSONInterface.h>
#include <RooFitHS3/JSONIO.h>
#include <RooFormulaVar.h>
#include <RooGenericPdf.h>
#include <RooHistFunc.h>
#include <RooHistPdf.h>
#include <RooLegacyExpPoly.h>
#include <RooLognormal.h>
#include <RooMultiVarGaussian.h>
#include <RooStats/HistFactory/ParamHistFunc.h>
#include <RooAddition.h>
#include <RooProduct.h>
#include <RooProdPdf.h>
#include <RooPoisson.h>
#include <RooPolynomial.h>
#include <RooPolyVar.h>
#include <RooAbsRealLValue.h>
#include <RooRealSumFunc.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>
#include <RooResolutionModel.h>
#include <RooTFnBinding.h>
#include <RooTruthModel.h>
#include <RooGaussModel.h>
#include <RooWrapperPdf.h>
#include <RooWorkspace.h>
#include <RooRealIntegral.h>
#include <RooSpline.h>
#include <RooUniformBinning.h>
#include <TSpline.h>

#include <TF1.h>
#include <TH1.h>

#include "JSONIOUtils.h"

#include "static_execute.h"

#include <algorithm>
#include <charconv>
#include <cctype>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <string_view>
#include <vector>

using RooFit::Detail::JSONNode;

///////////////////////////////////////////////////////////////////////////////////////////////////////
// individually implemented importers
///////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
bool isReservedExpressionIdentifier(const std::string &arg)
{
   return arg == "PI" || arg == "EULER" || arg == "TMath";
}

/**
 * Extracts arguments from a mathematical expression.
 *
 * This function takes a string representing a mathematical
 * expression and extracts the arguments from it.  The arguments are
 * defined as sequences of characters that do not contain digits,
 * spaces, or parentheses, and that start with a letter. Function
 * calls such as "exp( ... )", identified as being followed by an
 * opening parenthesis, are not treated as arguments. The extracted
 * arguments are returned as a vector of strings.
 *
 * @param expr A string representing a mathematical expression.
 * @return A set of unique strings representing the extracted arguments.
 */
std::set<std::string> extractArguments(std::string expr)
{
   // Get rid of whitespaces
   expr.erase(std::remove_if(expr.begin(), expr.end(), [](unsigned char c) { return std::isspace(c); }), expr.end());

   std::set<std::string> arguments;
   size_t startidx = expr.size();
   for (size_t i = 0; i < expr.size(); ++i) {
      if (startidx >= expr.size()) {
         if (isalpha(expr[i])) {
            startidx = i;
            // check this character is not part of scientific notation, e.g. 2e-5
            if (TFormula::IsScientificNotation(expr, i)) {
               // if it is, we ignore this character
               startidx = expr.size();
            }
         }
      } else {
         if (!isdigit(expr[i]) && !isalpha(expr[i]) && expr[i] != '_') {
            if (expr[i] == '(') {
               startidx = expr.size();
               continue;
            }
            std::string arg(expr.substr(startidx, i - startidx));
            startidx = expr.size();
            if (!isReservedExpressionIdentifier(arg)) {
               arguments.insert(arg);
            }
         }
      }
   }
   if (startidx < expr.size()) {
      std::string arg(expr.substr(startidx));
      if (!isReservedExpressionIdentifier(arg)) {
         arguments.insert(arg);
      }
   }
   return arguments;
}

void replaceIdentifier(TString &expr, std::string_view identifier, std::string_view replacement)
{
   std::string in(expr.Data());
   std::string out;
   out.reserve(in.size());

   for (std::size_t pos = 0; pos < in.size();) {
      const bool matches = in.compare(pos, identifier.size(), identifier) == 0;
      const bool beforeIdentifier =
         pos > 0 && (std::isalnum(static_cast<unsigned char>(in[pos - 1])) || in[pos - 1] == '_');
      const std::size_t end = pos + identifier.size();
      const bool afterIdentifier =
         end < in.size() && (std::isalnum(static_cast<unsigned char>(in[end])) || in[end] == '_');
      if (matches && !beforeIdentifier && !afterIdentifier) {
         out.append(replacement);
         pos = end;
      } else {
         out.push_back(in[pos]);
         ++pos;
      }
   }

   expr = out.c_str();
}

void translateImportedExpression(TString &expr)
{
   replaceIdentifier(expr, "PI", "TMath::Pi()");
   replaceIdentifier(expr, "EULER", "TMath::E()");
}

int readPositiveInteger(const JSONNode &node, const std::string &context)
{
   const std::string value = node.val();
   int out = 0;
   const auto result = std::from_chars(value.data(), value.data() + value.size(), out);
   if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || out <= 0) {
      RooJSONFactoryWSTool::error("\"nbins\" in " + context + " must be a positive integer");
   }
   return out;
}

std::unique_ptr<RooAbsBinning>
readFormulaAxisBinning(const JSONNode &axis, const std::string &axisName, const std::string &formulaName)
{
   const std::string context = "axis '" + axisName + "' of generic formula '" + formulaName + "'";
   const bool hasEdges = axis.has_child("edges");
   const bool hasMin = axis.has_child("min");
   const bool hasMax = axis.has_child("max");
   const bool hasNBins = axis.has_child("nbins");

   if (hasEdges && (hasMin || hasMax || hasNBins)) {
      RooJSONFactoryWSTool::error(context + " must use either \"edges\" or \"min\"/\"max\"/\"nbins\"");
   }

   if (hasEdges) {
      const JSONNode &edgesNode = axis["edges"];
      if (!edgesNode.is_seq()) {
         RooJSONFactoryWSTool::error("\"edges\" in " + context + " must be a sequence");
      }

      std::vector<double> edges;
      edges.reserve(edgesNode.num_children());
      for (const JSONNode &edgeNode : edgesNode.children()) {
         if (!edgeNode.is_number()) {
            RooJSONFactoryWSTool::error("\"edges\" in " + context + " must contain only finite values");
         }
         const double edge = edgeNode.val_double();
         if (!std::isfinite(edge)) {
            RooJSONFactoryWSTool::error("\"edges\" in " + context + " must contain only finite values");
         }
         if (!edges.empty() && edge <= edges.back()) {
            RooJSONFactoryWSTool::error("\"edges\" in " + context + " must be strictly increasing");
         }
         edges.push_back(edge);
      }
      if (edges.size() < 2) {
         RooJSONFactoryWSTool::error("\"edges\" in " + context + " must contain at least two values");
      }
      return std::make_unique<RooBinning>(static_cast<int>(edges.size() - 1), edges.data());
   }

   if (!hasMin || !hasMax || !hasNBins) {
      RooJSONFactoryWSTool::error(context + " must define \"min\", \"max\", and \"nbins\"");
   }

   if (!axis["min"].is_number() || !axis["max"].is_number()) {
      RooJSONFactoryWSTool::error("\"min\" and \"max\" in " + context + " must be finite and increasing");
   }
   const double min = axis["min"].val_double();
   const double max = axis["max"].val_double();
   if (!std::isfinite(min) || !std::isfinite(max) || max <= min) {
      RooJSONFactoryWSTool::error("\"min\" and \"max\" in " + context + " must be finite and increasing");
   }
   return std::make_unique<RooUniformBinning>(min, max, readPositiveInteger(axis["nbins"], context));
}

template <class RooArg_t>
void importFormulaBinnings(RooArg_t &arg, const JSONNode &node)
{
   if (!node.has_child("axes")) {
      return;
   }

   const JSONNode &axes = node["axes"];
   if (!axes.is_seq()) {
      RooJSONFactoryWSTool::error("\"axes\" in generic formula '" + std::string(arg.GetName()) +
                                  "' must be a sequence");
   }

   std::set<std::string> axisNames;
   for (const JSONNode &axis : axes.children()) {
      if (!axis.is_map() || !axis.has_child("name")) {
         RooJSONFactoryWSTool::error("each axis in generic formula '" + std::string(arg.GetName()) +
                                     "' must be a map with a \"name\"");
      }
      const std::string axisName = axis["name"].val();
      if (!axisNames.insert(axisName).second) {
         RooJSONFactoryWSTool::error("duplicate axis '" + axisName + "' in generic formula '" + arg.GetName() + "'");
      }

      auto *observable = dynamic_cast<RooAbsRealLValue *>(arg.getParameter(axisName.c_str()));
      if (!observable) {
         RooJSONFactoryWSTool::error(
            "axis '" + axisName + "' is not a real-valued formula variable of generic formula '" + arg.GetName() + "'");
      }

      std::unique_ptr<RooAbsBinning> binning = readFormulaAxisBinning(axis, axisName, arg.GetName());
      arg.setBinning(*observable, *binning, /*checkFlatness=*/false);
   }
}

template <class RooArg_t>
bool importFormulaArg(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   if (!p.has_child("expression")) {
      RooJSONFactoryWSTool::error("no expression given for '" + name + "'");
   }
   TString formula(p["expression"].val());
   translateImportedExpression(formula);
   RooArgList dependents;
   for (const auto &d : extractArguments(formula.Data())) {
      dependents.add(*tool->request<RooAbsReal>(d, name));
   }
   RooArg_t arg{name.c_str(), formula, dependents};
   importFormulaBinnings(arg, p);
   tool->wsImport(arg);
   return true;
}

// Fast-path importers for RooProduct, RooAddition, and RooProdPdf that
// bypass the generic factory-expression mechanism. The default path
// generates a string expression and passes it to gROOT->ProcessLineFast(),
// which invokes the Cling JIT for every single call. For workspaces with
// thousands of product/sum nodes (a common shape for HistFactory models)
// that JIT cost dominates JSON import time. Constructing the RooFit object
// directly here keeps the work O(N) of cheap C++ calls.
bool importProduct(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   tool->wsEmplace<RooProduct>(name, tool->requestArgList<RooAbsReal>(p, "factors"));
   return true;
}

bool importProdPdf(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   tool->wsEmplace<RooProdPdf>(name, tool->requestArgList<RooAbsPdf>(p, "factors"));
   return true;
}

bool importAddition(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   tool->wsEmplace<RooAddition>(name, tool->requestArgList<RooAbsReal>(p, "summands"));
   return true;
}

bool importAddPdf(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   if (!tool->requestArgList<RooAbsReal>(p, "coefficients").empty()) {
      tool->wsEmplace<RooAddPdf>(name, tool->requestArgList<RooAbsPdf>(p, "summands"),
                                 tool->requestArgList<RooAbsReal>(p, "coefficients"));
      return true;
   }
   tool->wsEmplace<RooAddPdf>(name, tool->requestArgList<RooAbsPdf>(p, "summands"));
   return true;
}

bool importAddModel(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   tool->wsEmplace<RooAddModel>(name, tool->requestArgList<RooAbsPdf>(p, "summands"),
                                tool->requestArgList<RooAbsReal>(p, "coefficients"));
   return true;
}

template <bool DivideByBinWidth>
bool importBinWidthFunction(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooHistFunc *hf = dynamic_cast<RooHistFunc *>(tool->request<RooAbsReal>(p["histogram"].val(), name));
   if (!hf) {
      RooJSONFactoryWSTool::error("histogram '" + p["histogram"].val() + "' of '" + name + "' is not a RooHistFunc");
   }
   tool->wsEmplace<RooBinWidthFunction>(name, *hf, DivideByBinWidth);
   return true;
}

bool importBinSamplingPdf(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));

   RooAbsPdf *pdf = tool->requestArg<RooAbsPdf>(p, "pdf");
   RooRealVar *obs = tool->requestArg<RooRealVar>(p, "observable");

   if (!pdf->dependsOn(*obs)) {
      RooJSONFactoryWSTool::error(std::string("pdf '") + pdf->GetName() + "' does not depend on observable '" +
                                  obs->GetName() + "' as indicated by parent RooBinSamplingPdf '" + name +
                                  "', please check!");
   }

   if (!p.has_child("epsilon")) {
      RooJSONFactoryWSTool::error("no epsilon given in '" + name + "'");
   }
   double epsilon(p["epsilon"].val_double());

   tool->wsEmplace<RooBinSamplingPdf>(name, *obs, *pdf, epsilon);

   return true;
}

bool importRealSumPdf(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));

   bool extended = false;
   if (p.has_child("extended") && p["extended"].val_bool()) {
      extended = true;
   }
   tool->wsEmplace<RooRealSumPdf>(name, tool->requestArgList<RooAbsReal>(p, "samples"),
                                  tool->requestArgList<RooAbsReal>(p, "coefficients"), extended);
   return true;
}

bool importRealSumFunc(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   tool->wsEmplace<RooRealSumFunc>(name, tool->requestArgList<RooAbsReal>(p, "samples"),
                                   tool->requestArgList<RooAbsReal>(p, "coefficients"));
   return true;
}

template <class RooArg_t>
bool importPolynomial(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   if (!p.has_child("coefficients")) {
      RooJSONFactoryWSTool::error("no coefficients given in '" + name + "'");
   }
   RooAbsReal *x = tool->requestArg<RooAbsReal>(p, "x");
   RooArgList coefs;
   int order = 0;
   int lowestOrder = 0;
   for (const auto &coef : p["coefficients"].children()) {
      // As long as the coefficients match the default coefficients in
      // RooFit, we don't have to instantiate RooFit objects but can
      // increase the lowestOrder flag.
      if (order == 0 && (coef.val() == "1.0" || coef.val() == "1")) {
         ++lowestOrder;
      } else if (coefs.empty() && (coef.val() == "0.0" || coef.val() == "0")) {
         ++lowestOrder;
      } else {
         coefs.add(*tool->request<RooAbsReal>(coef.val(), name));
      }
      ++order;
   }

   tool->wsEmplace<RooArg_t>(name, *x, coefs, lowestOrder);
   return true;
}

bool importPoisson(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooAbsReal *x = tool->requestArg<RooAbsReal>(p, "x");
   RooAbsReal *mean = tool->requestArg<RooAbsReal>(p, "mean");
   tool->wsEmplace<RooPoisson>(name, *x, *mean, !p["integer"].val_bool());
   return true;
}

bool importDecay(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooRealVar *t = tool->requestArg<RooRealVar>(p, "t");
   RooAbsReal *tau = tool->requestArg<RooAbsReal>(p, "tau");
   RooResolutionModel *model = dynamic_cast<RooResolutionModel *>(tool->requestArg<RooAbsPdf>(p, "resolutionModel"));
   if (!model) {
      RooJSONFactoryWSTool::error("resolutionModel of '" + name + "' is not a RooResolutionModel");
   }
   RooDecay::DecayType decayType = static_cast<RooDecay::DecayType>(p["decayType"].val_int());
   tool->wsEmplace<RooDecay>(name, *t, *tau, *model, decayType);
   return true;
}

bool importTruthModel(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooRealVar *x = tool->requestArg<RooRealVar>(p, "x");
   tool->wsEmplace<RooTruthModel>(name, *x);
   return true;
}

bool importGaussModel(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooRealVar *x = tool->requestArg<RooRealVar>(p, "x");
   RooRealVar *mean = tool->requestArg<RooRealVar>(p, "mean");
   RooRealVar *sigma = tool->requestArg<RooRealVar>(p, "sigma");
   tool->wsEmplace<RooGaussModel>(name, *x, *mean, *sigma);
   return true;
}

bool importRealIntegral(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooAbsReal *func = tool->requestArg<RooAbsReal>(p, "integrand");
   auto vars = tool->requestArgList<RooAbsReal>(p, "variables");
   RooArgSet normSet;
   RooArgSet const *normSetPtr = nullptr;
   if (p.has_child("normalization")) {
      normSet.add(tool->requestArgSet<RooAbsReal>(p, "normalization"));
      normSetPtr = &normSet;
   }
   std::string domain;
   bool hasDomain = p.has_child("domain");
   if (hasDomain) {
      domain = p["domain"].val();
   }
   // todo: at some point, take care of integrator configurations
   tool->wsEmplace<RooRealIntegral>(name, *func, vars, normSetPtr, static_cast<RooNumIntConfig *>(nullptr),
                                    hasDomain ? domain.c_str() : nullptr);
   return true;
}

bool importDerivative(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooAbsReal *func = tool->requestArg<RooAbsReal>(p, "function");
   RooRealVar *x = tool->requestArg<RooRealVar>(p, "x");
   Int_t order = p["order"].val_int();
   double eps = p["eps"].val_double();
   if (p.has_child("normalization")) {
      RooArgSet normSet;
      normSet.add(tool->requestArgSet<RooAbsReal>(p, "normalization"));
      tool->wsEmplace<RooDerivative>(name, *func, *x, normSet, order, eps);
      return true;
   }
   tool->wsEmplace<RooDerivative>(name, *func, *x, order, eps);
   return true;
}

bool importFFTConvPdf(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooRealVar *convVar = tool->requestArg<RooRealVar>(p, "conv_var");
   Int_t order = p["ipOrder"].val_int();
   RooAbsPdf *pdf1 = tool->requestArg<RooAbsPdf>(p, "pdf1");
   RooAbsPdf *pdf2 = tool->requestArg<RooAbsPdf>(p, "pdf2");
   if (p.has_child("conv_func")) {
      RooAbsReal *convFunc = tool->requestArg<RooAbsReal>(p, "conv_func");
      tool->wsEmplace<RooFFTConvPdf>(name, *convFunc, *convVar, *pdf1, *pdf2, order);
      return true;
   }
   tool->wsEmplace<RooFFTConvPdf>(name, *convVar, *pdf1, *pdf2, order);
   return true;
}

bool importExtendPdf(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooAbsPdf *pdf = tool->requestArg<RooAbsPdf>(p, "pdf");
   RooAbsReal *norm = tool->requestArg<RooAbsReal>(p, "norm");
   if (p.has_child("range")) {
      std::string rangeName = p["range"].val();
      tool->wsEmplace<RooExtendPdf>(name, *pdf, *norm, rangeName.c_str());
      return true;
   }
   tool->wsEmplace<RooExtendPdf>(name, *pdf, *norm);
   return true;
}

bool importLogNormal(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooAbsReal *x = tool->requestArg<RooAbsReal>(p, "x");

   // Same mechanism to undo the parameter transformation as in the
   // importExponential() function (see comments in that function for more info).
   const std::string muName = p["mu"].val();
   const std::string sigmaName = p["sigma"].val();
   const bool isTransformed = endsWith(muName, "_lognormal_log");
   const std::string suffixToRemove = isTransformed ? "_lognormal_log" : "";
   RooAbsReal *mu = tool->request<RooAbsReal>(removeSuffix(muName, suffixToRemove), name);
   RooAbsReal *sigma = tool->request<RooAbsReal>(removeSuffix(sigmaName, suffixToRemove), name);

   tool->wsEmplace<RooLognormal>(name, *x, *mu, *sigma, !isTransformed);

   return true;
}

bool importExponential(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooAbsReal *x = tool->requestArg<RooAbsReal>(p, "x");

   // If the parameter name ends with the "_exponential_inverted" suffix,
   // this means that it was exported from a RooFit object where the
   // parameter first needed to be transformed on export to match the HS3
   // specification. But when re-importing such a parameter, we can simply
   // skip the transformation and use the original RooFit parameter without
   // the suffix.
   //
   // A concrete example: take the following RooFit pdf in the factory language:
   //
   //    "Exponential::exponential_1(x[0, 10], c[-0.1])"
   //
   //  It defines en exponential exp(c * x). However, in HS3 the exponential
   //  is defined as exp(-c * x), to RooFit would export these dictionaries
   //  to the JSON:
   //
   //  {
   //      "name": "exponential_1",             // HS3 exponential_dist with transformed parameter
   //      "type": "exponential_dist",
   //      "x": "x",
   //      "c": "c_exponential_inverted"
   //  },
   //  {
   //      "name": "c_exponential_inverted",    // transformation function created on-the-fly on export
   //      "type": "generic_function",
   //      "expression": "-c"
   //  }
   //
   //  On import, we can directly take the non-transformed parameter, which is
   //  we check for the suffix and optionally remove it from the requested
   //  name next:

   const std::string constParamName = p["c"].val();
   const bool isInverted = endsWith(constParamName, "_exponential_inverted");
   const std::string suffixToRemove = isInverted ? "_exponential_inverted" : "";
   RooAbsReal *c = tool->request<RooAbsReal>(removeSuffix(constParamName, suffixToRemove), name);

   tool->wsEmplace<RooExponential>(name, *x, *c, !isInverted);

   return true;
}

bool importMultiVarGaussian(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   bool has_cov = p.has_child("covariances");
   bool has_corr = p.has_child("correlations") && p.has_child("standard_deviations");
   if (!has_cov && !has_corr) {
      RooJSONFactoryWSTool::error("no covariances or correlations+standard_deviations given in '" + name + "'");
   }

   TMatrixDSym covmat;

   if (has_cov) {
      int n = p["covariances"].num_children();
      int i = 0;
      covmat.ResizeTo(n, n);
      for (const auto &row : p["covariances"].children()) {
         int j = 0;
         for (const auto &val : row.children()) {
            covmat(i, j) = val.val_double();
            ++j;
         }
         ++i;
      }
   } else {
      std::vector<double> variances;
      for (const auto &v : p["standard_deviations"].children()) {
         variances.push_back(v.val_double());
      }
      covmat.ResizeTo(variances.size(), variances.size());
      int i = 0;
      for (const auto &row : p["correlations"].children()) {
         int j = 0;
         for (const auto &val : row.children()) {
            covmat(i, j) = val.val_double() * variances[i] * variances[j];
            ++j;
         }
         ++i;
      }
   }
   tool->wsEmplace<RooMultiVarGaussian>(name, tool->requestArgList<RooAbsReal>(p, "x"),
                                        tool->requestArgList<RooAbsReal>(p, "mean"), covmat);
   return true;
}

RooArgList readBinning(const JSONNode &topNode, const RooArgList &varList)
{
   // Temporary map from variable name → RooRealVar
   std::map<std::string, std::unique_ptr<RooRealVar>> varMap;

   // Build variables from JSON
   for (const JSONNode &node : topNode["axes"].children()) {
      const std::string name = node["name"].val();
      std::unique_ptr<RooRealVar> obs;

      if (node.has_child("edges")) {
         std::vector<double> edges;
         for (const auto &bound : node["edges"].children()) {
            edges.push_back(bound.val_double());
         }
         obs = std::make_unique<RooRealVar>(name.c_str(), name.c_str(), edges.front(), edges.back());
         RooBinning bins(obs->getMin(), obs->getMax());
         for (auto b : edges)
            bins.addBoundary(b);
         obs->setBinning(bins);
      } else {
         obs = std::make_unique<RooRealVar>(name.c_str(), name.c_str(), node["min"].val_double(),
                                            node["max"].val_double());
         obs->setBins(node["nbins"].val_int());
      }

      varMap[name] = std::move(obs);
   }

   // Now build the final list following the order in varList
   RooArgList vars;
   for (auto *refVar : dynamic_range_cast<RooRealVar *>(varList)) {
      if (!refVar)
         continue;

      auto it = varMap.find(refVar->GetName());
      if (it != varMap.end()) {
         vars.addOwned(std::move(it->second)); // preserve ownership
      }
   }
   return vars;
}

bool importParamHistFunc(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   if (!p.has_child("parameters")) {
      return false;
   }
   std::string name(RooJSONFactoryWSTool::name(p));
   RooArgList varList = tool->requestArgList<RooRealVar>(p, "variables");
   if (!p.has_child("axes")) {
      std::stringstream ss;
      ss << "No axes given in '" << name << "'"
         << ". Using default binning (uniform; nbins=100). If needed, export the Workspace to JSON with a newer "
         << "Root version that supports custom ParamHistFunc binnings(>=6.38.00)." << std::endl;
      RooJSONFactoryWSTool::warning(ss.str());
      tool->wsEmplace<ParamHistFunc>(name, varList, tool->requestArgList<RooAbsReal>(p, "parameters"));
      return true;
   }
   tool->wsEmplace<ParamHistFunc>(name, readBinning(p, varList), tool->requestArgList<RooAbsReal>(p, "parameters"));
   return true;
}

bool importSpline(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   const std::string name(RooJSONFactoryWSTool::name(p));

   // Mandatory fields
   if (!p.has_child("x")) {
      RooJSONFactoryWSTool::error("no x given in '" + name + "'");
   }
   if (!p.has_child("x0") || !p.has_child("y0")) {
      RooJSONFactoryWSTool::error("no x0/y0 given in '" + name + "'");
   }

   RooAbsReal *x = tool->requestArg<RooAbsReal>(p, "x");

   // Optional fields (defaults follow RooSpline ctor defaults)
   std::string algo = p.has_child("interpolation") ? p["interpolation"].val() : "poly3";
   int order = 0;
   if (algo == "poly3")
      order = 3;
   else if (algo == "poly5")
      order = 5;
   else {
      RooJSONFactoryWSTool::error("unsupported algo '" + algo + "' for RooSpline in '" + name +
                                  "': allowed are 'poly3' and 'poly5'");
   }
   const bool logx = p.has_child("logx") ? p["logx"].val_bool() : false;
   const bool logy = p.has_child("logy") ? p["logy"].val_bool() : false;

   // Read knots
   std::vector<double> x0;
   std::vector<double> y0;
   x0.reserve(p["x0"].num_children());
   y0.reserve(p["y0"].num_children());

   for (const auto &v : p["x0"].children())
      x0.push_back(v.val_double());
   for (const auto &v : p["y0"].children())
      y0.push_back(v.val_double());

   if (x0.size() != y0.size()) {
      RooJSONFactoryWSTool::error("x0/y0 size mismatch in '" + name + "': x0 has " + std::to_string(x0.size()) +
                                  ", y0 has " + std::to_string(y0.size()));
   }
   if (x0.size() < 2) {
      RooJSONFactoryWSTool::error("need at least 2 knots in '" + name + "'");
   }

   // Construct RooSpline(name,title, x, x0, y0, order, logx, logy)
   tool->wsEmplace<::RooSpline>(name.c_str(), *x, std::span<const double>(x0.data(), x0.size()),
                                std::span<const double>(y0.data(), y0.size()), order, logx, logy);

   return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// specialized exporter implementations
///////////////////////////////////////////////////////////////////////////////////////////////////////
template <class RooArg_t>
bool exportAddPdf(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   const RooArg_t *pdf = static_cast<const RooArg_t *>(func);
   elem["type"] << key;
   RooJSONFactoryWSTool::fillSeq(elem["summands"], pdf->pdfList());
   RooJSONFactoryWSTool::fillSeq(elem["coefficients"], pdf->coefList());
   elem["extended"] << (pdf->extendMode() != RooArg_t::CanNotBeExtended);
   return true;
}

bool exportRealSumPdf(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   const RooRealSumPdf *pdf = static_cast<const RooRealSumPdf *>(func);
   elem["type"] << key;
   RooJSONFactoryWSTool::fillSeq(elem["samples"], pdf->funcList());
   RooJSONFactoryWSTool::fillSeq(elem["coefficients"], pdf->coefList());
   elem["extended"] << (pdf->extendMode() != RooAbsPdf::CanNotBeExtended);
   return true;
}

bool exportRealSumFunc(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   const RooRealSumFunc *pdf = static_cast<const RooRealSumFunc *>(func);
   elem["type"] << key;
   RooJSONFactoryWSTool::fillSeq(elem["samples"], pdf->funcList());
   RooJSONFactoryWSTool::fillSeq(elem["coefficients"], pdf->coefList());
   return true;
}

template <class RooArg_t>
bool exportHist(RooJSONFactoryWSTool *tool, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   const RooArg_t *hf = static_cast<const RooArg_t *>(func);
   elem["type"] << key;
   RooDataHist const &dh = hf->dataHist();
   tool->exportHisto(*dh.get(), dh.numEntries(), dh.weightArray(), elem["data"].set_map());
   return true;
}

template <class RooArg_t>
bool importHist(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   if (!p.has_child("data")) {
      return false;
   }
   std::unique_ptr<RooDataHist> dataHist =
      RooJSONFactoryWSTool::readBinnedData(p["data"], name, RooJSONFactoryWSTool::readAxes(p["data"]));
   tool->wsEmplace<RooArg_t>(name, *dataHist->get(), *dataHist);
   return true;
}

bool exportBinSamplingPdf(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   const RooBinSamplingPdf *pdf = static_cast<const RooBinSamplingPdf *>(func);
   elem["type"] << key;
   elem["pdf"] << pdf->pdf().GetName();
   elem["observable"] << pdf->observable().GetName();
   elem["epsilon"] << pdf->epsilon();
   return true;
}

bool exportBinWidthFunction(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &)
{
   const RooBinWidthFunction *pdf = static_cast<const RooBinWidthFunction *>(func);
   elem["type"] << (pdf->divideByBinWidth() ? "inverse_binvolume" : "binvolume");
   elem["histogram"] << pdf->histFunc().GetName();
   return true;
}

void cleanExpression(TString &expr)
{
   // Plain substring replacement would also hit longer identifiers that
   // share a prefix (e.g. "TMath::Tan" in "TMath::TanH", or "TMath::Pi" in
   // "TMath::PiOver2"), corrupting the exported expression. Identifiers
   // without a replacement are kept as-is.
   replaceIdentifier(expr, "TMath::Exp", "exp");
   replaceIdentifier(expr, "TMath::Min", "min");
   replaceIdentifier(expr, "TMath::Max", "max");
   replaceIdentifier(expr, "TMath::Log", "log");
   replaceIdentifier(expr, "TMath::Log10", "log10");
   replaceIdentifier(expr, "TMath::Cos", "cos");
   replaceIdentifier(expr, "TMath::CosH", "cosh");
   replaceIdentifier(expr, "TMath::Sin", "sin");
   replaceIdentifier(expr, "TMath::SinH", "sinh");
   replaceIdentifier(expr, "TMath::Sqrt", "sqrt");
   replaceIdentifier(expr, "TMath::Power", "pow");
   replaceIdentifier(expr, "TMath::Erf", "erf");
   replaceIdentifier(expr, "TMath::Erfc", "erfc");
   replaceIdentifier(expr, "TMath::Floor", "floor");
   replaceIdentifier(expr, "TMath::Ceil", "ceil");
   replaceIdentifier(expr, "TMath::Abs", "abs");
   replaceIdentifier(expr, "TMath::Tan", "tan");
   replaceIdentifier(expr, "TMath::TanH", "tanh");
   replaceIdentifier(expr, "TMath::ASin", "asin");
   replaceIdentifier(expr, "TMath::ACos", "acos");
   replaceIdentifier(expr, "TMath::ATan", "atan");
   replaceIdentifier(expr, "TMath::ATan2", "atan2");
   replaceIdentifier(expr, "TMath::Pi()", "PI");
   replaceIdentifier(expr, "TMath::E()", "EULER");
}

template <class RooArg_t>
bool exportFormulaArg(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   const RooArg_t *pdf = static_cast<const RooArg_t *>(func);
   elem["type"] << key;
   TString expression(pdf->expression());
   cleanExpression(expression);
   // If the tokens follow the "x[#]" convention, the square braces enclosing each number
   // ensures that there is a unique mapping between the token and parameter name
   // If the tokens follow the "@#" convention, the numbers are not enclosed by braces.
   // So there may be tokens with numbers whose lower place value forms a subset string of ones with a higher place
   // value, e.g. "@1" is a subset of "@10". So the names of these parameters must be applied descending from the
   // highest place value in order to ensure each parameter name is uniquely applied to its token.
   for (size_t idx = pdf->nParameters(); idx--;) {
      const RooAbsArg *par = pdf->getParameter(idx);
      expression.ReplaceAll(("x[" + std::to_string(idx) + "]").c_str(), par->GetName());
      expression.ReplaceAll(("@" + std::to_string(idx)).c_str(), par->GetName());
   }
   elem["expression"] << expression.Data();

   for (const RooAbsArg *dependent : pdf->dependents()) {
      auto const *observable = dynamic_cast<const RooAbsRealLValue *>(dependent);
      if (!observable) {
         continue;
      }
      const RooAbsBinning *binning = pdf->getBinning(*observable);
      if (!binning) {
         continue;
      }

      auto &axes = elem["axes"];
      if (!axes.is_seq()) {
         axes.set_seq();
      }
      auto &axis = axes.append_child().set_map();
      axis["name"] << observable->GetName();
      writeAxisBinning(axis, *binning);
   }
   return true;
}

// Write the "x" reference and the coefficient list for polynomial-like
// pdfs/funcs, including the implicit defaults below "lowestOrder" so that the
// output is self-documenting.
template <class Pdf>
void writePolynomialBody(const Pdf *pdf, JSONNode &elem)
{
   elem["x"] << pdf->x().GetName();
   auto &coefs = elem["coefficients"].set_seq();
   for (int i = 0; i < pdf->lowestOrder(); ++i) {
      coefs.append_child() << (i == 0 ? 1.0 : 0.0);
   }
   for (const auto &coef : pdf->coefList()) {
      coefs.append_child() << coef->GetName();
   }
}

template <class RooArg_t>
bool exportPolynomial(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   elem["type"] << key;
   writePolynomialBody(static_cast<const RooArg_t *>(func), elem);
   return true;
}

bool exportPoisson(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   auto *pdf = static_cast<const RooPoisson *>(func);
   elem["type"] << key;
   elem["x"] << pdf->getX().GetName();
   elem["mean"] << pdf->getMean().GetName();
   elem["integer"] << !pdf->getNoRounding();
   return true;
}

bool exportDecay(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   auto *pdf = static_cast<const RooDecay *>(func);
   elem["type"] << key;
   elem["t"] << pdf->getT().GetName();
   elem["tau"] << pdf->getTau().GetName();
   elem["resolutionModel"] << pdf->getModel().GetName();
   elem["decayType"] << pdf->getDecayType();

   return true;
}

bool exportTruthModel(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   auto *pdf = static_cast<const RooTruthModel *>(func);
   elem["type"] << key;
   elem["x"] << pdf->convVar().GetName();

   return true;
}

bool exportGaussModel(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   auto *pdf = static_cast<const RooGaussModel *>(func);
   elem["type"] << key;
   elem["x"] << pdf->convVar().GetName();
   elem["mean"] << pdf->getMean().GetName();
   elem["sigma"] << pdf->getSigma().GetName();
   return true;
}

bool exportLogNormal(RooJSONFactoryWSTool *tool, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   auto *pdf = static_cast<const RooLognormal *>(func);

   elem["type"] << key;
   elem["x"] << pdf->getX().GetName();

   auto &m0 = pdf->getMedian();
   auto &k = pdf->getShapeK();

   if (pdf->useStandardParametrization()) {
      elem["mu"] << m0.GetName();
      elem["sigma"] << k.GetName();
   } else {
      elem["mu"] << tool->exportTransformed(&m0, "_lognormal_log", "log(%s)");
      elem["sigma"] << tool->exportTransformed(&k, "_lognormal_log", "log(%s)");
   }

   return true;
}

bool exportExponential(RooJSONFactoryWSTool *tool, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   auto *pdf = static_cast<const RooExponential *>(func);
   elem["type"] << key;
   elem["x"] << pdf->variable().GetName();
   auto &c = pdf->coefficient();
   if (pdf->negateCoefficient()) {
      elem["c"] << c.GetName();
   } else {
      elem["c"] << tool->exportTransformed(&c, "_exponential_inverted", "-%s");
   }

   return true;
}

bool exportMultiVarGaussian(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   auto *pdf = static_cast<const RooMultiVarGaussian *>(func);
   elem["type"] << key;
   RooJSONFactoryWSTool::fillSeq(elem["x"], pdf->xVec());
   RooJSONFactoryWSTool::fillSeq(elem["mean"], pdf->muVec());
   elem["covariances"].fill_mat(pdf->covarianceMatrix());
   return true;
}

bool exportTFnBinding(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   auto *pdf = static_cast<const RooTFnBinding *>(func);
   elem["type"] << key;

   TString formula(pdf->function().GetExpFormula());
   formula.ReplaceAll("x", pdf->observables()[0].GetName());
   formula.ReplaceAll("y", pdf->observables()[1].GetName());
   formula.ReplaceAll("z", pdf->observables()[2].GetName());
   for (size_t i = 0; i < pdf->parameters().size(); ++i) {
      TString pname(TString::Format("[%d]", (int)i));
      formula.ReplaceAll(pname, pdf->parameters()[i].GetName());
   }
   elem["expression"] << formula.Data();
   return true;
}

bool exportDerivative(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   auto *pdf = static_cast<const RooDerivative *>(func);
   elem["type"] << key;
   elem["x"] << pdf->getX().GetName();
   elem["function"] << pdf->getFunc().GetName();
   if (!pdf->getNset().empty()) {
      RooJSONFactoryWSTool::fillSeq(elem["normalization"], pdf->getNset());
   }
   elem["order"] << pdf->order();
   elem["eps"] << pdf->eps();
   return true;
}

bool exportRealIntegral(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   auto *integral = static_cast<const RooRealIntegral *>(func);
   elem["type"] << key;
   std::string integrand = integral->integrand().GetName();
   elem["integrand"] << integrand;
   if (integral->intRange()) {
      elem["domain"] << integral->intRange();
   }
   RooJSONFactoryWSTool::fillSeq(elem["variables"], integral->intVars());
   if (RooArgSet const *funcNormSet = integral->funcNormSet()) {
      RooJSONFactoryWSTool::fillSeq(elem["normalization"], *funcNormSet);
   }
   return true;
}

bool exportFFTConvPdf(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   auto *pdf = static_cast<const RooFFTConvPdf *>(func);
   elem["type"] << key;
   if (auto convFunc = pdf->getPdfConvVar()) {
      elem["conv_func"] << convFunc->GetName();
   }
   elem["conv_var"] << pdf->getConvVar().GetName();
   elem["pdf1"] << pdf->getPdf1().GetName();
   elem["pdf2"] << pdf->getPdf2().GetName();
   elem["ipOrder"] << pdf->getInterpolationOrder();
   return true;
}

bool exportExtendPdf(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   auto *pdf = static_cast<const RooExtendPdf *>(func);
   elem["type"] << key;
   if (auto rangeName = pdf->getRangeName()) {
      elem["range"] << rangeName->GetName();
   }
   elem["pdf"] << pdf->pdf().GetName();
   elem["norm"] << pdf->getN().GetName();
   return true;
}

bool exportParamHistFunc(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   auto *pdf = static_cast<const ParamHistFunc *>(func);
   elem["type"] << key;
   RooJSONFactoryWSTool::fillSeq(elem["variables"], pdf->dataVars());
   RooJSONFactoryWSTool::fillSeq(elem["parameters"], pdf->paramList());
   auto &observablesNode = elem["axes"].set_seq();
   // axes have to be ordered to get consistent bin indices
   for (auto *var : static_range_cast<RooRealVar *>(pdf->dataVars())) {
      RooJSONFactoryWSTool::exportAxis(observablesNode.append_child().set_map(), *var);
   }
   return true;
}

bool exportSpline(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem, std::string const &key)
{
   auto const *rs = static_cast<RooSpline const *>(func);

   elem["type"] << key;

   // Independent variable
   elem["x"] << rs->x().GetName();

   // Spline configuration
   // Canonical algo for RooSpline
   elem["interpolation"] << (rs->order() == 5 ? "poly5" : "poly3");
   elem["logx"] << rs->logx();
   elem["logy"] << rs->logy();

   // Serialize knots as primitive arrays
   TSpline const &sp = rs->spline();
   auto &x0 = elem["x0"].set_seq();
   auto &y0 = elem["y0"].set_seq();

   const int np = sp.GetNp();
   for (int i = 0; i < np; ++i) {
      double xk = 0.0, yk = 0.0;
      sp.GetKnot(i, xk, yk);
      x0.append_child() << xk;
      y0.append_child() << yk;
   }

   return true;
}

bool importWrapperPdf(RooJSONFactoryWSTool *tool, const JSONNode &node)
{
   if (node["type"].val() != "density_function_dist")
      return false;

   auto name = RooJSONFactoryWSTool::name(node);
   auto *func = tool->requestArg<RooAbsReal>(node, "function");

   bool selfNormalized = false;

   if (auto sn = node.find("selfNormalized"))
      selfNormalized = sn->val_bool();

   tool->wsEmplace<RooWrapperPdf>(name, *func, selfNormalized);
   return true;
}

bool exportWrapperPdf(RooJSONFactoryWSTool *, const RooAbsArg *arg, JSONNode &node, std::string const &key)
{
   auto const *pdf = dynamic_cast<RooWrapperPdf const *>(arg);
   if (!pdf)
      return false;

   node["type"] << key;

   // Proxy name in RooWrapperPdf is "_func" / "func" depending on accessor/proxy export.
   // Prefer a public accessor if one exists; otherwise inspect proxies as below.
   auto const *funcProxy = dynamic_cast<RooRealProxy const *>(pdf->getProxy(0));
   if (!funcProxy || !funcProxy->absArg())
      return false;

   node["function"] << funcProxy->absArg()->GetName();
   if (pdf->selfNormalized())
      node["selfnormalized"] << true;

   return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// instantiate all importers and exporters
///////////////////////////////////////////////////////////////////////////////////////////////////////

// Adapters that wrap the plain import/export functions above into the
// RooFit::JSONIO::Importer/Exporter interface. The exporter also owns the HS3
// type key, which is passed at registration time.
template <auto Func>
class FuncImporter : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override { return Func(tool, p); }
};

template <auto Func>
class FuncExporter : public RooFit::JSONIO::Exporter {
public:
   FuncExporter(std::string key) : _key{std::move(key)} {}
   std::string const &key() const override { return _key; }
   bool exportObject(RooJSONFactoryWSTool *tool, const RooAbsArg *func, JSONNode &elem) const override
   {
      return Func(tool, func, elem, _key);
   }

private:
   const std::string _key;
};

template <auto Func>
void registerImporter(const std::string &key, bool topPriority = true)
{
   RooFit::JSONIO::registerImporter(key, std::make_unique<FuncImporter<Func>>(), topPriority);
}

template <auto Func>
void registerExporter(TClass const *cl, std::string key, bool topPriority = true)
{
   RooFit::JSONIO::registerExporter(cl, std::make_unique<FuncExporter<Func>>(std::move(key)), topPriority);
}

STATIC_EXECUTE([]() {
   registerImporter<importWrapperPdf>("density_function_dist");
   registerImporter<importExtendPdf>("rate_extended_dist");
   registerImporter<importProduct>("product", false);
   registerImporter<importProdPdf>("product_dist", false);
   registerImporter<importAddition>("sum", false);
   registerImporter<importAddPdf>("mixture_dist", false);
   registerImporter<importAddModel>("mixture_resolution_model", false);
   registerImporter<importBinSamplingPdf>("binsampling_dist", false);
   registerImporter<importBinWidthFunction<false>>("binvolume", false);
   registerImporter<importBinWidthFunction<true>>("inverse_binvolume", false);
   registerImporter<importPolynomial<RooLegacyExpPoly>>("legacy_exp_poly_dist", false);
   registerImporter<importExponential>("exponential_dist", false);
   registerImporter<importFormulaArg<RooFormulaVar>>("generic", false);
   registerImporter<importFormulaArg<RooFormulaVar>>("generic_function", false);
   registerImporter<importFormulaArg<RooGenericPdf>>("generic_dist", false);
   registerImporter<importHist<RooHistFunc>>("histogram", false);
   registerImporter<importHist<RooHistFunc>>("step", false);
   registerImporter<importHist<RooHistPdf>>("histogram_dist", false);
   registerImporter<importLogNormal>("lognormal_dist", false);
   registerImporter<importMultiVarGaussian>("multivariate_normal_dist", false);
   registerImporter<importPoisson>("poisson_dist", false);
   registerImporter<importDecay>("decay_dist", false);
   registerImporter<importTruthModel>("delta_resolution_model", false);
   registerImporter<importGaussModel>("gauss_resolution_model", false);
   registerImporter<importPolynomial<RooPolynomial>>("polynomial_dist", false);
   registerImporter<importPolynomial<RooPolyVar>>("polynomial", false);
   registerImporter<importRealSumPdf>("weighted_sum_dist", false);
   registerImporter<importRealSumFunc>("weighted_sum", false);
   registerImporter<importRealIntegral>("integral", false);
   registerImporter<importDerivative>("derivative", false);
   registerImporter<importFFTConvPdf>("fft_convolution_dist", false);
   registerImporter<importExtendPdf>("extend_pdf", false);
   registerImporter<importParamHistFunc>("step", false);
   registerImporter<importSpline>("spline", false);

   registerExporter<exportWrapperPdf>(RooWrapperPdf::Class(), "density_function_dist");
   registerExporter<exportAddPdf<RooAddPdf>>(RooAddPdf::Class(), "mixture_dist", false);
   registerExporter<exportAddPdf<RooAddModel>>(RooAddModel::Class(), "mixture_resolution_model", false);
   registerExporter<exportBinSamplingPdf>(RooBinSamplingPdf::Class(), "binsampling", false);
   registerExporter<exportBinWidthFunction>(RooBinWidthFunction::Class(), "binvolume", false);
   registerExporter<exportPolynomial<RooLegacyExpPoly>>(RooLegacyExpPoly::Class(), "legacy_exp_poly_dist", false);
   registerExporter<exportExponential>(RooExponential::Class(), "exponential_dist", false);
   registerExporter<exportFormulaArg<RooFormulaVar>>(RooFormulaVar::Class(), "generic", false);
   registerExporter<exportFormulaArg<RooGenericPdf>>(RooGenericPdf::Class(), "generic_dist", false);
   registerExporter<exportHist<RooHistFunc>>(RooHistFunc::Class(), "step", false);
   registerExporter<exportHist<RooHistPdf>>(RooHistPdf::Class(), "histogram_dist", false);
   registerExporter<exportLogNormal>(RooLognormal::Class(), "lognormal_dist", false);
   registerExporter<exportMultiVarGaussian>(RooMultiVarGaussian::Class(), "multivariate_normal_dist", false);
   registerExporter<exportPoisson>(RooPoisson::Class(), "poisson_dist", false);
   registerExporter<exportDecay>(RooDecay::Class(), "decay_dist", false);
   registerExporter<exportTruthModel>(RooTruthModel::Class(), "delta_resolution_model", false);
   registerExporter<exportGaussModel>(RooGaussModel::Class(), "gauss_resolution_model", false);
   registerExporter<exportPolynomial<RooPolynomial>>(RooPolynomial::Class(), "polynomial_dist", false);
   registerExporter<exportPolynomial<RooPolyVar>>(RooPolyVar::Class(), "polynomial", false);
   registerExporter<exportRealSumFunc>(RooRealSumFunc::Class(), "weighted_sum", false);
   registerExporter<exportRealSumPdf>(RooRealSumPdf::Class(), "weighted_sum_dist", false);
   registerExporter<exportTFnBinding>(RooTFnBinding::Class(), "generic_function", false);
   registerExporter<exportRealIntegral>(RooRealIntegral::Class(), "integral", false);
   registerExporter<exportDerivative>(RooDerivative::Class(), "derivative", false);
   registerExporter<exportFFTConvPdf>(RooFFTConvPdf::Class(), "fft_convolution_dist", false);
   registerExporter<exportExtendPdf>(RooExtendPdf::Class(), "rate_extended_dist", false);
   registerExporter<exportParamHistFunc>(ParamHistFunc::Class(), "step", false);
   registerExporter<exportSpline>(RooSpline::Class(), "spline", false);
});

} // namespace

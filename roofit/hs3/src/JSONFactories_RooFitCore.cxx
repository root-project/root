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

#include <RooAddPdf.h>
#include <RooAddModel.h>
#include <RooArgusBG.h>
#include <RooBernstein.h>
#include <RooBifurGauss.h>
#include <RooBinSamplingPdf.h>
#include <RooBinWidthFunction.h>
#include <RooCBShape.h>
#include <RooChebychev.h>
#include <RooCrystalBall.h>
#include <RooDataHist.h>
#include <RooDecay.h>
#include <RooDerivative.h>
#include <RooEffProd.h>
#include <RooExponential.h>
#include <RooExtendPdf.h>
#include <RooFFTConvPdf.h>
#include <RooFit/Detail/JSONInterface.h>
#include <RooFitHS3/JSONIO.h>
#include <RooFormulaVar.h>
#include <RooGamma.h>
#include <RooGaussian.h>
#include <RooGenericPdf.h>
#include <RooHistFunc.h>
#include <RooHistPdf.h>
#include <RooLandau.h>
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
#include <RooPowerSum.h>
#include <RooProjectedPdf.h>
#include <RooAbsRealLValue.h>
#include <RooRealSumFunc.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>
#include <RooResolutionModel.h>
#include <RooTFnBinding.h>
#include <RooTruthModel.h>
#include <RooGaussModel.h>
#include <RooWrapperPdf.h>
#include <RooRealIntegral.h>
#include <RooSpline.h>
#include <RooUniform.h>
#include <TSpline.h>

#include <TF1.h>

#include "HS3Builder.h"
#include "JSONIOUtils.h"

#include <optional>
#include <type_traits>
#include <variant>

#include <algorithm>
#include <cctype>
#include <memory>
#include <initializer_list>
#include <set>
#include <string_view>
#include <vector>

using RooFit::Detail::JSONNode;

namespace HS3 = RooFit::HS3;

///////////////////////////////////////////////////////////////////////////////////////////////////////
// individually implemented importers
///////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
bool isReservedExpressionIdentifier(const std::string &arg)
{
   return arg == "PI" || arg == "EULER" || arg == "TMath";
}

bool isIdentifierChar(char c)
{
   return std::isalnum(static_cast<unsigned char>(c)) || c == '_';
}

/**
 * Extracts arguments from a mathematical expression.
 *
 * This function takes a string representing a mathematical
 * expression and extracts the arguments from it.  The arguments are
 * the identifiers: sequences of letters, digits and underscores that
 * start with a letter. Function calls such as "exp( ... )", identified
 * as being followed by an opening parenthesis, are not treated as
 * arguments, and neither are the reserved identifiers.
 *
 * @param expr A string representing a mathematical expression.
 * @return A set of unique strings representing the extracted arguments.
 */
std::set<std::string> extractArguments(std::string expr)
{
   // Get rid of whitespaces
   expr.erase(std::remove_if(expr.begin(), expr.end(), [](unsigned char c) { return std::isspace(c); }), expr.end());

   std::set<std::string> arguments;
   for (std::size_t i = 0; i < expr.size(); ++i) {
      // A letter starts an identifier, unless it is part of a number in scientific notation, e.g. 2e-5.
      if (!isalpha(expr[i]) || TFormula::IsScientificNotation(expr, i)) {
         continue;
      }
      const std::size_t start = i;
      while (i < expr.size() && isIdentifierChar(expr[i])) {
         ++i;
      }
      const std::string arg = expr.substr(start, i - start);
      const bool isFunctionCall = i < expr.size() && expr[i] == '(';
      if (!isFunctionCall && !isReservedExpressionIdentifier(arg)) {
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
      const bool beforeIdentifier = pos > 0 && isIdentifierChar(in[pos - 1]);
      const std::size_t end = pos + identifier.size();
      const bool afterIdentifier = end < in.size() && isIdentifierChar(in[end]);
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

std::string trimmed(std::string const &str)
{
   const auto begin = str.find_first_not_of(" \t\n");
   if (begin == std::string::npos) {
      return "";
   }
   return str.substr(begin, str.find_last_not_of(" \t\n") - begin + 1);
}

// Index of the bracket that closes the one at `open`, or std::string::npos if it is not closed.
std::size_t findClosingBracket(std::string const &str, std::size_t open)
{
   int depth = 0;
   for (std::size_t i = open; i < str.size(); ++i) {
      if (str[i] == '(' || str[i] == '[') {
         ++depth;
      } else if ((str[i] == ')' || str[i] == ']') && --depth == 0) {
         return i;
      }
   }
   return std::string::npos;
}

// Index of the first of the characters `chars` in `str` from `start` on that is not enclosed in brackets, or
// std::string::npos if there is none. The scope operator "::" is skipped: a ':' is the one of a ternary operator.
std::size_t findTopLevel(std::string const &str, std::string_view chars, std::size_t start = 0)
{
   int depth = 0;
   for (std::size_t i = start; i < str.size(); ++i) {
      if (str[i] == '(' || str[i] == '[') {
         ++depth;
      } else if (str[i] == ')' || str[i] == ']') {
         --depth;
      } else if (str[i] == ':' && i + 1 < str.size() && str[i + 1] == ':') {
         ++i;
      } else if (depth == 0 && chars.find(str[i]) != std::string_view::npos) {
         return i;
      }
   }
   return std::string::npos;
}

// Split `str` at the commas that are not enclosed in brackets.
std::vector<std::string> splitTopLevelCommas(std::string const &str)
{
   std::vector<std::string> parts;
   std::size_t start = 0;
   for (std::size_t comma = findTopLevel(str, ","); comma != std::string::npos; comma = findTopLevel(str, ",", start)) {
      parts.push_back(str.substr(start, comma - start));
      start = comma + 1;
   }
   parts.push_back(str.substr(start));
   return parts;
}

// Remove parentheses that enclose the whole string, e.g. "((a+b))" becomes "a+b".
std::string stripEnclosingParentheses(std::string str)
{
   str = trimmed(str);
   while (str.size() >= 2 && str.front() == '(' && findClosingBracket(str, 0) == str.size() - 1) {
      str = trimmed(str.substr(1, str.size() - 2));
   }
   return str;
}

// Rewrite the HS3 conditional "ifelse(c,a,b)" into the C-style ternary "((c)?(a):(b))" that TFormula understands.
// Malformed calls are left unchanged, so that TFormula reports them.
std::string ifElseToTernary(std::string const &expr)
{
   static const std::string keyword = "ifelse";
   std::string out;
   std::size_t pos = 0;
   while (pos < expr.size()) {
      const std::size_t found = expr.find(keyword, pos);
      if (found == std::string::npos) {
         break;
      }
      const std::size_t open = expr.find_first_not_of(" \t", found + keyword.size());
      const bool isCall =
         (found == 0 || !isIdentifierChar(expr[found - 1])) && open != std::string::npos && expr[open] == '(';
      const std::size_t close = isCall ? findClosingBracket(expr, open) : std::string::npos;
      const auto args = close != std::string::npos ? splitTopLevelCommas(expr.substr(open + 1, close - open - 1))
                                                   : std::vector<std::string>{};
      if (args.size() != 3) {
         out += expr.substr(pos, found + keyword.size() - pos);
         pos = found + keyword.size();
         continue;
      }
      out += expr.substr(pos, found - pos);
      out += "((" + ifElseToTernary(trimmed(args[0])) + ")?(" + ifElseToTernary(trimmed(args[1])) + "):(" +
             ifElseToTernary(trimmed(args[2])) + "))";
      pos = close + 1;
   }
   return out + expr.substr(pos);
}

void translateImportedExpression(TString &expr)
{
   replaceIdentifier(expr, "PI", "TMath::Pi()");
   replaceIdentifier(expr, "EULER", "TMath::E()");
   // The boolean literals, and the exact comparisons, which are the ones of TFormula.
   replaceIdentifier(expr, "TRUE", "1");
   replaceIdentifier(expr, "FALSE", "0");
   expr.ReplaceAll("!==", "!=");
   expr.ReplaceAll("===", "==");
   expr = ifElseToTernary(expr.Data()).c_str();
}

// Give the formula the binnings of its "axes", which are read like the axes of binned data.
template <class RooArg_t>
void importFormulaBinnings(RooArg_t &arg, const JSONNode &node)
{
   if (!node.has_child("axes")) {
      return;
   }
   const std::string context = "generic formula '" + std::string(arg.GetName()) + "'";
   HS3::Diagnostics diagnostics;
   const std::vector<HS3::BinnedAxis> axes = HS3::parseBinnedAxes(node["axes"], diagnostics);
   HS3::report(diagnostics, std::string(arg.GetName()) + ".axes");

   std::set<std::string> axisNames;
   for (HS3::BinnedAxis const &axis : axes) {
      if (!axisNames.insert(axis.name).second) {
         RooJSONFactoryWSTool::error("duplicate axis '" + axis.name + "' in " + context);
      }
      if (!dynamic_cast<RooAbsRealLValue *>(arg.getParameter(axis.name.c_str()))) {
         RooJSONFactoryWSTool::error("axis '" + axis.name + "' is not a real-valued formula variable of " + context);
      }
      auto const *regular = std::get_if<HS3::RegularBinning>(&axis.binning);
      if (regular && (regular->nbins < 1 || !(axis.min < axis.max))) {
         RooJSONFactoryWSTool::error("axis '" + axis.name + "' of " + context +
                                     " needs a positive \"nbins\" and \"min\" smaller than \"max\"");
      }
   }
   // The formula finds its variable by name, so the variables that are created for the axes can stand in for them.
   RooArgSet vars = HS3::createObservables(axes);
   for (auto *var : static_range_cast<RooRealVar *>(vars)) {
      arg.setBinning(*var, var->getBinning(), /*checkFlatness=*/false);
   }
}

template <class RooArg_t>
void importFormulaArg(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   if (!p.has_child("expression")) {
      RooJSONFactoryWSTool::error("no expression given for '" + name + "'");
   }
   TString formula(p["expression"].val());
   translateImportedExpression(formula);
   RooArgList dependents;
   for (const auto &d : extractArguments(formula.Data())) {
      RooAbsReal *dependent = tool->request<RooAbsReal>(d, name);
      // HS3 0.3 (HS3 PR #154) only allows expressions to refer to parameters, variables and functions.
      if (dynamic_cast<RooAbsPdf *>(dependent) && !(tool->hs3Version() < HS3::Version{0, 3})) {
         RooJSONFactoryWSTool::error("the expression of '" + name + "' refers to the distribution '" + d +
                                     "', which is not allowed in HS3");
      }
      dependents.add(*dependent);
   }
   RooArg_t arg{name.c_str(), formula, dependents};
   importFormulaBinnings(arg, p);
   tool->wsImport(arg);
}

void importProduct(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   tool->wsEmplace<RooProduct>(name, tool->requestArgList<RooAbsReal>(p, "factors"));
}

// RooLognormal has two parametrizations. In the standard one, its parameters are the mean "mu" and the width "sigma" of
// log(x), like in the lognormal_dist of HS3. In the other one, which is the default of RooFit, they are the median m0
// of x and the shape parameter k, with mu = log(m0) and sigma = log(k). Such a distribution is written with "mu" and
// "sigma" naming the functions log(m0) and log(k), which are named after m0 and k with the suffix below. The suffix
// is also how the parametrization is recognized when reading: the distribution is then created from m0 and k
// themselves instead of the functions. Only lognormalLogName() and importLogNormal() know this convention, for
// distributions on their own and for the components of array distributions alike.
constexpr const char *lognormalLogSuffix = "_lognormal_log";

// What a RooLognormal that is not in the standard parametrization writes as "mu" (for its median) or "sigma" (for its
// shape parameter): the name of the function log(`param`). The function is written as well if there is a `tool`;
// without one, only the name is given.
std::string lognormalLogName(RooAbsReal const &param, RooJSONFactoryWSTool *tool)
{
   if (tool) {
      return tool->exportTransformed(&param, lognormalLogSuffix, "log(%s)");
   }
   return transformedBaseName(param) + lognormalLogSuffix;
}

// Copy the value of a JSON node, keeping numbers and booleans as such.
void copyValue(JSONNode &target, JSONNode const &source)
{
   if (source.is_number()) {
      target << source.val_double();
      return;
   }
   const std::string value = source.val();
   if (value == "true" || value == "false") {
      target << (value == "true");
   } else {
      target << value;
   }
}

// An HS3 `array_dist` is a product of one-dimensional distributions of one type (`component_type`), each over its own
// observable `x`. RooFit has no dedicated class for it: every component is imported as an individual pdf of the
// component type, named after its observable, and a RooProdPdf with the attribute `arrayDistTag` that carries the
// name of the array holds them together, so that the array can be written back as `array_dist`. HistFactory modifiers
// that name an array as the constraint of a parameter pick their component out of it (see
// JSONFactories_HistFactory.cxx).
void importArrayDist(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   const std::string name = RooJSONFactoryWSTool::name(p);
   auto error = [&name](std::string const &message) {
      RooJSONFactoryWSTool::error("array_dist '" + name + "': " + message);
   };

   static const std::set<std::string> supportedComponentTypes{"poisson_dist", "poisson_continuous_dist",
                                                              "gaussian_dist", "normal_dist", "lognormal_dist"};
   if (!p.has_child("component_type")) {
      error("missing \"component_type\"");
   }
   const std::string componentType = p["component_type"].val();
   if (supportedComponentTypes.find(componentType) == supportedComponentTypes.end()) {
      error("the component_type '" + componentType +
            "' is not supported, supported are poisson_dist, poisson_continuous_dist, gaussian_dist and "
            "lognormal_dist");
   }

   const JSONNode *defaults = p.find("default_parameters");
   if (defaults) {
      if (!defaults->is_map()) {
         error("\"default_parameters\" must be a struct");
      }
      if (defaults->has_child("x")) {
         error("\"x\" must be given per component, not in \"default_parameters\"");
      }
   }
   const JSONNode *components = p.find("components");
   if (!components || !components->is_seq() || components->num_children() == 0) {
      error("\"components\" must be a non-empty list");
   }

   RooWorkspace &ws = *tool->workspace();

   // The components are imported through the importer of the component type, from a node per component that
   // combines its own fields with the defaults.
   auto tree = RooFit::Detail::JSONTree::create();
   JSONNode &nodes = tree->rootnode().set_seq();
   std::vector<std::string> observables;
   RooArgList pdfs;
   for (JSONNode const &component : components->children()) {
      const std::string context = "component " + std::to_string(observables.size());
      if (!component.is_map()) {
         error(context + " must be a struct");
      }
      const JSONNode *xNode = component.find("x");
      if (!xNode) {
         error(context + " has no \"x\"");
      }
      if (xNode->is_number() || !xNode->has_val()) {
         error(context + ": \"x\" must be the name of the observable");
      }
      const std::string x = xNode->val();
      if (std::find(observables.begin(), observables.end(), x) != observables.end()) {
         error("the observable '" + x + "' is used by two components");
      }
      observables.push_back(x);

      const std::string pdfName = x + "_constraint";
      if (ws.arg(pdfName)) {
         error("cannot create the component for '" + x + "' as '" + pdfName +
               "', the workspace already has an object of that name");
      }
      JSONNode &node = nodes.append_child().set_map();
      node["name"] << pdfName;
      node["type"] << componentType;
      if (defaults) {
         for (JSONNode const &field : defaults->children()) {
            copyValue(node[field.key()], field);
         }
      }
      for (JSONNode const &field : component.children()) {
         copyValue(node[field.key()], field);
      }
      tool->importFunction(node, true);
      RooAbsPdf *pdf = ws.pdf(pdfName);
      if (!pdf) {
         error(context + " did not produce a distribution");
      }
      pdfs.add(*pdf);
   }

   // The parameters of a component must not depend on any observable of the array, otherwise the product would not
   // be normalized.
   for (std::size_t i = 0; i < observables.size(); ++i) {
      RooAbsArg const *x = ws.arg(observables[i]);
      if (!x) {
         error("the observable '" + observables[i] + "' is not a parameter or function");
      }
      for (std::size_t k = 0; k < observables.size(); ++k) {
         if (k != i && pdfs[i].dependsOn(*ws.arg(observables[k]))) {
            error("component " + std::to_string(i) + " depends on the observable '" + observables[k] +
                  "' of component " + std::to_string(k));
         }
      }
      for (RooAbsArg const *server : pdfs[i].servers()) {
         if (server != x && server->dependsOn(*x)) {
            error("the parameters of component " + std::to_string(i) + " depend on its own observable '" +
                  observables[i] + "'");
         }
      }
   }

   RooProdPdf array{name.c_str(), name.c_str(), pdfs};
   array.setAttribute(arrayDistTag);
   // The product only exists to hold the components together.
   array.setAttribute(helperObjectTag);
   tool->wsImport(array);
}

// A "combined_dist" is interpreted by the document. This reads one that doesn't come with a document.
void importCombinedDist(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   HS3::Diagnostics diagnostics;
   HS3::Distribution const dist = HS3::parseDistribution(p, diagnostics);
   if (!HS3::hasErrors(diagnostics)) {
      HS3::validateCombinedDist(std::get<HS3::CombinedDist>(dist.body), dist.name, diagnostics);
   }
   HS3::report(diagnostics, dist.name);
   HS3::buildCombinedDist(*tool, dist.name, std::get<HS3::CombinedDist>(dist.body));
}

void importAddPdf(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   const std::size_t nSummands = p.has_child("summands") ? p["summands"].num_children() : 0;
   const std::size_t nCoefficients = p.has_child("coefficients") ? p["coefficients"].num_children() : 0;
   const bool sinceHS3v03 = !(tool->hs3Version() < HS3::Version{0, 3});
   if (p.has_child("extended")) {
      // Check the explicit "extended" flag against the number of coefficients, which determines the extended mode of
      // the RooAddPdf.
      const bool extended = p["extended"].val_bool();
      const std::string counts =
         std::to_string(nCoefficients) + " coefficients for " + std::to_string(nSummands) + " summands";
      if (extended && nCoefficients > 0 && nCoefficients + 1 == nSummands) {
         RooJSONFactoryWSTool::error("mixture_dist '" + name + "' is declared as extended, but it has " + counts +
                                     ", which defines a non-extended mixture");
      }
      if (!extended && nCoefficients == 0 && nSummands > 0) {
         RooJSONFactoryWSTool::error("mixture_dist '" + name + "' is declared as not extended, but it has " + counts +
                                     ", which defines an extended mixture of extended distributions");
      }
      if (!extended && nCoefficients == nSummands && nSummands > 0) {
         // A RooAddPdf with one coefficient per summand can always be extended. The non-extended mixture is the one
         // with the normalized coefficients c_i / sum(c), of which the last one follows from the others.
         RooArgList summands = tool->requestArgList<RooAbsPdf>(p, "summands");
         RooArgList coefficients = tool->requestArgList<RooAbsReal>(p, "coefficients");
         std::string sum;
         for (std::size_t i = 0; i < nCoefficients; ++i) {
            sum += (i == 0 ? "x[" : "+x[") + std::to_string(i) + "]";
         }
         RooArgList fractions;
         for (std::size_t i = 0; i + 1 < nCoefficients; ++i) {
            fractions.add(tool->wsEmplace<RooFormulaVar>(name + "_fraction_" + std::to_string(i),
                                                         ("x[" + std::to_string(i) + "]/(" + sum + ")").c_str(),
                                                         coefficients));
         }
         tool->wsEmplace<RooAddPdf>(name, summands, fractions);
         return;
      }
   } else if (nCoefficients == nSummands && nSummands > 0 && sinceHS3v03) {
      // HS3 0.3 requires the flag. Where it decides about the Poisson term, a document without it is invalid.
      RooJSONFactoryWSTool::error("mixture_dist '" + name +
                                  "' has one coefficient per summand, but no \"extended\" flag, which HS3 "
                                  "requires to decide whether the mixture is extended");
   }
   if (nSummands > 0 && nCoefficients == 0 && sinceHS3v03) {
      RooJSONFactoryWSTool::warning("RooFitHS3: mixture_dist '" + name +
                                    "' has no coefficients, which HS3 0.3 doesn't define. It is read as the sum of "
                                    "its extended summands.");
   }
   if (!tool->requestArgList<RooAbsReal>(p, "coefficients").empty()) {
      tool->wsEmplace<RooAddPdf>(name, tool->requestArgList<RooAbsPdf>(p, "summands"),
                                 tool->requestArgList<RooAbsReal>(p, "coefficients"));
   } else {
      tool->wsEmplace<RooAddPdf>(name, tool->requestArgList<RooAbsPdf>(p, "summands"));
   }
}

void importAddModel(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   tool->wsEmplace<RooAddModel>(name, tool->requestArgList<RooAbsPdf>(p, "summands"),
                                tool->requestArgList<RooAbsReal>(p, "coefficients"));
}

template <bool DivideByBinWidth>
void importBinWidthFunction(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooHistFunc *hf = dynamic_cast<RooHistFunc *>(tool->request<RooAbsReal>(p["histogram"].val(), name));
   if (!hf) {
      RooJSONFactoryWSTool::error("histogram '" + p["histogram"].val() + "' of '" + name + "' is not a RooHistFunc");
   }
   tool->wsEmplace<RooBinWidthFunction>(name, *hf, DivideByBinWidth);
}

void importBinSamplingPdf(RooJSONFactoryWSTool *tool, const JSONNode &p)
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
}

void importRealSumPdf(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooArgList samples = tool->requestArgList<RooAbsReal>(p, "samples");
   RooArgList coefs = tool->requestArgList<RooAbsReal>(p, "coefficients");
   const bool extended = p.has_child("extended") && p["extended"].val_bool();
   tool->wsEmplace<RooRealSumPdf>(name, samples, coefs, extended);
}

template <class RooArg_t>
void importPolynomial(RooJSONFactoryWSTool *tool, const JSONNode &p)
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
}

// HS3 0.3 distinguishes the discrete "poisson_dist" from the "poisson_continuous_dist" (HS3 #152), which are a
// RooPoisson without and with noRounding. Before, ROOT wrote both as "poisson_dist" with its own key "integer", where
// false denotes the continuous form. That key is read for documents before HS3 0.3, without it the discrete form is
// meant. From HS3 0.3 on, it must not be present.
template <bool Continuous>
void importPoisson(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   bool noRounding = Continuous;
   if (p.has_child("integer")) {
      std::string const type = Continuous ? "poisson_continuous_dist" : "poisson_dist";
      if (Continuous || !(tool->hs3Version() < HS3::Version{0, 3})) {
         RooJSONFactoryWSTool::error(
            type + " '" + name +
            "' has the key \"integer\", which is not part of HS3. The discrete Poisson distribution is "
            "\"poisson_dist\" and the continuous one is \"poisson_continuous_dist\".");
      }
      noRounding = !p["integer"].val_bool();
   }
   RooAbsReal *x = tool->requestArg<RooAbsReal>(p, "x");
   RooAbsReal *mean = tool->requestArg<RooAbsReal>(p, "mean");
   tool->wsEmplace<RooPoisson>(name, *x, *mean, noRounding);
}

void importDecay(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooRealVar *t = tool->requestArg<RooRealVar>(p, "t");
   RooAbsReal *tau = tool->requestArg<RooAbsReal>(p, "tau");
   // HS3 lists some resolution models as functions.
   RooResolutionModel *model = dynamic_cast<RooResolutionModel *>(tool->requestArg<RooAbsReal>(p, "resolutionModel"));
   if (!model) {
      RooJSONFactoryWSTool::error("resolutionModel of '" + name + "' is not a RooResolutionModel");
   }
   RooDecay::DecayType decayType = static_cast<RooDecay::DecayType>(p["decayType"].val_int());
   tool->wsEmplace<RooDecay>(name, *t, *tau, *model, decayType);
}

void importRealIntegral(RooJSONFactoryWSTool *tool, const JSONNode &p)
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
   JSONNode const *domain = p.find("domain");
   // todo: at some point, take care of integrator configurations
   tool->wsEmplace<RooRealIntegral>(name, *func, vars, normSetPtr, static_cast<RooNumIntConfig *>(nullptr),
                                    domain ? domain->val().c_str() : nullptr);
}

void importDerivative(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooAbsReal *func = tool->requestArg<RooAbsReal>(p, "function");
   RooRealVar *x = tool->requestArg<RooRealVar>(p, "x");
   Int_t order = p["order"].val_int();
   double eps = p["eps"].val_double();
   RooArgSet normSet;
   if (p.has_child("normalization")) {
      normSet.add(tool->requestArgSet<RooAbsReal>(p, "normalization"));
   }
   tool->wsEmplace<RooDerivative>(name, *func, *x, normSet, order, eps);
}

void importFFTConvPdf(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooRealVar *convVar = tool->requestArg<RooRealVar>(p, "conv_var");
   Int_t order = p["ipOrder"].val_int();
   RooAbsPdf *pdf1 = tool->requestArg<RooAbsPdf>(p, "pdf1");
   RooAbsPdf *pdf2 = tool->requestArg<RooAbsPdf>(p, "pdf2");
   if (p.has_child("conv_func")) {
      RooAbsReal *convFunc = tool->requestArg<RooAbsReal>(p, "conv_func");
      tool->wsEmplace<RooFFTConvPdf>(name, *convFunc, *convVar, *pdf1, *pdf2, order);
   } else {
      tool->wsEmplace<RooFFTConvPdf>(name, *convVar, *pdf1, *pdf2, order);
   }
}

void importExtendPdf(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   // HS3 calls the fields "dist" and "rate". RooFit used to write "pdf" and "norm", which are still accepted.
   RooAbsPdf *pdf = tool->requestArg<RooAbsPdf>(p, p.has_child("pdf") && !p.has_child("dist") ? "pdf" : "dist");
   RooAbsReal *norm = tool->requestArg<RooAbsReal>(p, p.has_child("norm") && !p.has_child("rate") ? "norm" : "rate");
   JSONNode const *range = p.find("range");
   tool->wsEmplace<RooExtendPdf>(name, *pdf, *norm, range ? range->val().c_str() : nullptr);
}

// HS3 0.3 calls the location parameter of the Landau distribution "mu", since it is neither the mean nor the most
// probable value. Documents before 0.3 used "mean", which is still read.
void importLandau(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   const std::string name(RooJSONFactoryWSTool::name(p));
   const bool legacyKey = !p.has_child("mu") && p.has_child("mean");
   if (legacyKey && !(tool->hs3Version() < HS3::Version{0, 3})) {
      RooJSONFactoryWSTool::warning("RooFitHS3: the landau_dist '" + name +
                                    "' uses the key \"mean\", which HS3 0.3 renamed to \"mu\".");
   }
   RooAbsReal *x = tool->requestArg<RooAbsReal>(p, "x");
   RooAbsReal *mu = tool->requestArg<RooAbsReal>(p, legacyKey ? "mean" : "mu");
   RooAbsReal *sigma = tool->requestArg<RooAbsReal>(p, "sigma");
   tool->wsEmplace<RooLandau>(name, *x, *mu, *sigma);
}

void importLogNormal(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooAbsReal *x = tool->requestArg<RooAbsReal>(p, "x");

   // Same mechanism to undo the parameter transformation as in the
   // importExponential() function (see comments in that function for more info).
   const std::string mu = p["mu"].val();
   const std::string sigma = p["sigma"].val();
   const bool isStandard = !endsWith(mu, lognormalLogSuffix);
   RooAbsReal *median = tool->request<RooAbsReal>(isStandard ? mu : transformedOriginal(mu, lognormalLogSuffix), name);
   RooAbsReal *shape =
      tool->request<RooAbsReal>(isStandard ? sigma : transformedOriginal(sigma, lognormalLogSuffix), name);

   tool->wsEmplace<RooLognormal>(name, *x, *median, *shape, isStandard);
}

void importExponential(RooJSONFactoryWSTool *tool, const JSONNode &p)
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
   //      "type": "generic",
   //      "expression": "-c"
   //  }
   //
   //  On import, we can directly take the non-transformed parameter, which is
   //  we check for the suffix and optionally remove it from the requested
   //  name next:

   const std::string constParamName = p["c"].val();
   const bool isInverted = endsWith(constParamName, "_exponential_inverted");
   RooAbsReal *c = tool->request<RooAbsReal>(
      isInverted ? transformedOriginal(constParamName, "_exponential_inverted") : constParamName, name);

   tool->wsEmplace<RooExponential>(name, *x, *c, !isInverted);
}

void importMultiVarGaussian(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   const bool hasCov = p.has_child("covariances");
   const bool hasCorr = p.has_child("correlations") && p.has_child("standard_deviations");
   if (!hasCov && !hasCorr) {
      RooJSONFactoryWSTool::error("no covariances or correlations+standard_deviations given in '" + name + "'");
   }
   // The covariances are given directly, or as correlations that are scaled with the standard deviations.
   const std::string key = hasCov ? "covariances" : "correlations";
   std::vector<double> sigmas;
   if (hasCov) {
      sigmas.assign(p[key].num_children(), 1.0);
   } else {
      sigmas << p["standard_deviations"];
   }
   const std::size_t n = sigmas.size();
   // The matrix must be square, otherwise the entries would be written out of bounds.
   if (p[key].num_children() != n) {
      RooJSONFactoryWSTool::error("\"" + key + "\" in '" + name + "' must have " + std::to_string(n) + " rows");
   }
   TMatrixDSym covmat;
   covmat.ResizeTo(n, n);
   int i = 0;
   for (const auto &row : p[key].children()) {
      if (!row.is_seq() || row.num_children() != n) {
         RooJSONFactoryWSTool::error("\"" + key + "\" in '" + name + "' must be a square matrix");
      }
      int j = 0;
      for (const auto &val : row.children()) {
         covmat(i, j) = val.val_double() * sigmas[i] * sigmas[j];
         ++j;
      }
      ++i;
   }
   tool->wsEmplace<RooMultiVarGaussian>(name, tool->requestArgList<RooAbsReal>(p, "x"),
                                        tool->requestArgList<RooAbsReal>(p, "mean"), covmat);
}

/// Read the binning variables from the "axes" node, ordered like in `varList`.
RooArgList readBinning(const JSONNode &topNode, const RooArgList &varList)
{
   RooArgSet axes = RooJSONFactoryWSTool::readAxes(topNode);
   RooArgList vars;
   for (RooAbsArg *refVar : varList) {
      if (RooAbsArg *axis = axes.find(*refVar)) {
         vars.addClone(*axis);
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
   const bool hasAxes = p.has_child("axes");
   if (!hasAxes) {
      RooJSONFactoryWSTool::warning(
         "No axes given in '" + name +
         "'. Using default binning (uniform; nbins=100). If needed, export the Workspace to JSON with a newer "
         "Root version that supports custom ParamHistFunc binnings(>=6.38.00).\n");
   }
   RooArgList binning = hasAxes ? readBinning(p, varList) : varList;
   checkParamHistFuncOrderIsDefined(tool->hs3Version(), binning, "the ParamHistFunc '" + name + "'");
   // The parameters are given in the order of HS3 (last axis fastest); ParamHistFunc expects TH1 order.
   tool->wsEmplace<ParamHistFunc>(name, binning,
                                  toParamHistFuncOrder(tool->requestArgList<RooAbsReal>(p, "parameters"), binning));
   return true;
}

void importSpline(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   const std::string name(RooJSONFactoryWSTool::name(p));
   RooAbsReal *x = tool->requestArg<RooAbsReal>(p, "x");
   if (!p.has_child("x0") || !p.has_child("y0")) {
      RooJSONFactoryWSTool::error("no x0/y0 given in '" + name + "'");
   }

   // The optional fields have the defaults of RooSpline.
   const std::string algo = p.has_child("interpolation") ? p["interpolation"].val() : "poly3";
   if (algo != "poly3" && algo != "poly5") {
      RooJSONFactoryWSTool::error("unsupported algo '" + algo + "' for RooSpline in '" + name +
                                  "': allowed are 'poly3' and 'poly5'");
   }
   const bool logx = p.has_child("logx") && p["logx"].val_bool();
   const bool logy = p.has_child("logy") && p["logy"].val_bool();

   std::vector<double> x0;
   std::vector<double> y0;
   x0 << p["x0"];
   y0 << p["y0"];

   if (x0.size() != y0.size()) {
      RooJSONFactoryWSTool::error("x0/y0 size mismatch in '" + name + "': x0 has " + std::to_string(x0.size()) +
                                  ", y0 has " + std::to_string(y0.size()));
   }
   if (x0.size() < 2) {
      RooJSONFactoryWSTool::error("need at least 2 knots in '" + name + "'");
   }

   tool->wsEmplace<::RooSpline>(name.c_str(), *x, x0, y0, algo == "poly5" ? 5 : 3, logx, logy);
}

void importBernstein(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   auto *x = dynamic_cast<RooAbsRealLValue *>(tool->requestArg<RooAbsReal>(p, "x"));
   if (!x) {
      RooJSONFactoryWSTool::error("\"x\" of '" + name + "' is not a variable");
   }
   tool->wsEmplace<RooBernstein>(name, *x, tool->requestArgList<RooAbsReal>(p, "coefficients"));
}

void importGaussian(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooAbsReal *x = tool->requestArg<RooAbsReal>(p, "x");
   RooAbsReal *mean = tool->requestArg<RooAbsReal>(p, "mean");
   RooAbsReal *sigma = tool->requestArg<RooAbsReal>(p, "sigma");
   tool->wsEmplace<RooGaussian>(name, *x, *mean, *sigma);
}

void importUniform(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   // A single observable may also be given by its name.
   if (p.has_child("x") && !p["x"].is_seq()) {
      tool->wsEmplace<RooUniform>(name, RooArgSet{*tool->requestArg<RooAbsReal>(p, "x")});
   } else {
      tool->wsEmplace<RooUniform>(name, tool->requestArgSet<RooAbsReal>(p, "x"));
   }
}

namespace CrystalBall {

// The keys of the tails of the double-sided Crystal Ball.
const std::initializer_list<const char *> tailKeys = {"alpha_L", "n_L", "alpha_R", "n_R"};

bool hasAny(const JSONNode &p, std::initializer_list<const char *> keys)
{
   return std::any_of(keys.begin(), keys.end(), [&p](const char *key) { return p.has_child(key); });
}

bool hasDoubleSidedKeys(const JSONNode &p)
{
   return hasAny(p, tailKeys) || hasAny(p, {"sigma_L", "sigma_R"});
}

// Create a RooCrystalBall with two tails. With `symmetricShorthand`, the tails may also be given as "alpha" and "n"
// for both sides, as in the "crystalball_dist" of HS3 versions before 0.3.
void importDoubleSided(RooJSONFactoryWSTool *tool, const JSONNode &p, std::string const &type, bool symmetricShorthand)
{
   const std::string name(RooJSONFactoryWSTool::name(p));
   const std::string context = type + " '" + name + "'";

   const bool asymmetricTails = hasAny(p, tailKeys);
   const bool symmetricTails = hasAny(p, {"alpha", "n"});
   if (symmetricTails && !symmetricShorthand) {
      RooJSONFactoryWSTool::error(context + " must not have \"alpha\" or \"n\", the tails are given by \"alpha_L\", " +
                                  "\"n_L\", \"alpha_R\" and \"n_R\"");
   }
   if (symmetricTails && asymmetricTails) {
      RooJSONFactoryWSTool::error(context +
                                  " must not combine \"alpha\"/\"n\" with \"alpha_L\"/\"n_L\"/\"alpha_R\"/\"n_R\"");
   }
   if (asymmetricTails || !symmetricTails) {
      for (const char *key : tailKeys) {
         if (!p.has_child(key)) {
            RooJSONFactoryWSTool::error(context + " is missing \"" + key +
                                        "\": \"alpha_L\", \"n_L\", \"alpha_R\" and \"n_R\" are all required");
         }
      }
   } else if (!p.has_child("alpha") || !p.has_child("n")) {
      RooJSONFactoryWSTool::error(context + " needs both \"alpha\" and \"n\"");
   }

   const bool asymmetricWidths = hasAny(p, {"sigma_L", "sigma_R"});
   if (asymmetricWidths && p.has_child("sigma")) {
      RooJSONFactoryWSTool::error(context + " must not combine \"sigma\" with \"sigma_L\"/\"sigma_R\"");
   }
   if (asymmetricWidths && !(p.has_child("sigma_L") && p.has_child("sigma_R"))) {
      RooJSONFactoryWSTool::error(context + " needs both \"sigma_L\" and \"sigma_R\"");
   }
   if (!asymmetricWidths && !p.has_child("sigma")) {
      RooJSONFactoryWSTool::error(context + " needs either \"sigma\" or both \"sigma_L\" and \"sigma_R\"");
   }

   RooAbsReal *m = tool->requestArg<RooAbsReal>(p, "m");
   RooAbsReal *m0 = tool->requestArg<RooAbsReal>(p, "m0");
   RooAbsReal *sigmaL = tool->requestArg<RooAbsReal>(p, asymmetricWidths ? "sigma_L" : "sigma");
   RooAbsReal *sigmaR = tool->requestArg<RooAbsReal>(p, asymmetricWidths ? "sigma_R" : "sigma");
   RooAbsReal *alphaL = tool->requestArg<RooAbsReal>(p, symmetricTails ? "alpha" : "alpha_L");
   RooAbsReal *nL = tool->requestArg<RooAbsReal>(p, symmetricTails ? "n" : "n_L");
   RooAbsReal *alphaR = tool->requestArg<RooAbsReal>(p, symmetricTails ? "alpha" : "alpha_R");
   RooAbsReal *nR = tool->requestArg<RooAbsReal>(p, symmetricTails ? "n" : "n_R");
   tool->wsEmplace<RooCrystalBall>(name, *m, *m0, *sigmaL, *sigmaR, *alphaL, *nL, *alphaR, *nR);
}

} // namespace CrystalBall

// "crystalball_dist" is the single-sided Crystal Ball (RooCBShape). Before HS3 0.3, it was also the double-sided
// Crystal Ball, recognizable by the "_L"/"_R" fields. Such documents are still read by this importer, with a
// deprecation warning. It declines the single-sided Crystal Ball.
bool importLegacyDoubleSidedCrystalBall(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   if (!CrystalBall::hasDoubleSidedKeys(p)) {
      return false;
   }
   static bool warned = false;
   if (!warned) {
      RooJSONFactoryWSTool::warning("RooFitHS3: \"crystalball_dist\" with the fields \"alpha_L\", \"n_L\", "
                                    "\"alpha_R\", \"n_R\", \"sigma_L\" or \"sigma_R\" is deprecated, please use "
                                    "\"crystalball_doublesided_dist\" for the double-sided Crystal Ball.");
      warned = true;
   }
   CrystalBall::importDoubleSided(tool, p, "crystalball_dist", /*symmetricShorthand=*/true);
   return true;
}

void importCrystalBallDoubleSided(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   CrystalBall::importDoubleSided(tool, p, "crystalball_doublesided_dist", /*symmetricShorthand=*/false);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// specialized exporter implementations
///////////////////////////////////////////////////////////////////////////////////////////////////////

// A RooCrystalBall with one tail is the single-sided "crystalball_dist" (read back as RooCBShape, which follows the
// same convention for the sign of alpha). With two tails, it is a "crystalball_doublesided_dist", whose tails are
// always written as "alpha_L"/"n_L"/"alpha_R"/"n_R": there is no shorthand for symmetric tails.
void exportCrystalBall(RooJSONFactoryWSTool *, RooCrystalBall const &pdf, JSONNode &elem)
{
   writeReference(elem["m"], pdf.x());
   writeReference(elem["m0"], pdf.x0());
   if (!pdf.alphaR() || !pdf.nR()) {
      elem["type"] << "crystalball_dist";
      writeReference(elem["sigma"], pdf.sigmaL());
      writeReference(elem["alpha"], pdf.alphaL());
      writeReference(elem["n"], pdf.nL());
      return;
   }
   if (&pdf.sigmaL() == &pdf.sigmaR()) {
      writeReference(elem["sigma"], pdf.sigmaL());
   } else {
      writeReference(elem["sigma_L"], pdf.sigmaL());
      writeReference(elem["sigma_R"], pdf.sigmaR());
   }
   writeReference(elem["alpha_L"], pdf.alphaL());
   writeReference(elem["n_L"], pdf.nL());
   writeReference(elem["alpha_R"], *pdf.alphaR());
   writeReference(elem["n_R"], *pdf.nR());
}

template <class RooArg_t>
void exportAddPdf(RooJSONFactoryWSTool *, RooArg_t const &pdf, JSONNode &elem)
{
   RooJSONFactoryWSTool::fillSeq(elem["summands"], pdf.pdfList());
   RooJSONFactoryWSTool::fillSeq(elem["coefficients"], pdf.coefList());
   elem["extended"] << (pdf.extendMode() != RooArg_t::CanNotBeExtended);
   // A RooAddModel is written as a mixture_model, which is not part of HS3.
   if (pdf.coefList().empty() && std::is_same_v<RooArg_t, RooAddPdf>) {
      RooJSONFactoryWSTool::warning("RooFitHS3: the RooAddPdf '" + std::string(pdf.GetName()) +
                                    "' has no coefficients. It is written as a mixture_dist without coefficients, "
                                    "which HS3 0.3 doesn't define, so other implementations might not read it.");
   }
}

void exportRealSumPdf(RooJSONFactoryWSTool *, RooRealSumPdf const &pdf, JSONNode &elem)
{
   RooJSONFactoryWSTool::fillSeq(elem["samples"], pdf.funcList());
   RooJSONFactoryWSTool::fillSeq(elem["coefficients"], pdf.coefList());
   elem["extended"] << (pdf.extendMode() != RooAbsPdf::CanNotBeExtended);
}

template <class RooArg_t>
void exportHist(RooJSONFactoryWSTool *tool, RooArg_t const &hf, JSONNode &elem)
{
   RooDataHist const &dh = hf.dataHist();
   // The axes are named after the observables of the function, which are not necessarily the variables of the
   // histogram.
   auto vars = histogramVariablesNamedAfterObservables(dh, hf.getHistObsList(), hf.variables());
   tool->exportHisto(*vars, dh.numEntries(), dh.weightArray(), elem["data"].set_map());
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

void exportBinSamplingPdf(RooJSONFactoryWSTool *, RooBinSamplingPdf const &pdf, JSONNode &elem)
{
   elem["pdf"] << pdf.pdf().GetName();
   elem["observable"] << pdf.observable().GetName();
   elem["epsilon"] << pdf.epsilon();
}

void exportBinWidthFunction(RooJSONFactoryWSTool *, RooBinWidthFunction const &pdf, JSONNode &elem)
{
   // It is registered for "binvolume".
   if (pdf.divideByBinWidth()) {
      elem["type"] << "inverse_binvolume";
   }
   elem["histogram"] << pdf.histFunc().GetName();
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

std::optional<std::string> ternariesToIfElse(std::string const &in);

// Rewrite one comma-free part of an expression, see ternariesToIfElse().
std::optional<std::string> ternaryPartToIfElse(std::string const &in)
{
   // Find the first '?' that is not enclosed in brackets. A ':' can only follow a '?', and a part has no commas.
   const std::size_t question = findTopLevel(in, "?:,");
   if (question != std::string::npos && in[question] != '?') {
      return std::nullopt;
   }

   if (question == std::string::npos) {
      // No ternary on this level: copy the expression and rewrite the ternaries inside of brackets.
      std::string out;
      for (std::size_t i = 0; i < in.size(); ++i) {
         if (in[i] == ')' || in[i] == ']') {
            return std::nullopt;
         }
         if (in[i] != '(' && in[i] != '[') {
            out += in[i];
            continue;
         }
         const std::size_t close = findClosingBracket(in, i);
         if (close == std::string::npos) {
            return std::nullopt;
         }
         const std::string inner = in.substr(i + 1, close - i - 1);
         const std::optional<std::string> innerOut = ternariesToIfElse(inner);
         if (!innerOut) {
            return std::nullopt;
         }
         // Parentheses that only group a ternary are redundant around the ifelse() function call.
         const std::string before = trimmed(in.substr(0, i));
         const bool isGrouping = in[i] == '(' && (before.empty() || !isIdentifierChar(before.back()));
         if (isGrouping && *innerOut != trimmed(inner) && startsWith(*innerOut, "ifelse(") &&
             findClosingBracket(*innerOut, std::string("ifelse").size()) == innerOut->size() - 1) {
            out += *innerOut;
         } else {
            out += in[i] + *innerOut + in[close];
         }
         i = close;
      }
      return out;
   }

   // Find the ':' that belongs to this '?': a nested '?' comes with a ':' of its own, which is skipped.
   std::size_t colon = question;
   for (int nQuestions = 1; nQuestions > 0;) {
      colon = findTopLevel(in, "?:", colon + 1);
      if (colon == std::string::npos) {
         return std::nullopt;
      }
      nQuestions += in[colon] == '?' ? 1 : -1;
   }

   const auto condition = ternaryPartToIfElse(stripEnclosingParentheses(in.substr(0, question)));
   const auto ifTrue = ternaryPartToIfElse(stripEnclosingParentheses(in.substr(question + 1, colon - question - 1)));
   const auto ifFalse = ternaryPartToIfElse(stripEnclosingParentheses(in.substr(colon + 1)));
   if (!condition || !ifTrue || !ifFalse || condition->empty() || ifTrue->empty() || ifFalse->empty()) {
      return std::nullopt;
   }
   return "ifelse(" + *condition + "," + *ifTrue + "," + *ifFalse + ")";
}

// Rewrite the C-style ternaries "c ? a : b" in a TFormula expression into the HS3 conditional "ifelse(c,a,b)". Returns
// std::nullopt if this is not safely possible, e.g. because of unbalanced brackets.
std::optional<std::string> ternariesToIfElse(std::string const &in)
{
   std::string out;
   bool first = true;
   for (std::string const &part : splitTopLevelCommas(in)) {
      const std::optional<std::string> partOut = ternaryPartToIfElse(part);
      if (!partOut) {
         return std::nullopt;
      }
      out += (first ? "" : ",") + *partOut;
      first = false;
   }
   return out;
}

// HS3 0.3 (HS3 PR #154) only allows expressions to refer to parameters, variables and functions.
void checkExpressionRefersToNoPdf(RooAbsArg const &formula, RooAbsCollection const &dependents)
{
   for (RooAbsArg const *dependent : dependents) {
      if (dynamic_cast<RooAbsPdf const *>(dependent)) {
         RooJSONFactoryWSTool::error(std::string("the expression of '") + formula.GetName() + "' refers to the pdf '" +
                                     dependent->GetName() + "', which HS3 doesn't allow");
      }
   }
}

template <class RooArg_t>
void exportFormulaArg(RooJSONFactoryWSTool *, RooArg_t const &pdf, JSONNode &elem)
{
   checkExpressionRefersToNoPdf(pdf, pdf.dependents());
   TString expression(pdf.expression());
   cleanExpression(expression);
   // If the tokens follow the "x[#]" convention, the square braces enclosing each number
   // ensures that there is a unique mapping between the token and parameter name
   // If the tokens follow the "@#" convention, the numbers are not enclosed by braces.
   // So there may be tokens with numbers whose lower place value forms a subset string of ones with a higher place
   // value, e.g. "@1" is a subset of "@10". So the names of these parameters must be applied descending from the
   // highest place value in order to ensure each parameter name is uniquely applied to its token.
   for (size_t idx = pdf.nParameters(); idx--;) {
      const RooAbsArg *par = pdf.getParameter(idx);
      expression.ReplaceAll(("x[" + std::to_string(idx) + "]").c_str(), par->GetName());
      expression.ReplaceAll(("@" + std::to_string(idx)).c_str(), par->GetName());
   }
   if (expression.Contains("?")) {
      if (const auto rewritten = ternariesToIfElse(expression.Data())) {
         expression = rewritten->c_str();
      } else {
         RooJSONFactoryWSTool::warning(std::string("RooFitHS3: could not translate the ternary operators in the "
                                                   "expression of '") +
                                       pdf.GetName() + "' to ifelse(), exporting the expression unchanged.");
      }
   }
   elem["expression"] << expression.Data();

   for (const RooAbsArg *dependent : pdf.dependents()) {
      auto const *observable = dynamic_cast<const RooAbsRealLValue *>(dependent);
      if (!observable) {
         continue;
      }
      const RooAbsBinning *binning = pdf.getBinning(*observable);
      if (!binning) {
         continue;
      }

      HS3::writeAxis(elem["axes"].set_seq().append_child(), HS3::toBinnedAxis(observable->GetName(), *binning));
   }
}

// Write the "x" reference and the coefficient list for polynomial-like
// pdfs/funcs, including the implicit defaults below "lowestOrder" so that the
// output is self-documenting.
template <class RooArg_t>
void exportPolynomial(RooJSONFactoryWSTool *, RooArg_t const &pdf, JSONNode &elem)
{
   elem["x"] << pdf.x().GetName();
   auto &coefs = elem["coefficients"].set_seq();
   for (int i = 0; i < pdf.lowestOrder(); ++i) {
      coefs.append_child() << (i == 0 ? 1.0 : 0.0);
   }
   for (const auto &coef : pdf.coefList()) {
      writeReference(coefs.append_child(), *coef);
   }
}

void exportPoisson(RooJSONFactoryWSTool *, RooPoisson const &pdf, JSONNode &elem)
{
   // It is registered for "poisson_dist", the discrete Poisson distribution.
   if (pdf.getNoRounding()) {
      elem["type"] << "poisson_continuous_dist";
   }
   writeReference(elem["x"], pdf.getX());
   writeReference(elem["mean"], pdf.getMean());
}

// The RooDecay servers are the internal resModel-times-basis convolutions,
// which are implementation details that should not leak into the JSON. We
// only export the actual dependents (t, tau and the original resolution
// model) explicitly.
void exportDecay(RooJSONFactoryWSTool *tool, RooDecay const &pdf, JSONNode &elem)
{
   elem["t"] << pdf.getT().GetName();
   writeReference(elem["tau"], pdf.getTau());
   elem["resolutionModel"] << pdf.getModel().GetName();
   elem["decayType"] << pdf.getDecayType();

   tool->queueExport(pdf.getT());
   tool->queueExport(pdf.getTau());
   tool->queueExport(pdf.getModel());
}

// A Gaussian constraint whose recorded observable (see RooJSONFactoryWSTool::setConstraintObservable()) is its
// mean is written with x and mean swapped, so that the observable is x.
void exportGaussian(RooJSONFactoryWSTool *tool, RooGaussian const &gauss, JSONNode &elem)
{
   std::string const *observable = tool->constraintObservable(gauss.GetName());
   const bool swap = observable && *observable == gauss.getMean().GetName() && !isLiteralConstVar(gauss.getMean());
   writeReference(elem["x"], swap ? gauss.getMean() : gauss.getX());
   writeReference(elem["mean"], swap ? gauss.getX() : gauss.getMean());
   writeReference(elem["sigma"], gauss.getSigma());
}

void exportLogNormal(RooJSONFactoryWSTool *tool, RooLognormal const &pdf, JSONNode &elem)
{
   writeReference(elem["x"], pdf.getX());

   auto &m0 = pdf.getMedian();
   auto &k = pdf.getShapeK();

   if (pdf.useStandardParametrization()) {
      writeReference(elem["mu"], m0);
      writeReference(elem["sigma"], k);
   } else {
      elem["mu"] << lognormalLogName(m0, tool);
      elem["sigma"] << lognormalLogName(k, tool);
   }
}

void exportExponential(RooJSONFactoryWSTool *tool, RooExponential const &pdf, JSONNode &elem)
{
   elem["x"] << pdf.variable().GetName();
   auto &c = pdf.coefficient();
   if (pdf.negateCoefficient()) {
      writeReference(elem["c"], c);
   } else {
      elem["c"] << tool->exportTransformed(&c, "_exponential_inverted", "-%s");
   }
}

void exportMultiVarGaussian(RooJSONFactoryWSTool *, RooMultiVarGaussian const &pdf, JSONNode &elem)
{
   RooJSONFactoryWSTool::fillSeq(elem["x"], pdf.xVec());
   RooJSONFactoryWSTool::fillSeq(elem["mean"], pdf.muVec());
   elem["covariances"].fill_mat(pdf.covarianceMatrix());
}

void exportTFnBinding(RooJSONFactoryWSTool *, RooTFnBinding const &pdf, JSONNode &elem)
{
   checkExpressionRefersToNoPdf(pdf, pdf.observables());
   checkExpressionRefersToNoPdf(pdf, pdf.parameters());
   TString formula(pdf.function().GetExpFormula());
   formula.ReplaceAll("x", pdf.observables()[0].GetName());
   formula.ReplaceAll("y", pdf.observables()[1].GetName());
   formula.ReplaceAll("z", pdf.observables()[2].GetName());
   for (size_t i = 0; i < pdf.parameters().size(); ++i) {
      TString pname(TString::Format("[%d]", (int)i));
      formula.ReplaceAll(pname, pdf.parameters()[i].GetName());
   }
   elem["expression"] << formula.Data();
}

void exportDerivative(RooJSONFactoryWSTool *, RooDerivative const &pdf, JSONNode &elem)
{
   elem["x"] << pdf.getX().GetName();
   elem["function"] << pdf.getFunc().GetName();
   if (!pdf.getNset().empty()) {
      RooJSONFactoryWSTool::fillSeq(elem["normalization"], pdf.getNset());
   }
   elem["order"] << pdf.order();
   elem["eps"] << pdf.eps();
}

void exportRealIntegral(RooJSONFactoryWSTool *, RooRealIntegral const &integral, JSONNode &elem)
{
   std::string integrand = integral.integrand().GetName();
   elem["integrand"] << integrand;
   if (integral.intRange()) {
      elem["domain"] << integral.intRange();
   }
   RooJSONFactoryWSTool::fillSeq(elem["variables"], integral.intVars());
   if (RooArgSet const *funcNormSet = integral.funcNormSet()) {
      RooJSONFactoryWSTool::fillSeq(elem["normalization"], *funcNormSet);
   }
}

void exportFFTConvPdf(RooJSONFactoryWSTool *, RooFFTConvPdf const &pdf, JSONNode &elem)
{
   if (auto convFunc = pdf.getPdfConvVar()) {
      elem["conv_func"] << convFunc->GetName();
   }
   elem["conv_var"] << pdf.getConvVar().GetName();
   elem["pdf1"] << pdf.getPdf1().GetName();
   elem["pdf2"] << pdf.getPdf2().GetName();
   elem["ipOrder"] << pdf.getInterpolationOrder();
}

void exportExtendPdf(RooJSONFactoryWSTool *, RooExtendPdf const &pdf, JSONNode &elem)
{
   if (auto rangeName = pdf.getRangeName()) {
      elem["range"] << rangeName->GetName();
   }
   elem["dist"] << pdf.pdf().GetName();
   writeReference(elem["rate"], pdf.getN());
}

void exportParamHistFunc(RooJSONFactoryWSTool *, ParamHistFunc const &pdf, JSONNode &elem)
{
   RooJSONFactoryWSTool::fillSeq(elem["variables"], pdf.dataVars());
   // One parameter per bin, in the order of HS3 (last axis fastest), not in ParamHistFunc's internal TH1 order.
   auto &paramsNode = elem["parameters"].set_seq();
   for (RooAbsReal *param : paramHistFuncParamsInBinOrder(pdf)) {
      writeReference(paramsNode.append_child(), *param);
   }
   auto &observablesNode = elem["axes"].set_seq();
   // axes have to be ordered to get consistent bin indices
   for (auto *var : static_range_cast<RooRealVar *>(pdf.dataVars())) {
      HS3::writeAxis(observablesNode.append_child(), HS3::toBinnedAxis(*var));
   }
}

void exportSpline(RooJSONFactoryWSTool *, RooSpline const &rs, JSONNode &elem)
{
   elem["x"] << rs.x().GetName();
   elem["interpolation"] << (rs.order() == 5 ? "poly5" : "poly3");
   elem["logx"] << rs.logx();
   elem["logy"] << rs.logy();

   // The knots
   TSpline const &sp = rs.spline();
   auto &x0 = elem["x0"].set_seq();
   auto &y0 = elem["y0"].set_seq();
   for (int i = 0; i < sp.GetNp(); ++i) {
      double xk = 0.0;
      double yk = 0.0;
      sp.GetKnot(i, xk, yk);
      x0.append_child() << xk;
      y0.append_child() << yk;
   }
}

void exportBernstein(RooJSONFactoryWSTool *, RooBernstein const &pdf, JSONNode &elem)
{
   writeReference(elem["x"], pdf.x());
   RooJSONFactoryWSTool::fillSeq(elem["coefficients"], pdf.coefList());
}

void exportLandau(RooJSONFactoryWSTool *, RooLandau const &pdf, JSONNode &elem)
{
   writeReference(elem["x"], pdf.getX());
   writeReference(elem["mu"], pdf.getMean());
   writeReference(elem["sigma"], pdf.getSigma());
}

void exportProduct(RooJSONFactoryWSTool *, RooProduct const &prod, JSONNode &elem)
{
   RooJSONFactoryWSTool::fillSeq(elem["factors"], prod.realComponents());
   RooJSONFactoryWSTool::fillSeq(elem["factors"], prod.categoricalComponents());
}

void exportUniform(RooJSONFactoryWSTool *, RooUniform const &pdf, JSONNode &elem)
{
   RooJSONFactoryWSTool::fillSeq(elem["x"], pdf.observables());
}

void importWrapperPdf(RooJSONFactoryWSTool *tool, const JSONNode &p)
{
   std::string name(RooJSONFactoryWSTool::name(p));
   RooAbsReal *func = tool->requestArg<RooAbsReal>(p, "function");
   // ROOT previously exported this key without an underscore.
   const char *key = p.has_child("self_normalized") ? "self_normalized" : "selfnormalized";
   const bool selfNormalized = p.has_child(key) && p[key].val_bool();
   tool->wsEmplace<RooWrapperPdf>(name, *func, selfNormalized);
}

void exportWrapperPdf(RooJSONFactoryWSTool *, RooWrapperPdf const &pdf, JSONNode &elem)
{
   // RooWrapperPdf has no accessor for the wrapped function, which is its only proxy.
   auto const &funcProxy = dynamic_cast<RooRealProxy const &>(*pdf.getProxy(0));
   elem["function"] << funcProxy.absArg()->GetName();
   if (pdf.selfNormalized()) {
      elem["self_normalized"] << true;
   }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// instantiate all importers and exporters
///////////////////////////////////////////////////////////////////////////////////////////////////////

// A class whose constructor takes nothing but objects and lists of objects, each of which it has a getter for, is
// read and written field by field. A field describes a constructor argument: its key in the document, and the getter
// that gives what is written.

// An argument that is one object: requested as a T, and written as a reference to it.
template <class T, class Getter_t>
struct ObjectField {
   const char *key;
   Getter_t getter;

   T &request(RooJSONFactoryWSTool &tool, const JSONNode &p) const { return *tool.requestArg<T>(p, key); }
   template <class RooArg_t>
   void write(RooArg_t const &arg, JSONNode &elem) const
   {
      writeReference(elem[key], (arg.*getter)());
   }
};

// An argument that is a list of objects: requested as a Collection_t of T, and written as a list of references.
template <class T, class Collection_t, class Getter_t>
struct ListField {
   const char *key;
   Getter_t getter;

   Collection_t request(RooJSONFactoryWSTool &tool, const JSONNode &p) const
   {
      return tool.requestCollection<T, Collection_t>(p, key);
   }
   template <class RooArg_t>
   void write(RooArg_t const &arg, JSONNode &elem) const
   {
      RooJSONFactoryWSTool::fillSeq(elem[key], (arg.*getter)());
   }
};

template <class T = RooAbsReal, class Getter_t>
ObjectField<T, Getter_t> object(const char *key, Getter_t getter)
{
   return {key, getter};
}

template <class T = RooAbsReal, class Collection_t = RooArgList, class Getter_t>
ListField<T, Collection_t, Getter_t> list(const char *key, Getter_t getter)
{
   return {key, getter};
}

// Register the importer and the exporter of the class RooArg_t, whose constructor takes the name, the title and then
// the fields in the given order.
template <class RooArg_t, class... Fields_t>
void registerFields(const std::string &key, Fields_t... fields)
{
   auto importFields = [=](RooJSONFactoryWSTool *tool, const JSONNode &p) {
      const std::string name = RooJSONFactoryWSTool::name(p);
      // The braces guarantee that the fields are requested in their order.
      tool->wsImport(RooArg_t{name.c_str(), name.c_str(), fields.request(*tool, p)...});
   };
   auto exportFields = [=](RooJSONFactoryWSTool *, RooArg_t const &arg, JSONNode &elem) {
      (fields.write(arg, elem), ...);
   };
   registerImporter(key, importFields);
   registerExporter<RooArg_t>(key, exportFields);
}

// A deprecated type key is still read by the importers of its replacement, with a warning the first time.
void registerDeprecatedKey(const std::string &key, const std::string &replacement)
{
   registerImporter(key, [=](RooJSONFactoryWSTool *tool, const JSONNode &p) {
      static std::set<std::string> warned;
      if (warned.insert(key).second) {
         RooJSONFactoryWSTool::warning("RooFitHS3: the type \"" + key + "\" is deprecated, please use \"" +
                                       replacement + "\" instead.");
      }
      auto const &importers = RooFit::JSONIO::importers().at(replacement);
      return std::any_of(importers.begin(), importers.end(),
                         [&](auto const &importer) { return importer->importArg(tool, p); });
   });
}

// Registers the importers and exporters when the library is loaded.
const bool registered = []() {
   registerType("bernstein_poly_dist", importBernstein, exportBernstein);
   registerType("binsampling_dist", importBinSamplingPdf, exportBinSamplingPdf);
   registerType("binvolume", importBinWidthFunction<false>, exportBinWidthFunction);
   registerType("crystalball_doublesided_dist", importCrystalBallDoubleSided, exportCrystalBall);
   registerType("decay_dist", importDecay, exportDecay, /*autoExportDependants=*/false);
   registerType("density_function_dist", importWrapperPdf, exportWrapperPdf);
   registerType("derivative", importDerivative, exportDerivative);
   registerType("exponential_dist", importExponential, exportExponential);
   registerType("fft_convolution_dist", importFFTConvPdf, exportFFTConvPdf);
   registerType("gaussian_dist", importGaussian, exportGaussian);
   registerType("generic", importFormulaArg<RooFormulaVar>, exportFormulaArg<RooFormulaVar>);
   registerType("generic_dist", importFormulaArg<RooGenericPdf>, exportFormulaArg<RooGenericPdf>);
   registerType("histogram_dist", importHist<RooHistPdf>, exportHist<RooHistPdf>);
   registerType("integral", importRealIntegral, exportRealIntegral);
   registerType("landau_dist", importLandau, exportLandau);
   registerType("legacy_exp_poly_dist", importPolynomial<RooLegacyExpPoly>, exportPolynomial<RooLegacyExpPoly>);
   registerType("lognormal_dist", importLogNormal, exportLogNormal);
   registerType("mixture_dist", importAddPdf, exportAddPdf<RooAddPdf>);
   registerType("mixture_model", importAddModel, exportAddPdf<RooAddModel>);
   registerType("multivariate_normal_dist", importMultiVarGaussian, exportMultiVarGaussian);
   registerType("poisson_dist", importPoisson<false>, exportPoisson);
   registerType("polynomial", importPolynomial<RooPolyVar>, exportPolynomial<RooPolyVar>);
   registerType("polynomial_dist", importPolynomial<RooPolynomial>, exportPolynomial<RooPolynomial>);
   registerType("product", importProduct, exportProduct);
   registerType("rate_extended_dist", importExtendPdf, exportExtendPdf);
   registerType("spline", importSpline, exportSpline);
   // A "step" with "data" is a histogram, otherwise it is a ParamHistFunc.
   registerType("step", importHist<RooHistFunc>, exportHist<RooHistFunc>);
   registerType("step", importParamHistFunc, exportParamHistFunc);
   registerType("uniform_dist", importUniform, exportUniform);
   registerType("weighted_sum_dist", importRealSumPdf, exportRealSumPdf);

   // The classes that are read and written field by field.
   registerFields<RooArgusBG>("argus_dist", object("mass", &RooArgusBG::getM), object("resonance", &RooArgusBG::getM0),
                              object("slope", &RooArgusBG::getC), object("power", &RooArgusBG::getP));
   registerFields<RooBifurGauss>("bifurkated_gaussian_dist", object("x", &RooBifurGauss::getX),
                                 object("mean", &RooBifurGauss::getMean), object("sigmaL", &RooBifurGauss::getSigmaL),
                                 object("sigmaR", &RooBifurGauss::getSigmaR));
   // A "crystalball_dist" with the fields of two tails is the double-sided Crystal Ball of documents before HS3 0.3.
   registerImporter("crystalball_dist", importLegacyDoubleSidedCrystalBall);
   registerFields<RooCBShape>("crystalball_dist", object("m", &RooCBShape::getM), object("m0", &RooCBShape::getM0),
                              object("sigma", &RooCBShape::getSigma), object("alpha", &RooCBShape::getAlpha),
                              object("n", &RooCBShape::getN));
   registerFields<RooChebychev>("chebychev_dist", object("x", &RooChebychev::x),
                                list("coefficients", &RooChebychev::coefList));
   registerFields<RooEffProd>("efficiency_product_pdf_dist", object<RooAbsPdf>("pdf", &RooEffProd::pdf),
                              object("eff", &RooEffProd::eff));
   registerFields<RooGamma>("gamma_dist", object("x", &RooGamma::getX), object("gamma", &RooGamma::getGamma),
                            object("beta", &RooGamma::getBeta), object("mu", &RooGamma::getMu));
   registerFields<RooGaussModel>("gauss_model_function", object<RooRealVar>("x", &RooGaussModel::convVar),
                                 object("mean", &RooGaussModel::getMean), object("sigma", &RooGaussModel::getSigma));
   registerFields<RooPowerSum>("power_sum_dist", object("x", &RooPowerSum::base),
                               list("coefficients", &RooPowerSum::coefList), list("exponents", &RooPowerSum::expList));
   registerFields<RooProdPdf>("product_dist", list<RooAbsPdf>("factors", &RooProdPdf::pdfList));
   registerFields<RooProjectedPdf>("projected_dist", object<RooAbsPdf>("input_pdf", &RooProjectedPdf::intPdf),
                                   list<RooAbsReal, RooArgSet>("observables", &RooProjectedPdf::intObs));
   registerFields<RooAddition>("sum", list("summands", &RooAddition::list));
   registerFields<RooTruthModel>("truth_model_function", object<RooRealVar>("x", &RooTruthModel::convVar));
   registerFields<RooRealSumFunc>("weighted_sum", list("samples", &RooRealSumFunc::funcList),
                                  list("coefficients", &RooRealSumFunc::coefList));

   // A RooTFnBinding is written as a generic formula.
   registerExporter<RooTFnBinding>("generic", exportTFnBinding);

   // The types that are only read: the ones that RooFit writes in another way, and other names of the types above.
   registerImporter("array_dist", importArrayDist);
   registerImporter("combined_dist", importCombinedDist);
   registerImporter("extend_pdf", importExtendPdf);
   registerImporter("generic_function", importFormulaArg<RooFormulaVar>);
   registerImporter("histogram", importHist<RooHistFunc>);
   registerImporter("inverse_binvolume", importBinWidthFunction<true>);
   registerImporter("normal_dist", importGaussian);
   registerImporter("poisson_continuous_dist", importPoisson<true>);
   registerDeprecatedKey("ARGUS_dist", "argus_dist");
   registerDeprecatedKey("binsampling", "binsampling_dist");
   registerDeprecatedKey("delta_resolution_model", "truth_model_function");
   registerDeprecatedKey("gauss_resolution_model", "gauss_model_function");
   registerDeprecatedKey("mixture_resolution_model", "mixture_model");
   return true;
}();

} // namespace

std::optional<ArrayComponentInfo>
arrayComponentInfo(RooAbsPdf const &pdf, std::string const *observable, RooJSONFactoryWSTool *tool)
{
   ArrayComponentInfo out;
   RooAbsReal const *x = nullptr;
   if (auto const *poisson = dynamic_cast<RooPoisson const *>(&pdf)) {
      out.type = poisson->getNoRounding() ? "poisson_continuous_dist" : "poisson_dist";
      x = &poisson->getX();
      out.fields = {{"mean", HS3::toNameOrNumber(poisson->getMean())}};
   } else if (auto const *gauss = dynamic_cast<RooGaussian const *>(&pdf)) {
      out.type = "gaussian_dist";
      x = &gauss->getX();
      RooAbsReal const *mean = &gauss->getMean();
      // A Gaussian constraint built by HistFactory has the parameter as x and the global observable as mean. The
      // Gaussian is symmetric in the two, so it is written with the observable as x where the observable is known.
      if (observable && *observable == mean->GetName() && !isLiteralConstVar(*mean)) {
         std::swap(x, mean);
      }
      out.fields = {{"mean", HS3::toNameOrNumber(*mean)}, {"sigma", HS3::toNameOrNumber(gauss->getSigma())}};
   } else if (auto const *lognormal = dynamic_cast<RooLognormal const *>(&pdf)) {
      out.type = "lognormal_dist";
      x = &lognormal->getX();
      if (lognormal->useStandardParametrization()) {
         out.fields = {{"mu", HS3::toNameOrNumber(lognormal->getMedian())},
                       {"sigma", HS3::toNameOrNumber(lognormal->getShapeK())}};
      } else {
         out.fields = {{"mu", lognormalLogName(lognormal->getMedian(), tool)},
                       {"sigma", lognormalLogName(lognormal->getShapeK(), tool)}};
      }
   } else {
      return std::nullopt;
   }
   // The observable of a component must be named: the observables are pairwise distinct and are paired with data.
   if (isLiteralConstVar(*x)) {
      return std::nullopt;
   }
   out.observable = x->GetName();
   return out;
}

void writeArrayDist(RooJSONFactoryWSTool &tool, JSONNode &elem, std::string const &name,
                    std::vector<RooAbsPdf const *> const &components)
{
   auto error = [&name](std::string const &message) {
      RooJSONFactoryWSTool::error("cannot write the array distribution '" + name + "': " + message);
   };
   if (components.empty()) {
      error("it has no components");
   }
   std::vector<ArrayComponentInfo> comps;
   std::set<std::string> observables;
   for (RooAbsPdf const *pdf : components) {
      auto comp = arrayComponentInfo(*pdf, tool.constraintObservable(pdf->GetName()), &tool);
      if (!comp) {
         error("'" + std::string(pdf->GetName()) +
               "' cannot be a component, only RooPoisson, RooGaussian and RooLognormal over a named observable "
               "are supported");
      }
      if (!comps.empty() && comp->type != comps.front().type) {
         error("the components '" + std::string(components.front()->GetName()) + "' (" + comps.front().type +
               ") and '" + pdf->GetName() + "' (" + comp->type + ") have different types");
      }
      if (!observables.insert(comp->observable).second) {
         error("two components have the observable '" + comp->observable + "'");
      }
      comps.push_back(std::move(*comp));
   }

   elem["type"] << "array_dist";
   elem["component_type"] << comps.front().type;

   auto write = [](JSONNode &node, HS3::NameOrNumber const &value) {
      std::visit([&node](auto const &v) { node << v; }, value);
   };

   // A field with the same value in every component is written once in "default_parameters".
   std::set<std::string> defaultKeys;
   if (comps.size() > 1) {
      for (std::size_t i = 0; i < comps.front().fields.size(); ++i) {
         auto const &field = comps.front().fields[i];
         if (std::all_of(comps.begin(), comps.end(),
                         [&](ArrayComponentInfo const &comp) { return comp.fields[i].second == field.second; })) {
            defaultKeys.insert(field.first);
            write(elem["default_parameters"].set_map()[field.first], field.second);
         }
      }
   }

   JSONNode &seq = elem["components"].set_seq();
   for (ArrayComponentInfo const &comp : comps) {
      JSONNode &node = seq.append_child().set_map();
      node["x"] << comp.observable;
      for (auto const &field : comp.fields) {
         if (defaultKeys.find(field.first) == defaultKeys.end()) {
            write(node[field.first], field.second);
         }
      }
   }
}

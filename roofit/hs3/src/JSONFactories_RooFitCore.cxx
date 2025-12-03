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
#include <RooPoisson.h>
#include <RooPolynomial.h>
#include <RooPolyVar.h>
#include <RooRealSumFunc.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>
#include <RooResolutionModel.h>
#include <RooTFnBinding.h>
#include <RooTruthModel.h>
#include <RooGaussModel.h>
#include <RooWorkspace.h>
#include <RooRealIntegral.h>

#include <TF1.h>
#include <TH1.h>

#include "JSONIOUtils.h"

#include "static_execute.h"

#include <algorithm>
#include <cctype>

using RooFit::Detail::JSONNode;

///////////////////////////////////////////////////////////////////////////////////////////////////////
// individually implemented importers
///////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
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
            arguments.insert(arg);
         }
      }
   }
   if (startidx < expr.size()) {
      arguments.insert(expr.substr(startidx));
   }
   return arguments;
}

template <class RooArg_t>
class RooFormulaArgFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("expression")) {
         RooJSONFactoryWSTool::error("no expression given for '" + name + "'");
      }
      TString formula(p["expression"].val());
      RooArgList dependents;
      for (const auto &d : extractArguments(formula.Data())) {
         dependents.add(*tool->request<RooAbsReal>(d, name));
      }
      tool->wsImport(RooArg_t{name.c_str(), formula, dependents});
      return true;
   }
};

class RooAddPdfFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
};

class RooAddModelFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      tool->wsEmplace<RooAddModel>(name, tool->requestArgList<RooAbsPdf>(p, "summands"),
                                   tool->requestArgList<RooAbsReal>(p, "coefficients"));
      return true;
   }
};

class RooBinWidthFunctionFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      RooHistFunc *hf = static_cast<RooHistFunc *>(tool->request<RooAbsReal>(p["histogram"].val(), name));
      tool->wsEmplace<RooBinWidthFunction>(name, *hf, p["divideByBinWidth"].val_bool());
      return true;
   }
};

class RooBinSamplingPdfFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
};

class RooRealSumPdfFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
};

class RooRealSumFuncFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      tool->wsEmplace<RooRealSumFunc>(name, tool->requestArgList<RooAbsReal>(p, "samples"),
                                      tool->requestArgList<RooAbsReal>(p, "coefficients"));
      return true;
   }
};
template <class RooArg_t>
class RooPolynomialFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
         if (order == 0 && coef.val() == "1.0") {
            ++lowestOrder;
         } else if (coefs.empty() && coef.val() == "0.0") {
            ++lowestOrder;
         } else {
            coefs.add(*tool->request<RooAbsReal>(coef.val(), name));
         }
         ++order;
      }

      tool->wsEmplace<RooArg_t>(name, *x, coefs, lowestOrder);
      return true;
   }
};

class RooPoissonFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      RooAbsReal *x = tool->requestArg<RooAbsReal>(p, "x");
      RooAbsReal *mean = tool->requestArg<RooAbsReal>(p, "mean");
      tool->wsEmplace<RooPoisson>(name, *x, *mean, !p["integer"].val_bool());
      return true;
   }
};

class RooDecayFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      RooRealVar *t = tool->requestArg<RooRealVar>(p, "t");
      RooAbsReal *tau = tool->requestArg<RooAbsReal>(p, "tau");
      RooResolutionModel *model = dynamic_cast<RooResolutionModel *>(tool->requestArg<RooAbsPdf>(p, "resolutionModel"));
      RooDecay::DecayType decayType = static_cast<RooDecay::DecayType>(p["decayType"].val_int());
      tool->wsEmplace<RooDecay>(name, *t, *tau, *model, decayType);
      return true;
   }
};

class RooTruthModelFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      RooRealVar *x = tool->requestArg<RooRealVar>(p, "x");
      tool->wsEmplace<RooTruthModel>(name, *x);
      return true;
   }
};

class RooGaussModelFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      RooRealVar *x = tool->requestArg<RooRealVar>(p, "x");
      RooRealVar *mean = tool->requestArg<RooRealVar>(p, "mean");
      RooRealVar *sigma = tool->requestArg<RooRealVar>(p, "sigma");
      tool->wsEmplace<RooGaussModel>(name, *x, *mean, *sigma);
      return true;
   }
};

class RooRealIntegralFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
};

class RooDerivativeFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
};

class RooFFTConvPdfFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
};

class RooExtendPdfFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
};

class RooLogNormalFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      RooAbsReal *x = tool->requestArg<RooAbsReal>(p, "x");

      // Same mechanism to undo the parameter transformation as in the
      // RooExponentialFactory (see comments in that class for more info).
      const std::string muName = p["mu"].val();
      const std::string sigmaName = p["sigma"].val();
      const bool isTransformed = endsWith(muName, "_lognormal_log");
      const std::string suffixToRemove = isTransformed ? "_lognormal_log" : "";
      RooAbsReal *mu = tool->request<RooAbsReal>(removeSuffix(muName, suffixToRemove), name);
      RooAbsReal *sigma = tool->request<RooAbsReal>(removeSuffix(sigmaName, suffixToRemove), name);

      tool->wsEmplace<RooLognormal>(name, *x, *mu, *sigma, !isTransformed);

      return true;
   }
};

class RooExponentialFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
};

class RooLegacyExpPolyFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
         if (order == 0 && coef.val() == "1.0") {
            ++lowestOrder;
         } else if (coefs.empty() && coef.val() == "0.0") {
            ++lowestOrder;
         } else {
            coefs.add(*tool->request<RooAbsReal>(coef.val(), name));
         }
         ++order;
      }

      tool->wsEmplace<RooLegacyExpPoly>(name, *x, coefs, lowestOrder);
      return true;
   }
};

class RooMultiVarGaussianFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
};

class ParamHistFuncFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
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

private:
   RooArgList readBinning(const JSONNode &topNode, const RooArgList &varList) const
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
      for (int i = 0; i < varList.getSize(); ++i) {
         const auto *refVar = dynamic_cast<RooRealVar *>(varList.at(i));
         if (!refVar)
            continue;

         auto it = varMap.find(refVar->GetName());
         if (it != varMap.end()) {
            vars.addOwned(std::move(it->second)); // preserve ownership
         }
      }
      return vars;
   }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// specialized exporter implementations
///////////////////////////////////////////////////////////////////////////////////////////////////////
template <class RooArg_t>
class RooAddPdfStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooArg_t *pdf = static_cast<const RooArg_t *>(func);
      elem["type"] << key();
      RooJSONFactoryWSTool::fillSeq(elem["summands"], pdf->pdfList());
      RooJSONFactoryWSTool::fillSeq(elem["coefficients"], pdf->coefList());
      elem["extended"] << (pdf->extendMode() != RooArg_t::CanNotBeExtended);
      return true;
   }
};

class RooRealSumPdfStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooRealSumPdf *pdf = static_cast<const RooRealSumPdf *>(func);
      elem["type"] << key();
      RooJSONFactoryWSTool::fillSeq(elem["samples"], pdf->funcList());
      RooJSONFactoryWSTool::fillSeq(elem["coefficients"], pdf->coefList());
      elem["extended"] << (pdf->extendMode() != RooAbsPdf::CanNotBeExtended);
      return true;
   }
};

class RooRealSumFuncStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooRealSumFunc *pdf = static_cast<const RooRealSumFunc *>(func);
      elem["type"] << key();
      RooJSONFactoryWSTool::fillSeq(elem["samples"], pdf->funcList());
      RooJSONFactoryWSTool::fillSeq(elem["coefficients"], pdf->coefList());
      return true;
   }
};

class RooHistFuncStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *tool, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooHistFunc *hf = static_cast<const RooHistFunc *>(func);
      elem["type"] << key();
      RooDataHist const &dh = hf->dataHist();
      tool->exportHisto(*dh.get(), dh.numEntries(), dh.weightArray(), elem["data"].set_map());
      return true;
   }
};

class RooHistFuncFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("data")) {
         RooJSONFactoryWSTool::error("function '" + name + "' is of histogram type, but does not define a 'data' key");
      }
      std::unique_ptr<RooDataHist> dataHist =
         RooJSONFactoryWSTool::readBinnedData(p["data"], name, RooJSONFactoryWSTool::readAxes(p["data"]));
      tool->wsEmplace<RooHistFunc>(name, *dataHist->get(), *dataHist);
      return true;
   }
};

class RooHistPdfStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *tool, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooHistPdf *hf = static_cast<const RooHistPdf *>(func);
      elem["type"] << key();
      RooDataHist const &dh = hf->dataHist();
      tool->exportHisto(*dh.get(), dh.numEntries(), dh.weightArray(), elem["data"].set_map());
      return true;
   }
};

class RooHistPdfFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("data")) {
         RooJSONFactoryWSTool::error("function '" + name + "' is of histogram type, but does not define a 'data' key");
      }
      std::unique_ptr<RooDataHist> dataHist =
         RooJSONFactoryWSTool::readBinnedData(p["data"], name, RooJSONFactoryWSTool::readAxes(p["data"]));
      tool->wsEmplace<RooHistPdf>(name, *dataHist->get(), *dataHist);
      return true;
   }
};

class RooBinSamplingPdfStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooBinSamplingPdf *pdf = static_cast<const RooBinSamplingPdf *>(func);
      elem["type"] << key();
      elem["pdf"] << pdf->pdf().GetName();
      elem["observable"] << pdf->observable().GetName();
      elem["epsilon"] << pdf->epsilon();
      return true;
   }
};

class RooBinWidthFunctionStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooBinWidthFunction *pdf = static_cast<const RooBinWidthFunction *>(func);
      elem["type"] << key();
      elem["histogram"] << pdf->histFunc().GetName();
      elem["divideByBinWidth"] << pdf->divideByBinWidth();
      return true;
   }
};

template <class RooArg_t>
class RooFormulaArgStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooArg_t *pdf = static_cast<const RooArg_t *>(func);
      elem["type"] << key();
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
      return true;
   }

private:
   void cleanExpression(TString &expr) const
   {
      expr.ReplaceAll("TMath::Exp", "exp");
      expr.ReplaceAll("TMath::Min", "min");
      expr.ReplaceAll("TMath::Max", "max");
      expr.ReplaceAll("TMath::Log", "log");
      expr.ReplaceAll("TMath::Cos", "cos");
      expr.ReplaceAll("TMath::Sin", "sin");
      expr.ReplaceAll("TMath::Sqrt", "sqrt");
      expr.ReplaceAll("TMath::Power", "pow");
      expr.ReplaceAll("TMath::Erf", "erf");
   }
};
template <class RooArg_t>
class RooPolynomialStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const RooArg_t *>(func);
      elem["type"] << key();
      elem["x"] << pdf->x().GetName();
      auto &coefs = elem["coefficients"].set_seq();
      // Write out the default coefficient that RooFit uses for the lower
      // orders before the order of the first coefficient. Like this, the
      // output is more self-documenting.
      for (int i = 0; i < pdf->lowestOrder(); ++i) {
         coefs.append_child() << (i == 0 ? "1.0" : "0.0");
      }
      for (const auto &coef : pdf->coefList()) {
         coefs.append_child() << coef->GetName();
      }
      return true;
   }
};

class RooLegacyExpPolyStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const RooLegacyExpPoly *>(func);
      elem["type"] << key();
      elem["x"] << pdf->x().GetName();
      auto &coefs = elem["coefficients"].set_seq();
      // Write out the default coefficient that RooFit uses for the lower
      // orders before the order of the first coefficient. Like this, the
      // output is more self-documenting.
      for (int i = 0; i < pdf->lowestOrder(); ++i) {
         coefs.append_child() << (i == 0 ? "1.0" : "0.0");
      }
      for (const auto &coef : pdf->coefList()) {
         coefs.append_child() << coef->GetName();
      }
      return true;
   }
};

class RooPoissonStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const RooPoisson *>(func);
      elem["type"] << key();
      elem["x"] << pdf->getX().GetName();
      elem["mean"] << pdf->getMean().GetName();
      elem["integer"] << !pdf->getNoRounding();
      return true;
   }
};

class RooDecayStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const RooDecay *>(func);
      elem["type"] << key();
      elem["t"] << pdf->getT().GetName();
      elem["tau"] << pdf->getTau().GetName();
      elem["resolutionModel"] << pdf->getModel().GetName();
      elem["decayType"] << pdf->getDecayType();

      return true;
   }
};

class RooTruthModelStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const RooTruthModel *>(func);
      elem["type"] << key();
      elem["x"] << pdf->convVar().GetName();

      return true;
   }
};

class RooGaussModelStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const RooGaussModel *>(func);
      elem["type"] << key();
      elem["x"] << pdf->convVar().GetName();
      elem["mean"] << pdf->getMean().GetName();
      elem["sigma"] << pdf->getSigma().GetName();
      return true;
   }
};

class RooLogNormalStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *tool, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const RooLognormal *>(func);

      elem["type"] << key();
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
};

class RooExponentialStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *tool, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const RooExponential *>(func);
      elem["type"] << key();
      elem["x"] << pdf->variable().GetName();
      auto &c = pdf->coefficient();
      if (pdf->negateCoefficient()) {
         elem["c"] << c.GetName();
      } else {
         elem["c"] << tool->exportTransformed(&c, "_exponential_inverted", "-%s");
      }

      return true;
   }
};

class RooMultiVarGaussianStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const RooMultiVarGaussian *>(func);
      elem["type"] << key();
      RooJSONFactoryWSTool::fillSeq(elem["x"], pdf->xVec());
      RooJSONFactoryWSTool::fillSeq(elem["mean"], pdf->muVec());
      elem["covariances"].fill_mat(pdf->covarianceMatrix());
      return true;
   }
};

class RooTFnBindingStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const RooTFnBinding *>(func);
      elem["type"] << key();

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
};

class RooDerivativeStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const RooDerivative *>(func);
      elem["type"] << key();
      elem["x"] << pdf->getX().GetName();
      elem["function"] << pdf->getFunc().GetName();
      if (!pdf->getNset().empty()) {
         RooJSONFactoryWSTool::fillSeq(elem["normalization"], pdf->getNset());
      }
      elem["order"] << pdf->order();
      elem["eps"] << pdf->eps();
      return true;
   }
};

class RooRealIntegralStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *integral = static_cast<const RooRealIntegral *>(func);
      elem["type"] << key();
      std::string integrand = integral->integrand().GetName();
      // elem["integrand"] << RooJSONFactoryWSTool::sanitizeName(integrand);
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
};

class RooFFTConvPdfStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const RooFFTConvPdf *>(func);
      elem["type"] << key();
      if (auto convFunc = pdf->getPdfConvVar()) {
         elem["conv_func"] << convFunc->GetName();
      }
      elem["conv_var"] << pdf->getConvVar().GetName();
      elem["pdf1"] << pdf->getPdf1().GetName();
      elem["pdf2"] << pdf->getPdf2().GetName();
      elem["ipOrder"] << pdf->getInterpolationOrder();
      return true;
   }
};

class RooExtendPdfStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const RooExtendPdf *>(func);
      elem["type"] << key();
      if (auto rangeName = pdf->getRangeName()) {
         elem["range"] << rangeName->GetName();
      }
      elem["pdf"] << pdf->pdf().GetName();
      elem["norm"] << pdf->getN().GetName();
      return true;
   }
};

class ParamHistFuncStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const ParamHistFunc *>(func);
      elem["type"] << key();
      RooJSONFactoryWSTool::fillSeq(elem["variables"], pdf->dataVars());
      RooJSONFactoryWSTool::fillSeq(elem["parameters"], pdf->paramList());
      writeBinningInfo(pdf, elem);
      return true;
   }

private:
   void writeBinningInfo(const ParamHistFunc *pdf, JSONNode &elem) const
   {
      auto &observablesNode = elem["axes"].set_seq();
      // axes have to be ordered to get consistent bin indices
      for (auto *var : static_range_cast<RooRealVar *>(pdf->dataVars())) {
         std::string name = var->GetName();
         RooJSONFactoryWSTool::testValidName(name, false);
         JSONNode &obsNode = observablesNode.append_child().set_map();
         obsNode["name"] << name;
         if (var->getBinning().isUniform()) {
            obsNode["min"] << var->getMin();
            obsNode["max"] << var->getMax();
            obsNode["nbins"] << var->getBins();
         } else {
            auto &edges = obsNode["edges"];
            edges.set_seq();
            double val = var->getBinning().binLow(0);
            edges.append_child() << val;
            for (int i = 0; i < var->getBinning().numBins(); ++i) {
               val = var->getBinning().binHigh(i);
               edges.append_child() << val;
            }
         }
      }
   }
};

#define DEFINE_EXPORTER_KEY(class_name, name)    \
   std::string const &class_name::key() const    \
   {                                             \
      const static std::string keystring = name; \
      return keystring;                          \
   }
template <>
DEFINE_EXPORTER_KEY(RooAddPdfStreamer<RooAddPdf>, "mixture_dist");
template <>
DEFINE_EXPORTER_KEY(RooAddPdfStreamer<RooAddModel>, "mixture_model");
DEFINE_EXPORTER_KEY(RooBinSamplingPdfStreamer, "binsampling");
DEFINE_EXPORTER_KEY(RooBinWidthFunctionStreamer, "binwidth");
DEFINE_EXPORTER_KEY(RooLegacyExpPolyStreamer, "legacy_exp_poly_dist");
DEFINE_EXPORTER_KEY(RooExponentialStreamer, "exponential_dist");
template <>
DEFINE_EXPORTER_KEY(RooFormulaArgStreamer<RooFormulaVar>, "generic_function");
template <>
DEFINE_EXPORTER_KEY(RooFormulaArgStreamer<RooGenericPdf>, "generic_dist");
DEFINE_EXPORTER_KEY(RooHistFuncStreamer, "histogram");
DEFINE_EXPORTER_KEY(RooHistPdfStreamer, "histogram_dist");
DEFINE_EXPORTER_KEY(RooLogNormalStreamer, "lognormal_dist");
DEFINE_EXPORTER_KEY(RooMultiVarGaussianStreamer, "multivariate_normal_dist");
DEFINE_EXPORTER_KEY(RooPoissonStreamer, "poisson_dist");
DEFINE_EXPORTER_KEY(RooDecayStreamer, "decay_dist");
DEFINE_EXPORTER_KEY(RooTruthModelStreamer, "truth_model_function");
DEFINE_EXPORTER_KEY(RooGaussModelStreamer, "gauss_model_function");
template <>
DEFINE_EXPORTER_KEY(RooPolynomialStreamer<RooPolynomial>, "polynomial_dist");
template <>
DEFINE_EXPORTER_KEY(RooPolynomialStreamer<RooPolyVar>, "polynomial");
DEFINE_EXPORTER_KEY(RooRealSumFuncStreamer, "weighted_sum");
DEFINE_EXPORTER_KEY(RooRealSumPdfStreamer, "weighted_sum_dist");
DEFINE_EXPORTER_KEY(RooTFnBindingStreamer, "generic_function");
DEFINE_EXPORTER_KEY(RooRealIntegralStreamer, "integral");
DEFINE_EXPORTER_KEY(RooDerivativeStreamer, "derivative");
DEFINE_EXPORTER_KEY(RooFFTConvPdfStreamer, "fft_conv_pdf");
DEFINE_EXPORTER_KEY(RooExtendPdfStreamer, "extend_pdf");
DEFINE_EXPORTER_KEY(ParamHistFuncStreamer, "step");

///////////////////////////////////////////////////////////////////////////////////////////////////////
// instantiate all importers and exporters
///////////////////////////////////////////////////////////////////////////////////////////////////////

STATIC_EXECUTE([]() {
   using namespace RooFit::JSONIO;

   registerImporter<RooAddPdfFactory>("mixture_dist", false);
   registerImporter<RooAddModelFactory>("mixture_model", false);
   registerImporter<RooBinSamplingPdfFactory>("binsampling_dist", false);
   registerImporter<RooBinWidthFunctionFactory>("binwidth", false);
   registerImporter<RooLegacyExpPolyFactory>("legacy_exp_poly_dist", false);
   registerImporter<RooExponentialFactory>("exponential_dist", false);
   registerImporter<RooFormulaArgFactory<RooFormulaVar>>("generic_function", false);
   registerImporter<RooFormulaArgFactory<RooGenericPdf>>("generic_dist", false);
   registerImporter<RooHistFuncFactory>("histogram", false);
   registerImporter<RooHistPdfFactory>("histogram_dist", false);
   registerImporter<RooLogNormalFactory>("lognormal_dist", false);
   registerImporter<RooMultiVarGaussianFactory>("multivariate_normal_dist", false);
   registerImporter<RooPoissonFactory>("poisson_dist", false);
   registerImporter<RooDecayFactory>("decay_dist", false);
   registerImporter<RooTruthModelFactory>("truth_model_function", false);
   registerImporter<RooGaussModelFactory>("gauss_model_function", false);
   registerImporter<RooPolynomialFactory<RooPolynomial>>("polynomial_dist", false);
   registerImporter<RooPolynomialFactory<RooPolyVar>>("polynomial", false);
   registerImporter<RooRealSumPdfFactory>("weighted_sum_dist", false);
   registerImporter<RooRealSumFuncFactory>("weighted_sum", false);
   registerImporter<RooRealIntegralFactory>("integral", false);
   registerImporter<RooDerivativeFactory>("derivative", false);
   registerImporter<RooFFTConvPdfFactory>("fft_conv_pdf", false);
   registerImporter<RooExtendPdfFactory>("extend_pdf", false);
   registerImporter<ParamHistFuncFactory>("step", false);

   registerExporter<RooAddPdfStreamer<RooAddPdf>>(RooAddPdf::Class(), false);
   registerExporter<RooAddPdfStreamer<RooAddModel>>(RooAddModel::Class(), false);
   registerExporter<RooBinSamplingPdfStreamer>(RooBinSamplingPdf::Class(), false);
   registerExporter<RooBinWidthFunctionStreamer>(RooBinWidthFunction::Class(), false);
   registerExporter<RooLegacyExpPolyStreamer>(RooLegacyExpPoly::Class(), false);
   registerExporter<RooExponentialStreamer>(RooExponential::Class(), false);
   registerExporter<RooFormulaArgStreamer<RooFormulaVar>>(RooFormulaVar::Class(), false);
   registerExporter<RooFormulaArgStreamer<RooGenericPdf>>(RooGenericPdf::Class(), false);
   registerExporter<RooHistFuncStreamer>(RooHistFunc::Class(), false);
   registerExporter<RooHistPdfStreamer>(RooHistPdf::Class(), false);
   registerExporter<RooLogNormalStreamer>(RooLognormal::Class(), false);
   registerExporter<RooMultiVarGaussianStreamer>(RooMultiVarGaussian::Class(), false);
   registerExporter<RooPoissonStreamer>(RooPoisson::Class(), false);
   registerExporter<RooDecayStreamer>(RooDecay::Class(), false);
   registerExporter<RooTruthModelStreamer>(RooTruthModel::Class(), false);
   registerExporter<RooGaussModelStreamer>(RooGaussModel::Class(), false);
   registerExporter<RooPolynomialStreamer<RooPolynomial>>(RooPolynomial::Class(), false);
   registerExporter<RooPolynomialStreamer<RooPolyVar>>(RooPolyVar::Class(), false);
   registerExporter<RooRealSumFuncStreamer>(RooRealSumFunc::Class(), false);
   registerExporter<RooRealSumPdfStreamer>(RooRealSumPdf::Class(), false);
   registerExporter<RooTFnBindingStreamer>(RooTFnBinding::Class(), false);
   registerExporter<RooRealIntegralStreamer>(RooRealIntegral::Class(), false);
   registerExporter<RooDerivativeStreamer>(RooDerivative::Class(), false);
   registerExporter<RooFFTConvPdfStreamer>(RooFFTConvPdf::Class(), false);
   registerExporter<RooExtendPdfStreamer>(RooExtendPdf::Class(), false);
   registerExporter<ParamHistFuncStreamer>(ParamHistFunc::Class(), false);
});

} // namespace

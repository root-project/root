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
#include <RooBinSamplingPdf.h>
#include <RooBinWidthFunction.h>
#include <RooCategory.h>
#include <RooDataHist.h>
#include <RooExponential.h>
#include <RooFit/Detail/JSONInterface.h>
#include <RooFitHS3/JSONIO.h>
#include <RooFormulaVar.h>
#include <RooGenericPdf.h>
#include <RooHistFunc.h>
#include <RooHistPdf.h>
#include <RooLegacyExpPoly.h>
#include <RooLognormal.h>
#include <RooMultiVarGaussian.h>
#include <RooPoisson.h>
#include <RooPolynomial.h>
#include <RooRealSumFunc.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>
#include <RooTFnBinding.h>
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
 * @return A vector of strings representing the extracted arguments.
 */
std::vector<std::string> extractArguments(std::string expr)
{
   // Get rid of whitespaces
   expr.erase(std::remove_if(expr.begin(), expr.end(), [](unsigned char c) { return std::isspace(c); }), expr.end());

   std::vector<std::string> arguments;
   size_t startidx = expr.size();
   for (size_t i = 0; i < expr.size(); ++i) {
      if (startidx >= expr.size()) {
         if (isalpha(expr[i])) {
            startidx = i;
         }
      } else {
         if (!isdigit(expr[i]) && !isalpha(expr[i]) && expr[i] != '_') {
            if (expr[i] == '(') {
               startidx = expr.size();
               continue;
            }
            std::string arg(expr.substr(startidx, i - startidx));
            startidx = expr.size();
            arguments.push_back(arg);
         }
      }
   }
   if (startidx < expr.size()) {
      arguments.push_back(expr.substr(startidx));
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
      tool->wsEmplace<RooAddPdf>(name, tool->requestArgList<RooAbsPdf>(p, "summands"),
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

      tool->wsEmplace<RooPolynomial>(name, *x, coefs, lowestOrder);
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

///////////////////////////////////////////////////////////////////////////////////////////////////////
// specialized exporter implementations
///////////////////////////////////////////////////////////////////////////////////////////////////////

class RooAddPdfStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooAddPdf *pdf = static_cast<const RooAddPdf *>(func);
      elem["type"] << key();
      RooJSONFactoryWSTool::fillSeq(elem["summands"], pdf->pdfList());
      RooJSONFactoryWSTool::fillSeq(elem["coefficients"], pdf->coefList());
      elem["extended"] << (pdf->extendMode() != RooAbsPdf::CanNotBeExtended);
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
};

class RooPolynomialStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const RooPolynomial *>(func);
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

class RooRealIntegralStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *integral = static_cast<const RooRealIntegral *>(func);
      elem["type"] << key();
      elem["integrand"] << integral->integrand().GetName();
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

#define DEFINE_EXPORTER_KEY(class_name, name)    \
   std::string const &class_name::key() const    \
   {                                             \
      const static std::string keystring = name; \
      return keystring;                          \
   }

DEFINE_EXPORTER_KEY(RooAddPdfStreamer, "mixture_dist");
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
DEFINE_EXPORTER_KEY(RooPolynomialStreamer, "polynomial_dist");
DEFINE_EXPORTER_KEY(RooRealSumFuncStreamer, "weighted_sum");
DEFINE_EXPORTER_KEY(RooRealSumPdfStreamer, "weighted_sum_dist");
DEFINE_EXPORTER_KEY(RooTFnBindingStreamer, "generic_function");
DEFINE_EXPORTER_KEY(RooRealIntegralStreamer, "integral");

///////////////////////////////////////////////////////////////////////////////////////////////////////
// instantiate all importers and exporters
///////////////////////////////////////////////////////////////////////////////////////////////////////

STATIC_EXECUTE([]() {
   using namespace RooFit::JSONIO;

   registerImporter<RooAddPdfFactory>("mixture_dist", false);
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
   registerImporter<RooPolynomialFactory>("polynomial_dist", false);
   registerImporter<RooRealSumPdfFactory>("weighted_sum_dist", false);
   registerImporter<RooRealSumFuncFactory>("weighted_sum", false);
   registerImporter<RooRealIntegralFactory>("integral", false);

   registerExporter<RooAddPdfStreamer>(RooAddPdf::Class(), false);
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
   registerExporter<RooPolynomialStreamer>(RooPolynomial::Class(), false);
   registerExporter<RooRealSumFuncStreamer>(RooRealSumFunc::Class(), false);
   registerExporter<RooRealSumPdfStreamer>(RooRealSumPdf::Class(), false);
   registerExporter<RooTFnBindingStreamer>(RooTFnBinding::Class(), false);
   registerExporter<RooRealIntegralStreamer>(RooRealIntegral::Class(), false);
});

} // namespace

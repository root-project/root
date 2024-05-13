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
#include <RooExpPoly.h>
#include <RooExponential.h>
#include <RooFit/Detail/JSONInterface.h>
#include <RooFitHS3/JSONIO.h>
#include <RooFormulaVar.h>
#include <RooGenericPdf.h>
#include <RooHistFunc.h>
#include <RooHistPdf.h>
#include <RooLognormal.h>
#include <RooMultiVarGaussian.h>
#include <RooPoisson.h>
#include <RooPolynomial.h>
#include <RooRealSumFunc.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>
#include <RooTFnBinding.h>
#include <RooWorkspace.h>

#include <TF1.h>
#include <TH1.h>

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

class RooLogNormalFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      RooAbsReal *x = tool->requestArg<RooAbsReal>(p, "x");
      RooAbsReal *mu = tool->requestArg<RooAbsReal>(p, "mu");
      RooAbsReal *sigma = tool->requestArg<RooAbsReal>(p, "sigma");

      // TODO: check if the pdf was originally exported by ROOT, in which case
      // it can be imported back without using the standard parametrization.
      tool->wsEmplace<RooLognormal>(name, *x, *mu, *sigma, true);
      return true;
   }
};

class RooExponentialFactory : public RooFit::JSONIO::Importer {
public:
   bool importArg(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      RooAbsReal *x = tool->requestArg<RooAbsReal>(p, "x");
      RooAbsReal *c = tool->requestArg<RooAbsReal>(p, "c");

      // TODO: check if the pdf was originally exported by ROOT, in which case
      // it can be imported back without using the standard parametrization.
      tool->wsEmplace<RooExponential>(name, *x, *c, true);
      return true;
   }
};

class RooExpPolyFactory : public RooFit::JSONIO::Importer {
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

      tool->wsEmplace<RooExpPoly>(name, *x, coefs, lowestOrder);
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
      for (size_t i = 0; i < pdf->nParameters(); ++i) {
         RooAbsArg *par = pdf->getParameter(i);
         std::stringstream ss_1;
         ss_1 << "x[" << i << "]";
         std::stringstream ss_2;
         ss_2 << "@" << i << "";
         expression.ReplaceAll(ss_1.str().c_str(), par->GetName());
         expression.ReplaceAll(ss_2.str().c_str(), par->GetName());
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

class RooExpPolyStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override;
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const RooExpPoly *>(func);
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

      if(pdf->useStandardParametrization()) {
         elem["mu"] << m0.GetName();
         elem["sigma"] << k.GetName();
      } else {
         elem["mu"] << tool->exportTransformed(&m0, "lognormal", "log", "log(%s)");
         elem["sigma"] << tool->exportTransformed(&k, "lognormal", "log", "log(%s)");
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
         elem["c"] << tool->exportTransformed(&c, "exponential", "inverted", "-%s");
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

#define DEFINE_EXPORTER_KEY(class_name, name)    \
   std::string const &class_name::key() const    \
   {                                             \
      const static std::string keystring = name; \
      return keystring;                          \
   }

DEFINE_EXPORTER_KEY(RooAddPdfStreamer, "mixture_dist");
DEFINE_EXPORTER_KEY(RooBinSamplingPdfStreamer, "binsampling");
DEFINE_EXPORTER_KEY(RooBinWidthFunctionStreamer, "binwidth");
DEFINE_EXPORTER_KEY(RooExpPolyStreamer, "exp_poly_dist");
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

///////////////////////////////////////////////////////////////////////////////////////////////////////
// instantiate all importers and exporters
///////////////////////////////////////////////////////////////////////////////////////////////////////

STATIC_EXECUTE([]() {
   using namespace RooFit::JSONIO;

   registerImporter<RooAddPdfFactory>("mixture_dist", false);
   registerImporter<RooBinSamplingPdfFactory>("binsampling_dist", false);
   registerImporter<RooBinWidthFunctionFactory>("binwidth", false);
   registerImporter<RooExpPolyFactory>("exp_poly_dist", false);
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

   registerExporter<RooAddPdfStreamer>(RooAddPdf::Class(), false);
   registerExporter<RooBinSamplingPdfStreamer>(RooBinSamplingPdf::Class(), false);
   registerExporter<RooBinWidthFunctionStreamer>(RooBinWidthFunction::Class(), false);
   registerExporter<RooExpPolyStreamer>(RooExpPoly::Class(), false);
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
});

} // namespace

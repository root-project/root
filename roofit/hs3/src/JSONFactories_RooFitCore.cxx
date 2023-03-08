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
#include <RooFit/Detail/JSONInterface.h>
#include <RooFitHS3/JSONIO.h>
#include <RooFormulaVar.h>
#include <RooGenericPdf.h>
#include <RooMultiVarGaussian.h>
#include <RooTFnBinding.h>
#include <RooHistFunc.h>
#include <RooHistPdf.h>
#include <RooProdPdf.h>
#include <RooPolynomial.h>
#include <RooRealSumFunc.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <TF1.h>
#include <TH1.h>

#include "static_execute.h"

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
 * @param expression A string representing a mathematical expression.
 * @return A vector of strings representing the extracted arguments.
 */
std::vector<std::string> extract_arguments(const std::string &expression)
{
   std::vector<std::string> arguments;
   size_t startidx = expression.size();
   for (size_t i = 0; i < expression.size(); ++i) {
      if (startidx >= expression.size()) {
         if (isalpha(expression[i])) {
            startidx = i;
         }
      } else {
         if (!isdigit(expression[i]) && !isalpha(expression[i]) && expression[i] != '_') {
            if (expression[i] == ' ')
               continue;
            if (expression[i] == '(') {
               startidx = expression.size();
               continue;
            }
            std::string arg(expression.substr(startidx, i - startidx));
            startidx = expression.size();
            arguments.push_back(arg);
         }
      }
   }
   if (startidx < expression.size())
      arguments.push_back(expression.substr(startidx));
   return arguments;
}

class RooGenericPdfFactory : public RooFit::JSONIO::Importer {
public:
   bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("expression")) {
         RooJSONFactoryWSTool::error("no expression given for '" + name + "'");
      }
      TString formula(p["expression"].val());
      RooArgList dependents;
      for (const auto &d : extract_arguments(formula.Data())) {
         TObject *obj = tool->workspace()->obj(d);
         if (obj->InheritsFrom(RooAbsArg::Class())) {
            dependents.add(*static_cast<RooAbsArg *>(obj));
         }
      }
      tool->wsEmplace<RooGenericPdf>(name, formula, dependents);
      return true;
   }
};

class RooFormulaVarFactory : public RooFit::JSONIO::Importer {
public:
   bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("expression")) {
         RooJSONFactoryWSTool::error("no expression given for '" + name + "'");
      }
      TString formula(p["expression"].val());
      RooArgList dependents;
      for (const auto &d : extract_arguments(formula.Data())) {
         TObject *obj = tool->workspace()->obj(d);
         if (obj->InheritsFrom(RooAbsArg::Class())) {
            dependents.add(*static_cast<RooAbsArg *>(obj));
         }
      }
      tool->wsEmplace<RooFormulaVar>(name, formula, dependents);
      return true;
   }
};

class RooProdPdfFactory : public RooFit::JSONIO::Importer {
public:
   bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      tool->wsEmplace<RooProdPdf>(name, name, tool->requestArgSet<RooAbsPdf>(p, "pdfs"));
      return true;
   }
};

class RooAddPdfFactory : public RooFit::JSONIO::Importer {
public:
   bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      tool->wsEmplace<RooAddPdf>(name, name, tool->requestArgList<RooAbsPdf>(p, "summands"),
                                 tool->requestArgList<RooAbsReal>(p, "coefficients"));
      return true;
   }
};

class RooBinWidthFunctionFactory : public RooFit::JSONIO::Importer {
public:
   bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      RooHistFunc *hf = static_cast<RooHistFunc *>(tool->request<RooAbsReal>(p["histogram"].val(), name));
      tool->wsEmplace<RooBinWidthFunction>(name, name, *hf, p["divideByBinWidth"].val_bool());
      return true;
   }
};

class RooBinSamplingPdfFactory : public RooFit::JSONIO::Importer {
public:
   bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));

      if (!p.has_child("pdf")) {
         RooJSONFactoryWSTool::error("no pdf given in '" + name + "'");
      }
      RooAbsPdf *pdf = tool->request<RooAbsPdf>(p["pdf"].val(), name);

      if (!p.has_child("observable")) {
         RooJSONFactoryWSTool::error("no observable given in '" + name + "'");
      }
      RooRealVar *obs = tool->request<RooRealVar>(p["observable"].val(), name);

      if (!pdf->dependsOn(*obs)) {
         pdf->Print("t");
         RooJSONFactoryWSTool::error(std::string("pdf '") + pdf->GetName() + "' does not depend on observable '" +
                                     obs->GetName() + "' as indicated by parent RooBinSamplingPdf '" + name +
                                     "', please check!");
      }

      if (!p.has_child("epsilon")) {
         RooJSONFactoryWSTool::error("no epsilon given in '" + name + "'");
      }
      double epsilon(p["epsilon"].val_double());

      tool->wsEmplace<RooBinSamplingPdf>(name, name, *obs, *pdf, epsilon);

      return true;
   }
};

class RooRealSumPdfFactory : public RooFit::JSONIO::Importer {
public:
   bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));

      bool extended = false;
      if (p.has_child("extended") && p["extended"].val_bool()) {
         extended = true;
      }
      tool->wsEmplace<RooRealSumPdf>(name, name, tool->requestArgList<RooAbsReal>(p, "samples"),
                                     tool->requestArgList<RooAbsReal>(p, "coefficients"), extended);
      return true;
   }
};

class RooRealSumFuncFactory : public RooFit::JSONIO::Importer {
public:
   bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      tool->wsEmplace<RooRealSumFunc>(name, name, tool->requestArgList<RooAbsReal>(p, "samples"),
                                      tool->requestArgList<RooAbsReal>(p, "coefficients"));
      return true;
   }
};

class RooPolynomialFactory : public RooFit::JSONIO::Importer {
public:
   bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("x")) {
         RooJSONFactoryWSTool::error("no x given in '" + name + "'");
      }
      if (!p.has_child("coefficients")) {
         RooJSONFactoryWSTool::error("no coefficients given in '" + name + "'");
      }
      RooAbsReal *x = tool->request<RooAbsReal>(p["x"].val(), name);
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

      tool->wsEmplace<RooPolynomial>(name, name, *x, coefs, lowestOrder);
      return true;
   }
};

class RooMultiVarGaussianFactory : public RooFit::JSONIO::Importer {
public:
   bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
      tool->wsEmplace<RooMultiVarGaussian>(name, name, tool->requestArgList<RooAbsReal>(p, "x"),
                                           tool->requestArgList<RooAbsReal>(p, "mean"), covmat);
      return true;
   }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// specialized exporter implementations
///////////////////////////////////////////////////////////////////////////////////////////////////////

class RooRealSumPdfStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override
   {
      const static std::string keystring = "sumpdf";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooRealSumPdf *pdf = static_cast<const RooRealSumPdf *>(func);
      elem["type"] << key();
      elem["samples"].fill_seq(pdf->funcList(), [](auto const &item) { return item->GetName(); });
      elem["coefficients"].fill_seq(pdf->coefList(), [](auto const &item) { return item->GetName(); });
      elem["extended"] << (pdf->extendMode() == RooAbsPdf::CanBeExtended);
      return true;
   }
};

class RooRealSumFuncStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override
   {
      const static std::string keystring = "sumfunc";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooRealSumFunc *pdf = static_cast<const RooRealSumFunc *>(func);
      elem["type"] << key();
      elem["samples"].fill_seq(pdf->funcList(), [](auto const &item) { return item->GetName(); });
      elem["coefficients"].fill_seq(pdf->coefList(), [](auto const &item) { return item->GetName(); });
      return true;
   }
};

class RooHistFuncStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override
   {
      static const std::string keystring = "histogram";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooHistFunc *hf = static_cast<const RooHistFunc *>(func);
      elem["type"] << key();
      RooArgList vars(*std::unique_ptr<RooArgSet>{hf->getVariables()});
      std::unique_ptr<TH1> hist{hf->createHistogram(RooJSONFactoryWSTool::concat(&vars))};
      if (!hist)
         return false;
      auto &data = elem["data"];
      RooJSONFactoryWSTool::exportHistogram(*hist, data, RooJSONFactoryWSTool::names(&vars));
      return true;
   }
};

class RooHistFuncFactory : public RooFit::JSONIO::Importer {
public:
   bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("data")) {
         RooJSONFactoryWSTool::error("function '" + name + "' is of histogram type, but does not define a 'data' key");
      }
      std::unique_ptr<RooDataHist> dataHist = RooJSONFactoryWSTool::readBinnedData(p["data"], name);
      tool->wsEmplace<RooHistFunc>(name, name, *dataHist->get(), *dataHist);
      return true;
   }
};

class RooHistPdfStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override
   {
      static const std::string keystring = "histogramPdf";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooHistPdf *hf = static_cast<const RooHistPdf *>(func);
      elem["type"] << key();
      RooArgList vars(*std::unique_ptr<RooArgSet>{hf->getVariables()});
      std::unique_ptr<TH1> hist{hf->createHistogram(RooJSONFactoryWSTool::concat(&vars))};
      if (!hist)
         return false;
      auto &data = elem["data"];
      RooJSONFactoryWSTool::exportHistogram(*hist, data, RooJSONFactoryWSTool::names(&vars));
      return true;
   }
};

class RooHistPdfFactory : public RooFit::JSONIO::Importer {
public:
   bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("data")) {
         RooJSONFactoryWSTool::error("function '" + name + "' is of histogram type, but does not define a 'data' key");
      }
      std::unique_ptr<RooDataHist> dataHist = RooJSONFactoryWSTool::readBinnedData(p["data"], name);
      tool->wsEmplace<RooHistPdf>(name, name, *dataHist->get(), *dataHist);
      return true;
   }
};

class RooBinSamplingPdfStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override
   {
      static const std::string keystring = "binsampling";
      return keystring;
   }
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

class RooProdPdfStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override
   {
      static const std::string keystring = "product_dist";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooProdPdf *pdf = static_cast<const RooProdPdf *>(func);
      elem["type"] << key();
      elem["pdfs"].fill_seq(pdf->pdfList(), [](auto const &f) { return f->GetName(); });
      return true;
   }
};

class RooGenericPdfStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override
   {
      static const std::string keystring = "generic_dist";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooGenericPdf *pdf = static_cast<const RooGenericPdf *>(func);
      elem["type"] << key();
      elem["expression"] << pdf->expression();
      return true;
   }
};

class RooBinWidthFunctionStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override
   {
      static const std::string keystring = "binwidth";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooBinWidthFunction *pdf = static_cast<const RooBinWidthFunction *>(func);
      elem["type"] << key();
      elem["histogram"] << pdf->histFunc().GetName();
      elem["divideByBinWidth"] << pdf->divideByBinWidth();
      return true;
   }
};

class RooFormulaVarStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override
   {
      static const std::string keystring = "generic_function";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooFormulaVar *var = static_cast<const RooFormulaVar *>(func);
      elem["type"] << key();
      elem["expression"] << var->expression();
      return true;
   }
};

class RooPolynomialStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override
   {
      static const std::string keystring = "polynomial_dist";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const RooPolynomial *>(func);
      elem["type"] << key();
      elem["x"] << pdf->x().GetName();
      auto &coefs = elem["coefficients"];
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

class RooMultiVarGaussianStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override
   {
      static const std::string keystring = "multinormal_dist";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      auto *pdf = static_cast<const RooMultiVarGaussian *>(func);
      elem["type"] << key();
      elem["x"].fill_seq(pdf->xVec(), [](auto const &item) { return item->GetName(); });
      elem["mean"].fill_seq(pdf->muVec(), [](auto const &item) { return item->GetName(); });
      elem["covariances"].fill_mat(pdf->covarianceMatrix());
      return true;
   }
};

class RooTFnBindingStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override
   {
      static const std::string keystring = "generic_function";
      return keystring;
   }
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

///////////////////////////////////////////////////////////////////////////////////////////////////////
// instantiate all importers and exporters
///////////////////////////////////////////////////////////////////////////////////////////////////////

STATIC_EXECUTE([]() {
   using namespace RooFit::JSONIO;

   registerImporter<RooProdPdfFactory>("product_dist", false);
   registerImporter<RooGenericPdfFactory>("generic_dist", false);
   registerImporter<RooFormulaVarFactory>("generic_function", false);
   registerImporter<RooBinSamplingPdfFactory>("binsampling", false);
   registerImporter<RooAddPdfFactory>("sum_dist", false);
   registerImporter<RooHistFuncFactory>("histogram", false);
   registerImporter<RooHistPdfFactory>("histogramPdf", false);
   registerImporter<RooBinWidthFunctionFactory>("binwidth", false);
   registerImporter<RooRealSumPdfFactory>("sumpdf", false);
   registerImporter<RooRealSumFuncFactory>("sumfunc", false);
   registerImporter<RooPolynomialFactory>("polynomial_dist", false);
   registerImporter<RooMultiVarGaussianFactory>("multinormal_dist", false);

   registerExporter<RooBinWidthFunctionStreamer>(RooBinWidthFunction::Class(), false);
   registerExporter<RooProdPdfStreamer>(RooProdPdf::Class(), false);
   registerExporter<RooBinSamplingPdfStreamer>(RooBinSamplingPdf::Class(), false);
   registerExporter<RooHistFuncStreamer>(RooHistFunc::Class(), false);
   registerExporter<RooHistPdfStreamer>(RooHistPdf::Class(), false);
   registerExporter<RooGenericPdfStreamer>(RooGenericPdf::Class(), false);
   registerExporter<RooFormulaVarStreamer>(RooFormulaVar::Class(), false);
   registerExporter<RooRealSumPdfStreamer>(RooRealSumPdf::Class(), false);
   registerExporter<RooRealSumFuncStreamer>(RooRealSumFunc::Class(), false);
   registerExporter<RooPolynomialStreamer>(RooPolynomial::Class(), false);
   registerExporter<RooMultiVarGaussianStreamer>(RooMultiVarGaussian::Class(), false);
   registerExporter<RooTFnBindingStreamer>(RooTFnBinding::Class(), false);
});

} // namespace

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
#include <RooHistFunc.h>
#include <RooHistPdf.h>
#include <RooProdPdf.h>
#include <RooRealSumFunc.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooWorkspace.h>

#include <TH1.h>

#include "static_execute.h"

using RooFit::Detail::JSONNode;

///////////////////////////////////////////////////////////////////////////////////////////////////////
// individually implemented importers
///////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

class RooGenericPdfFactory : public RooFit::JSONIO::Importer {
public:
   bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("dependents")) {
         RooJSONFactoryWSTool::error("no dependents of '" + name + "'");
      }
      if (!p.has_child("formula")) {
         RooJSONFactoryWSTool::error("no formula given for '" + name + "'");
      }
      RooArgList dependents;
      for (const auto &d : p["dependents"].children()) {
         std::string objname(RooJSONFactoryWSTool::name(d));
         TObject *obj = tool->workspace()->obj(objname);
         if (obj->InheritsFrom(RooAbsArg::Class())) {
            dependents.add(*static_cast<RooAbsArg *>(obj));
         }
      }
      TString formula(p["formula"].val());
      RooGenericPdf thepdf(name.c_str(), formula.Data(), dependents);
      tool->workspace()->import(thepdf, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
      return true;
   }
};

class RooFormulaVarFactory : public RooFit::JSONIO::Importer {
public:
   bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("dependents")) {
         RooJSONFactoryWSTool::error("no dependents of '" + name + "'");
      }
      if (!p.has_child("formula")) {
         RooJSONFactoryWSTool::error("no formula given for '" + name + "'");
      }
      RooArgList dependents;
      for (const auto &d : p["dependents"].children()) {
         std::string objname(RooJSONFactoryWSTool::name(d));
         TObject *obj = tool->workspace()->obj(objname);
         if (obj->InheritsFrom(RooAbsArg::Class())) {
            dependents.add(*static_cast<RooAbsArg *>(obj));
         }
      }
      TString formula(p["formula"].val());
      RooFormulaVar thevar(name.c_str(), formula.Data(), dependents);
      tool->workspace()->import(thevar, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
      return true;
   }
};

class RooProdPdfFactory : public RooFit::JSONIO::Importer {
public:
   bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      RooArgSet factors;
      if (!p.has_child("pdfs")) {
         RooJSONFactoryWSTool::error("no pdfs of '" + name + "'");
      }
      if (!p["pdfs"].is_seq()) {
         RooJSONFactoryWSTool::error("pdfs '" + name + "' are not a list.");
      }
      for (const auto &comp : p["pdfs"].children()) {
         std::string pdfname(comp.val());
         RooAbsPdf *pdf = tool->request<RooAbsPdf>(pdfname, name);
         factors.add(*pdf);
      }
      RooProdPdf prod(name.c_str(), name.c_str(), factors);
      tool->workspace()->import(prod, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
      return true;
   }
};

class RooAddPdfFactory : public RooFit::JSONIO::Importer {
public:
   bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      RooArgList pdfs;
      RooArgList coefs;
      if (!p.has_child("summands")) {
         RooJSONFactoryWSTool::error("no summands of '" + name + "'");
      }
      if (!p["summands"].is_seq()) {
         RooJSONFactoryWSTool::error("summands '" + name + "' are not a list.");
      }
      if (!p.has_child("coefficients")) {
         RooJSONFactoryWSTool::error("no coefficients of '" + name + "'");
      }
      if (!p["coefficients"].is_seq()) {
         RooJSONFactoryWSTool::error("coefficients '" + name + "' are not a list.");
      }
      for (const auto &comp : p["summands"].children()) {
         std::string pdfname(comp.val());
         RooAbsPdf *pdf = tool->request<RooAbsPdf>(pdfname, name);
         pdfs.add(*pdf);
      }
      for (const auto &comp : p["coefficients"].children()) {
         std::string coefname(comp.val());
         RooAbsReal *coef = tool->request<RooAbsReal>(coefname, name);
         coefs.add(*coef);
      }
      RooAddPdf add(name.c_str(), name.c_str(), pdfs, coefs);
      tool->workspace()->import(add, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
      return true;
   }
};

class RooBinWidthFunctionFactory : public RooFit::JSONIO::Importer {
public:
   bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      bool divideByBinWidth = p["divideByBinWidth"].val_bool();
      RooHistFunc *hf = dynamic_cast<RooHistFunc *>(tool->request<RooAbsReal>(p["histogram"].val(), name));
      RooBinWidthFunction func(name.c_str(), name.c_str(), *hf, divideByBinWidth);
      tool->workspace()->import(func, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
      return true;
   }
};

class RooSimultaneousFactory : public RooFit::JSONIO::Importer {
public:
   bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("channels")) {
         RooJSONFactoryWSTool::error("no channel components of '" + name + "'");
      }
      std::map<std::string, RooAbsPdf *> components;
      std::string indexname(p["index"].val());
      RooCategory cat(indexname.c_str(), indexname.c_str());
      for (const auto &comp : p["channels"].children()) {
         std::string catname(RooJSONFactoryWSTool::name(comp));
         RooJSONFactoryWSTool::log(RooFit::INFO) << "importing category " << catname << std::endl;
         std::string pdfname(comp.has_val() ? comp.val() : RooJSONFactoryWSTool::name(comp));
         RooAbsPdf *pdf = tool->request<RooAbsPdf>(pdfname, name);
         components[catname] = pdf;
         cat.defineType(catname.c_str());
      }
      RooSimultaneous simpdf(name.c_str(), name.c_str(), components, cat);
      tool->workspace()->import(simpdf, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
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
      std::string pdfname(p["pdf"].val());
      RooAbsPdf *pdf = tool->request<RooAbsPdf>(pdfname, name);

      if (!p.has_child("observable")) {
         RooJSONFactoryWSTool::error("no observable given in '" + name + "'");
      }
      std::string obsname(p["observable"].val());
      RooRealVar *obs = tool->request<RooRealVar>(obsname, name);

      if (!pdf->dependsOn(*obs)) {
         pdf->Print("t");
         RooJSONFactoryWSTool::error("pdf '" + pdfname + "' does not depend on observable '" + obsname +
                                     "' as indicated by parent RooBinSamplingPdf '" + name + "', please check!");
      }

      if (!p.has_child("epsilon")) {
         RooJSONFactoryWSTool::error("no epsilon given in '" + name + "'");
      }
      double epsilon(p["epsilon"].val_double());

      RooBinSamplingPdf thepdf(name.c_str(), name.c_str(), *obs, *pdf, epsilon);
      tool->workspace()->import(thepdf, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));

      return true;
   }
};

class RooRealSumPdfFactory : public RooFit::JSONIO::Importer {
public:
   bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("samples")) {
         RooJSONFactoryWSTool::error("no samples given in '" + name + "'");
      }
      if (!p.has_child("coefficients")) {
         RooJSONFactoryWSTool::error("no coefficients given in '" + name + "'");
      }
      RooArgList samples;
      for (const auto &sample : p["samples"].children()) {
         RooAbsReal *s = tool->request<RooAbsReal>(sample.val(), name);
         samples.add(*s);
      }
      RooArgList coefficients;
      for (const auto &coef : p["coefficients"].children()) {
         RooAbsReal *c = tool->request<RooAbsReal>(coef.val(), name);
         coefficients.add(*c);
      }

      bool extended = false;
      if (p.has_child("extended") && p["extended"].val_bool()) {
         extended = true;
      }
      RooRealSumPdf thepdf(name.c_str(), name.c_str(), samples, coefficients, extended);
      tool->workspace()->import(thepdf, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
      return true;
   }
};

class RooRealSumFuncFactory : public RooFit::JSONIO::Importer {
public:
   bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("samples")) {
         RooJSONFactoryWSTool::error("no samples given in '" + name + "'");
      }
      if (!p.has_child("coefficients")) {
         RooJSONFactoryWSTool::error("no coefficients given in '" + name + "'");
      }
      RooArgList samples;
      for (const auto &sample : p["samples"].children()) {
         RooAbsReal *s = tool->request<RooAbsReal>(sample.val(), name);
         samples.add(*s);
      }
      RooArgList coefficients;
      for (const auto &coef : p["coefficients"].children()) {
         RooAbsReal *c = tool->request<RooAbsReal>(coef.val(), name);
         coefficients.add(*c);
      }

      RooRealSumFunc thefunc(name.c_str(), name.c_str(), samples, coefficients);
      tool->workspace()->import(thefunc, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
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

class RooSimultaneousStreamer : public RooFit::JSONIO::Exporter {
public:
   std::string const &key() const override
   {
      const static std::string keystring = "simultaneous";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooSimultaneous *sim = static_cast<const RooSimultaneous *>(func);
      elem["type"] << key();
      elem["index"] << sim->indexCat().GetName();
      auto &channels = elem["channels"];
      channels.set_map();
      const auto &indexCat = sim->indexCat();
      for (const auto &cat : indexCat) {
         const auto catname = cat.first.c_str();
         RooAbsPdf *pdf = sim->getPdf(catname);
         if (!pdf)
            RooJSONFactoryWSTool::error("no pdf found for category");
         channels[catname] << pdf->GetName();
      }
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
      const RooDataHist &dh = hf->dataHist();
      elem["type"] << key();
      RooArgList vars(*dh.get());
      std::unique_ptr<TH1> hist{hf->createHistogram(RooJSONFactoryWSTool::concat(&vars).c_str())};
      auto &data = elem["data"];
      RooJSONFactoryWSTool::exportHistogram(*hist, data, RooJSONFactoryWSTool::names(&vars));
      return true;
   }
};

class RooHistFuncFactory : public RooFit::JSONIO::Importer {
public:
   bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      RooWorkspace &ws = *tool->workspace();

      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("data")) {
         RooJSONFactoryWSTool::error("function '" + name + "' is of histogram type, but does not define a 'data' key");
      }
      RooArgSet varlist;
      RooJSONFactoryWSTool::getObservables(ws, p["data"], name, varlist);
      RooDataHist *dh = dynamic_cast<RooDataHist *>(ws.embeddedData(name));
      if (!dh) {
         auto dhForImport = RooJSONFactoryWSTool::readBinnedData(ws, p["data"], name, varlist);
         ws.import(*dhForImport, RooFit::Silence(true), RooFit::Embedded());
         dh = static_cast<RooDataHist *>(ws.embeddedData(dhForImport->GetName()));
      }
      RooHistFunc hf(name.c_str(), name.c_str(), *(dh->get()), *dh);
      ws.import(hf, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
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
      const RooDataHist &dh = hf->dataHist();
      elem["type"] << key();
      RooArgList vars(*dh.get());
      std::unique_ptr<TH1> hist{hf->createHistogram(RooJSONFactoryWSTool::concat(&vars).c_str())};
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
      RooArgSet varlist;
      tool->getObservables(*tool->workspace(), p["data"], name, varlist);
      RooDataHist *dh = dynamic_cast<RooDataHist *>(tool->workspace()->embeddedData(name));
      if (!dh) {
         auto dhForImport = tool->readBinnedData(*tool->workspace(), p["data"], name, varlist);
         tool->workspace()->import(*dhForImport, RooFit::Silence(true), RooFit::Embedded());
         dh = static_cast<RooDataHist *>(tool->workspace()->embeddedData(dhForImport->GetName()));
      }
      RooHistPdf hf(name.c_str(), name.c_str(), *(dh->get()), *dh);
      tool->workspace()->import(hf, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
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
      static const std::string keystring = "pdfprod";
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
      static const std::string keystring = "genericpdf";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooGenericPdf *pdf = static_cast<const RooGenericPdf *>(func);
      elem["type"] << key();
      elem["formula"] << pdf->expression();
      elem["dependents"].fill_seq(pdf->dependents(), [](auto const &f) { return f->GetName(); });
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
      static const std::string keystring = "formulavar";
      return keystring;
   }
   bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooFormulaVar *var = static_cast<const RooFormulaVar *>(func);
      elem["type"] << key();
      elem["formula"] << var->expression();
      elem["dependents"].fill_seq(var->dependents(), [](auto const &f) { return f->GetName(); });
      return true;
   }
};

///////////////////////////////////////////////////////////////////////////////////////////////////////
// instantiate all importers and exporters
///////////////////////////////////////////////////////////////////////////////////////////////////////

STATIC_EXECUTE([]() {
   using namespace RooFit::JSONIO;

   registerImporter<RooProdPdfFactory>("pdfprod", false);
   registerImporter<RooGenericPdfFactory>("genericpdf", false);
   registerImporter<RooFormulaVarFactory>("formulavar", false);
   registerImporter<RooBinSamplingPdfFactory>("binsampling", false);
   registerImporter<RooAddPdfFactory>("pdfsum", false);
   registerImporter<RooHistFuncFactory>("histogram", false);
   registerImporter<RooHistPdfFactory>("histogramPdf", false);
   registerImporter<RooSimultaneousFactory>("simultaneous", false);
   registerImporter<RooBinWidthFunctionFactory>("binwidth", false);
   registerImporter<RooRealSumPdfFactory>("sumpdf", false);
   registerImporter<RooRealSumFuncFactory>("sumfunc", false);

   registerExporter<RooBinWidthFunctionStreamer>(RooBinWidthFunction::Class(), false);
   registerExporter<RooProdPdfStreamer>(RooProdPdf::Class(), false);
   registerExporter<RooSimultaneousStreamer>(RooSimultaneous::Class(), false);
   registerExporter<RooBinSamplingPdfStreamer>(RooBinSamplingPdf::Class(), false);
   registerExporter<RooHistFuncStreamer>(RooHistFunc::Class(), false);
   registerExporter<RooHistPdfStreamer>(RooHistPdf::Class(), false);
   registerExporter<RooGenericPdfStreamer>(RooGenericPdf::Class(), false);
   registerExporter<RooFormulaVarStreamer>(RooFormulaVar::Class(), false);
   registerExporter<RooRealSumPdfStreamer>(RooRealSumPdf::Class(), false);
   registerExporter<RooRealSumFuncStreamer>(RooRealSumFunc::Class(), false);
});

} // namespace

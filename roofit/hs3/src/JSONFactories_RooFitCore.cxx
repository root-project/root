#include <RooFitHS3/RooJSONFactoryWSTool.h>
#include <RooFitHS3/JSONInterface.h>

#include <RooDataHist.h>
#include <RooWorkspace.h>

#include "static_execute.h"

using RooFit::Detail::JSONNode;

///////////////////////////////////////////////////////////////////////////////////////////////////////
// individually implemented importers
///////////////////////////////////////////////////////////////////////////////////////////////////////

#include <RooGenericPdf.h>

namespace {
class RooGenericPdfFactory : public RooJSONFactoryWSTool::Importer {
public:
   virtual bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
         TObject *obj = tool->workspace()->obj(objname.c_str());
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
} // namespace

#include <RooFormulaVar.h>

namespace {
class RooFormulaVarFactory : public RooJSONFactoryWSTool::Importer {
public:
   virtual bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
         TObject *obj = tool->workspace()->obj(objname.c_str());
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
} // namespace

#include <RooProdPdf.h>

namespace {
class RooProdPdfFactory : public RooJSONFactoryWSTool::Importer {
public:
   virtual bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
} // namespace

#include <RooAddPdf.h>

namespace {
class RooAddPdfFactory : public RooJSONFactoryWSTool::Importer {
public:
   virtual bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
} // namespace

#include <RooBinWidthFunction.h>

namespace {
class RooBinWidthFunctionFactory : public RooJSONFactoryWSTool::Importer {
public:
   virtual bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      bool divideByBinWidth = p["divideByBinWidth"].val_bool();
      RooHistFunc *hf = dynamic_cast<RooHistFunc *>(tool->request<RooAbsReal>(p["histogram"].val(), name));
      RooBinWidthFunction func(name.c_str(), name.c_str(), *hf, divideByBinWidth);
      tool->workspace()->import(func, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
      return true;
   }
};
} // namespace

#include <RooSimultaneous.h>
#include <RooCategory.h>

namespace {
class RooSimultaneousFactory : public RooJSONFactoryWSTool::Importer {
public:
   virtual bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("channels")) {
         RooJSONFactoryWSTool::error("no channel components of '" + name + "'");
      }
      tool->importPdfs(p["channels"]);
      std::map<std::string, RooAbsPdf *> components;
      std::string indexname(p["index"].val());
      RooCategory cat(indexname.c_str(), indexname.c_str());
      for (const auto &comp : p["channels"].children()) {
         std::string catname(RooJSONFactoryWSTool::name(comp));
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
} // namespace

#include <RooBinSamplingPdf.h>
#include <RooRealVar.h>

namespace {
class RooBinSamplingPdfFactory : public RooJSONFactoryWSTool::Importer {
public:
   virtual bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
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
      double epsilon(p["epsilon"].val_float());

      RooBinSamplingPdf thepdf(name.c_str(), name.c_str(), *obs, *pdf, epsilon);
      tool->workspace()->import(thepdf, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));

      return true;
   }
};
} // namespace

#include <RooRealSumPdf.h>

namespace {
class RooRealSumPdfFactory : public RooJSONFactoryWSTool::Importer {
public:
   virtual bool importPdf(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("samples")) {
         RooJSONFactoryWSTool::error("no samples given in '" + name + "'");
      }
      if (!p.has_child("coefficients")) {
         RooJSONFactoryWSTool::error("no coefficients given in '" + name + "'");
      }
      RooArgList samples;
      for(const auto& sample:p["samples"].children()){
        RooAbsReal *s = tool->request<RooAbsReal>(sample.val(), name);
        samples.add(*s);
      }
      RooArgList coefficients;
      for(const auto& coef:p["coefficients"].children()){
        RooAbsReal *c = tool->request<RooAbsReal>(coef.val(), name);
        coefficients.add(*c);
      }      
      
      bool extended = p["extended"].val_bool();
      RooRealSumPdf thepdf(name.c_str(), name.c_str(), samples, coefficients, extended);      
      tool->workspace()->import(thepdf, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
      return true;
   }
};
} // namespace

///////////////////////////////////////////////////////////////////////////////////////////////////////
// specialized exporter implementations
///////////////////////////////////////////////////////////////////////////////////////////////////////

#include <RooRealSumPdf.h>

namespace {
  class RooRealSumPdfStreamer : public RooJSONFactoryWSTool::Exporter {
public:
   virtual std::string key() const { return "sumpdf"; }
   virtual bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooRealSumPdf *pdf = static_cast<const RooRealSumPdf *>(func);
      auto &samples = elem["samples"];
      samples.set_seq();      
      auto &coefs = elem["coefficients"];
      coefs.set_seq();
      for(const auto& s:pdf->funcList()){
        samples.append_child() << s->GetName();
      }
      for(const auto& c:pdf->coefList()){
        coefs.append_child() << c->GetName();
      }
      elem["extended"] << (pdf->extendMode() == RooAbsPdf::CanBeExtended);
      return true;
   }
};
}

namespace {
class RooSimultaneousStreamer : public RooJSONFactoryWSTool::Exporter {
public:
   virtual std::string key() const { return "simultaneous"; }
   bool autoExportDependants() const override { return false; }
   virtual bool exportObject(RooJSONFactoryWSTool *tool, const RooAbsArg *func, JSONNode &elem) const override
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
         tool->exportObject(pdf, channels);
      }
      return true;
   }
};
} // namespace

#include <RooHistFunc.h>
#include <TH1.h>

namespace {
class RooHistFuncStreamer : public RooJSONFactoryWSTool::Exporter {
public:
   virtual std::string key() const { return "histogram"; }
   virtual bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooHistFunc *hf = static_cast<const RooHistFunc *>(func);
      const RooDataHist &dh = hf->dataHist();
      elem["type"] << key();
      RooArgList vars(*dh.get());
      TH1 *hist = hf->createHistogram(RooJSONFactoryWSTool::concat(&vars).c_str());
      auto &data = elem["data"];
      RooJSONFactoryWSTool::exportHistogram(*hist, data, RooJSONFactoryWSTool::names(&vars));
      delete hist;
      return true;
   }
};
} // namespace

namespace {
class RooHistFuncFactory : public RooJSONFactoryWSTool::Importer {
public:
   virtual bool importFunction(RooJSONFactoryWSTool *tool, const JSONNode &p) const override
   {
      std::string name(RooJSONFactoryWSTool::name(p));
      if (!p.has_child("data")) {
         RooJSONFactoryWSTool::error("function '" + name + "' is of histogram type, but does not define a 'data' key");
      }
      auto varlist = tool->getObservables(p["data"], name);
      RooDataHist *dh = dynamic_cast<RooDataHist *>(tool->workspace()->embeddedData(name.c_str()));
      if (!dh) {
         dh = tool->readBinnedData(p["data"], name, varlist);
         tool->workspace()->import(*dh, RooFit::Silence(true), RooFit::Embedded());
      }
      RooHistFunc hf(name.c_str(), name.c_str(), *(dh->get()), *dh);
      tool->workspace()->import(hf, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
      return true;
   }
};
} // namespace

#include <RooBinSamplingPdf.h>

namespace {
class RooBinSamplingPdfStreamer : public RooJSONFactoryWSTool::Exporter {
public:
   virtual std::string key() const { return "binsampling"; }
   virtual bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooBinSamplingPdf *pdf = static_cast<const RooBinSamplingPdf *>(func);
      elem["type"] << key();
      elem["pdf"] << pdf->pdf().GetName();
      elem["observable"] << pdf->observable().GetName();
      elem["epsilon"] << pdf->epsilon();
      return true;
   }
};
} // namespace

#include <RooProdPdf.h>

namespace {
class RooProdPdfStreamer : public RooJSONFactoryWSTool::Exporter {
public:
   virtual std::string key() const { return "pdfprod"; }
   virtual bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooProdPdf *pdf = static_cast<const RooProdPdf *>(func);
      elem["type"] << key();
      auto &factors = elem["pdfs"];
      for (const auto &f : pdf->pdfList()) {
         factors.append_child() << f->GetName();
      }
      return true;
   }
};
} // namespace

#include <RooGenericPdf.h>

namespace {
class RooGenericPdfStreamer : public RooJSONFactoryWSTool::Exporter {
public:
   virtual std::string key() const { return "genericpdf"; }
   virtual bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooGenericPdf *pdf = static_cast<const RooGenericPdf *>(func);
      elem["type"] << key();
      elem["formula"] << pdf->expression().Data();
      auto &factors = elem["dependents"];
      for (const auto &f : pdf->dependents()) {
         factors.append_child() << f->GetName();
      }
      return true;
   }
};
} // namespace

#include <RooBinWidthFunction.h>

namespace {
class RooBinWidthFunctionStreamer : public RooJSONFactoryWSTool::Exporter {
public:
   virtual std::string key() const { return "binwidth"; }
   virtual bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooBinWidthFunction *pdf = static_cast<const RooBinWidthFunction *>(func);
      elem["type"] << key();
      elem["histogram"] << pdf->histFunc()->GetName();
      elem["divideByBinWidth"] << pdf->divideByBinWidth();
      return true;
   }
};
} // namespace

#include <RooFormulaVar.h>

namespace {
class RooFormulaVarStreamer : public RooJSONFactoryWSTool::Exporter {
public:
   virtual std::string key() const { return "formulavar"; }
   virtual bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooFormulaVar *var = static_cast<const RooFormulaVar *>(func);
      elem["type"] << key();
      elem["formula"] << var->expression().Data();
      auto &factors = elem["dependents"];
      for (const auto &f : var->dependents()) {
         factors.append_child() << f->GetName();
      }
      return true;
   }
};
} // namespace

///////////////////////////////////////////////////////////////////////////////////////////////////////
// instantiate all importers and exporters
///////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
STATIC_EXECUTE(

               RooJSONFactoryWSTool::registerImporter("pdfprod", new RooProdPdfFactory(),false);
               RooJSONFactoryWSTool::registerImporter("genericpdf", new RooGenericPdfFactory(),false);
               RooJSONFactoryWSTool::registerImporter("formulavar", new RooFormulaVarFactory(),false);
               RooJSONFactoryWSTool::registerImporter("binsampling", new RooBinSamplingPdfFactory(),false);
               RooJSONFactoryWSTool::registerImporter("pdfsum", new RooAddPdfFactory(),false);
               RooJSONFactoryWSTool::registerImporter("histogram", new RooHistFuncFactory(),false);
               RooJSONFactoryWSTool::registerImporter("simultaneous", new RooSimultaneousFactory(),false);
               RooJSONFactoryWSTool::registerImporter("binwidth", new RooBinWidthFunctionFactory(),false);
               RooJSONFactoryWSTool::registerImporter("sumpdf", new RooRealSumPdfFactory(),false);               

               RooJSONFactoryWSTool::registerExporter(RooBinWidthFunction::Class(), new RooBinWidthFunctionStreamer(),false);
               RooJSONFactoryWSTool::registerExporter(RooProdPdf::Class(), new RooProdPdfStreamer(),false);
               RooJSONFactoryWSTool::registerExporter(RooSimultaneous::Class(), new RooSimultaneousStreamer(),false);
               RooJSONFactoryWSTool::registerExporter(RooBinSamplingPdf::Class(), new RooBinSamplingPdfStreamer(),false);
               RooJSONFactoryWSTool::registerExporter(RooHistFunc::Class(), new RooHistFuncStreamer(),false);
               RooJSONFactoryWSTool::registerExporter(RooGenericPdf::Class(), new RooGenericPdfStreamer(),false);
               RooJSONFactoryWSTool::registerExporter(RooFormulaVar::Class(), new RooFormulaVarStreamer(),false);
               RooJSONFactoryWSTool::registerExporter(RooRealSumPdf::Class(), new RooRealSumPdfStreamer(),false);               
               
)
} // namespace

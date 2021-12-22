#include <RooFitHS3/RooJSONFactoryWSTool.h>

#include <RooDataHist.h>
#include <RooWorkspace.h>


#include "JSONInterface.h"
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
      RooArgSet dependents;
      for(const auto& d:p["dependents"].children()){
        std::string objname(RooJSONFactoryWSTool::name(d));
        TObject* obj = tool->workspace()->obj(objname.c_str());
        if(obj->InheritsFrom(RooAbsArg::Class())){
          dependents.add(*static_cast<RooAbsArg*>(obj));
        }
      }
      TString formula(p["formula"].val());
      RooGenericPdf thepdf(name.c_str(), formula.Data(), dependents);
      tool->workspace()->import(thepdf);
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
      if (!p.has_child("factors")) {
         RooJSONFactoryWSTool::error("no pdfs of '" + name + "'");
      }
      if (!p["factors"].is_seq()) {
         RooJSONFactoryWSTool::error("pdfs '" + name + "' are not a list.");
      }
      for (const auto &comp : p["factors"].children()) {
         std::string pdfname(comp.val());
         RooAbsPdf *pdf = tool->workspace()->pdf(pdfname.c_str());
         if (!pdf) {
            RooJSONFactoryWSTool::error("unable to obtain component '" + pdfname + "' of '" + name + "'.");
         }
         factors.add(*pdf);
      }
      RooProdPdf prod(name.c_str(), name.c_str(), factors);
      tool->workspace()->import(prod);
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
         RooAbsPdf *pdf = tool->workspace()->pdf(pdfname.c_str());
         if (!pdf) {
            RooJSONFactoryWSTool::error("unable to obtain component '" + pdfname + "' of '" + name + "'.");
         }
         pdfs.add(*pdf);
      }
      for (const auto &comp : p["coefficients"].children()) {
         std::string coefname(comp.val());
         RooAbsArg *coef = tool->workspace()->arg(coefname.c_str());
         if (!coef) {
            RooJSONFactoryWSTool::error("unable to obtain component '" + coefname + "' of '" + name + "'.");
         }
         coefs.add(*coef);
      }
      RooAddPdf add(name.c_str(), name.c_str(), pdfs, coefs);
      tool->workspace()->import(add);
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
      RooCategory cat("channelCat", "channelCat");
      for (const auto &comp : p["channels"].children()) {
         std::string catname(RooJSONFactoryWSTool::name(comp));
         std::string pdfname(comp.has_val() ? comp.val() : RooJSONFactoryWSTool::name(comp));
         RooAbsPdf *pdf = tool->workspace()->pdf(pdfname.c_str());
         if (!pdf) {
            RooJSONFactoryWSTool::error("unable to obtain component '" + pdfname + "' of '" + name + "'");
         }
         components[catname] = pdf;
         cat.defineType(catname.c_str());
      }
      RooSimultaneous simpdf(name.c_str(), name.c_str(), components, cat);
      tool->workspace()->import(simpdf);
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

      std::string pdfname(p["pdf"].val());
      RooAbsPdf *pdf = tool->workspace()->pdf(pdfname.c_str());
      if (!pdf) {
        RooJSONFactoryWSTool::error("unable to obtain component '" + pdfname + "' of '" + name + "'");
      }
      
      std::string obsname(p["observable"].val());
      RooRealVar *obs = tool->workspace()->var(obsname.c_str());
      if (!obs) {
        RooJSONFactoryWSTool::error("unable to obtain observable '" + obsname + "' of '" + name + "'");
      }
      
      double epsilon(p["epsilon"].val_float());            

      RooBinSamplingPdf thepdf(name.c_str(), name.c_str(), *obs, *pdf, epsilon); 
      tool->workspace()->import(thepdf);
      
      return true;
   }
};
} // namespace

///////////////////////////////////////////////////////////////////////////////////////////////////////
// specialized exporter implementations
///////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
class RooSimultaneousStreamer : public RooJSONFactoryWSTool::Exporter {
public:
   bool autoExportDependants() const override { return false; }
   virtual bool exportObject(RooJSONFactoryWSTool *tool, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooSimultaneous *sim = static_cast<const RooSimultaneous *>(func);
      elem["type"] << "simultaneous";
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
   virtual bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooHistFunc *hf = static_cast<const RooHistFunc *>(func);
      const RooDataHist &dh = hf->dataHist();
      elem["type"] << "histogram";
      RooArgList vars(*dh.get());
      TH1 *hist = hf->createHistogram(RooJSONFactoryWSTool::concat(&vars).c_str());
      auto &data = elem["data"];
      RooJSONFactoryWSTool::exportHistogram(*hist, data, RooJSONFactoryWSTool::names(&vars));
      delete hist;
      return true;
   }
};
} // namespace

#include <RooBinSamplingPdf.h>

namespace {
class RooBinSamplingPdfStreamer : public RooJSONFactoryWSTool::Exporter {
public:
   virtual bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooBinSamplingPdf *pdf = static_cast<const RooBinSamplingPdf *>(func);     
      elem["type"] << "binsampling";
      elem["pdf"] << pdf->pdf().GetName();
      elem["epsilon"] << pdf->epsilon(); 
      return true;
   }
};
} // namespace

#include <RooProdPdf.h>

namespace {
class RooProdPdfStreamer : public RooJSONFactoryWSTool::Exporter {
public:
   virtual bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooProdPdf *pdf = static_cast<const RooProdPdf *>(func);     
      elem["type"] << "pdfprod";
      auto& factors = elem["pdfs"];
      for(const auto& f:pdf->pdfList()){
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
   virtual bool exportObject(RooJSONFactoryWSTool *, const RooAbsArg *func, JSONNode &elem) const override
   {
      const RooGenericPdf *pdf = static_cast<const RooGenericPdf *>(func);     
      elem["type"] << "genericpdf";
      elem["formula"] << pdf->expression().Data();
      auto& factors = elem["dependents"];
      for(const auto& f:pdf->dependents()){
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

   RooJSONFactoryWSTool::registerImporter("pdfprod", new RooProdPdfFactory());
   RooJSONFactoryWSTool::registerImporter("genericpdf", new RooGenericPdfFactory());   
   RooJSONFactoryWSTool::registerImporter("binsampling", new RooBinSamplingPdfFactory());   
   RooJSONFactoryWSTool::registerImporter("pdfsum", new RooAddPdfFactory());
   RooJSONFactoryWSTool::registerImporter("simultaneous", new RooSimultaneousFactory());

   RooJSONFactoryWSTool::registerExporter(RooProdPdf::Class(), new RooProdPdfStreamer());   
   RooJSONFactoryWSTool::registerExporter(RooSimultaneous::Class(), new RooSimultaneousStreamer());
   RooJSONFactoryWSTool::registerExporter(RooBinSamplingPdf::Class(), new RooBinSamplingPdfStreamer());   
   RooJSONFactoryWSTool::registerExporter(RooHistFunc::Class(), new RooHistFuncStreamer());
   RooJSONFactoryWSTool::registerExporter(RooGenericPdf::Class(), new RooGenericPdfStreamer());   
)
} // namespace


#include <RooStats/RooJSONFactoryWSTool.h>
#include <RooDataHist.h>
#include <RooProdPdf.h>
#include <RooAddPdf.h>
#include <RooSimultaneous.h>
#include <RooCategory.h>
#include <RooHistFunc.h>



///////////////////////////////////////////////////////////////////////////////////////////////////////
// individually implemented importers
///////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef INCLUDE_RYML
#include <ryml.hpp>
#include <c4/yml/std/map.hpp>
#include <c4/yml/std/string.hpp>

namespace {
  // error handling helpers
  void error(const char* s){
    throw std::runtime_error(s);
  }
  void error(const std::string& s){
    throw std::runtime_error(s);
  }
  inline std::string name(const c4::yml::NodeRef& n){
    std::stringstream ss;
    if(n.has_key()){
      ss << n.key();
    } else if(n.has_child("name")){
      ss << n["name"].val();
    }    
    return ss.str();
  }
  inline std::string val_s(const c4::yml::NodeRef& n){
    std::stringstream ss;    
    ss << n.val();
    return ss.str();
  }
  inline double val_d(const c4::yml::NodeRef& n){
    float d;
    c4::atof(n.val(),&d);
    return d;    
  }
  inline int val_i(const c4::yml::NodeRef& n){
    int i;
    c4::atoi(n.val(),&i);

    return i;
  }
  inline bool val_b(const c4::yml::NodeRef& n){
    int i;
    c4::atoi(n.val(),&i);
    return i;
  }
  template<class T> static std::string concat(const T* items, const std::string& sep=",") {
    // Returns a string being the concatenation of strings in input list <items>
    // (names of objects obtained using GetName()) separated by string <sep>.
    bool first = true;
    std::string text;
    
    // iterate over strings in list
    for(auto it:*items){
      if (!first) {
        // insert separator string
        text += sep;
      } else {
        first = false;
      }
      if(!it) text+="NULL";
      else text+=it->GetName();
    }
    return text;
  }
  template<class T> static std::vector<std::string> names(const T* items) {
    // Returns a string being the concatenation of strings in input list <items>
    // (names of objects obtained using GetName()) separated by string <sep>.
    std::vector<std::string> names;
    // iterate over strings in list
    for(auto it:*items){
      if(!it) names.push_back("NULL");
      else names.push_back(it->GetName());
    }
    return names;
  }
}

namespace {
  class RooProdPdfFactory : public RooJSONFactoryWSTool::Importer<c4::yml::NodeRef> {
  public:
    virtual bool importPdf(RooWorkspace* ws, const c4::yml::NodeRef& p) const override {
      std::string name(::name(p));
      RooArgSet factors;
      if(!p.has_child("pdfs")){
        error("no pdfs of '" + name + "'");
      }
      if(!p["pdfs"].is_seq()){
        error("pdfs '" + name + "' are not a list.");
      }      
      for(auto comp:p["pdfs"].children()){
        std::string pdfname(::val_s(comp));
        RooAbsPdf* pdf = ws->pdf(pdfname.c_str());
        if(!pdf){
          error("unable to obtain component '" + pdfname + "' of '" + name + "'.");
        }
        factors.add(*pdf);
      }
      RooProdPdf prod(name.c_str(),name.c_str(),factors);
      ws->import(prod);
      return true;
    }
  };
  bool _rooprodpdffactory = RooJSONFactoryWSTool::registerImporter("pdfprod",new RooProdPdfFactory());

  class RooAddPdfFactory : public RooJSONFactoryWSTool::Importer<c4::yml::NodeRef> {
  public:
    virtual bool importPdf(RooWorkspace* ws, const c4::yml::NodeRef& p) const override {
      std::string name(::name(p));
      RooArgList pdfs;
      RooArgList coefs;      
      if(!p.has_child("summands")){
        error("no summands of '" + name + "'");
      }
      if(!p["summands"].is_seq()){
        error("summands '" + name + "' are not a list.");
      }      
      if(!p.has_child("coefficients")){
        error("no coefficients of '" + name + "'");
      }
      if(!p["coefficients"].is_seq()){
        error("coefficients '" + name + "' are not a list.");
      }      
      for(auto comp:p["summands"].children()){
        std::string pdfname(::val_s(comp));
        RooAbsPdf* pdf = ws->pdf(pdfname.c_str());
        if(!pdf){
          error("unable to obtain component '" + pdfname + "' of '" + name + "'.");
        }
        pdfs.add(*pdf);
      }
      for(auto comp:p["coefficients"].children()){
        std::string coefname(::val_s(comp));
        RooAbsArg* coef = ws->arg(coefname.c_str());
        if(!coef){
          error("unable to obtain component '" + coefname + "' of '" + name + "'.");
        }
        coefs.add(*coef);
      }      
      RooAddPdf add(name.c_str(),name.c_str(),pdfs,coefs);
      ws->import(add);
      return true;
    }
  };
  bool _rooaddpdffactory = RooJSONFactoryWSTool::registerImporter("pdfsum",new RooAddPdfFactory());  

  
  class RooSimultaneousFactory : public RooJSONFactoryWSTool::Importer<c4::yml::NodeRef> {
  public:
    virtual bool importPdf(RooWorkspace* ws, const c4::yml::NodeRef& p) const override {
      std::string name(::name(p));
      if(!p.has_child("channels")){
        error("no channel components of '" + name + "'");
      }
      std::map< std::string, RooAbsPdf *> components;
      RooCategory cat("channelCat","channelCat");
      for(auto comp:p["channels"]){
        std::string catname(::name(comp));
        std::string pdfname(::val_s(comp));
        RooAbsPdf* pdf = ws->pdf(pdfname.c_str());
        if(!pdf){
          error("unable to obtain component '" + pdfname + "' of '" + name + "'");
        }
        components[catname] = pdf;
        cat.defineType(pdfname.c_str());
      }
      RooSimultaneous simpdf(name.c_str(),name.c_str(),components,cat);
      ws->import(simpdf);
      return true;
    }
  };
  bool _roosimultaneousfactory = RooJSONFactoryWSTool::registerImporter("simultaneous",new RooSimultaneousFactory()); 
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// specialized exporter implementations
///////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
  class RooSimultaneousStreamer : public RooJSONFactoryWSTool::Exporter<c4::yml::NodeRef> {
  public:
    virtual bool exportObject(RooAbsReal* func, c4::yml::NodeRef& elem) const override {
      RooSimultaneous* sim = static_cast<RooSimultaneous*>(func);
      elem["type"] << "simultaneous";
      auto channels = elem["channels"];
      channels |= c4::yml::MAP;
      const auto& indexCat = sim->indexCat();
      for(const auto& cat:indexCat){
        channels[c4::to_csubstr(cat->GetName())] << sim->getPdf(cat->GetName())->GetName();
      }
      return true;
    }
  };
  bool _roosimultaneousstreamer = RooJSONFactoryWSTool::registerExporter(RooSimultaneous::Class(),new RooSimultaneousStreamer());

  class RooHistFuncStreamer : public RooJSONFactoryWSTool::Exporter<c4::yml::NodeRef> {
  public:
    virtual bool exportObject(RooAbsReal* func, c4::yml::NodeRef& elem) const override {
      RooHistFunc* hf = static_cast<RooHistFunc*>(func);
      const RooDataHist& dh = hf->dataHist();
      elem["type"] << "histogram";
      RooArgList vars(*dh.get());
      TH1* hist = func->createHistogram(::concat(&vars).c_str());
      auto data = elem["data"];
      RooJSONFactoryWSTool::exportHistogram(*hist,data,::names(&vars));
      delete hist;
      return true;
    }
  };
  bool _roohistfuncstreamer = RooJSONFactoryWSTool::registerExporter(RooHistFunc::Class(),new RooHistFuncStreamer());
}
#endif


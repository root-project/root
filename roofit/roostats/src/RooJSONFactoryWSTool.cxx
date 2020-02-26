#include <RooStats/RooJSONFactoryWSTool.h>

#include <iostream>
#include <fstream>

#include <RooConstVar.h>
#include <RooCategory.h>
#include <RooRealVar.h>
#include <RooDataHist.h>
#include <RooHistFunc.h>
#include <RooStats/ModelConfig.h>

#include "TROOT.h"

std::vector<std::string> RooJSONFactoryWSTool::_strcache = std::vector<std::string>(1000);

namespace {
  // error handling helpers
  void error(const char* s){
    throw std::runtime_error(s);
  }
  void error(const std::string& s){
    throw std::runtime_error(s);
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
  // helpers for serializing / deserializing binned datasets
  inline void genIndicesHelper(std::vector<std::vector<int> >& combinations, RooArgList& vars, size_t curridx){
    if(curridx == vars.size()){
      std::vector<int> indices(curridx);
      for(size_t i=0; i<curridx; ++i){
        RooRealVar* v = (RooRealVar*)(vars.at(i));
        indices[i] = v->getBinning().binNumber(v->getVal());
      }
      combinations.push_back(indices);
    } else {
      RooRealVar* v = (RooRealVar*)(vars.at(curridx));
      for(int i=0; i<v->numBins(); ++i){      
        v->setVal(v->getBinning().binCenter(i));
        ::genIndicesHelper(combinations,vars,curridx+1);
      }
    }
  }
}

#ifdef INCLUDE_RYML
#include <ryml.hpp>
#include <c4/yml/std/map.hpp>
#include <c4/yml/std/string.hpp>

// maps to hold the importers and exporters for runtime lookup
template<> std::map<std::string,const RooJSONFactoryWSTool::Importer<c4::yml::NodeRef>*> RooJSONFactoryWSTool::_importers<c4::yml::NodeRef> = std::map<std::string,const RooJSONFactoryWSTool::Importer<c4::yml::NodeRef>*>();
template<> std::map<const TClass*,const RooJSONFactoryWSTool::Exporter<c4::yml::NodeRef>*> RooJSONFactoryWSTool::_exporters<c4::yml::NodeRef> = std::map<const TClass*,const RooJSONFactoryWSTool::Exporter<c4::yml::NodeRef>*>();

///////////////////////////////////////////////////////////////////////////////////////////////////////
// helper functions specific to RYML
///////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
  inline std::string key(const c4::yml::NodeRef& n){
    std::stringstream ss;
    ss << n.key();
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
  inline std::string genPrefix(const c4::yml::NodeRef& p,bool trailing_underscore){
    std::string prefix;
    if(!p.is_map()) return prefix;
    if(p.has_child("namespaces")){
      for(auto ns:p["namespaces"]){
        if(prefix.size() > 0) prefix+="_";
        prefix += ::val_s(ns);
      }
    }
    if(trailing_underscore && prefix.size()>0) prefix += "_";
    return prefix;
  }

  inline void importAttributes(RooAbsArg* arg, const c4::yml::NodeRef& n){
    if(!n.is_map()) return;
    if(n.has_child("dict")){
      for(auto attr:n["dict"]){
        arg->setStringAttribute(::key(attr).c_str(),::val_s(attr).c_str());
      }
    }
  }
  inline void writeAxis(c4::yml::NodeRef& bounds, const TAxis& ax){
    if(!ax.IsVariableBinSize()){
      bounds |= c4::yml::MAP;
      bounds["nbins"] << ax.GetNbins();                
      bounds["min"] << ax.GetXmin();
      bounds["max"] << ax.GetXmax();
    } else {
      bounds |= c4::yml::SEQ;              
      for(int i=1; i<=ax.GetNbins(); ++i){
        bounds.append_child() << ax.GetBinLowEdge(i);      
      }
    }
  }  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// RooWSFactoryTool expression handling
///////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
  struct RYML_Factory_Expression {
    TClass* tclass;
    std::vector<std::string> arguments;
    std::string generate(const c4::yml::NodeRef& p){
      std::string name(::key(p));
      std::stringstream expression;
      std::string classname(this->tclass->GetName());
      size_t colon = classname.find_last_of(":");
      if(colon < classname.size()){
        expression << classname.substr(colon+1);
      } else {
        expression << classname;
      }
      expression << "::" << name << "(";
      bool first = true;
      for(auto k:this->arguments){
        if(!first) expression << ",";
        first = false;
        if(k == "true"){
          expression << "1";
          continue;
        } else if(k == "false"){
          expression << "0";
          continue;          
        } else if(!p.has_child(c4::to_csubstr(k))){
          std::stringstream err;
          err << "factory expression for class '" << this->tclass->GetName() << "', which expects key '" << k << "' missing from input for object '" << name << "', skipping.";
          error(err.str().c_str());
        }
        if(p[c4::to_csubstr(k)].is_seq()){
          expression << "{";
          bool f = true;
          for(auto x:p[c4::to_csubstr(k)]){
            if(!f) expression << ",";
            f=false;
            expression << ::val_s(x);
          }
          expression << "}";
        } else {
          expression << ::val_s(p[c4::to_csubstr(k)]);
        }          
      }
      expression << ")";
      return expression.str();
    }
  };
  std::map<std::string,RYML_Factory_Expression> _rymlPdfFactoryExpressions;
  std::map<std::string,RYML_Factory_Expression> _rymlFuncFactoryExpressions;
}

void RooJSONFactoryWSTool::loadFactoryExpressions(const std::string& fname){
  // load a yml file defining the factory expressions
#ifdef INCLUDE_RYML
  std::ifstream infile(fname);  
  std::string s(std::istreambuf_iterator<char>(infile), {});
  ryml::Tree t = c4::yml::parse(c4::to_csubstr(s.c_str()));    
  c4::yml::NodeRef n = t.rootref();
  for(const auto& cl:n.children()){
    std::string key(::key(cl));
    if(!cl.has_child("class")){
      std::cerr << "error in file '" << fname << "' for entry '" << key << "': 'class' key is required!" << std::endl;
      continue;
    }
    std::string classname(::val_s(cl["class"]));    
    TClass* c = TClass::GetClass(classname.c_str());
    if(!c){
      std::cerr << "unable to find class " << classname << ", skipping." << std::endl;
    } else {
      RYML_Factory_Expression ex;
      ex.tclass = c;
      for(const auto& arg:cl["arguments"]){
        ex.arguments.push_back(::val_s(arg));
      }
      if(c->InheritsFrom(RooAbsPdf::Class())){
        _rymlPdfFactoryExpressions[key] = ex;
      } else if(c->InheritsFrom(RooAbsReal::Class())){
        _rymlFuncFactoryExpressions[key] = ex;        
      } else {
        std::cerr << "class " << classname << " seems to not inherit from any suitable class, skipping" << std::endl;
      }
    }
  }  
#endif
}
void RooJSONFactoryWSTool::clearFactoryExpressions(){
  // clear all factory expressions
#ifdef INCLUDE_RYML    
  _rymlPdfFactoryExpressions.clear();
  _rymlFuncFactoryExpressions.clear();
#endif
}
void RooJSONFactoryWSTool::printFactoryExpressions(){
  // print all factory expressions
#ifdef INCLUDE_RYML    
  for(auto it:_rymlPdfFactoryExpressions){
    std::cout << it.first;
    std::cout << " " << it.second.tclass->GetName();    
    for(auto v:it.second.arguments){
      std::cout << " " << v;
    }
    std::cout << std::endl;
  }
  for(auto it:_rymlFuncFactoryExpressions){
    std::cout << it.first;
    std::cout << " " << it.second.tclass->GetName();    
    for(auto v:it.second.arguments){
      std::cout << " " << v;
    }
    std::cout << std::endl;
  }  
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// individually implemented importers
///////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef INCLUDE_RYML
#include <RooProdPdf.h>
#include <RooAddPdf.h>
#include <RooSimultaneous.h>

namespace {
  class RooProdPdfFactory : public RooJSONFactoryWSTool::Importer<c4::yml::NodeRef> {
  public:
    virtual bool importPdf(RooWorkspace* ws, const c4::yml::NodeRef& p) const override {
      std::string name(::key(p));
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
      std::string name(::key(p));
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
      std::string name(::key(p));
      if(!p.has_child("channels")){
        error("no channel components of '" + name + "'");
      }
      std::map< std::string, RooAbsPdf *> components;
      RooCategory cat("channelCat","channelCat");
      for(auto comp:p["channels"]){
        std::string catname(::key(comp));
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
#endif 

///////////////////////////////////////////////////////////////////////////////////////////////////////
// static helper for exporting RooDataHist
///////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::vector<int> > RooJSONFactoryWSTool::generateBinIndices(RooArgList& vars){
  std::vector<std::vector<int> > combinations;
  ::genIndicesHelper(combinations,vars,0);
  return combinations;
}

template<> void  RooJSONFactoryWSTool::exportHistogram(const TH1& h, c4::yml::NodeRef& n, const std::vector<std::string>& varnames){
  n |= c4::yml::MAP;
  auto bounds = n["binning"];
  bounds |= c4::yml::MAP;
  auto weights = n["counts"];
  weights |= c4::yml::SEQ;    
  auto errors = n["errors"];    
  errors |= c4::yml::SEQ;
  auto x = bounds[ c4::to_csubstr(RooJSONFactoryWSTool::incache(varnames[0]))];
  writeAxis(x,*(h.GetXaxis()));
  if(h.GetDimension()>1){
    auto y = bounds[ c4::to_csubstr(RooJSONFactoryWSTool::incache(varnames[1]))];
    writeAxis(y,*(h.GetYaxis()));
    if(h.GetDimension()>2){
      auto z = bounds[ c4::to_csubstr(RooJSONFactoryWSTool::incache(varnames[2]))];
      writeAxis(z,*(h.GetZaxis()));              
    }
  }
  for(int i=1; i<=h.GetNbinsX(); ++i){
    if(h.GetDimension()==1){
      weights.append_child() << h.GetBinContent(i);
      errors.append_child() << h.GetBinError(i);
    } else {
      for(int j=1; j<=h.GetNbinsY(); ++j){
        if(h.GetDimension()==2){
          weights.append_child() << h.GetBinContent(i,j);
          errors.append_child() << h.GetBinError(i,j);
        } else {
          for(int k=1; k<=h.GetNbinsZ(); ++k){        
            weights.append_child() << h.GetBinContent(i,j,k);
            errors.append_child() << h.GetBinError(i,j,k);              
          }
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// RooProxy-based export handling
///////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {
  struct RYML_Export_Keys {
    std::string type;
    std::map<std::string,std::string> proxies;
  };  
  std::map<TClass*,RYML_Export_Keys> _rymlExportKeys;
}
void RooJSONFactoryWSTool::loadExportKeys(const std::string& fname){
  // load a yml file defining the export keys
#ifdef INCLUDE_RYML
  std::ifstream infile(fname);  
  std::string s(std::istreambuf_iterator<char>(infile), {});
  ryml::Tree t = c4::yml::parse(c4::to_csubstr(s.c_str()));
  c4::yml::NodeRef n = t.rootref();
  for(const auto& cl:n.children()){
    std::string classname(::key(cl));
    TClass* c = TClass::GetClass(classname.c_str());
    if(!c){
      std::cerr << "unable to find class " << classname << ", skipping." << std::endl;
    } else {
      RYML_Export_Keys ex;
      ex.type = ::val_s(cl["type"]);
      for(const auto& k:cl["proxies"].children()){
        std::string key(::key(k));
        std::string val(::val_s(k));                
        ex.proxies[key] = val;
      }
      _rymlExportKeys[c] = ex;
    }
  }  
#endif
}
void RooJSONFactoryWSTool::clearExportKeys(){
  // clear all export keys
#ifdef INCLUDE_RYML    
  _rymlExportKeys.clear();
#endif
}
void RooJSONFactoryWSTool::printExportKeys(){
  // print all export keys
#ifdef INCLUDE_RYML    
  for(const auto& it:_rymlExportKeys){
    std::cout << it.first->GetName() << ": " << it.second.type;
    for(const auto& kv:it.second.proxies){
      std::cout << " " << kv.first << "=" << kv.second;
    }
    std::cout << std::endl;
  }
#endif
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// specialized exporter implementations
///////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef INCLUDE_RYML    
#include <RooProdPdf.h>
#include <RooAddPdf.h>
#include <RooSimultaneous.h>

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

///////////////////////////////////////////////////////////////////////////////////////////////////////
// helper namespace
///////////////////////////////////////////////////////////////////////////////////////////////////////

class RooJSONFactoryWSTool::Helpers {
public:
#ifdef INCLUDE_RYML      
  static void exportAttributes(const RooAbsArg* arg, c4::yml::NodeRef& n){
    // exporta ll string attributes of an object
    if(arg->stringAttributes().size() > 0){
      auto dict = n["dict"];
      dict |= c4::yml::MAP;      
      for(const auto& it:arg->stringAttributes()){
        dict[c4::to_csubstr(it.first.c_str())] << it.second;
      }
    }
  }
  
  static void exportVariables(const RooArgSet& allElems, c4::yml::NodeRef& n) {
    // export a list of RooRealVar objects
    for(auto* arg:allElems){
      RooRealVar* v = dynamic_cast<RooRealVar*>(arg);
      if(!v) continue;
      auto var = n[c4::to_csubstr(v->GetName())];
      var |= c4::yml::MAP;  
      var["value"] << v->getVal();
      if(v->getMin() > -1e30){
        var["min"] << v->getMin();
      }
      if(v->getMax() < 1e30){
        var["max"] << v->getMax();
      }
      var["nbins"] << v->numBins();
      if(v->getError() > 0){
        var["err"] << v->getError();
      }
      if(v->isConstant()){
        var["const"] << v->isConstant();
      }
      Helpers::exportAttributes(arg,var);      
    }  
  }

  static void exportFunctions(const RooArgSet& allElems, c4::yml::NodeRef& n){
    // export a list of functions
    // note: this function assumes that all the dependants of these objects have already been exported
    for(auto* arg:allElems){
      RooAbsReal* func = dynamic_cast<RooAbsReal*>(arg);
      if(!func) continue;

      if(n.has_child(c4::to_csubstr(func->GetName()))) continue;
      
      auto elem = n[c4::to_csubstr(func->GetName())];
      elem |= c4::yml::MAP;

      TClass* cl = TClass::GetClass(func->ClassName());
      
      auto it = _exporters<c4::yml::NodeRef>.find(cl);
      if(it != _exporters<c4::yml::NodeRef>.end()){
        try {
          if(!it->second->exportObject(func,elem)){
            std::cerr << "exporter for type " << cl->GetName() << " does not export objects!" << std::endl;          
          }
        } catch (const std::exception& ex){
          std::cerr << ex.what() << ". skipping." << std::endl;
        }
      } else { // generic import using the factory expressions      
        const auto& dict = _rymlExportKeys.find(cl);
        if(dict == _rymlExportKeys.end()){
          std::cerr << "unable to export class '" << cl->GetName() << "' - no export keys available!" << std::endl;
          std::cerr << "there are several possible reasons for this:" << std::endl;
          std::cerr << " 1. " << cl->GetName() << " is a custom class that you or some package you are using added." << std::endl;
          std::cerr << " 2. " << cl->GetName() << " is a ROOT class that nobody ever bothered to write a serialization definition for." << std::endl;
          std::cerr << " 3. something is wrong with your setup, e.g. you might have called RooJSONFactoryWSTool::clearExportKeys() and/or never successfully read a file defining these keys with RooJSONFactoryWSTool::loadExportKeys(filename)" << std::endl;
          std::cerr << "either way, please make sure that:" << std::endl;
          std::cerr << " 3: you are reading a file with export keys - call RooJSONFactoryWSTool::printExportKeys() to see what is available" << std::endl;
          std::cerr << " 2 & 1: you might need to write a serialization definition yourself. check INSERTLINKHERE to see how to do this!" << std::endl;                              
          continue;
        }
        elem["type"] << dict->second.type;
        
        size_t nprox = func->numProxies();
        for(size_t i=0; i<nprox; ++i){
          RooAbsProxy* p = func->getProxy(i);
          
          std::string pname(p->name());
          if(pname[0] == '!') pname.erase(0,1);
          
          auto k = dict->second.proxies.find(pname);
          if(k == dict->second.proxies.end()){
            std::cerr << "failed to find key matching proxy '" << pname << "' for type '" << dict->second.type << "', skipping" << std::endl;
            continue;
          }
          
          RooListProxy* l = dynamic_cast<RooListProxy*>(p);
          if(l){
            auto items = elem[c4::to_csubstr(k->second)];
            items |= c4::yml::SEQ;
            for(auto e:*l){
              items.append_child() << e->GetName();
            }
          }
          RooRealProxy* r = dynamic_cast<RooRealProxy*>(p);
          if(r){
            elem[c4::to_csubstr(k->second)] << r->arg().GetName();
          }
        }
      }
      Helpers::exportAttributes(func,elem);
    }
  }
  #endif
};

// forward declaration needed for alternating recursion
template<> void RooJSONFactoryWSTool::importDependants(const c4::yml::NodeRef& n);

///////////////////////////////////////////////////////////////////////////////////////////////////////
// importing functions
template<> void RooJSONFactoryWSTool::importFunctions(const c4::yml::NodeRef& n) {
  // import a list of RooAbsReal objects
  if(!n.is_map()) return;
  for(const auto& p:n.children()){
    // some preparations: what type of function are we dealing with here?
    std::string name(::key(p));
    if(name.size() == 0) continue;
    if(this->_workspace->pdf(name.c_str())) continue;    
    if(!p.is_map()){
      coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") node '" << name << "' is not a map, skipping." << std::endl;
      continue;
    }
    std::string prefix = ::genPrefix(p,true);
    if(prefix.size() > 0) name = prefix+name;    
    if(!p.has_child("type")){
      coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") no type given for '" << name << "', skipping." << std::endl;
      continue;
    }
    std::string functype(::val_s(p["type"]));
    this->importDependants(p);    
    // check for specific implementations
    auto it = _importers<c4::yml::NodeRef>.find(functype);
    if(it != _importers<c4::yml::NodeRef>.end()){
      try {
        if(!it->second->importFunction(this->_workspace,p)){
          coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") importer for type " << functype << " does not import functions!" << std::endl;          
        }
      } catch (const std::exception& ex){
        coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") " << ex.what() << ". skipping." << std::endl;
      }
    } else { // generic import using the factory expressions
      auto expr = _rymlFuncFactoryExpressions.find(functype);
      if(expr != _rymlFuncFactoryExpressions.end()){
        std::string expression = expr->second.generate(p);
        if(!this->_workspace->factory(expression.c_str())){
          coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") failed to create " << expr->second.tclass->GetName() << " '" << name << "', skipping. expression was\n"
                                << expression << std::endl;          
        }
      } else {
        coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") no handling for functype '" << functype << "' implemented, skipping." << "\n"
                              << "there are several possible reasons for this:\n"
                              << " 1. " << functype << " is a custom type that is not available in RooFit.\n"
                              << " 2. " << functype << " is a ROOT class that nobody ever bothered to write a deserialization definition for.\n"
                              << " 3. something is wrong with your setup, e.g. you might have called RooJSONFactoryWSTool::clearFactoryExpressions() and/or never successfully read a file defining these expressions with RooJSONFactoryWSTool::loadFactoryExpressions(filename)\n"
                              << "either way, please make sure that:\n"
                              << " 3: you are reading a file with export keys - call RooJSONFactoryWSTool::printFactoryExpressions() to see what is available\n" 
                              << " 2 & 1: you might need to write a serialization definition yourself. check INSERTLINKHERE to see how to do this!" << std::endl ;
          continue;        
      }
    }
    RooAbsReal* func = this->_workspace->function(name.c_str());
    if(!func){
      coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") something went wrong importing function '" << name << "'." << std::endl;      
    } else {
      ::importAttributes(func,p);
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// importing pdfs
template<> void RooJSONFactoryWSTool::importPdfs(const c4::yml::NodeRef& n) {
  // import a list of RooAbsPdf objects
  if(!n.is_map()) return;  
  for(const auto& p:n.children()){
    // general preparations: what type of pdf should we build?
    std::string name(::key(p));
    if(name.size() == 0) continue;
    if(this->_workspace->pdf(name.c_str())) continue;    
    if(!p.is_map()){
      coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") node '" << name << "' is not a map, skipping." << std::endl;
      continue;
    }
    std::string prefix = ::genPrefix(p,true);
    if(prefix.size() > 0) name = prefix+name;
    if(!p.has_child("type")){
      coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") no type given for '" << name << "', skipping." << std::endl;
      continue;
    }
    bool toplevel = false;
    if(p.has_child("dict")){
      auto dict = p["dict"];
      if(dict.has_child("toplevel") && ::val_b(dict["toplevel"])){
        toplevel = true;
      }
    }    
    std::string pdftype(::val_s(p["type"]));
    this->importDependants(p);

    // check for specific implementations
    auto it = _importers<c4::yml::NodeRef>.find(pdftype);
    if(it != _importers<c4::yml::NodeRef>.end()){
      try {
        if(!it->second->importPdf(this->_workspace,p)){
          coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") importer for type " << pdftype << " does not import pdfs!" << std::endl;          
        }
      } catch (const std::exception& ex){
        coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") " << ex.what() << ". skipping." << std::endl;
      }
    } else { // default implementation using the factory expressions
      auto expr = _rymlPdfFactoryExpressions.find(pdftype);
      if(expr != _rymlPdfFactoryExpressions.end()){
        std::string expression = expr->second.generate(p);
        if(!this->_workspace->factory(expression.c_str())){
          coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") failed to create " << expr->second.tclass->GetName() << " '" << name << "', skipping. expression was\n"
                                << expression << std::endl;
        }
      } else {
        coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") no handling for pdftype '" << pdftype << "' implemented, skipping." << "\n"
                              << "there are several possible reasons for this:\n"
                              << " 1. " << pdftype << " is a custom type that is not available in RooFit.\n"
                              << " 2. " << pdftype << " is a ROOT class that nobody ever bothered to write a deserialization definition for.\n"
                              << " 3. something is wrong with your setup, e.g. you might have called RooJSONFactoryWSTool::clearFactoryExpressions() and/or never successfully read a file defining these expressions with RooJSONFactoryWSTool::loadFactoryExpressions(filename)\n"
                              << "either way, please make sure that:\n"
                              << " 3: you are reading a file with export keys - call RooJSONFactoryWSTool::printFactoryExpressions() to see what is available\n" 
                              << " 2 & 1: you might need to write a serialization definition yourself. check INSERTLINKHERE to see how to do this!" << std::endl;
        continue;
      }
    }
    // post-processing: make sure that the pdf has been created, and attach needed attributes
    RooAbsPdf* pdf = this->_workspace->pdf(name.c_str());
    if(!pdf){
      coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") something went wrong importing pdf '" << name << "'." << std::endl;      
    } else {
      ::importAttributes(pdf,p);
      if(toplevel){
        // if this is a toplevel pdf, also cereate a modelConfig for it
        std::string mcname = name+"_modelConfig";
        RooStats::ModelConfig* mc = new RooStats::ModelConfig(mcname.c_str(),name.c_str());
        this->_workspace->import(*mc);
        RooStats::ModelConfig* inwsmc = dynamic_cast<RooStats::ModelConfig*>(this->_workspace->obj(mcname.c_str()));
        if(inwsmc){
          inwsmc->SetWS(*(this->_workspace));
          inwsmc->SetPdf(*pdf);
        } else {
          coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") object '" << mcname << "' in workspace is not of type RooStats::ModelConfig!" << std::endl;        
        }
      }
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// importing variables
template<> void RooJSONFactoryWSTool::importVariables(const c4::yml::NodeRef& n) {
  // import a list of RooRealVar objects
  if(!n.is_map()) return;  
  for(const auto& p:n.children()){
    std::string name(::key(p));
    if(this->_workspace->var(name.c_str())) continue;
    if(!p.is_map()){
      coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") node '" << name << "' is not a map, skipping." << std::endl;
      continue;
    }
    double val(p.has_child("value") ? ::val_d(p["value"]) : 1.);
    RooRealVar v(name.c_str(),name.c_str(),val);
    if(p.has_child("min"))    v.setMin     (::val_d(p["min"]));
    if(p.has_child("max"))    v.setMax     (::val_d(p["max"]));
    if(p.has_child("nbins"))  v.setBins     (::val_i(p["nbins"]));    
    if(p.has_child("relErr")) v.setError   (v.getVal()*::val_d(p["relErr"]));
    if(p.has_child("err"))    v.setError   (           ::val_d(p["err"]));
    if(p.has_child("const"))  v.setConstant(::val_b(p["const"]));
    else v.setConstant(false);
    ::importAttributes(&v,p);    
    this->_workspace->import(v);
  }  
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// obtain a const pointer from the string cache
const char* RooJSONFactoryWSTool::incache(const std::string& str){
  auto it = std::find(RooJSONFactoryWSTool::_strcache.begin(),RooJSONFactoryWSTool::_strcache.end(),str);
  if(it == RooJSONFactoryWSTool::_strcache.end()){
    size_t idx = RooJSONFactoryWSTool::_strcache.size();
    RooJSONFactoryWSTool::_strcache.push_back(str);
    return RooJSONFactoryWSTool::_strcache[idx].c_str();
  } else {
    return it->c_str();
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// clear the string cache
void RooJSONFactoryWSTool::clearcache(){
  RooJSONFactoryWSTool::_strcache.clear();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// export all dependants (servers) of a RooAbsArg
template<> void RooJSONFactoryWSTool::exportDependants(RooAbsArg* source, c4::yml::NodeRef& n) {
  // export all the servers of a given RooAbsArg
  auto servers(source->servers());
  RooArgSet pdfset;
  RooArgSet functionset;
  RooArgSet variableset;  
  for(auto s:servers){
    if(s->InheritsFrom(RooConstVar::Class())){
      // for RooConstVar, name and value are the same, so we don't need to do anything
      continue;
    }
    if(s->InheritsFrom(RooRealVar::Class())){
      variableset.add(*s);
      continue;
    }
    this->exportDependants(s,n);
    if(s->InheritsFrom(RooAbsPdf::Class())){
      pdfset.add(*s);
    } else {
      functionset.add(*s);
    }
  }
  auto vars = n["variables"];
  vars |= c4::yml::MAP;  
  Helpers::exportVariables(variableset,vars);
  
  auto funcs = n["functions"];
  funcs |= c4::yml::MAP;    
  Helpers::exportFunctions(functionset,funcs);
  
  auto pdfs = n["pdfs"];
  pdfs |= c4::yml::MAP;    
  Helpers::exportFunctions(pdfset,pdfs);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////
// import all dependants (servers) of a node
template<> void RooJSONFactoryWSTool::importDependants(const c4::yml::NodeRef& n) {
  // import all the dependants of an object
  if(n.has_child("variables")){
    auto vars = n["variables"];
    this->importVariables(vars);
  }
  if(n.has_child("functions")){
    auto funcs = n["functions"];
    this->importFunctions(funcs);
  }    
  if(n.has_child("pdfs")){
    auto pdfs = n["pdfs"];    
    this->importPdfs(pdfs);
  }
}

#endif

template<> void RooJSONFactoryWSTool::exportAll( c4::yml::NodeRef& n) {
  // export all ModelConfig objects and attached Pdfs
  RooArgSet main;
  for(auto obj:this->_workspace->allGenericObjects()){
    if(obj->InheritsFrom(RooStats::ModelConfig::Class())){
      RooStats::ModelConfig* mc = static_cast<RooStats::ModelConfig*>(obj);
      RooAbsPdf* pdf = mc->GetPdf();
      this->exportDependants(pdf,n);
      main.add(*pdf);     
    }
  }
  for(auto obj:this->_workspace->allPdfs()){
    RooAbsPdf* pdf = dynamic_cast<RooAbsPdf*>(obj);
    if(!pdf) continue;
    if((pdf->getAttribute("toplevel") || pdf->clients().size()==0) && !main.find(*pdf)){
      this->exportDependants(pdf,n);
      main.add(*pdf);
    }
  }
  if(main.size() > 0){
    auto pdfs = n["pdfs"];
    pdfs |= c4::yml::MAP;  
    RooJSONFactoryWSTool::Helpers::exportFunctions(main,pdfs);
    for(auto pdf:main){
      auto node = pdfs[c4::to_csubstr(pdf->GetName())];
      auto dict = node["dict"];
      dict |= c4::yml::MAP;
      dict["toplevel"] << true;
    }
  } else {
    std::cerr << "no ModelConfig found in workspace and no pdf identified as toplevel by 'toplevel' attribute or an empty client list. nothing exported!" << std::endl;    
  }
}

Bool_t RooJSONFactoryWSTool::exportJSON( std::ostream& os ) {
  // export the workspace in JSON
#ifdef INCLUDE_RYML  
  ryml::Tree t;
  c4::yml::NodeRef n = t.rootref();
  n |= c4::yml::MAP;

  os << t;
  return true;
#else
  std::cerr << "JSON export only support with rapidyaml!" << std::endl;
  return false;
#endif
}
Bool_t RooJSONFactoryWSTool::exportJSON( const char* filename ) {
  // export the workspace in JSON  
  std::ofstream out(filename);
  return this->exportJSON(out);
}

Bool_t RooJSONFactoryWSTool::exportYML( std::ostream& os ) {
  // export the workspace in YML  
#ifdef INCLUDE_RYML  
  ryml::Tree t;
  c4::yml::NodeRef n = t.rootref();
  n |= c4::yml::MAP;
  this->exportAll(n);
  os << t;
  return true;
#else
  std::cerr << "YAML export only support with rapidyaml!" << std::endl;
  return false;
#endif
}
Bool_t RooJSONFactoryWSTool::exportYML( const char* filename ) {
  // export the workspace in YML    
  std::ofstream out(filename);
  return this->exportYML(out);
}

void RooJSONFactoryWSTool::prepare(){
  gROOT->ProcessLine("using namespace RooStats::HistFactory;");
}

Bool_t RooJSONFactoryWSTool::importJSON( std::istream& is ) {
  // import a JSON file to the workspace
#ifdef INCLUDE_RYML
  std::string s(std::istreambuf_iterator<char>(is), {});
  ryml::Tree t = c4::yml::parse(c4::to_csubstr(s.c_str()));  
  c4::yml::NodeRef n = t.rootref();
  this->prepare();
  this->importDependants(n);
  return true;
#else
  std::cerr << "JSON import only support with rapidyaml!" << std::endl;
  return false;
#endif
}
Bool_t RooJSONFactoryWSTool::importJSON( const char* filename ) {
  // import a JSON file to the workspace  
  std::ifstream infile(filename);
  return this->importJSON(infile);
}

Bool_t RooJSONFactoryWSTool::importYML( std::istream& is ) {
  // import a YML file to the workspace  
#ifdef INCLUDE_RYML
  std::string s(std::istreambuf_iterator<char>(is), {});
  ryml::Tree t = c4::yml::parse(c4::to_csubstr(s.c_str()));    
  c4::yml::NodeRef n = t.rootref();
  this->prepare();
  this->importDependants(n);  
  return true;
#else
  std::cerr << "YAML import only support with rapidyaml!" << std::endl;
  return false;
#endif
}
Bool_t RooJSONFactoryWSTool::importYML( const char* filename ) {
  // import a YML file to the workspace    
  std::ifstream out(filename);
  return this->importYML(out);
}

RooJSONFactoryWSTool::RooJSONFactoryWSTool(RooWorkspace& ws) : _workspace(&ws){
  // default constructor
}


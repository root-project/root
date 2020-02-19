#include <RooStats/HistFactory/RooJSONFactoryWSTool.h>

#include <iostream>
#include <fstream>

#include <RooConstVar.h>
#include <RooCategory.h>
#include <RooRealVar.h>
#include <RooDataHist.h>
#include <RooHistFunc.h>
#include <RooRealSumPdf.h>
#include <RooProdPdf.h>
#include <RooSimultaneous.h>
#include <RooPoisson.h>
#include <RooConstraintSum.h>
#include <RooProduct.h>
#include <RooStats/ModelConfig.h>
#include <RooStats/HistFactory/ParamHistFunc.h>
#include <RooStats/HistFactory/PiecewiseInterpolation.h>
#include <RooStats/HistFactory/FlexibleInterpVar.h>

#include "TROOT.h"

std::vector<std::string> RooJSONFactoryWSTool::_strcache = std::vector<std::string>(1000);

#ifdef INCLUDE_RYML
#include <ryml.hpp>
#include <c4/yml/std/map.hpp>
#include <c4/yml/std/string.hpp>

template<> std::map<std::string,const RooJSONFactoryWSTool::Importer<c4::yml::NodeRef>*> RooJSONFactoryWSTool::_importers<c4::yml::NodeRef> = std::map<std::string,const RooJSONFactoryWSTool::Importer<c4::yml::NodeRef>*>();
template<> std::map<const TClass*,const RooJSONFactoryWSTool::Exporter<c4::yml::NodeRef>*> RooJSONFactoryWSTool::_exporters<c4::yml::NodeRef> = std::map<const TClass*,const RooJSONFactoryWSTool::Exporter<c4::yml::NodeRef>*>();

namespace c4 { namespace yml {
    template<class T> void read(c4::yml::NodeRef const& n, std::vector<T> *v){
      for(size_t i=0; i<n.num_children(); ++i){
        T e;
        n[i]>>e;
        v->push_back(e);
      }
    }
    
    template<class T> void write(c4::yml::NodeRef *n, std::vector<T> const& v){
      *n |= c4::yml::SEQ;
      for(auto e:v){
        n->append_child() << e;
      }
    }
  }
}

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
  inline std::vector<std::vector<int> > genIndices(RooArgList& vars){
    std::vector<std::vector<int> > combinations;
    ::genIndicesHelper(combinations,vars,0);
    return combinations;
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

  inline std::map<std::string,c4::yml::NodeRef> readVars(const c4::yml::NodeRef& n,const std::string& obsnamecomp){
    std::map<std::string,c4::yml::NodeRef> vars;
    if(!n.is_map()) return vars;
    if(!n.has_child("binning")) throw "no binning given";
    c4::yml::NodeRef bounds(n["binning"]);
    if(!bounds.is_map()) return vars;    
    if(bounds.has_child("nbins")){
      vars["obs_x_"+obsnamecomp] = bounds;
    } else {
      for(c4::yml::NodeRef p:bounds.children()){
        vars[::key(p)] = p;      
      }
    }
    return vars;
  }
  inline void collectNames(const c4::yml::NodeRef& n,std::vector<std::string>& names){
    for(auto c:n.children()){
      names.push_back(::key(c));
    }
  }
  
  inline void collectObsNames(const c4::yml::NodeRef& n,std::vector<std::string>& obsnames,const std::string& obsnamecomp){
    auto vars = ::readVars(n,obsnamecomp);
    if(obsnames.size() == 0){
      for(auto it:vars){
        obsnames.push_back(it.first);
      }
    }
    if(vars.size() != obsnames.size()){
      throw "inconsistent number of variabels";
    }
  }
  
  inline void stackError(const c4::yml::NodeRef& n,std::vector<double>& sumW,std::vector<double>& sumW2){
    if(!n.is_map()) return;    
    if(!n.has_child("counts")) throw "no counts given";
    if(!n.has_child("errors")) throw "no errors given";    
    if(n["counts"].num_children() != n["errors"].num_children()){
      throw "inconsistent bin numbers";
    }
    const size_t nbins = n["counts"].num_children();
    for(size_t ibin=0; ibin<nbins; ++ibin){
      double w = ::val_d(n["counts"][ibin]);
      double e = ::val_d(n["errors"][ibin]);
      if(ibin<sumW.size()) sumW[ibin] += w;
      else sumW.push_back(w);
      if(ibin<sumW2.size()) sumW2[ibin] += e*e;
      else sumW2.push_back(e*e);
    }
  }

  void error(const char* s){
    throw std::runtime_error(s);
  }
  void error(const std::string& s){
    throw std::runtime_error(s);
  }  
  
  inline RooDataHist* readData(RooWorkspace* ws, const c4::yml::NodeRef& n,const std::string& namecomp,const std::string& obsnamecomp){
    if(!n.is_map()) throw "data is not a map!";
    auto vars = ::readVars(n,obsnamecomp);
    RooArgList varlist;
    for(auto v:vars){
      std::string name(v.first);
      if(ws->var(name.c_str())){
        varlist.add(*(ws->var(name.c_str())));
      } else {
        auto& val = v.second;
        if(!val.is_map()) error("variable is not a map!");          
        if(!val.has_child("nbins")) error("no nbins given");
        if(!val.has_child("min"))   error("no min given");
        if(!val.has_child("max"))   error("no max given");
        double nbins = ::val_i(val["nbins"]);      
        double min   = ::val_d(val["min"]);
        double max   = ::val_d(val["max"]);
        RooRealVar* var = new RooRealVar(name.c_str(),name.c_str(),min);
        var->setMin(min);
        var->setMax(max);
        var->setConstant(true);
        var->setBins(nbins);
        varlist.addOwned(*var);
      }
    }
    RooDataHist* dh = new RooDataHist(("dataHist_"+namecomp).c_str(),namecomp.c_str(),varlist);
    auto bins = genIndices(varlist);
    if(!n.has_child("counts")) error("no counts given");
    auto counts = n["counts"];
    if(counts.num_children() != bins.size()) error(TString::Format("inconsistent bin numbers: counts=%d, bins=%d",(int)counts.num_children(),(int)(bins.size())));
    for(size_t ibin=0; ibin<bins.size(); ++ibin){
      for(size_t i = 0; i<bins[ibin].size(); ++i){
        RooRealVar* v = (RooRealVar*)(varlist.at(i));
        v->setVal(v->getBinning().binCenter(bins[ibin][i]));
      }
      dh->add(varlist,::val_d(counts[ibin]));
    }
    return dh;
  }
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
        if(!p.has_child(c4::to_csubstr(k))){
          std::stringstream err;
          err << "factory expression for class '" << this->tclass->GetName() << "', which expects key '" << k << "' missing from input for object '" << name << "', skipping.";
          error(err.str().c_str());
        }
        if(!first) expression << ",";
        first = false;
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
  struct RYML_Export_Keys {
    std::string type;
    std::map<std::string,std::string> proxies;
  };  
  std::map<std::string,RYML_Factory_Expression> _rymlPdfFactoryExpressions;
  std::map<std::string,RYML_Factory_Expression> _rymlFuncFactoryExpressions;
  std::map<TClass*,RYML_Export_Keys> _rymlExportKeys;

  void importAttributes(RooAbsArg* arg, const c4::yml::NodeRef& n){
    if(!n.is_map()) return;
    if(n.has_child("dict")){
      for(auto attr:n["dict"]){
        arg->setStringAttribute(::key(attr).c_str(),::val_s(attr).c_str());
      }
    }
  }
  void writeAxis(c4::yml::NodeRef& bounds, const TAxis& ax){
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


template<> void RooJSONFactoryWSTool::importDependants(const c4::yml::NodeRef& n);

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
          coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") failed to create " << expr->second.tclass->GetName() << " '" << name << "', skipping." << std::endl;
        }
      } else {
        coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") no handling for functype '" << functype << "' implemented, skipping." << std::endl;
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



class RooHistogramFactory : public RooJSONFactoryWSTool::Importer<c4::yml::NodeRef> {
public:
  virtual bool importFunction(RooWorkspace* ws, const c4::yml::NodeRef& p) const override {
    std::string name(::key(p));
    std::string prefix = ::genPrefix(p,true);
    if(prefix.size() > 0) name = prefix+name;    
    if(!p.has_child("data")){
      error("function '" + name + "' is of histogram type, but does not define a 'data' key");
    }
    try {
      RooArgSet prodElems;
      RooDataHist* dh = ::readData(ws,p["data"],name,prefix);       
      RooHistFunc* hf = new RooHistFunc(name.c_str(),::key(p).c_str(),*(dh->get()),*dh);          
      if(p.has_child("normfactors")){
        for(auto nf:p["normfactors"].children()){
          std::string nfname(::val_s(nf));
          RooAbsReal* r = ws->var(nfname.c_str());
          if(!r){
            error("unable to find normalization factor '" + nfname + "'");
          } else {
            prodElems.add(*r);
          }
        }
      }
      if(p.has_child("overallSystematics")){
        RooArgList nps;
        std::vector<double> low;
          std::vector<double> high;
          for(auto sys:p["overallSystematics"].children()){
            std::string sysname(::key(sys));
            std::string parname(::val_s(sys["parameter"]));
            RooAbsReal* par = ws->var(parname.c_str());
            RooAbsPdf* pdf = ws->pdf(sysname.c_str());
            if(!par){
              error("unable to find nuisance parameter '" + parname + "'.");
            } else if(!pdf){
              error("unable to find pdf '" + sysname + "'.");
            } else {
              nps.add(*par);
              low.push_back(::val_d(sys["low"]));
              high.push_back(::val_d(sys["high"]));
            }
          }
          RooStats::HistFactory::FlexibleInterpVar* v = new RooStats::HistFactory::FlexibleInterpVar(("overallSys_"+name).c_str(),("overallSys_"+name).c_str(),nps,1.,low,high);
          prodElems.addOwned(*v);
        }
        if(p.has_child("histogramSystematics")){
          RooArgList nps;
          RooArgList low;
          RooArgList high;            
          for(auto sys:p["histogramSystematics"].children()){
            std::string sysname(::key(sys));
            std::string parname(::val_s(sys["parameter"]));            
            RooAbsReal* par = ws->var(parname.c_str());
            RooAbsPdf* pdf = ws->pdf(sysname.c_str());
            if(!par){
              error("unable to find nuisance parameter '" + parname + "'");
            } else if(!pdf){
              error("unable to find pdf '" + sysname + "'.");
            } else {
              nps.add(*par);
              RooDataHist* dh_low = ::readData(ws,p["dataLow"],sysname+"Low_"+name,prefix);
              RooHistFunc hf_low((sysname+"Low_"+name).c_str(),::key(p).c_str(),*(dh_low->get()),*dh_low);              
              low.add(hf_low);
              RooDataHist* dh_high = ::readData(ws,p["dataHigh"],sysname+"High_"+name,prefix);
              RooHistFunc hf_high((sysname+"High_"+name).c_str(),::key(p).c_str(),*(dh_high->get()),*dh_high);              
              high.add(hf_high);              
            }
          }
          PiecewiseInterpolation* v = new PiecewiseInterpolation(("histoSys_"+name).c_str(),("histoSys_"+name).c_str(),*hf,nps,low,high,true);
          prodElems.addOwned(*v);
        }
        if(prodElems.size() > 0){
          hf->SetName(("hist_"+name).c_str());
          prodElems.addOwned(*hf);          
          RooProduct prod(name.c_str(),name.c_str(),prodElems);
          ws->import(prod);
        } else {
          ws->import(*hf);
        }
      } catch (const std::runtime_error& e){
      error("function '" + name + "' is of histogram type, but 'data' is not a valid definition. " + e.what() + ".");
    }
    return true;
  }
};
namespace {
  bool _roohistogramfactory = RooJSONFactoryWSTool::registerImporter("histogram",new RooHistogramFactory());
}

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
namespace {
  bool _rooprodpdffactory = RooJSONFactoryWSTool::registerImporter("pdfprod",new RooProdPdfFactory());
}

  
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
 namespace {
   bool _roosimultaneousfactory = RooJSONFactoryWSTool::registerImporter("simultaneous",new RooSimultaneousFactory());
 }
 
class RooRealSumPdfFactory : public RooJSONFactoryWSTool::Importer<c4::yml::NodeRef> {
public:
  virtual bool importPdf(RooWorkspace* ws, const c4::yml::NodeRef& p) const override {
    std::string name(::key(p));
    RooArgList funcs;
    RooArgList coefs;
    if(!p.has_child("sum")){
      error("no sum components of '" + name + "', skipping.");
    }
    RooArgList constraints;
    RooArgList nps;      
    RooConstVar* c = new RooConstVar("1","1",1.);
    std::vector<std::string> usesStatError;
    double statErrorThreshold = 0;
    if(p.has_child("statError")){
      auto staterr = p["statError"];
      statErrorThreshold = ::val_d(staterr["relThreshold"]);      
      std::string constraint = ::val_s(staterr["constraint"]);
      std::vector<double> relValues;
      if(staterr.has_child("stack")){
        for(auto comp:staterr["stack"]){
          std::string elem = ::val_s(comp);
          usesStatError.push_back(elem);
        }
      }
    }
    std::vector<double> sumW;
    std::vector<double> sumW2;
    std::vector<std::string> obsnames;
    std::vector<std::string> sysnames;            
    for(auto& comp:p["sum"]){
      std::string fprefix;
      std::string fname(::val_s(comp));
      if(p.has_child("functions")){
        auto funcdefs = p["functions"];
        if(funcdefs.has_child(c4::to_csubstr(fname.c_str()))){
          auto def = funcdefs[c4::to_csubstr(fname.c_str())];
          fprefix = ::genPrefix(def,true);
          if(::val_s(def["type"]) == "histogram"){
            try {
              ::collectObsNames(def["data"],obsnames,name);
              if(def.has_child("overallSystematics"))  ::collectNames(def["overallSystematics"],sysnames);
              if(def.has_child("histogramSystematics"))::collectNames(def["histogramSystematics"],sysnames);                                
            } catch (const char* s){
              error("function '" + name + "' unable to collect observables from function " + fname + ". " + s );
            }
            if(std::find(usesStatError.begin(),usesStatError.end(),fname) != usesStatError.end()){
              try {              
                ::stackError(def["data"],sumW,sumW2);
              } catch (const char* s){                
                error("function '" + name + "' unable to sum statError from function " + fname + ". " + s );
              }                
            }
          }
        }
      }
      RooAbsReal* func = ws->function((fprefix+fname).c_str());
      if(!func){
        error("unable to obtain component '" + fprefix+fname + "' of '" + name + "'");
      }
      funcs.add(*func);
    }
    RooArgList observables;
    for(auto& obsname:obsnames){
      RooRealVar* obs = ws->var(obsname.c_str());
      if(!obs){
        error("unable to obtain observable '" + obsname + "' of '" + name + "',");
      }
      observables.add(*obs);
    }
    ParamHistFunc* phf = NULL;
    if(usesStatError.size() > 0){
      RooArgList gammas;
      for(size_t i=0; i<sumW.size(); ++i){
        TString gname = TString::Format("gamma_stat_%s_bin_%d",name.c_str(),(int)i);
        TString tname = TString::Format("tau_stat_%s_bin_%d",name.c_str(),(int)i);
        TString prodname = TString::Format("nExp_stat_%s_bin_%d",name.c_str(),(int)i);
        TString poisname = TString::Format("Constraint_stat_%s_bin_%d",name.c_str(),(int)i);                        
        double tauCV = sumW2[i];
        double err = sqrt(sumW2[i])/sumW[i];
        RooRealVar* g = new RooRealVar(gname.Data(),gname.Data(),1.);
        if(err < statErrorThreshold) g->setConstant(true);
        RooRealVar* tau = new RooRealVar(tname.Data(),tname.Data(),tauCV);
        RooArgSet elems;
        elems.add(*g);
        elems.add(*tau);
        g->setError(err);
        RooProduct* prod = new RooProduct(prodname.Data(),prodname.Data(),elems);
        RooPoisson* pois = new RooPoisson(poisname.Data(),poisname.Data(),*prod,*tau);;        
        gammas.add(*g,true);
        constraints.add(*pois,true);
      }
      nps.add(gammas);
      phf = new ParamHistFunc(TString::Format("%s_mcstat",name.c_str()),"staterror",observables,gammas);
      phf->recursiveRedirectServers(observables);
    }
    for(auto& comp:p["sum"]){
      std::string fname(::val_s(comp));        
      if(std::find(usesStatError.begin(),usesStatError.end(),fname) != usesStatError.end()){
        coefs.add(*phf);
      } else {
        coefs.add(*c);
      }
    }
    for(auto& np:nps){
      for(auto client:np->clients()){
        if(client->InheritsFrom(RooAbsPdf::Class()) && !constraints.find(*client)){
          constraints.add(*client);
        }
      }
    }
    for(auto sysname:sysnames){
      RooAbsPdf* pdf = ws->pdf(sysname.c_str());
      if(pdf){
        constraints.add(*pdf);
      } else {
        error("unable to find constraint term '" + sysname + "'");
      }
    }
    if(constraints.getSize() == 0){
      RooRealSumPdf sum(name.c_str(),name.c_str(),funcs,coefs);
      ws->import(sum);
    } else {
      RooRealSumPdf sum((name+"_model").c_str(),name.c_str(),funcs,coefs);
      constraints.add(sum);
      RooProdPdf prod(name.c_str(),name.c_str(),constraints);
      ws->import(prod);        
    }
    return true;
  }
};

namespace {
  bool _roorealsumpdffactory = RooJSONFactoryWSTool::registerImporter("sum",new RooRealSumPdfFactory());
}

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
          coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") failed to create " << expr->second.tclass->GetName() << " '" << name << "', skipping." << std::endl;
        }
      } else {
        coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") no handling for pdftype '" << pdftype << "' available, skipping." << std::endl;
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

class RooJSONFactoryWSTool::Helpers {
public:
  
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
};

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
namespace {
  bool _roosimultaneousstreamer = RooJSONFactoryWSTool::registerExporter(RooSimultaneous::Class(),new RooSimultaneousStreamer());
}

class RooHistFuncStreamer : public RooJSONFactoryWSTool::Exporter<c4::yml::NodeRef> {
public:
  virtual bool exportObject(RooAbsReal* func, c4::yml::NodeRef& elem) const override {
    RooHistFunc* hf = static_cast<RooHistFunc*>(func);
    const RooDataHist& dh = hf->dataHist();
    elem["type"] << "histogram";
    RooArgList vars(*dh.get());
    TH1* hist = func->createHistogram(RooJSONFactoryWSTool::Helpers::concat(&vars).c_str());
    auto data = elem["data"];
    RooJSONFactoryWSTool::exportHistogram(*hist,data,RooJSONFactoryWSTool::Helpers::names(&vars));
    delete hist;
    return true;
  }
};
namespace {
  bool _roohistfuncstreamer = RooJSONFactoryWSTool::registerExporter(RooHistFunc::Class(),new RooHistFuncStreamer());
}

class FlexibleInterpVarStreamer : public RooJSONFactoryWSTool::Exporter<c4::yml::NodeRef> {
public:
  virtual bool exportObject(RooAbsReal* func, c4::yml::NodeRef& elem) const override {
    RooStats::HistFactory::FlexibleInterpVar* fip = static_cast<RooStats::HistFactory::FlexibleInterpVar*>(func);
    elem["type"] << "interpolation0d";        
    auto vars = elem["vars"];
    vars |= c4::yml::SEQ;        
    for(const auto& v:fip->variables()){
      vars.append_child() << v->GetName();
    }
    elem["nom"] << fip->nominal();                
    elem["high"] << fip->high();        
    elem["low"] << fip->low();
    return true;
  }
};
namespace {
  bool _flexibleinterpvarstreamer = RooJSONFactoryWSTool::registerExporter(RooStats::HistFactory::FlexibleInterpVar::Class(),new FlexibleInterpVarStreamer());
}

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

void RooJSONFactoryWSTool::clearcache(){
  RooJSONFactoryWSTool::_strcache.clear();
}


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

void RooJSONFactoryWSTool::clearFactoryExpressions(){
  // clear all factory expressions
#ifdef INCLUDE_RYML    
  _rymlPdfFactoryExpressions.clear();
  _rymlFuncFactoryExpressions.clear();
#endif
}
void RooJSONFactoryWSTool::clearExportKeys(){
  // clear all export keys
#ifdef INCLUDE_RYML    
  _rymlExportKeys.clear();
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


template<> void RooJSONFactoryWSTool::exportAll( c4::yml::NodeRef& n) {
  // export all ModelConfig objects and attached Pdfs
  RooArgSet main;
  for(auto obj:this->_workspace->allGenericObjects()){
    if(obj->InheritsFrom(RooStats::ModelConfig::Class())){
      RooStats::ModelConfig* mc = dynamic_cast<RooStats::ModelConfig*>(obj);
      RooAbsPdf* pdf = mc->GetPdf();
      this->exportDependants(pdf,n);
      main.add(*pdf);     
    }
  }
  auto pdfs = n["pdfs"];
  pdfs |= c4::yml::MAP;  
  RooJSONFactoryWSTool::Helpers::exportFunctions(main,pdfs);
  for(auto pdf:main){
    auto node = pdfs[c4::to_csubstr(pdf->GetName())];
    auto dict = node["dict"];
    dict |= c4::yml::MAP;
    dict["toplevel"] << true;
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


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

std::vector<std::string> RooJSONFactoryWSTool::_strcache = std::vector<std::string>();

#ifdef INCLUDE_RYML
#include <ryml.hpp>
#include <c4/yml/std/map.hpp>
#include <c4/yml/std/string.hpp>

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
    if(p.has_child("namespaces")){
      for(auto ns:p["namespaces"]){
        if(prefix.size() > 0) prefix+="_";
        prefix += ::val_s(ns);
      }
    }
    if(trailing_underscore) prefix += "_";
    return prefix;
  }

  inline std::map<std::string,c4::yml::NodeRef> readVars(const c4::yml::NodeRef& n,const std::string& obsnamecomp){
    std::map<std::string,c4::yml::NodeRef> vars;
    if(!n.has_child("binning")) throw "no binning given";
    c4::yml::NodeRef bounds(n["binning"]);
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
  
  inline RooDataHist* readData(const c4::yml::NodeRef& n,const std::string& namecomp,const std::string& obsnamecomp){
    auto vars = ::readVars(n,obsnamecomp);
    RooArgList varlist;
    for(auto v:vars){
      std::string name(v.first);
      auto& val = v.second;
      if(!val.has_child("nbins")) throw "no nbins given";
      if(!val.has_child("min"))   throw "no min given";
      if(!val.has_child("max"))   throw "no max given";
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
    RooDataHist* dh = new RooDataHist(("dataHist_"+namecomp).c_str(),namecomp.c_str(),varlist);
    auto bins = genIndices(varlist);
    if(!n.has_child("counts")) throw "no counts given";
    auto counts = n["counts"];
    if(counts.num_children() != bins.size()) throw "inconsistent bin numbers";
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
  };
  struct RYML_Export_Keys {
    std::string type;
    std::map<std::string,std::string> proxies;
  };  
  std::map<std::string,RYML_Factory_Expression> _rymlPdfFactoryExpressions;
  std::map<std::string,RYML_Factory_Expression> _rymlFuncFactoryExpressions;
  std::map<TClass*,RYML_Export_Keys> _rymlExportKeys;

  void importAttributes(RooAbsArg* arg, const c4::yml::NodeRef& n){
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
  RooJSONFactoryWSTool::_strcache.push_back(varnames[0]);
  auto x = bounds[ c4::to_csubstr(RooJSONFactoryWSTool::_strcache[ RooJSONFactoryWSTool::_strcache.size()-1 ])];
  writeAxis(x,*(h.GetXaxis()));
  if(h.GetDimension()>1){
    RooJSONFactoryWSTool::_strcache.push_back(varnames[1]);
    auto y = bounds[ c4::to_csubstr(RooJSONFactoryWSTool::_strcache[ RooJSONFactoryWSTool::_strcache.size()-1 ])];    
    writeAxis(y,*(h.GetYaxis()));
    if(h.GetDimension()>2){
      RooJSONFactoryWSTool::_strcache.push_back(varnames[2]);
      auto z = bounds[ c4::to_csubstr(RooJSONFactoryWSTool::_strcache[ RooJSONFactoryWSTool::_strcache.size()-1 ])];    
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
  for(const auto& p:n.children()){
    // some preparations: what type of function are we dealing with here?
    std::string prefix = ::genPrefix(p,false);
    std::string name(prefix + "_" + ::key(p));
    if(!p.has_child("type")){
      coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") no type given for '" << name << "', skipping." << std::endl;
      continue;
    }    
    std::string functype(::val_s(p["type"]));
    this->importDependants(p);    
    if(functype=="histogram"){ // special import for histfactory macros
      if(!p.has_child("data")){
        coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") function '" << name << "' is of histogram type, but does not define a 'data' key, skipping.";
        continue;
      }
      try {
        RooDataHist* dh = ::readData(p["data"],name,prefix);
        RooHistFunc* hf = new RooHistFunc(("hist_"+name).c_str(),::key(p).c_str(),*(dh->get()),*dh);
        RooArgSet prodElems;
        prodElems.addOwned(*hf);
        if(p.has_child("normfactors")){
          for(auto nf:p["normfactors"].children()){
            std::string nfname(::val_s(nf));
            RooAbsReal* r = this->_workspace->var(nfname.c_str());
            if(!r){
              coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") unable to find normalization factor '" << nfname << "', skipping." << std::endl;
              continue;
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
            RooAbsReal* par = this->_workspace->var(parname.c_str());
            RooAbsPdf* pdf = this->_workspace->pdf(sysname.c_str());
            if(!par){
              coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") unable to find nuisance parameter '" << parname << "', skipping." << std::endl;
              continue;
            } else if(!pdf){
              coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") unable to find pdf '" << sysname << "', skipping." << std::endl;
              continue;
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
            RooAbsReal* par = this->_workspace->var(parname.c_str());
            RooAbsPdf* pdf = this->_workspace->pdf(sysname.c_str());
            if(!par){
              coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") unable to find nuisance parameter '" << parname << "', skipping." << std::endl;
              continue;
            } else if(!pdf){
              coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") unable to find pdf '" << sysname << "', skipping." << std::endl;
              continue;            
            } else {
              nps.add(*par);
              RooDataHist* dh_low = ::readData(p["dataLow"],sysname+"Low_"+name,prefix);
              RooHistFunc hf_low((sysname+"Low_"+name).c_str(),::key(p).c_str(),*(dh_low->get()),*dh_low);              
              low.add(hf_low);
              RooDataHist* dh_high = ::readData(p["dataHigh"],sysname+"High_"+name,prefix);
              RooHistFunc hf_high((sysname+"High_"+name).c_str(),::key(p).c_str(),*(dh_high->get()),*dh_high);              
              high.add(hf_high);              
            }
          }
          PiecewiseInterpolation* v = new PiecewiseInterpolation(("histoSys_"+name).c_str(),("histoSys_"+name).c_str(),*hf,nps,low,high,true);
          prodElems.addOwned(*v);
        }                
        RooProduct prod(name.c_str(),name.c_str(),prodElems);
        this->_workspace->import(prod);
      } catch (const char* s){
        coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") function '" << name << "' is of histogram type, but 'data' is not a valid definition. " << s << ". skipping." << std::endl;
        continue;        
      }
    } else { // generic import using the factory expressions
      auto expr = _rymlFuncFactoryExpressions.find(functype);
      if(expr != _rymlFuncFactoryExpressions.end()){
        bool ok = true;
        bool first = true;
        std::stringstream expression;
        expression << expr->second.tclass->GetName() << "::" << name << "(";
        for(auto k:expr->second.arguments){
          if(!p.has_child(c4::to_csubstr(k))){
            coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") factory expression for '" << expr->first << "' maps to class '" << expr->second.tclass->GetName() << "', which expects key '" << k << "' missing from input for object '" << name << "', skipping." << std::endl;
            ok=false;
            break;
          }
          if(!first) expression << ",";
          first = false;
          expression << ::val_s(p[c4::to_csubstr(k)]);
        }
        expression << ")";
        if(ok){
          std::cout << expression.str() << std::endl;
          if(!this->_workspace->factory(expression.str().c_str())){
            coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") failed to create " << expr->second.tclass->GetName() << " '" << name << "', skipping." << std::endl;
          }
        }
      } else {
        coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") no handling for functype '" << functype << "' implemented, skipping." << std::endl;
      }
    }
    RooAbsReal* func = this->_workspace->function(name.c_str());
    if(!func){
      coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") something went wrong importing '" << name << "'." << std::endl;      
    } else {
      ::importAttributes(func,p);
    }
  }
}

template<> void RooJSONFactoryWSTool::importPdfs(const c4::yml::NodeRef& n) {
  // import a list of RooAbsPdf objects
  for(const auto& p:n.children()){
    // general preparations: what type of pdf should we build?
    std::string prefix = ::genPrefix(p,false);
    std::string name(::key(p));
    std::cout << name << std::endl;
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

    // a few specific implementations
    if(pdftype=="sum"){ // RooRealSumPdf
      RooArgList funcs;
      RooArgList coefs;
      if(!p.has_child("sum")){
        coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") no sum components of '" << name << "', skipping." << std::endl;
        continue;        
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
                coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") function '" << name << "' unable to collect observables from function " << fname << ". " << s << ". skipping." << std::endl;
                continue;        
              }
              if(std::find(usesStatError.begin(),usesStatError.end(),fname) != usesStatError.end()){
                try {              
                  ::stackError(def["data"],sumW,sumW2);
                } catch (const char* s){                
                  coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") function '" << name << "' unable to sum statError from function " << fname << ". " << s << ". skipping." << std::endl;
                  continue;        
                }                
              }
            }
          }
        }
        RooAbsReal* func = this->_workspace->function((fprefix+fname).c_str());
        if(!func){
          coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") unable to obtain component '" << fprefix+fname << "' of '" << name << "', skipping." << std::endl;
          continue;
        }
        funcs.add(*func);
      }
      RooArgList observables;
      for(auto& obsname:obsnames){
        RooRealVar* obs = this->_workspace->var(obsname.c_str());
        if(!obs){
          coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") unable to obtain observable '" << obsname << "' of '" << name << "', skipping." << std::endl;
          continue;
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
        RooAbsPdf* pdf = this->_workspace->pdf(sysname.c_str());
        if(pdf){
          constraints.add(*pdf);
        } else {
          coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") unable to find constraint term '" << sysname << "', skipping." << std::endl;
        }
      }
      if(constraints.getSize() == 0){
        RooRealSumPdf sum(name.c_str(),name.c_str(),funcs,coefs);
        this->_workspace->import(sum);
      } else {
        RooRealSumPdf sum((name+"_model").c_str(),name.c_str(),funcs,coefs);
        constraints.add(sum);
        RooProdPdf prod(name.c_str(),name.c_str(),constraints);
        this->_workspace->import(prod);        
      }
    } else if(pdftype=="simultaneous"){ // RooSimultaneous
      if(!p.has_child("channels")){
        coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") no channel components of '" << name << "', skipping." << std::endl;
        continue;        
      }
      std::map< std::string, RooAbsPdf *> components;
      RooCategory cat("channelCat","channelCat");
      for(auto comp:p["channels"]){
        std::string pdfname(::val_s(comp));
        RooAbsPdf* pdf = this->_workspace->pdf(pdfname.c_str());
        if(!pdf){
          coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") unable to obtain component '" << pdfname << "' of '" << name << "', skipping." << std::endl;
          continue;
        }
        components[pdfname] = pdf;
        cat.defineType(pdfname.c_str());
      }
      RooSimultaneous simpdf(name.c_str(),name.c_str(),components,cat);
      this->_workspace->import(simpdf);      
    } else { // default implementation using the factory expressions
      auto expr = _rymlPdfFactoryExpressions.find(pdftype);
      if(expr!= _rymlPdfFactoryExpressions.end()){
        bool ok = true;
        bool first = true;
        std::stringstream expression;
        expression << expr->second.tclass->GetName() << "::" << name << "(";
        for(auto k:expr->second.arguments){
          if(!p.has_child(c4::to_csubstr(k))){
            coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") factory expression for '" << expr->first << "' maps to class '" << expr->second.tclass->GetName() << "', which expects key '" << k << "' missing from input for object '" << name << "', skipping." << std::endl;
            ok=false;
            break;
          }
          if(!first) expression << ",";
          first = false;
          expression << ::val_s(p[c4::to_csubstr(k)]);
        }
        expression << ")";
        if(ok){
          std::cout << expression.str() << std::endl;          
          if(!this->_workspace->factory(expression.str().c_str())){
            coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") failed to create " << expr->second.tclass->GetName() << " '" << name << "', skipping." << std::endl;
          }
        }
      } else {
        coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") no handling for pdftype '" << pdftype << "' implemented, skipping." << std::endl;
      }
    }
    // post-processing: make sure that the pdf has been created, and attach needed attributes
    RooAbsPdf* pdf = this->_workspace->pdf(name.c_str());
    if(!pdf){
      coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") something went wrong importing '" << name << "'." << std::endl;      
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
  for(const auto& p:n.children()){
    std::string name(::key(p));
    std::cout << name << std::endl;
    if(this->_workspace->var(name.c_str())) continue;
    double val(p.has_child("value") ? ::val_d(p["value"]) : 1.);
    RooRealVar v(name.c_str(),name.c_str(),val);
    if(p.has_child("min"))    v.setMin     (::val_d(p["min"]));
    if(p.has_child("max"))    v.setMax     (::val_d(p["max"]));
    if(p.has_child("relErr")) v.setError   (v.getVal()*::val_d(p["relErr"]));
    if(p.has_child("err"))    v.setError   (           ::val_d(p["err"]));
    if(p.has_child("const"))  v.setConstant(::val_b(p["const"]));
    else v.setConstant(false);
    ::importAttributes(&v,p);    
    this->_workspace->import(v);
  }  
}

namespace {
  
  template<class T> std::string concat(const T* items, const std::string& sep=",") {
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


  template<class T> std::vector<std::string> names(const T* items) {
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

  
  void exportAttributes(const RooAbsArg* arg, c4::yml::NodeRef& n){
    // exporta ll string attributes of an object
    if(arg->stringAttributes().size() > 0){
      auto dict = n["dict"];
      dict |= c4::yml::MAP;      
      for(const auto& it:arg->stringAttributes()){
        dict[c4::to_csubstr(it.first.c_str())] << it.second;
      }
    }
  }
  
  void exportVariables(const RooArgSet& allElems, c4::yml::NodeRef& n) {
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
      if(v->getError() > 0){
        var["err"] << v->getError();
      }
      if(v->isConstant()){
        var["const"] << v->isConstant();
      }
      ::exportAttributes(arg,var);      
    }  
  }

  void exportFunctions(const RooArgSet& allElems, c4::yml::NodeRef& n){
    // export a list of functions
    // note: this function assumes that all the dependants of these objects have already been exported
    for(auto* arg:allElems){
      RooAbsReal* func = dynamic_cast<RooAbsReal*>(arg);
      if(!func) continue;

      if(n.has_child(c4::to_csubstr(func->GetName()))) continue;
      
      auto elem = n[c4::to_csubstr(func->GetName())];
      elem |= c4::yml::MAP;
      
      if(func->InheritsFrom(RooSimultaneous::Class())){
        RooSimultaneous* sim = static_cast<RooSimultaneous*>(func);
        elem["type"] << "simultaneous";
        auto pdfs = elem["pdfs"];
        pdfs |= c4::yml::MAP;
        const auto& indexCat = sim->indexCat();
        for(const auto& cat:indexCat){
          pdfs[c4::to_csubstr(cat->GetName())] << sim->getPdf(cat->GetName())->GetName();
        }
      } else if(func->InheritsFrom(RooHistFunc::Class())){
        RooHistFunc* hf = static_cast<RooHistFunc*>(func);
        const RooDataHist& dh = hf->dataHist();
        elem["type"] << "histogram";
        RooArgList vars(*dh.get());
        TH1* hist = func->createHistogram(::concat(&vars).c_str());
        auto data = elem["data"];
        RooJSONFactoryWSTool::exportHistogram(*hist,data,::names(&vars));
        delete hist;
      } else {
        TClass* cl = TClass::GetClass(func->ClassName());

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
      ::exportAttributes(func,elem);
    }
  }
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
  ::exportVariables(variableset,vars);
  
  auto funcs = n["functions"];
  funcs |= c4::yml::MAP;    
  ::exportFunctions(functionset,funcs);
  
  auto pdfs = n["pdfs"];
  pdfs |= c4::yml::MAP;    
  ::exportFunctions(pdfset,pdfs);
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
        if(arg.is_container()){
          std::stringstream ss;
          ss << "{";
          for(const auto& k:arg){
            ss << ::key(k);
          }
          ss << "}";
          ex.arguments.push_back(ss.str());
        } else {
          ex.arguments.push_back(::val_s(arg));
        }
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
  ::exportFunctions(main,pdfs);
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

Bool_t RooJSONFactoryWSTool::importJSON( std::istream& is ) {
  // import a JSON file to the workspace
#ifdef INCLUDE_RYML
  std::string s(std::istreambuf_iterator<char>(is), {});
  ryml::Tree t = c4::yml::parse(c4::to_csubstr(s.c_str()));  
  c4::yml::NodeRef n = t.rootref();
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


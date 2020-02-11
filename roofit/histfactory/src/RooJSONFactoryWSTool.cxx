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
#include <RooStats/HistFactory/ParamHistFunc.h>
#include <RooStats/HistFactory/PiecewiseInterpolation.h>
#include <RooStats/HistFactory/FlexibleInterpVar.h>

#ifdef INCLUDE_RYML
#include <ryml.hpp>
#include <c4/yml/std/map.hpp>
#include <c4/yml/std/string.hpp>
namespace {
  inline const char* key(const c4::yml::NodeRef& n){
    std::stringstream ss;
    ss << n.key();
    return ss.str().c_str();
  }
  inline const char* val_s(const c4::yml::NodeRef& n){
    std::stringstream ss;    
    ss << n.val();
    return ss.str().c_str();
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
  std::map<std::string,RYML_Factory_Expression> _rymlPdfFactoryExpressions;
  std::map<std::string,RYML_Factory_Expression> _rymlFuncFactoryExpressions;  
  
}


template<> void RooJSONFactoryWSTool::importDependants(const c4::yml::NodeRef& n);

template<> void RooJSONFactoryWSTool::importFunctions(const c4::yml::NodeRef& n) {
  for(const auto& p:n.children()){
    std::string prefix = ::genPrefix(p,false);
    std::string name(prefix + "_" + ::key(p));
    if(!p.has_child("type")){
      coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") no type given for '" << name << "', skipping." << std::endl;
      continue;
    }    
    std::string functype(::val_s(p["type"]));
    this->importDependants(p);    
    if(functype=="histogram"){
      if(!p.has_child("data")){
        coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") function '" << name << "' is of histogram type, but does not define a 'data' key, skipping.";
        continue;
      }
      try {
        RooDataHist* dh = ::readData(p["data"],name,prefix);
        RooHistFunc* hf = new RooHistFunc(("hist_"+name).c_str(),::key(p),*(dh->get()),*dh);
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
              RooHistFunc hf_low((sysname+"Low_"+name).c_str(),::key(p),*(dh_low->get()),*dh_low);              
              low.add(hf_low);
              RooDataHist* dh_high = ::readData(p["dataHigh"],sysname+"High_"+name,prefix);
              RooHistFunc hf_high((sysname+"High_"+name).c_str(),::key(p),*(dh_high->get()),*dh_high);              
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
    } else {
      auto expr = _rymlFuncFactoryExpressions.find(functype);
      if(expr!= _rymlFuncFactoryExpressions.end()){
        bool ok = true;
        bool first = true;
        std::stringstream expression;
        expression << expr->second.tclass->GetName() << "::" << name << "(";
        for(auto k:expr->second.arguments){
          if(!p.has_child(c4::to_substr(k))){
            coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") factory expression for '" << expr->first << "' maps to class '" << expr->second.tclass->GetName() << "', which expects key '" << k << "' missing from input for object '" << name << "', skipping." << std::endl;
            ok=false;
            break;
          }
          if(!first) expression << ",";
          first = false;
          expression << ::val_s(p[c4::to_substr(k)]);
        }
        expression << ")";
        if(ok){
          if(!this->_workspace->factory(expression.str().c_str())){
            coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") failed to create " << expr->second.tclass->GetName() << " '" << name << "', skipping." << std::endl;
          }
        }
      } else {
        coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") no handling for functype '" << functype << "' implemented, skipping." << std::endl;
      }
    } 
  }
}

template<> void RooJSONFactoryWSTool::importPdfs(const c4::yml::NodeRef& n) {
  for(const auto& p:n.children()){
    std::string prefix = ::genPrefix(p,false);
    std::string name(::key(p));
    if(!p.has_child("type")){
      coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") no type given for '" << name << "', skipping." << std::endl;
      continue;
    }
    std::string pdftype(::val_s(p["type"]));
    this->importDependants(p);
    if(pdftype=="sum"){
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
            if(strcmp(::val_s(def["type"]),"histogram")==0){
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
          coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") unable to obtain component '" << fprefix << "::" << fname << "' of '" << name << "', skipping." << std::endl;
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
    } else if(pdftype=="simultaneous"){
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
    } else {
      auto expr = _rymlPdfFactoryExpressions.find(pdftype);
      if(expr!= _rymlPdfFactoryExpressions.end()){
        bool ok = true;
        bool first = true;
        std::stringstream expression;
        expression << expr->second.tclass->GetName() << "::" << name << "(";
        for(auto k:expr->second.arguments){
          if(!p.has_child(c4::to_substr(k))){
            coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") factory expression for '" << expr->first << "' maps to class '" << expr->second.tclass->GetName() << "', which expects key '" << k << "' missing from input for object '" << name << "', skipping." << std::endl;
            ok=false;
            break;
          }
          if(!first) expression << ",";
          first = false;
          expression << ::val_s(p[c4::to_substr(k)]);
        }
        expression << ")";
        if(ok){
          if(!this->_workspace->factory(expression.str().c_str())){
            coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") failed to create " << expr->second.tclass->GetName() << " '" << name << "', skipping." << std::endl;
          }
        }
      } else {
        coutE(InputArguments) << "RooJSONFactoryWSTool(" << GetName() << ") no handling for pdftype '" << pdftype << "' implemented, skipping." << std::endl;
      }
    }
  }
}

template<> void RooJSONFactoryWSTool::importVariables(const c4::yml::NodeRef& n) {
  for(const auto& p:n.children()){
    std::string name(::key(p));
    double val(p.has_child("value") ? ::val_d(p["value"]) : 1.);
    RooRealVar v(name.c_str(),name.c_str(),val);
    if(p.has_child("min"))    v.setMin     (::val_d(p["min"]));
    if(p.has_child("max"))    v.setMax     (::val_d(p["max"]));
    if(p.has_child("relErr")) v.setError   (v.getVal()*::val_d(p["relErr"]));
    if(p.has_child("err"))    v.setError   (           ::val_d(p["err"]));
    if(p.has_child("const"))  v.setConstant(::val_b(p["const"]));
    this->_workspace->import(v);
  }  
}

template<> void RooJSONFactoryWSTool::exportVariables(const c4::yml::NodeRef& n) {
  for(auto* arg:this->_workspace->allVars()){
    RooRealVar* v = dynamic_cast<RooRealVar*>(arg);
    if(!v) continue;
    auto var = n[c4::to_csubstr(v->GetName())];
    var |= c4::yml::MAP;  
    var["value"] << v->getVal();
    var["min"] << v->getMin();
    var["max"] << v->getMin();
    var["err"] << v->getError();
    var["const"] << v->isConstant();    
  }  
}

template<> void RooJSONFactoryWSTool::exportFunctions(const c4::yml::NodeRef& n) {
}

template<> void RooJSONFactoryWSTool::exportPdfs(const c4::yml::NodeRef& n) {
}

template<> void RooJSONFactoryWSTool::importDependants(const c4::yml::NodeRef& n) {
  if(n.has_child("variables")){
    this->importVariables(n["variables"]);
  }
  if(n.has_child("functions")){
    this->importFunctions(n["functions"]);
  }    
  if(n.has_child("pdfs")){
    this->importPdfs(n["pdfs"]);
  }
}

template<> void RooJSONFactoryWSTool::exportAll(const c4::yml::NodeRef& n) {
  auto vars = n["variables"];
  vars |= c4::yml::MAP;  
  this->exportVariables(vars);
  auto funcs = n["functions"];
  funcs |= c4::yml::MAP;    
  this->exportFunctions(funcs);
  auto pdfs = n["pdfs"];
  pdfs |= c4::yml::MAP;    
  this->exportPdfs(pdfs);
}

 

#endif

void RooJSONFactoryWSTool::loadFactoryExpressions(const std::string& fname){
#ifdef INCLUDE_RYML    
  std::ifstream infile(fname);
  std::string line;
  while (std::getline(infile, line)){
    std::istringstream iss(line);
    std::string key;
    iss >> key;
    TString classname;
    iss >> classname;
    TClass* c = TClass::GetClass(classname);
    if(!c){
      std::cerr << "unable to find class " << classname << ", skipping." << std::endl;
    } else {
      RYML_Factory_Expression ex;
      ex.tclass = c;
      while(iss.good()){
        std::string value;
        iss >> value;
        ex.arguments.push_back(value);
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
#ifdef INCLUDE_RYML    
  _rymlPdfFactoryExpressions.clear();
  _rymlFuncFactoryExpressions.clear();
#endif
}
void RooJSONFactoryWSTool::printFactoryExpressions(){
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


Bool_t RooJSONFactoryWSTool::exportJSON( std::ostream& os ) {
#ifdef INCLUDE_RYML  
  ryml::Tree t;
  c4::yml::NodeRef n = t.rootref();
  n |= c4::yml::MAP;
  this->exportAll(n);
  os << t;
  return true;
#else
  std::cerr << "JSON export only support with rapidyaml!" << std::endl;
  return false;
#endif
}
Bool_t RooJSONFactoryWSTool::exportJSON( const char* filename ) {
  std::ofstream out(filename);
  return this->exportJSON(out);
}

Bool_t RooJSONFactoryWSTool::exportYML( std::ostream& os ) {
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
  std::ofstream out(filename);
  return this->exportYML(out);
}

Bool_t RooJSONFactoryWSTool::importJSON( std::istream& is ) {
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
  std::ifstream out(filename);
  return this->importJSON(out);
}

Bool_t RooJSONFactoryWSTool::importYML( std::istream& is ) {
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
  std::ifstream out(filename);
  return this->importYML(out);
}

RooJSONFactoryWSTool::RooJSONFactoryWSTool(RooWorkspace& ws) : _workspace(&ws){
  // default constructor
}


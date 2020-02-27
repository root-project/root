#include <RooStats/RooJSONFactoryWSTool.h>

#include <RooStats/HistFactory/ParamHistFunc.h>
#include <RooStats/HistFactory/PiecewiseInterpolation.h>
#include <RooStats/HistFactory/FlexibleInterpVar.h>
#include <RooConstVar.h>
#include <RooCategory.h>
#include <RooRealVar.h>
#include <RooDataHist.h>
#include <RooHistFunc.h>
#include <RooRealSumPdf.h>
#include <RooProdPdf.h>
#include <RooSimultaneous.h>
#include <RooPoisson.h>
#include <RooProduct.h>

namespace {
  // error handling helpers
  void error(const char* s){
    throw std::runtime_error(s);
  }
  void error(const std::string& s){
    throw std::runtime_error(s);
  }
}

#ifdef INCLUDE_RYML

#include <ryml.hpp>
#include <c4/yml/std/map.hpp>
#include <c4/yml/std/string.hpp>

//namespace c4 { namespace yml {
//    template<class T> void read(NodeRef const& n, std::vector<T> *v){
//      for(size_t i=0; i<n.num_children(); ++i){
//        std::string e;
//        n[i]>>e;
//        v->push_back(e);
//      }
//    }
//    
//    template<class T> void write(NodeRef *n, std::vector<T> const& v){
//      *n |= c4::yml::SEQ;
//      for(auto e:v){
//        n->append_child() << e;
//      }
//    }
//  }
//}


namespace {
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
}


namespace {
  inline void collectNames(const c4::yml::NodeRef& n,std::vector<std::string>& names){
    for(auto c:n.children()){
      names.push_back(::name(c));
    }
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
        vars[::name(p)] = p;      
      }
    }
    return vars;
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

  inline RooDataHist* readData(RooWorkspace* ws, const c4::yml::NodeRef& n,const std::string& namecomp,const std::string& obsnamecomp){
    if(!n.is_map()) throw "data is not a map!";
    auto vars = readVars(n,obsnamecomp);
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
    auto bins = RooJSONFactoryWSTool::generateBinIndices(varlist);
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

  
  class RooHistogramFactory : public RooJSONFactoryWSTool::Importer<c4::yml::NodeRef> {
  public:
    virtual bool importFunction(RooWorkspace* ws, const c4::yml::NodeRef& p) const override {
      std::string name(::name(p));
      std::string prefix = ::genPrefix(p,true);
      if(prefix.size() > 0) name = prefix+name;    
      if(!p.has_child("data")){
        error("function '" + name + "' is of histogram type, but does not define a 'data' key");
      }
      try {
        RooArgSet prodElems;
        RooDataHist* dh = ::readData(ws,p["data"],name,prefix);       
        RooHistFunc* hf = new RooHistFunc(name.c_str(),::name(p).c_str(),*(dh->get()),*dh);          
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
            std::string sysname(::name(sys));
            std::string parname( sys.has_child("parameter") ? ::val_s(sys["parameter"]) : "alpha_"+sysname);
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
            std::string sysname(::name(sys));
            std::string parname( sys.has_child("parameter") ? ::val_s(sys["parameter"]) : "alpha_"+sysname);            
            RooAbsReal* par = ws->var(parname.c_str());
            RooAbsPdf* pdf = ws->pdf(sysname.c_str());
            if(!par){
              error("unable to find nuisance parameter '" + parname + "'");
            } else if(!pdf){
              error("unable to find pdf '" + sysname + "'.");
            } else {
              nps.add(*par);
              RooDataHist* dh_low = ::readData(ws,p["dataLow"],sysname+"Low_"+name,prefix);
              RooHistFunc hf_low((sysname+"Low_"+name).c_str(),::name(p).c_str(),*(dh_low->get()),*dh_low);              
              low.add(hf_low);
              RooDataHist* dh_high = ::readData(ws,p["dataHigh"],sysname+"High_"+name,prefix);
              RooHistFunc hf_high((sysname+"High_"+name).c_str(),::name(p).c_str(),*(dh_high->get()),*dh_high);              
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
  bool _roohistogramfactory = RooJSONFactoryWSTool::registerImporter("histogram",new RooHistogramFactory());

  class RooRealSumPdfFactory : public RooJSONFactoryWSTool::Importer<c4::yml::NodeRef> {
  public:
    virtual bool importPdf(RooWorkspace* ws, const c4::yml::NodeRef& p) const override {
      std::string name(::name(p));
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

  bool _roorealsumpdffactory = RooJSONFactoryWSTool::registerImporter("sum",new RooRealSumPdfFactory());
}

namespace {
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
  bool _flexibleinterpvarstreamer = RooJSONFactoryWSTool::registerExporter(RooStats::HistFactory::FlexibleInterpVar::Class(),new FlexibleInterpVarStreamer());
}


#endif

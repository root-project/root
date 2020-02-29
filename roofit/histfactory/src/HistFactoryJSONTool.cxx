#ifdef INCLUDE_RYML
#include <ryml.hpp>
#include <c4/yml/std/map.hpp>
#include <c4/yml/std/string.hpp>
#include "RooStats/RooJSONFactoryWSTool.h"
#include "RooStats/HistFactory/HistFactoryJSONTool.h"
#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/Channel.h"
#include "RooStats/HistFactory/Sample.h"

namespace RooStats { namespace HistFactory {
    void read(c4::yml::NodeRef const& n, PreprocessFunction *v)
    {
      std::string name;
      std::string expression;
      std::string dependents;
      std::string command;   

      n["name"]       >> name;      
      n["expression"] >> expression;
      n["dependents"] >> dependents;
      n["command"]    >> command;   
      
      v->SetName      (name      );
      v->SetExpression(expression);
      v->SetDependents(dependents);
      v->SetCommand   (command   );
    }
    
    void write(c4::yml::NodeRef *n, PreprocessFunction const& v)
    {
      *n |= c4::yml::MAP;

      auto ch = n->append_child();
      ch["name"] << v.GetName();
      ch["expression"] << v.GetExpression();
      ch["dependents"] << v.GetDependents();
      ch["command"] << v.GetCommand();
    }
  }
}

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

RooStats::HistFactory::JSONTool::JSONTool( RooStats::HistFactory::Measurement* m ) : _measurement(m) {};

template<> void RooStats::HistFactory::JSONTool::Export(const RooStats::HistFactory::Sample& sample, c4::yml::NodeRef& n) const {
  std::vector<std::string> obsnames;
  obsnames.push_back("obs_x_"+sample.GetChannelName());
  obsnames.push_back("obs_y_"+sample.GetChannelName());
  obsnames.push_back("obs_z_"+sample.GetChannelName());
  
  auto s = n[c4::to_csubstr(RooJSONFactoryWSTool::incache(sample.GetName()))];
  s |= c4::yml::MAP;
  s["type"] << "histogram";
  
  if(sample.GetOverallSysList().size() > 0){
    auto overallSys = s["overallSystematics"];
    overallSys |= c4::yml::MAP;
    for(const auto& sys:sample.GetOverallSysList()){
      auto node = overallSys[c4::to_csubstr(sys.GetName())];
      node |= c4::yml::MAP;        
      node["low"] << sys.GetLow();
      node["high"] << sys.GetHigh();
    }
  }

  if(sample.GetNormFactorList().size()>0){
    auto normFactors = s["normFactors"];
    normFactors |= c4::yml::SEQ;
    for(auto& sys:sample.GetNormFactorList()){
      normFactors.append_child() << sys.GetName();
    }
  }

  if(sample.GetHistoSysList().size()>0){
    auto histoSys = s["histogramSystematics"];
    histoSys |= c4::yml::MAP;
    for(size_t i=0; i<sample.GetHistoSysList().size(); ++i){
      auto sys = sample.GetHistoSysList()[i];
      auto node = histoSys[c4::to_csubstr(sys.GetName())];
      node |= c4::yml::MAP;        
      auto dataLow = node["dataLow"];
      auto dataHigh = node["dataHigh"];          
      RooJSONFactoryWSTool::exportHistogram(*(sys.GetHistoLow()),dataLow,obsnames);
      RooJSONFactoryWSTool::exportHistogram(*(sys.GetHistoHigh()),dataHigh,obsnames);                    
    }
  }

  // std::vector< RooStats::HistFactory::ShapeSys >    fShapeSysList;
  // std::vector< RooStats::HistFactory::ShapeFactor > fShapeFactorList;
  
  auto tags = s["dict"];
  tags |= c4::yml::MAP;      
  tags["normalizeByTheory"] << sample.GetNormalizeByTheory();
  tags["statErrorActivate"] << sample.HasStatError();

  auto data = s["data"];
  RooJSONFactoryWSTool::exportHistogram(*sample.GetHisto(),data,obsnames);
}


template<> void RooStats::HistFactory::JSONTool::Export(const RooStats::HistFactory::Channel& c, c4::yml::NodeRef& n) const {
  auto ch = n[c4::to_csubstr(RooJSONFactoryWSTool::incache(c.GetName()))];
  ch |= c4::yml::MAP;
  ch["type"] << "sum";      
  
  auto staterr = ch["statError"];
  staterr |= c4::yml::MAP;
  staterr["relThreshold"] << c.GetStatErrorConfig().GetRelErrorThreshold();      
  staterr["constraint"] <<RooStats::HistFactory::Constraint::Name(c.GetStatErrorConfig().GetConstraintType());
  auto stack = staterr["stack"];
  stack |= c4::yml::SEQ;      
  
  auto samples = ch["samples"];
  samples |= c4::yml::MAP;
  for(const auto& s:c.GetSamples()){
    this->Export(s,samples);
    auto ns = samples.last_child()["namespaces"];
    ns |= c4::yml::SEQ;
    ns.append_child() << c.GetName();
    stack.append_child() << s.GetName();
  }
}

template<> void RooStats::HistFactory::JSONTool::Export(c4::yml::NodeRef& n) const {
  RooJSONFactoryWSTool::clearcache();
  for(const auto& ch:this->_measurement->GetChannels()){
    if(!ch.CheckHistograms()) throw std::runtime_error("unable to export histograms, please call CollectHistograms first");
  }

  //   auto parlist = n["variables"];
  //   parlist |= c4::yml::MAP;

  auto pdflist = n["pdfs"];
  pdflist |= c4::yml::MAP;

  // collect information
  std::map<std::string,RooStats::HistFactory::Constraint::Type> constraints;
  std::map<std::string,NormFactor > normfactors;
  for(const auto& ch:this->_measurement->GetChannels()){
    for(const auto&s: ch.GetSamples()){
      for(const auto& sys:s.GetOverallSysList()){
        constraints[sys.GetName()] = RooStats::HistFactory::Constraint::Gaussian;
      }
      for(const auto& sys:s.GetHistoSysList()){
        constraints[sys.GetName()] = RooStats::HistFactory::Constraint::Gaussian;
      }
      for(const auto& sys:s.GetShapeSysList()){
        constraints[sys.GetName()] = sys.GetConstraintType();
      }
      for(const auto& norm:s.GetNormFactorList()){
        normfactors[norm.GetName()] = norm;
      }
    }
  }

  // parameters
  
  //  auto lumi = parlist["Lumi"];
  //  lumi |= c4::yml::MAP;  
  //  lumi["value"] << this->_measurement->GetLumi();  
  //  lumi["relErr"] << this->_measurement->GetLumiRelErr();

  //   for(const auto& par:this->_measurement->GetPOIList()){
  //     auto node = parlist[c4::to_csubstr(par)];
  //     node |= c4::yml::MAP;    
  //     if(this->_measurement->GetParamValues().find(par) != this->_measurement->GetParamValues().end()){
  //       node["value"] << this->_measurement->GetParamValues().at(par);
  //     }    
  //   }
  //   for(const auto& par:this->_measurement->GetParamValues()){
  //     auto node = parlist[c4::to_csubstr(par.first)];
  //     node |= c4::yml::MAP;    
  //     node["value"] << par.second;
  //     if(std::find(this->_measurement->GetConstantParams().begin(),this->_measurement->GetConstantParams().end(),par.first) != this->_measurement->GetConstantParams().end()){
  //       node["const"] << 1;
  //     }
  //   }
  //   
  //   for(const auto& norm:normfactors){
  //     auto node = parlist[c4::to_csubstr(norm.second.GetName())];
  //     node |= c4::yml::MAP;
  //     node["value"] << norm.second.GetVal();        
  //     node["min"] << norm.second.GetLow();
  //     node["max"] << norm.second.GetHigh();
  //     if(norm.second.GetConst()){
  //       node["const"] << norm.second.GetConst();
  //     }
  //   }
  
  // pdfs
  //   for(const auto& sys:constraints){
  //     auto node = pdflist[c4::to_csubstr(sys.first)];
  //     node |= c4::yml::MAP;
  //     node["type"] << RooStats::HistFactory::Constraint::Name(sys.second);
  //     if(sys.second == RooStats::HistFactory::Constraint::Gaussian){
  //       std::string xname = std::string("alpha_")+sys.first;
  //       auto xpar = parlist[c4::to_csubstr(RooJSONFactoryWSTool::incache(xname))];
  //       xpar |= c4::yml::MAP;
  //       xpar["value"] << 0;      
  //       xpar["max"] << 5.;
  //       xpar["min"] << -5.;
  //       xpar["err"] << 1;            
  //       node["x"] << xname;
  //       node["mean"] << "0.";
  //       node["sigma"] << "1.";
  //     }
  //   }

  //  for(const auto& sys:this->_measurement->GetGammaSyst()){
  //    auto node = pdflist[c4::to_csubstr(sys.first)];
  //    node |= c4::yml::MAP;
  //    node["value"] << sys.second;    
  //  }
  //  for(auto sys:this->_measurement->GetUniformSyst()){
  //    auto node = pdflist[c4::to_csubstr(sys.first)];
  //    node |= c4::yml::MAP;
  //    node["value"] << sys.second;    
  //  }
  //  for(auto sys:this->_measurement->GetLogNormSyst()){
  //    auto node = pdflist[c4::to_csubstr(sys.first)];    
  //    node |= c4::yml::MAP;
  //    node["value"] << sys.second;    
  //  }
  //  for(auto sys:this->_measurement->GetNoSyst()){
  //    auto node = pdflist[c4::to_csubstr(sys.first)];    
  //    node |= c4::yml::MAP;    
  //    node["value"] << sys.second;
  //  }

  if(this->_measurement->GetFunctionObjects().size() > 0){
    auto funclist = n["functions"];
    funclist |= c4::yml::MAP;    
    funclist << this->_measurement->GetFunctionObjects();
  }

  // and finally, the simpdf
  
  auto sim = pdflist[c4::to_csubstr(this->_measurement->GetName())];
  sim |= c4::yml::MAP;
  sim["type"] << "simultaneous";
  auto simdict = sim["dict"];
  simdict |= c4::yml::MAP;  
  simdict["InterpolationScheme"] << this->_measurement->GetInterpolationScheme();
  auto simtags = sim["tags"];
  simtags |= c4::yml::SEQ;  
  simtags.append_child() << "toplevel";
  auto ch = sim["channels"];
  ch |= c4::yml::MAP;  
  for(const auto& c:this->_measurement->GetChannels()){
    this->Export(c,ch);
  }
}
#endif

void RooStats::HistFactory::JSONTool::PrintJSON( std::ostream& os ) {
#ifdef INCLUDE_RYML  
  ryml::Tree t;
  c4::yml::NodeRef n = t.rootref();
  n |= c4::yml::MAP;
  this->Export(n);
  os << c4::yml::as_json(t);
#else
  std::cerr << "JSON export only support with rapidyaml!" << std::endl;
#endif
}
void RooStats::HistFactory::JSONTool::PrintJSON( std::string filename ) {
  std::ofstream out(filename);
  this->PrintJSON(out);
}

void RooStats::HistFactory::JSONTool::PrintYAML( std::ostream& os ) {
#ifdef INCLUDE_RYML  
  ryml::Tree t;
  c4::yml::NodeRef n = t.rootref();
  n |= c4::yml::MAP;
  this->Export(n);
  os << t;
#else
  std::cerr << "YAML export only support with rapidyaml!" << std::endl;
#endif
}
void RooStats::HistFactory::JSONTool::PrintYAML( std::string filename ) {
  std::ofstream out(filename);
  this->PrintYAML(out);
}



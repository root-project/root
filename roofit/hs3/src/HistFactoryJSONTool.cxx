#include "RooJSONFactoryWSTool.h"
#include "HistFactoryJSONTool.h"

#include "RooStats/HistFactory/Measurement.h"
#include "RooStats/HistFactory/Channel.h"
#include "RooStats/HistFactory/Sample.h"

#include "JSONInterface.h"

#ifdef INCLUDE_RYML  
#include "RYMLParser.h"
typedef TRYMLTree tree_t;
#else
#include "JSONParser.h"
typedef TJSONTree tree_t;
#endif

RooStats::HistFactory::JSONTool::JSONTool( RooStats::HistFactory::Measurement* m ) : _measurement(m) {};

void RooStats::HistFactory::JSONTool::Export(const RooStats::HistFactory::Sample& sample, JSONNode& s) const {
  std::vector<std::string> obsnames;
  obsnames.push_back("obs_x_"+sample.GetChannelName());
  obsnames.push_back("obs_y_"+sample.GetChannelName());
  obsnames.push_back("obs_z_"+sample.GetChannelName());
  
  s.set_map();
  s["type"] << "histogram";
  
  if(sample.GetOverallSysList().size() > 0){
    auto& overallSys = s["overallSystematics"];
    overallSys.set_map();
    for(const auto& sys:sample.GetOverallSysList()){
      auto& node = overallSys[sys.GetName()];
      node.set_map();
      node["low"] << sys.GetLow();
      node["high"] << sys.GetHigh();
    }
  }

  if(sample.GetNormFactorList().size()>0){
    auto& normFactors = s["normFactors"];
    normFactors.set_seq();
    for(auto& sys:sample.GetNormFactorList()){
      normFactors.append_child() << sys.GetName();
    }
  }

  if(sample.GetHistoSysList().size()>0){
    auto& histoSys = s["histogramSystematics"];
    histoSys.set_map();
    for(size_t i=0; i<sample.GetHistoSysList().size(); ++i){
      auto& sys = sample.GetHistoSysList()[i];
      auto& node = histoSys[sys.GetName()];
      node.set_map();
      auto& dataLow = node["dataLow"];
      auto& dataHigh = node["dataHigh"];          
      RooJSONFactoryWSTool::exportHistogram(*(sys.GetHistoLow()),dataLow,obsnames);
      RooJSONFactoryWSTool::exportHistogram(*(sys.GetHistoHigh()),dataHigh,obsnames);                    
    }
  }

  // std::vector< RooStats::HistFactory::ShapeSys >    fShapeSysList;
  // std::vector< RooStats::HistFactory::ShapeFactor > fShapeFactorList;
  
  auto& tags = s["dict"];
  tags.set_map();
  tags["normalizeByTheory"] << sample.GetNormalizeByTheory();
  tags["statErrorActivate"] << sample.HasStatError();

  auto& data = s["data"];
  RooJSONFactoryWSTool::exportHistogram(*sample.GetHisto(),data,obsnames);
}


void RooStats::HistFactory::JSONTool::Export(const RooStats::HistFactory::Channel& c, JSONNode& ch) const {
  ch.set_map();
  ch["type"] << "histfactory";      
  
  auto& staterr = ch["statError"];
  staterr.set_map();
  staterr["relThreshold"] << c.GetStatErrorConfig().GetRelErrorThreshold();      
  staterr["constraint"] <<RooStats::HistFactory::Constraint::Name(c.GetStatErrorConfig().GetConstraintType());
  auto& stack = staterr["stack"];
  stack.set_seq();
  
  auto& samples = ch["samples"];
  samples.set_map();
  for(const auto& s:c.GetSamples()){
    auto& sample = samples[s.GetName()];
    this->Export(s,sample);
    auto& ns = sample["namespaces"];
    ns.set_seq();
    ns.append_child() << c.GetName();
    stack.append_child() << s.GetName();
  }
}

void RooStats::HistFactory::JSONTool::Export(JSONNode& n) const {
  for(const auto& ch:this->_measurement->GetChannels()){
    if(!ch.CheckHistograms()) throw std::runtime_error("unable to export histograms, please call CollectHistograms first");
  }

  //   auto parlist = n["variables"];
  //   parlist |= c4::yml::MAP;

  auto& pdflist = n["pdfs"];
  pdflist.set_map();

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
    auto& funclist = n["functions"];
    funclist.set_map();
    for(const auto& func:this->_measurement->GetFunctionObjects()){
      auto& f = funclist[func.GetName()];
      f.set_map();
      f["name"] << func.GetName();
      f["expression"] << func.GetExpression();
      f["dependents"] << func.GetDependents();
      f["command"] << func.GetCommand();      
    }
  }
    
  // and finally, the simpdf
  
  auto& sim = pdflist[this->_measurement->GetName()];
  sim.set_map();
  sim["type"] << "simultaneous";
  auto& simdict = sim["dict"];
  simdict.set_map();
  simdict["InterpolationScheme"] << this->_measurement->GetInterpolationScheme();
  auto& simtags = sim["tags"];
  simtags.set_seq();
  simtags.append_child() << "toplevel";
  auto& ch = sim["channels"];
  ch.set_map();
  for(const auto& c:this->_measurement->GetChannels()){
    auto& thisch = ch[c.GetName()];
    this->Export(c,thisch);
  }
}

void RooStats::HistFactory::JSONTool::PrintJSON( std::ostream& os ) {
  tree_t p;
  auto& n = p.rootnode();
  n.set_map();
  this->Export(n);
  n.writeJSON(os);
}
void RooStats::HistFactory::JSONTool::PrintJSON( std::string filename ) {
  std::ofstream out(filename);
  this->PrintJSON(out);
}

void RooStats::HistFactory::JSONTool::PrintYAML( std::ostream& os ) {
#ifdef INCLUDE_RYML
  TRYMLTree p;
  auto& n = p.rootnode();
  n.set_map();
  this->Export(n);
  n.writeYML(os);  
#else
  std::cerr << "YAML export only support with rapidyaml!" << std::endl;
#endif
}
void RooStats::HistFactory::JSONTool::PrintYAML( std::string filename ) {
  std::ofstream out(filename);
  this->PrintYAML(out);
}



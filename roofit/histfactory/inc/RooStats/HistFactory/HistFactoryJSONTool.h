#ifndef HFJSONTOOL_H
#define HFJSONTOOL_H
#include "Measurement.h"
#include "TH1.h"
#include "RooStats/JSONInterface.h"
#include <string>

namespace RooStats { namespace HistFactory {
class JSONTool : public TNamed, RooPrintable {
 protected:
  RooStats::HistFactory::Measurement* _measurement;
  static std::vector<std::string> _strcache;  
  
  void Export(const RooStats::HistFactory::Channel& c, TJSONNode& t) const;
  void Export(const RooStats::HistFactory::Sample& s, TJSONNode& t) const;  
  
 public:
  JSONTool( RooStats::HistFactory::Measurement* );
  
  void PrintJSON( std::ostream& os = std::cout );    
  void PrintJSON( std::string filename );
  void PrintYAML( std::ostream& os = std::cout );    
  void PrintYAML( std::string filename );  
  void Export(TJSONNode& t) const;  

};
  }
}
#endif

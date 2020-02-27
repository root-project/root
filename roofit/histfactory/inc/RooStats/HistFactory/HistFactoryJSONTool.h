#ifndef HFJSONTOOL_H
#define HFJSONTOOL_H
#include "Measurement.h"
#include "TH1.h"
#include <string>

namespace RooStats { namespace HistFactory {
class JSONTool : public TNamed, RooPrintable {
 protected:
  RooStats::HistFactory::Measurement* _measurement;
  static std::vector<std::string> _strcache;  
  
  template<class T> void Export(const RooStats::HistFactory::Channel& c, T& t) const;
  template<class T> void Export(const RooStats::HistFactory::Sample& s, T& t) const;  
  
 public:
  JSONTool( RooStats::HistFactory::Measurement* );
  
  void PrintJSON( std::ostream& os = std::cout );    
  void PrintJSON( std::string filename );
  void PrintYAML( std::ostream& os = std::cout );    
  void PrintYAML( std::string filename );  
  template<class T> void Export(T& t) const;  

};
  }
}
#endif

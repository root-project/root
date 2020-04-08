#ifndef RooFitHS3_HistFactoryJSONTool_h
#define RooFitHS3_HistFactoryJSONTool_h

#include <iostream>
#include <string>

namespace RooFit {
namespace Detail {
class JSONNode;
}
} // namespace RooFit

namespace RooStats {
namespace HistFactory {

class Channel;
class Measurement;
class Sample;

class JSONTool {
protected:
   RooStats::HistFactory::Measurement *_measurement;

   void Export(const RooStats::HistFactory::Channel &c, RooFit::Detail::JSONNode &t) const;
   void Export(const RooStats::HistFactory::Sample &s, RooFit::Detail::JSONNode &t) const;

public:
   JSONTool(RooStats::HistFactory::Measurement *);

   void PrintJSON(std::ostream &os = std::cout);
   void PrintJSON(std::string const &filename);
   void PrintYAML(std::ostream &os = std::cout);
   void PrintYAML(std::string const &filename);
   void Export(RooFit::Detail::JSONNode &t) const;
};

} // namespace HistFactory
} // namespace RooStats
#endif

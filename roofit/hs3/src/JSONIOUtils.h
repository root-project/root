#ifndef JSONIOUtils_h
#define JSONIOUtils_h

#include <string_view>
#include <vector>
#include <RooFit/Detail/JSONInterface.h>

class RooAbsBinning;

bool startsWith(std::string_view str, std::string_view prefix);
bool endsWith(std::string_view str, std::string_view suffix);
std::string removeSuffix(std::string_view str, std::string_view suffix);
std::unique_ptr<RooFit::Detail::JSONTree> varJSONString(const RooFit::Detail::JSONNode &treeRoot);
void writeAxisBinning(RooFit::Detail::JSONNode &node, const RooAbsBinning &binning);
bool parseHS3Version(std::string const &version, int &major, int &minor);
std::vector<double> readEdges(const RooFit::Detail::JSONNode &node, std::string const &context);

#endif

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
double readBound(RooFit::Detail::JSONNode const &node, const char *key, double defaultValue);

// The key of a fixed domain axis (HS3 #131): the parameter only takes this value, and RooFit reads it as a
// RooConstVar.
inline constexpr const char *fixedValueKey = "value";
bool parseHS3Version(std::string const &version, int &major, int &minor);
bool isHS3VersionBefore(std::string const &version, int major, int minor);
std::vector<double> readEdges(const RooFit::Detail::JSONNode &node, std::string const &context);

#endif

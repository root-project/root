#ifndef JSONIOUtils_h
#define JSONIOUtils_h

#include <string_view>
#include <vector>
#include <RooFit/Detail/JSONInterface.h>

class RooAbsArg;
class RooAbsBinning;

struct ToNumberOutput {
   bool valid = false;
   double value = 0.0;
};

ToNumberOutput toNumber(const std::string &str);
bool isLiteralConstVar(RooAbsArg const &arg);
void writeReference(RooFit::Detail::JSONNode &node, RooAbsArg const &arg);
bool startsWith(std::string_view str, std::string_view prefix);
bool endsWith(std::string_view str, std::string_view suffix);
std::string removeSuffix(std::string_view str, std::string_view suffix);
void writeAxisBinning(RooFit::Detail::JSONNode &node, const RooAbsBinning &binning);

// The key of a fixed domain axis (HS3 #131): the parameter only takes this value, and RooFit reads it as a
// RooConstVar.
inline constexpr const char *fixedValueKey = "value";
bool parseHS3Version(std::string const &version, int &major, int &minor);
bool isHS3VersionBefore(std::string const &version, int major, int minor);

#endif

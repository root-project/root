#ifndef JSONIOUtils_h
#define JSONIOUtils_h

#include <string_view>
#include <vector>
#include <RooFit/Detail/JSONInterface.h>

class RooAbsBinning;
class RooAbsCollection;
class RooAbsReal;
class RooArgList;
class ParamHistFunc;

bool startsWith(std::string_view str, std::string_view prefix);
bool endsWith(std::string_view str, std::string_view suffix);
std::string removeSuffix(std::string_view str, std::string_view suffix);
void writeAxisBinning(RooFit::Detail::JSONNode &node, const RooAbsBinning &binning);

// The key of a fixed domain axis (HS3 #131): the parameter only takes this value, and RooFit reads it as a
// RooConstVar.
inline constexpr const char *fixedValueKey = "value";
bool parseHS3Version(std::string const &version, int &major, int &minor);
bool isHS3VersionBefore(std::string const &version, int major, int minor);

// ParamHistFunc orders its parameters like the bins of a TH1 (first variable fastest), while HS3 lays out per-bin
// arrays with the last axis varying fastest, like RooDataHist. These helpers convert at the HS3 boundary.
std::vector<RooAbsReal *> paramHistFuncParamsInBinOrder(ParamHistFunc const &func);
RooArgList toParamHistFuncOrder(RooArgList const &paramsInBinOrder, RooAbsCollection const &vars);
// Documents before HS3 0.3 wrote the parameters of multi-dimensional ParamHistFuncs in TH1 order. No such documents
// are known; they are rejected instead of being guessed.
void checkParamHistFuncOrderIsDefined(std::string const &hs3Version, RooAbsCollection const &vars,
                                      std::string const &context);

#endif

#ifndef JSONIOUtils_h
#define JSONIOUtils_h

#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <RooFit/Detail/JSONInterface.h>

class RooAbsArg;
class RooAbsBinning;
class RooAbsCollection;
class RooAbsPdf;
class RooAbsReal;
class RooArgList;
class ParamHistFunc;

// A RooProdPdf with this attribute represents an HS3 `array_dist`: a product of one-dimensional distributions of one
// type, each over its own observable. The factors are the components, imported as individual pdfs named after their
// observable.
inline constexpr const char *arrayDistTag = "hs3_array_dist";

// True if `arg` is a RooConstVar whose name is a plain number (i.e. a literal constant).
bool isLiteralConstVar(RooAbsArg const &arg);

// The HS3 type of `pdf` as a component of an array distribution (RooPoisson, RooGaussian or RooLognormal in the
// standard parametrization, over a named observable), or std::nullopt if `pdf` cannot be a component.
std::optional<std::string> arrayComponentType(RooAbsPdf const &pdf);

// Write the array distribution `name` with the given components into `elem`.
void writeArrayDist(RooFit::Detail::JSONNode &elem, std::string const &name,
                    std::vector<RooAbsPdf const *> const &components);

bool startsWith(std::string_view str, std::string_view prefix);
bool endsWith(std::string_view str, std::string_view suffix);
std::string removeSuffix(std::string_view str, std::string_view suffix);
std::unique_ptr<RooFit::Detail::JSONTree> varJSONString(const RooFit::Detail::JSONNode &treeRoot);
void writeAxisBinning(RooFit::Detail::JSONNode &node, const RooAbsBinning &binning);
double readBound(RooFit::Detail::JSONNode const &node, const char *key, double defaultValue);
bool parseHS3Version(std::string const &version, int &major, int &minor);
bool isHS3VersionBefore(std::string const &version, int major, int minor);
std::vector<double> readEdges(const RooFit::Detail::JSONNode &node, std::string const &context);

// ParamHistFunc orders its parameters like the bins of a TH1 (first variable fastest), while HS3 lays out per-bin
// arrays with the last axis varying fastest, like RooDataHist. These helpers convert at the HS3 boundary.
std::vector<RooAbsReal *> paramHistFuncParamsInBinOrder(ParamHistFunc const &func);
RooArgList toParamHistFuncOrder(RooArgList const &paramsInBinOrder, RooAbsCollection const &vars);
// Documents before HS3 0.3 wrote the parameters of multi-dimensional ParamHistFuncs in TH1 order. No such documents
// are known; they are rejected instead of being guessed.
void checkParamHistFuncOrderIsDefined(std::string const &hs3Version, RooAbsCollection const &vars,
                                      std::string const &context);

#endif

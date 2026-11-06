#ifndef JSONIOUtils_h
#define JSONIOUtils_h

#include <string_view>
#include <vector>
#include <RooFit/Detail/JSONInterface.h>

class RooAbsBinning;
class RooAbsCategory;
class RooAbsCollection;
class RooAbsReal;
class RooArgList;
class RooCategory;
class RooWorkspace;
class ParamHistFunc;

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

// A RooSimultaneous with this attribute is a "category_conditional_dist": its index category is a parameter, and it
// evaluates to the component selected by the current state ("switch mode"). Without it, a RooSimultaneous is a
// "combined_dist", whose index category is the column of the paired combined data set. RooFit decides between the
// two by the normalization set, i.e. by context; the attribute makes the export context-free and is set when a
// "category_conditional_dist" is imported.
inline constexpr const char *categoryConditionalTag = "hs3_category_conditional_dist";

// Selection keys (HS3 "combined" data sets and "combined_dist"): a key is one label per column, and the columns are
// categories declared by categorical axes of the domains. In RooFit, the columns are the index category of a
// RooSimultaneous (a RooCategory), or the input categories of a RooSuperCategory index, or the category observables
// of a combined data set. They are sorted by name, so that a distribution and a data set agree on the order of the
// columns without looking at each other.
std::vector<RooCategory const *> selectionColumns(RooAbsCategory const &indexCat);
std::vector<RooCategory const *> selectionColumns(RooAbsCollection const &vars);
// All keys of the columns, in the order of the columns and of their states, with the first column varying slowest.
std::vector<std::vector<std::string>> selectionKeys(std::vector<RooCategory const *> const &columns);
// The state label of a RooCategory (one column) or of a RooSuperCategory over the columns in that order.
std::string selectionLabel(std::vector<std::string> const &key);
// The state label of `indexCat` for a key with the labels in the order of `columns`: the label itself for a
// RooCategory, or "{a;b}" with the labels in the order of the inputs for a RooSuperCategory.
std::string selectionLabel(RooAbsCategory const &indexCat, std::vector<RooCategory const *> const &columns,
                           std::vector<std::string> const &key);
// A key as written in the document, for messages: ["a", "b"].
std::string selectionKeyToString(std::vector<std::string> const &key);
// Read the columns of a combined distribution or data set from the list `field` of `node`: each entry must be the
// name of a category in the workspace, i.e. declared by a categorical axis of a domain.
std::vector<RooCategory *> readSelectionColumns(RooWorkspace const &ws, RooFit::Detail::JSONNode const &node,
                                                const char *field, std::string const &context);
// Read and validate the "key" of a component: one label per column, each one of the categories of its column.
std::vector<std::string> readSelectionKey(RooFit::Detail::JSONNode const &component,
                                          std::vector<RooCategory *> const &columns, std::string const &context);

#endif

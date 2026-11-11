#ifndef JSONIOUtils_h
#define JSONIOUtils_h

#include <memory>
#include <string_view>
#include <vector>
#include <RooFit/Detail/JSONInterface.h>

class RooAbsBinning;
class RooAbsCategory;
class RooAbsCollection;
class RooAbsReal;
class RooArgList;
class RooArgSet;
class RooCategory;
class RooDataHist;
class RooWorkspace;
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

// A RooHistFunc or RooHistPdf is evaluated with its own observables (`funcObs`), which are mapped by position onto
// the variables of its histogram (`histObs`) and need not have the same names, e.g. after a RooCustomizer replaced
// them. The variables of the histogram `dh`, in the order of the histogram, each named after the observable that is
// mapped onto it, so that the axes are written with the names of the observables. A variable onto which no variable
// or category is mapped keeps its name.
std::unique_ptr<RooArgSet>
histogramVariablesNamedAfterObservables(RooDataHist const &dh, RooArgSet const &histObs, RooArgSet const &funcObs);

// ParamHistFunc orders its parameters like the bins of a TH1 (first variable fastest), while HS3 lays out per-bin
// arrays with the last axis varying fastest, like RooDataHist. These helpers convert at the HS3 boundary.
std::vector<RooAbsReal *> paramHistFuncParamsInBinOrder(ParamHistFunc const &func);
RooArgList toParamHistFuncOrder(RooArgList const &paramsInBinOrder, RooAbsCollection const &vars);
// Documents before HS3 0.3 wrote the parameters of multi-dimensional ParamHistFuncs in TH1 order. No such documents
// are known; they are rejected instead of being guessed.
void checkParamHistFuncOrderIsDefined(std::string const &hs3Version, RooAbsCollection const &vars,
                                      std::string const &context);

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

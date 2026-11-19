#ifndef JSONIOUtils_h
#define JSONIOUtils_h

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <RooFit/Detail/JSONInterface.h>

#include "HS3Document.h"

class RooAbsArg;
class RooAbsBinning;
class RooAbsCategory;
class RooAbsCollection;
class RooAbsPdf;
class RooAbsReal;
class RooArgList;
class RooArgSet;
class RooCategory;
class RooDataHist;
class RooJSONFactoryWSTool;
class RooWorkspace;
class ParamHistFunc;

// A RooProdPdf with this attribute represents an HS3 `array_dist`: a product of one-dimensional distributions of one
// type, each over its own observable. The factors are the components, imported as individual pdfs named after their
// observable.
inline constexpr const char *arrayDistTag = "hs3_array_dist";

// An object with this attribute is a helper: RooFit created it to represent something of an HS3 document that it has
// no class for, e.g. the RooProdPdf that holds the components of an array distribution together, or the parts of a
// composite structure that stands for one HS3 object. A helper is not a statement of the user about the model, so:
//  - it is only written if another object that is written refers to it by name, or if the HS3 object that it stands
//    for is written (see isSkippedHelper()), but not just because it is in the workspace;
//  - a helper that is not written is ignored wherever the export looks at the users of an object, e.g. to decide
//    whether the object is a top-level object or whether a constraint can be a component of an array.
// The attribute is set by the importers and is not written to the document. Objects created by users don't have it
// and are always written.
inline constexpr const char *helperObjectTag = "hs3_helper";

// True if `arg` is a RooConstVar whose name is a plain number (i.e. a literal constant).
bool isLiteralConstVar(RooAbsArg const &arg);

// Whether `arg` is a helper (see helperObjectTag) that the export of the workspace `ws` does not write: nothing that
// is written refers to it, i.e. it has no clients in the workspace other than helpers that are skipped themselves and
// no ModelConfig names it, and the HS3 object that it stands for is not written either. Such an object is treated as
// if it were not in the workspace.
bool isSkippedHelper(RooAbsArg const &arg, RooWorkspace const &ws);

// The HS3 type, the observable and the other fields of `pdf` as a component of an array distribution (RooPoisson,
// RooGaussian or RooLognormal, over a named observable), or std::nullopt if `pdf` cannot be a component. The fields
// are the ones of the exporter of its class.
// If `observable` names the mean of a RooGaussian, the Gaussian is written with x and mean swapped (see
// RooJSONFactoryWSTool::setConstraintObservable()). The functions that a field names, e.g. for a RooLognormal that is
// not in the standard parametrization, are only written if there is a `tool`; without one, this only describes the
// component.
struct ArrayComponentInfo {
   std::string type;
   std::string observable;
   std::vector<std::pair<std::string, RooFit::HS3::NameOrNumber>> fields;
};
std::optional<ArrayComponentInfo>
arrayComponentInfo(RooAbsPdf const &pdf, std::string const *observable = nullptr, RooJSONFactoryWSTool *tool = nullptr);

// Write the array distribution `name` with the given components into `elem`. `observableOf` gives the observable
// recorded for a component by name, if any. The `tool` writes the functions that the components need besides, e.g.
// for a RooLognormal that is not in the standard parametrization.
void writeArrayDist(RooJSONFactoryWSTool &tool, RooFit::Detail::JSONNode &elem, std::string const &name,
                    std::vector<RooAbsPdf const *> const &components,
                    std::map<std::string, std::string> const &observableOf);

// The number that a string spells as a whole, if it does.
std::optional<double> toNumber(const std::string &str);
void writeReference(RooFit::Detail::JSONNode &node, RooAbsArg const &arg);
bool startsWith(std::string_view str, std::string_view prefix);
bool endsWith(std::string_view str, std::string_view suffix);
std::string removeSuffix(std::string_view str, std::string_view suffix);

// Some RooFit classes are parametrized differently than their HS3 distribution, e.g. RooLognormal by default. They
// are written with a function of the parameter (RooJSONFactoryWSTool::exportTransformed()), which is named after the
// parameter with a suffix that says what the function is. A literal constant has no name that could be part of an
// identifier ("1.2", "-3", "1e-05"), so its value is spelled as one: "const_1p2", "const_m3", "const_1em05".
//
// What a transformed function is named after: the name of the parameter, or the identifier for the value of a
// literal constant.
std::string transformedBaseName(RooAbsReal const &param);
// What a transformed function is a function of: the name of the parameter, or the value of a literal constant with
// as many digits as it takes to read it back unchanged (in parentheses if it is negative).
std::string transformedArgument(RooAbsReal const &param);
// The inverse: what to create the RooFit object from, given the name of a transformed function with this suffix. This
// is the name of the parameter, or the value of a literal constant as a string.
std::string transformedOriginal(std::string_view name, std::string_view suffix);
// Whether a name is the identifier for the value of a literal constant.
bool isLiteralIdentifier(std::string_view name);

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
void checkParamHistFuncOrderIsDefined(RooFit::HS3::Version const &version, RooAbsCollection const &vars,
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

#endif

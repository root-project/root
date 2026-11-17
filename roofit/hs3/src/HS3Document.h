/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN, Nov 2026
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#ifndef RooFit_HS3Document_h
#define RooFit_HS3Document_h

#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace RooFit {

namespace Detail {
class JSONNode;
class JSONTree;
} // namespace Detail

/// Types that represent an HS3 document. They follow the structure of the HS3 standard, and this header must not depend
/// on RooFit: creating RooFit objects from a document is a separate step.
namespace HS3 {

struct Version {
   int major = 0;
   int minor = 0;
   int patch = 0;
};

inline bool operator<(Version const &l, Version const &r)
{
   return std::tie(l.major, l.minor, l.patch) < std::tie(r.major, r.minor, r.patch);
}

/// Combines lambdas into one visitor for std::visit(). The compiler complains about the alternatives without lambda.
template <class... Ts>
struct overloaded : Ts... {
   using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

/// The name of the alternative that a variant holds, e.g. of an axis or of a modifier.
template <class... Ts>
std::string const &nameOf(std::variant<Ts...> const &v)
{
   return std::visit([](auto const &alternative) -> std::string const & { return alternative.name; }, v);
}

/// A name that says what it refers to, so that for example a DataRef can't be used as a DomainRef.
template <class Tag>
struct Ref {
   std::string name;
};

using DistributionRef = Ref<struct DistributionTag>;
using DataRef = Ref<struct DataTag>;
using DomainRef = Ref<struct DomainTag>;
using LikelihoodRef = Ref<struct LikelihoodTag>;
using ParameterPointRef = Ref<struct ParameterPointTag>;
using ParameterRef = Ref<struct ParameterTag>;

struct Package {
   std::string name;
   std::string version;
};

struct Metadata {
   std::string hs3Version; ///< As it was written in the document, empty for fragments of a document.
   Version version;        ///< The parsed version. Versions that can't be parsed count as the oldest one.
   std::vector<Package> packages;
   std::vector<std::string> authors;
   std::vector<std::string> publications;
   std::optional<std::string> description;
};

struct RegularBinning {
   int nbins = 0;
};
struct EdgesBinning {
   std::vector<double> edges;
};
using Binning = std::variant<RegularBinning, EdgesBinning>;

/// The axis of a domain. Infinite bounds are written as null, and bounds can be missing in older documents. The binning
/// is a RooFit extension. A fixed axis has a single value instead, which excludes bounds and a binning.
struct DomainAxis {
   std::string name;
   std::optional<double> fixedValue;
   std::optional<double> min;
   std::optional<double> max;
   std::optional<Binning> binning;
};

/// The axis of point data or unbinned data, which can't have a binning.
struct ContinuousAxis {
   std::string name;
   std::optional<double> min;
   std::optional<double> max;
};

/// The axis of binned data.
struct BinnedAxis {
   std::string name;
   double min = 0.0;
   double max = 0.0;
   Binning binning;
};

struct CategoricalAxis {
   std::string name;
   std::vector<std::string> categories;
};

struct Domain {
   std::string name;
   std::vector<std::variant<DomainAxis, CategoricalAxis>> axes;
};

/// The domain in which RooFit writes the ranges of all variables. It is a convention of RooFit, not of the standard.
inline constexpr char const *defaultDomainName = "default_domain";

/// Where RooFit takes the ranges of variables from: the "default_domain" if there is one, otherwise the first domain by
/// name that has the variable. The domains must outlive this object.
class DomainRanges {
public:
   explicit DomainRanges(std::vector<Domain> const &domains);
   /// The axis that defines the range of a variable, or null.
   DomainAxis const *find(std::string const &name) const;
   /// Whether any domain has the variable.
   bool contains(std::string const &name) const { return _names.count(name); }

private:
   std::map<std::string, DomainAxis const *> _ranges;
   std::set<std::string> _names;
};

struct ParameterValue {
   std::string name;
   std::optional<std::variant<double, std::string>> value; ///< A string selects the state of a category.
   bool isConst = false;
   // RooFit extensions that were written by older versions:
   std::optional<double> error;
   std::optional<double> relError;
   std::optional<int> nbins;
};

struct ParameterPoint {
   std::string name;
   std::vector<ParameterValue> parameters;
};

struct GaussianUncertainty {
   std::vector<double> sigma;
   std::vector<std::vector<double>> correlation; ///< Empty if there is no correlation.
};

struct PointData {
   double value = 0.0;
   std::optional<double> uncertainty;
   std::vector<ContinuousAxis> axes;
};

struct UnbinnedData {
   std::vector<std::variant<ContinuousAxis, CategoricalAxis>> axes;
   /// The coordinates, row by row. A value on a categorical axis is the index in the list of categories of that axis.
   std::vector<double> entries;
   std::optional<std::vector<double>> weights;
   bool hasEntriesUncertainties = false;

   std::size_t numEntries() const { return axes.empty() ? 0 : entries.size() / axes.size(); }
};

struct BinnedData {
   std::vector<std::variant<BinnedAxis, CategoricalAxis>> axes;
   std::vector<double> contents;
   std::optional<GaussianUncertainty> uncertainty;
};

/// The sub-data set of a combined data set for one key, which has one label per column.
struct CombinedDataComponent {
   std::vector<std::string> key;
   DataRef data;
};

/// Data sets that are selected by a key (HS3 #119). The columns are categorical axes of the domains.
struct CombinedData {
   std::vector<std::string> columns;
   std::vector<CombinedDataComponent> components;
};

struct Data {
   std::string name;
   std::variant<PointData, UnbinnedData, BinnedData, CombinedData> body;
};

struct LikelihoodTerm {
   DistributionRef distribution;
   std::variant<DataRef, double> data;
};

struct Likelihood {
   std::string name;
   std::vector<LikelihoodTerm> terms;
   std::vector<DistributionRef> auxDistributions;
};

struct Analysis {
   std::string name;
   LikelihoodRef likelihood;
   std::vector<ParameterRef> parametersOfInterest;
   std::vector<DomainRef> domains;
   std::optional<ParameterPointRef> init;
   std::optional<DistributionRef> prior;
};

/// A distribution or function that is not interpreted by the document, but by the importers that are registered for
/// its type in RooFit::JSONIO::importers().
struct GenericObject {
   std::string type;
   /// The object in the JSON tree that the document was read from. In a document that is exported, the distributions
   /// refer to what the exporters wrote once all of them are written, and the functions don't have a node.
   Detail::JSONNode const *node = nullptr;
};

enum class InterpolationType {
   Add,
   Mult
};

enum class InterpolationFunction {
   Poly1,
   Poly2,
   Poly6,
   Exp
};

/// How a modifier of a HistFactory distribution interpolates and extrapolates its effect.
struct Interpolation {
   InterpolationType type = InterpolationType::Add;
   InterpolationFunction in = InterpolationFunction::Poly1;
   std::optional<InterpolationFunction> out; ///< If not set, the function inside the bounds is used outside as well.
};

inline bool operator==(Interpolation const &l, Interpolation const &r)
{
   return l.type == r.type && l.in == r.in && l.out == r.out;
}
inline bool operator!=(Interpolation const &l, Interpolation const &r)
{
   return !(l == r);
}
inline bool operator<(Interpolation const &l, Interpolation const &r)
{
   return std::tie(l.type, l.in, l.out) < std::tie(r.type, r.in, r.out);
}

inline constexpr Interpolation additivePiecewiseLinear{InterpolationType::Add, InterpolationFunction::Poly1,
                                                       std::nullopt};
inline constexpr Interpolation multiplicativePiecewiseExponential{InterpolationType::Mult, InterpolationFunction::Exp,
                                                                  std::nullopt};
inline constexpr Interpolation additiveQuadraticLinear{InterpolationType::Add, InterpolationFunction::Poly2,
                                                       InterpolationFunction::Poly1};
inline constexpr Interpolation additivePolynomialLinear{InterpolationType::Add, InterpolationFunction::Poly6,
                                                        InterpolationFunction::Poly1};
inline constexpr Interpolation multiplicativePolynomialExponential{
   InterpolationType::Mult, InterpolationFunction::Poly6, InterpolationFunction::Exp};
inline constexpr Interpolation multiplicativePolynomialLinear{InterpolationType::Mult, InterpolationFunction::Poly6,
                                                              InterpolationFunction::Poly1};

char const *toString(InterpolationType type);
char const *toString(InterpolationFunction function);

/// The RooFit classes that implement the interpolations, which encode them with integer codes.
enum class InterpolationClass {
   Piecewise,
   Flexible
};

/// The interpolation for an integer code, or nothing if the class doesn't support the code.
std::optional<Interpolation> interpolationFromCode(int code, InterpolationClass interpolationClass);
/// The integer code of an interpolation, or nothing if the class can't represent the interpolation.
std::optional<int> codeFromInterpolation(Interpolation const &interpolation, InterpolationClass interpolationClass);

/// The constraint types that older documents named, for which RooFit creates the constraint terms.
enum class ConstraintType {
   Gauss,
   Poisson,
   LogNormal
};

char const *toString(ConstraintType type);

/// The parameter of the modifier is not constrained.
struct Unconstrained {};

/// The constraint term is a distribution of the document, named in "constraints".
struct ConstraintPdf {
   DistributionRef pdf;
   /// Named in "constraint_name" by an older document, whose constraint distributions of normsys and histosys
   /// modifiers may use the global observable "nom_<parameter>" without defining it, because RooFit created it.
   bool legacyName = false;
};

using Constraint = std::variant<Unconstrained, ConstraintType, ConstraintPdf>;

inline bool operator==(Unconstrained, Unconstrained)
{
   return true;
}
inline bool operator==(ConstraintPdf const &l, ConstraintPdf const &r)
{
   return l.pdf.name == r.pdf.name;
}

struct NormFactor {
   std::string name;
   ParameterRef parameter;
   Constraint constraint;
};

struct NormSys {
   std::string name;
   ParameterRef parameter;
   double hi = 1.0;
   double lo = 1.0;
   Constraint constraint;
   std::optional<Interpolation> interpolation;
};

struct HistoSysVariation {
   std::vector<double> contents;
   std::optional<std::vector<double>> errors; ///< Not in the standard, but older versions of RooFit read them.
};

struct HistoSys {
   std::string name;
   ParameterRef parameter;
   HistoSysVariation hi;
   HistoSysVariation lo;
   Constraint constraint;
   std::optional<Interpolation> interpolation;
};

/// The gamma parameters are constant, which is a RooFit extension.
struct ConstantGammas {};

/// Every gamma parameter has its own constraint term, or none, as named in "constraints".
struct GammaConstraintPdfs {
   std::vector<std::optional<DistributionRef>> pdfs;
};

using GammaConstraint = std::variant<Unconstrained, ConstraintType, ConstantGammas, GammaConstraintPdfs>;

/// Scales each bin by its own parameter. The modifier type "shapesys" is a deprecated alias of "shapefactor".
struct ShapeSys {
   std::string name;
   std::vector<ParameterRef> parameters; ///< If empty, the parameters are named after the modifier.
   std::vector<double> vals;
   GammaConstraint constraint;
};

/// Scales each bin by a parameter, that is constrained by the statistical uncertainty of all samples that have this
/// modifier.
struct StatError {
   std::string name;
   std::vector<ParameterRef> parameters; ///< If empty, the parameters are named after the channel.
   GammaConstraint constraint;
};

/// Multiplies the sample with a function of the document.
struct CustomModifier {
   std::string name;
};

using Modifier = std::variant<NormFactor, NormSys, HistoSys, ShapeSys, StatError, CustomModifier>;

struct Sample {
   std::string name;
   std::vector<double> contents;
   std::optional<std::vector<double>> errors;
   std::vector<Modifier> modifiers;
};

/// Settings of the staterror modifiers that apply to the whole channel. This is a RooFit extension.
struct StatErrorConfig {
   double relThreshold = 0.0;
   GammaConstraint constraint;
   /// Samples without "errors" have Poisson uncertainties, as in documents before HS3 0.3.
   bool impliedPoissonErrors = false;
};

struct HistFactoryDist {
   std::vector<BinnedAxis> axes;
   std::optional<Interpolation> defaultInterpolation;
   std::vector<Sample> samples;
   StatErrorConfig statError;
};

struct Distribution {
   std::string name;
   std::variant<GenericObject, HistFactoryDist> body;
};

struct Function {
   std::string name;
   GenericObject body;
};

/// A name, or a number for a constant.
using NameOrNumber = std::variant<std::string, double>;

/// The function "interpolation" of RooFit, implemented by PiecewiseInterpolation: a nominal function plus the
/// interpolated differences to the variations "high" and "low" for each parameter.
struct InterpolatedFunction {
   std::string nominal;
   std::vector<NameOrNumber> parameters;
   std::vector<NameOrNumber> high;
   std::vector<NameOrNumber> low;
   std::vector<Interpolation> interpolations; ///< One per parameter.
   bool positiveDefinite = false;
};

/// The function "interpolation0d" of RooFit, implemented by FlexibleInterpVar: a nominal value plus the interpolated
/// differences to the variations "high" and "low" for each parameter.
struct InterpolatedNumber {
   double nominal = 0.0;
   std::vector<std::string> parameters;
   std::vector<double> high;
   std::vector<double> low;
   std::vector<Interpolation> interpolations; ///< One per parameter.
};

// The following types are stored in the "misc" section, for which the HS3 standard doesn't prescribe a format.

struct Attributes {
   std::map<std::string, std::string> dict;
   std::vector<std::string> tags;
   bool isConstVar = false; ///< How ROOT versions before HS3 0.3 marked named RooConstVars.
};

struct CombinedDistribution {
   std::string indexCat;
   std::vector<std::string> labels;
   std::vector<int> indices;
   std::vector<DistributionRef> distributions;
};

struct CombinedDataset {
   std::string indexCat;
   std::vector<std::string> labels;
   std::vector<int> indices;
};

struct ModelConfigInfo {
   std::optional<std::string> mcName;
   std::optional<std::string> pdfName;
   std::optional<std::string> combinedDataName;
   /// The observables of the ModelConfig, written if no data set of the likelihood gives them.
   std::optional<std::vector<std::string>> observables;
};

struct StepWidth {
   std::string name;
   double width = 0.0;
};

/// The content of misc.ROOT_internal and misc.minimization.
struct RooFitInternal {
   std::map<std::string, Attributes> attributes;
   std::map<std::string, CombinedDistribution> combinedDistributions;
   std::map<std::string, CombinedDataset> combinedDatasets;
   std::map<std::string, ModelConfigInfo> modelConfigs;
   std::vector<StepWidth> stepWidths;
   /// The constraint terms of HistFactory distributions that no modifier names, by distribution. They are
   /// also listed in the "aux_distributions" of the likelihoods, which is what other implementations use.
   std::map<std::string, std::vector<DistributionRef>> channelConstraints;
   /// Axes that the components of a combined data set share, but that are no variate of any distribution, e.g. an
   /// event number, by combined data set. They have no statistical meaning, and may be shared.
   std::map<std::string, std::vector<std::string>> bookkeepingColumns;
};

struct Diagnostic {
   enum class Severity {
      Warning,
      Error
   };
   Severity severity = Severity::Warning;
   std::string path; ///< Where in the document, for example "distributions[3](pdf).x".
   std::string message;
};

using Diagnostics = std::vector<Diagnostic>;

/// Whether there is a diagnostic with error severity.
bool hasErrors(Diagnostics const &diagnostics);

} // namespace HS3

class HS3Document;
class ValidatedHS3Document;

namespace HS3 {
struct ValidateOptions;
std::optional<ValidatedHS3Document>
validate(HS3Document doc, Diagnostics &diagnostics, ValidateOptions const &options);
} // namespace HS3

/// An HS3 document. It is move-only, because generic objects refer to the JSON tree of the document.
class HS3Document {
public:
   HS3::Metadata metadata;
   std::vector<HS3::Distribution> distributions;
   std::vector<HS3::Function> functions;
   std::vector<HS3::Data> data;
   std::vector<HS3::Likelihood> likelihoods;
   std::vector<HS3::Domain> domains;
   std::vector<HS3::ParameterPoint> parameterPoints;
   std::vector<HS3::Analysis> analyses;
   HS3::RooFitInternal rooFit;

   HS3Document() = default;
   HS3Document(HS3Document &&) = default;
   HS3Document &operator=(HS3Document &&) = default;
   HS3Document(HS3Document const &) = delete;
   HS3Document &operator=(HS3Document const &) = delete;
};

namespace HS3 {

/// Options for upgradeDocument() and parseDocument().
struct ParseOptions {
   /// Fragments of a document can be imported without metadata.
   bool requireMetadata = true;
   /// If false, HistFactory distributions stay generic objects, because a user registered their own importer.
   bool typedHistFactory = true;
};

/// Rewrites everything that older versions of RooFit wrote in a way that is not part of the standard into the form that
/// parseDocument() understands.
void upgradeDocument(Detail::JSONNode &root, Diagnostics &diagnostics, ParseOptions const &options = {});

/// Same as upgradeDocument(), but for a single HistFactory distribution that doesn't come with a document. Names that
/// can't be resolved in the document are assumed to refer to distributions.
void upgradeHistFactoryDist(Detail::JSONNode &distribution, Version const &version, Diagnostics &diagnostics);

/// Reads a document. The generic objects in the document refer to nodes of the JSON tree, so the tree must outlive the
/// document. Problems are reported in `diagnostics`, and if there are errors the document must not be used.
HS3Document parseDocument(Detail::JSONNode const &root, Diagnostics &diagnostics, ParseOptions const &options = {});

/// Reads a HistFactory distribution that doesn't come with a document.
HistFactoryDist parseHistFactoryDist(Detail::JSONNode const &distribution, Diagnostics &diagnostics);

/// Reads a data set that doesn't come with a document.
Data parseData(Detail::JSONNode const &data, Diagnostics &diagnostics);

/// Reads the axes of binned data that doesn't come with a document.
std::vector<BinnedAxis> parseBinnedAxes(Detail::JSONNode const &axes, Diagnostics &diagnostics);

/// Reads the functions "interpolation" and "interpolation0d", which don't come with a document. Older versions of
/// RooFit wrote integer "interpolationCodes" instead of "interpolations".
InterpolatedFunction parseInterpolatedFunction(Detail::JSONNode const &node, Diagnostics &diagnostics);
InterpolatedNumber parseInterpolatedNumber(Detail::JSONNode const &node, Diagnostics &diagnostics);

/// Writes everything but the distributions and functions, which are written one by one while they are exported: the
/// generic ones by the exporters, and the others with functions like writeHistFactoryDist().
void writeDocument(Detail::JSONNode &root, HS3Document const &doc);

/// Writes a HistFactory distribution, without its name.
void writeHistFactoryDist(Detail::JSONNode &node, HistFactoryDist const &dist);

/// Writes the axes, contents and uncertainties of binned data, without its name and type.
void writeBinnedData(Detail::JSONNode &node, BinnedData const &data);

/// Writes an axis of binned data.
void writeAxis(Detail::JSONNode &node, BinnedAxis const &axis);

/// Writes the functions "interpolation" and "interpolation0d", with a single interpolation if they are all the same.
void writeInterpolatedFunction(Detail::JSONNode &node, InterpolatedFunction const &function);
void writeInterpolatedNumber(Detail::JSONNode &node, InterpolatedNumber const &function);

void writeInterpolation(Detail::JSONNode &node, Interpolation const &interpolation);

/// Writes a number, as an integer if it has no fractional part.
void writeNumber(Detail::JSONNode &node, double value);

/// Writes an array of numbers, each as an integer if it has no fractional part.
void writeContents(Detail::JSONNode &node, std::vector<double> const &values);

/// Options for validate().
struct ValidateOptions {
   /// Whether the document has to follow the current standard, or if older conventions of RooFit are tolerated.
   bool strict = false;
   /// Whether a name that is not defined in the document is known in some other way, e.g. in the workspace.
   std::function<bool(std::string const &)> isExternal;
};

} // namespace HS3

/// A document that passed validate(). Only validate() can create it, so the code that builds RooFit objects can rely on
/// the checks.
class ValidatedHS3Document {
public:
   HS3Document const &doc() const { return _doc; }

private:
   explicit ValidatedHS3Document(HS3Document doc) : _doc{std::move(doc)} {}
   friend std::optional<ValidatedHS3Document>
   HS3::validate(HS3Document doc, HS3::Diagnostics &diagnostics, HS3::ValidateOptions const &options);

   HS3Document _doc;
};

namespace HS3 {

/// Checks what can be checked without knowing how the objects are built: unique names, that the references can be
/// resolved, and that the sizes of arrays are consistent. Gives nothing if there are errors.
std::optional<ValidatedHS3Document>
validate(HS3Document doc, Diagnostics &diagnostics, ValidateOptions const &options = {});

/// The checks of validate(), for a document that is still needed if it doesn't pass them.
void checkDocument(HS3Document const &doc, Diagnostics &diagnostics, ValidateOptions const &options = {});

/// Same as validate(), but for a single HistFactory distribution that doesn't come with a document.
void validateHistFactoryDist(HistFactoryDist const &dist, std::string const &name, Diagnostics &diagnostics);

/// Reads the leading "MAJOR.MINOR" and the optional ".PATCH" of an HS3 version string.
bool parseVersion(std::string const &version, Version &out);

/// Combines the modifiers of a sample that have the same type and name, which HistFactory models can have but
/// documents can't: normsys by multiplying their variations, and histosys by adding up their differences to the
/// nominal contents. The modifiers of a type must be sorted by name. Other duplicates are errors.
void mergeDuplicateModifiers(HistFactoryDist &dist, std::string const &channel, Diagnostics &diagnostics);

/// Makes the most common interpolation of the modifiers the "default_interpolation", which they then don't repeat.
void factorOutDefaultInterpolation(HistFactoryDist &dist);

// HS3 requires the terms of a likelihood, and the components of a combined distribution or data set, to have distinct
// variates. For this, the generic distributions are read as far as the standard says what they consist of (e.g. the
// factors of a product), through the nodes that the document refers to.

/// Describes each variate that is shared although it must be distinct: by the terms of a likelihood, by the
/// components of a combined distribution that no likelihood lists (the others are checked through the likelihood), or
/// by the components of a combined data set, except for bookkeeping columns.
std::vector<std::string> sharedVariates(HS3Document const &doc);

/// In HS3 0.3 (HS3 PR #154, decision on HS3 #156), the "constraints" of a HistFactory modifier add no term to a
/// likelihood: every likelihood that contains the channel must list each named distribution exactly once, in
/// "distributions" or "aux_distributions", or as a factor of a product_dist or a component of a combined_dist reached
/// from them. One message for each named distribution of each likelihood that doesn't. For older documents, RooFit
/// adds the named constraints itself (see sharedVariates()).
std::vector<std::string> unlistedConstraints(HS3Document const &doc);

/// The variates of the "distributions" of a likelihood, which it pairs with data, with the columns of its combined
/// distributions.
std::set<std::string> variatesOf(HS3Document const &doc, Likelihood const &likelihood);

/// The variates of the distribution `distribution`, without the columns of a combined distribution.
std::set<std::string> variatesOf(HS3Document const &doc, std::string const &distribution);

/// The axes that several components of a combined data set have, by combined data set.
std::map<std::string, std::set<std::string>> sharedAxes(HS3Document const &doc);

/// The bookkeeping columns by combined data set: the declared ones (RooFitInternal::bookkeepingColumns), except for
/// those that are a variate of a distribution after all. They are exempt from the requirement of distinct variates.
std::map<std::string, std::set<std::string>> bookkeepingColumns(HS3Document const &doc);

/// Writes each term of the likelihoods once: a factor of a product that is a term of the likelihood as well would enter
/// it twice, so it is removed from the "factors" of the product in `root`, the JSON tree that the document is written
/// to. Only products that no other distribution uses are changed (being a component of a combined distribution is not
/// a use), only factors that are terms of every likelihood that has the product as a term, and a product whose factors
/// are all such terms is left as it is. Returns the removed factors by product.
std::map<std::string, std::set<std::string>> writeLikelihoodTermsOnce(HS3Document const &doc, Detail::JSONNode &root);

} // namespace HS3
} // namespace RooFit

#endif

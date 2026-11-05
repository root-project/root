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

#include <map>
#include <memory>
#include <optional>
#include <string>
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
   if (l.major != r.major) {
      return l.major < r.major;
   }
   if (l.minor != r.minor) {
      return l.minor < r.minor;
   }
   return l.patch < r.patch;
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

struct Data {
   std::string name;
   std::variant<PointData, UnbinnedData, BinnedData> body;
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
/// its type in RooFit::JSONIO::importers(), or by the factory expressions.
struct GenericObject {
   std::string type;
   Detail::JSONNode const *node = nullptr;
};

struct Distribution {
   std::string name;
   GenericObject body;
};

struct Function {
   std::string name;
   GenericObject body;
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

/// An HS3 document. It is move-only, because generic objects refer to the JSON tree that the document was read from.
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

/// Options for parseDocument().
struct ParseOptions {
   /// Fragments of a document can be imported without metadata.
   bool requireMetadata = true;
};

/// Reads a document. The generic objects in the document refer to nodes of the JSON tree, so the tree must outlive the
/// document. Problems are reported in `diagnostics`, and if there are errors the document must not be used.
HS3Document parseDocument(Detail::JSONNode const &root, Diagnostics &diagnostics, ParseOptions const &options = {});

/// Reads the leading "MAJOR.MINOR" and the optional ".PATCH" of an HS3 version string.
bool parseVersion(std::string const &version, Version &out);

} // namespace HS3
} // namespace RooFit

#endif

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

#include "HS3Schema.h"
#include "JSONIOUtils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <exception>
#include <sstream>

using RooFit::Detail::JSONNode;

namespace RooFit {
namespace HS3 {

constexpr auto supportedVersion = "0.3";

bool hasErrors(Diagnostics const &diagnostics)
{
   return std::any_of(diagnostics.begin(), diagnostics.end(),
                      [](Diagnostic const &d) { return d.severity == Diagnostic::Severity::Error; });
}

std::string Cursor::path() const
{
   if (!_parent) {
      return "";
   }
   std::string out = _parent->path();
   if (_key) {
      if (!out.empty()) {
         out += ".";
      }
      out += _key;
      return out;
   }
   out += "[" + std::to_string(_index) + "]";
   if (_node.is_map() && _node.has_child("name") && !_node["name"].is_container() && !_node["name"].is_null()) {
      out += "(" + _node["name"].val() + ")";
   }
   return out;
}

void Cursor::error(std::string const &message) const
{
   _diagnostics.push_back({Diagnostic::Severity::Error, path(), message});
}

void Cursor::warning(std::string const &message) const
{
   _diagnostics.push_back({Diagnostic::Severity::Warning, path(), message});
}

void readValue(Cursor const &c, std::string &out)
{
   if (c.node().is_container() || c.node().is_null()) {
      c.error("must be a string");
      return;
   }
   out = c.node().val();
}

void readValue(Cursor const &c, double &out)
{
   if (!c.node().is_number()) {
      c.error("must be a number");
      return;
   }
   out = c.node().val_double();
}

void readValue(Cursor const &c, int &out)
{
   // Read through double, such that an integer that is encoded as a JSON float like 1e6 is accepted.
   const double value = c.node().is_number() ? c.node().val_double() : 0.5;
   if (value != std::floor(value) || std::abs(value) > static_cast<double>(std::numeric_limits<int>::max())) {
      c.error("must be an integer");
      return;
   }
   out = static_cast<int>(value);
}

void readValue(Cursor const &c, bool &out)
{
   if (c.node().is_container() || c.node().is_null()) {
      c.error("must be a boolean");
      return;
   }
   try {
      out = c.node().val_bool();
   } catch (std::exception const &) {
      c.error("must be a boolean");
   }
}

bool parseVersion(std::string const &version, Version &out)
{
   std::size_t pos = 0;
   auto readNumber = [&](int &number) {
      const std::size_t begin = pos;
      while (pos < version.size() && std::isdigit(static_cast<unsigned char>(version[pos]))) {
         ++pos;
      }
      // Reject empty and absurdly long numbers, which std::stoi could not represent.
      if (pos == begin || pos - begin > 9) {
         return false;
      }
      number = std::stoi(version.substr(begin, pos - begin));
      return true;
   };
   Version parsed;
   if (!readNumber(parsed.major) || pos >= version.size() || version[pos] != '.') {
      return false;
   }
   ++pos;
   if (!readNumber(parsed.minor)) {
      return false;
   }
   if (pos < version.size() && version[pos] == '.') {
      ++pos;
      if (!readNumber(parsed.patch)) {
         parsed.patch = 0;
      }
   }
   out = parsed;
   return true;
}

namespace {

// Read a bound of an axis. Infinite bounds are written as null.
void readBound(Reader &ar, const char *key, std::optional<double> &out, double valueIfNull)
{
   if (!ar.has(key)) {
      return;
   }
   Cursor c{ar.cursor().node()[key], ar.cursor(), key};
   if (c.node().is_null()) {
      out = valueIfNull;
      return;
   }
   out.emplace();
   readValue(c, *out);
}

constexpr double infinity = std::numeric_limits<double>::infinity();

// Read bin edges, which must be at least two strictly ascending numbers.
void readEdges(Reader &ar, std::string const &axisName, std::vector<double> &edges)
{
   Cursor c{ar.cursor().node()["edges"], ar.cursor(), "edges"};
   const std::string context = "\"edges\" of axis '" + axisName + "' must ";
   if (!c.node().is_seq()) {
      c.error(context + "be a list");
      return;
   }
   readValue(c, edges);
   for (std::size_t i = 1; i < edges.size(); ++i) {
      if (!(edges[i] > edges[i - 1])) {
         c.error(context + "be strictly ascending");
         return;
      }
   }
   if (edges.size() < 2) {
      c.error(context + "contain at least two values");
   }
}

} // namespace

void describe(Reader &ar, Package &p)
{
   ar("name", p.name);
   ar("version", p.version);
}

void describe(Reader &ar, DomainAxis &axis)
{
   ar("name", axis.name);
   if (ar.has(fixedValueKey)) {
      // The domain of the parameter is the set with this single value: no range, binning or categories.
      for (const char *key : {"min", "max", "nbins", "edges", "categories"}) {
         if (ar.has(key)) {
            ar.cursor().error("the axis '" + axis.name + "' has \"" + fixedValueKey + "\" and \"" + key +
                              "\", which exclude each other");
         }
      }
      // Not read as an optional component, for which null would be the same as leaving it out.
      ar(fixedValueKey, axis.fixedValue.emplace());
      return;
   }
   readBound(ar, "min", axis.min, -infinity);
   readBound(ar, "max", axis.max, infinity);
   if (ar.has("edges")) {
      // Not necessarily ascending, because RooFit also writes parameterized ranges as edges.
      EdgesBinning binning;
      ar("edges", binning.edges);
      if (!binning.edges.empty()) {
         if (!axis.min) {
            axis.min = binning.edges.front();
         }
         if (!axis.max) {
            axis.max = binning.edges.back();
         }
      }
      axis.binning = std::move(binning);
   } else if (ar.has("nbins")) {
      RegularBinning binning;
      ar("nbins", binning.nbins);
      if (binning.nbins != 0) {
         axis.binning = binning;
      }
   }
}

void describe(Reader &ar, ContinuousAxis &axis)
{
   ar("name", axis.name);
   readBound(ar, "min", axis.min, -infinity);
   readBound(ar, "max", axis.max, infinity);
}

void describe(Reader &ar, BinnedAxis &axis)
{
   ar("name", axis.name);
   if (ar.has("edges")) {
      EdgesBinning binning;
      readEdges(ar, axis.name, binning.edges);
      if (binning.edges.size() >= 2) {
         axis.min = binning.edges.front();
         axis.max = binning.edges.back();
      }
      axis.binning = std::move(binning);
      return;
   }
   ar("min", axis.min);
   ar("max", axis.max);
   RegularBinning binning;
   ar("nbins", binning.nbins);
   axis.binning = binning;
}

void describe(Reader &ar, CategoricalAxis &axis)
{
   ar("name", axis.name);
   ar("categories", axis.categories);
}

namespace {

// The kind of an axis is determined by its components.
template <class ContinuousAxis_t>
void readAxis(Cursor const &c, std::variant<ContinuousAxis_t, CategoricalAxis> &out)
{
   if (!c.node().is_map()) {
      c.error("must be a struct");
      return;
   }
   Reader reader{c};
   // A fixed axis is not categorical. That it has no categories is checked when it is read.
   if (reader.has("categories") && !(std::is_same_v<ContinuousAxis_t, DomainAxis> && reader.has(fixedValueKey))) {
      CategoricalAxis axis;
      describe(reader, axis);
      out = std::move(axis);
   } else {
      ContinuousAxis_t axis;
      describe(reader, axis);
      out = std::move(axis);
   }
}

} // namespace

void readValue(Cursor const &c, std::variant<DomainAxis, CategoricalAxis> &out)
{
   readAxis(c, out);
}

void readValue(Cursor const &c, std::variant<ContinuousAxis, CategoricalAxis> &out)
{
   readAxis(c, out);
}

void readValue(Cursor const &c, std::variant<BinnedAxis, CategoricalAxis> &out)
{
   readAxis(c, out);
}

void describe(Reader &ar, Domain &domain)
{
   ar("name", domain.name);
   std::string type;
   if (ar.has("type")) {
      ar("type", type);
   }
   if (type != "product_domain") {
      ar.cursor().error("only domains of type \"product_domain\" are currently supported!");
      return;
   }
   ar.list("axes", domain.axes);
}

void readValue(Cursor const &c, ParameterValue &p)
{
   if (!c.node().is_map()) {
      c.error("must be a struct");
      return;
   }
   Reader ar{c};
   ar("name", p.name);
   if (ar.has("value")) {
      Cursor crsr{ar.cursor().node()["value"], ar.cursor(), "value"};
      if (crsr.node().is_number()) {
         p.value = crsr.node().val_double();
      } else {
         std::string label;
         readValue(crsr, label);
         p.value = std::move(label);
      }
   }
   std::optional<bool> isConst;
   ar("const", isConst);
   p.isConst = isConst.value_or(false);
   ar("err", p.error);
   ar("relErr", p.relError);
   ar("nbins", p.nbins);
}

void describe(Reader &ar, ParameterPoint &point)
{
   ar("name", point.name);
   ar.list("parameters", point.parameters);
}

void describe(Reader &ar, PointData &data)
{
   ar("value", data.value);
   ar("uncertainty", data.uncertainty);
   ar.list("axes", data.axes);
}

void describe(Reader &ar, UnbinnedData &data)
{
   ar.list("axes", data.axes);
   ar("weights", data.weights);
   data.hasEntriesUncertainties = ar.has("entries_uncertainties");

   if (!ar.has("entries")) {
      ar.cursor().error("does not define the required component \"entries\"");
      return;
   }
   Cursor entries{ar.cursor().node()["entries"], ar.cursor(), "entries"};
   if (!entries.node().is_seq()) {
      entries.error("key 'entries' is not a list!");
      return;
   }
   if (data.weights && data.weights->size() != entries.node().num_children()) {
      ar.cursor().error("inconsistent number of entries and weights!");
      return;
   }
   data.entries.reserve(entries.node().num_children() * data.axes.size());
   std::size_t iRow = 0;
   for (JSONNode const &row : entries.node().children()) {
      Cursor rowCursor{row, entries, iRow};
      if (!row.is_seq()) {
         rowCursor.error("coordinate point '" + std::to_string(iRow) + "' is not a list!");
         return;
      }
      if (row.num_children() != data.axes.size()) {
         rowCursor.error("inconsistent number of entries and observables!");
         return;
      }
      std::size_t iAxis = 0;
      for (JSONNode const &value : row.children()) {
         Cursor valueCursor{value, rowCursor, iAxis};
         if (auto const *category = std::get_if<CategoricalAxis>(&data.axes[iAxis])) {
            std::string label;
            readValue(valueCursor, label);
            auto found = std::find(category->categories.begin(), category->categories.end(), label);
            if (found == category->categories.end()) {
               valueCursor.error("'" + label + "' is not a category of the axis '" + category->name + "'");
               return;
            }
            data.entries.push_back(static_cast<double>(std::distance(category->categories.begin(), found)));
         } else {
            double coordinate = 0.0;
            readValue(valueCursor, coordinate);
            data.entries.push_back(coordinate);
         }
         ++iAxis;
      }
      ++iRow;
   }
}

void describe(Reader &ar, GaussianUncertainty &uncertainty)
{
   if (ar.has("type")) {
      std::string type;
      ar("type", type);
      if (type != "gaussian_uncertainty") {
         ar.cursor().error("unsupported type of uncertainty '" + type + "'");
      }
   }
   ar("sigma", uncertainty.sigma);
   if (ar.has("correlation")) {
      Cursor c{ar.cursor().node()["correlation"], ar.cursor(), "correlation"};
      // Besides a matrix, it can be the number 0 to indicate that there is no correlation.
      if (c.node().is_seq()) {
         readValue(c, uncertainty.correlation);
      } else if (!c.node().is_number() || c.node().val_double() != 0.0) {
         c.error("must be a matrix or 0");
      }
   }
}

void describe(Reader &ar, BinnedData &data)
{
   ar.list("axes", data.axes);
   ar("contents", data.contents);
   ar("uncertainty", data.uncertainty);
   // RooFit extension: older versions read the errors like in the samples of a HistFactory distribution.
   if (ar.has("errors")) {
      Cursor errors{ar.cursor().node()["errors"], ar.cursor(), "errors"};
      if (!errors.node().is_seq()) {
         errors.error("errors are not in list form");
      } else if (!data.uncertainty) {
         data.uncertainty.emplace();
         readValue(errors, data.uncertainty->sigma);
      }
   }
}

void readValue(Cursor const &c, Data &out)
{
   if (!c.node().is_map()) {
      c.error("must be a struct");
      return;
   }
   Reader ar{c};
   ar("name", out.name);
   std::string type;
   ar("type", type);
   if (type == "binned") {
      BinnedData body;
      describe(ar, body);
      out.body = std::move(body);
   } else if (type == "unbinned") {
      UnbinnedData body;
      describe(ar, body);
      out.body = std::move(body);
   } else if (type == "point") {
      PointData body;
      describe(ar, body);
      out.body = std::move(body);
   } else if (ar.has("type")) {
      c.error("unknown type of dataset '" + type + "'");
   }
}

void describe(Reader &ar, Likelihood &likelihood)
{
   ar("name", likelihood.name);
   ar.list("aux_distributions", likelihood.auxDistributions);

   std::vector<DistributionRef> distributions;
   ar("distributions", distributions);
   if (!ar.has("data")) {
      ar.cursor().error("does not define the required component \"data\"");
      return;
   }
   Cursor data{ar.cursor().node()["data"], ar.cursor(), "data"};
   if (!data.node().is_seq()) {
      data.error("must be an array");
      return;
   }
   if (data.node().num_children() != distributions.size()) {
      ar.cursor().error("\"distributions\" and \"data\" must have the same length");
      return;
   }
   likelihood.terms.reserve(distributions.size());
   std::size_t i = 0;
   for (JSONNode const &node : data.node().children()) {
      LikelihoodTerm term;
      term.distribution = std::move(distributions[i]);
      // Older versions of RooFit used the string "0" to say that there is no dataset.
      if (node.is_number() || (!node.is_container() && !node.is_null() && node.val() == "0")) {
         term.data = node.is_number() ? node.val_double() : 0.0;
      } else {
         DataRef ref;
         readValue(Cursor{node, data, i}, ref);
         term.data = std::move(ref);
      }
      likelihood.terms.push_back(std::move(term));
      ++i;
   }
}

void describe(Reader &ar, Analysis &analysis)
{
   ar("name", analysis.name);
   ar("likelihood", analysis.likelihood);
   // HS3 v0.2.9 used the singular keys "parameter_of_interest" (a list) and "domain" (a single name). They are still
   // accepted, but the plural keys take precedence.
   auto isLegacy = [&](const char *key, const char *legacyKey) {
      if (ar.has(key) || !ar.has(legacyKey)) {
         return false;
      }
      ar.cursor().warning(std::string("the key \"") + legacyKey + "\" of analysis '" + analysis.name +
                          "' is deprecated, please use \"" + key + "\" instead.");
      return true;
   };
   ar.list(isLegacy("parameters_of_interest", "parameter_of_interest") ? "parameter_of_interest"
                                                                       : "parameters_of_interest",
           analysis.parametersOfInterest);
   if (!isLegacy("domains", "domain")) {
      ar.list("domains", analysis.domains);
   } else if (ar.cursor().node()["domain"].is_seq()) {
      ar.list("domain", analysis.domains);
   } else {
      analysis.domains.emplace_back();
      ar("domain", analysis.domains.back());
   }
   ar("init", analysis.init);
   ar("prior", analysis.prior);
}

namespace {

template <class Object_t>
void readGenericObject(Cursor const &c, Object_t &out)
{
   if (!c.node().is_map()) {
      c.error("must be a struct");
      return;
   }
   Reader ar{c};
   ar("name", out.name);
   ar("type", out.body.type);
   out.body.node = &c.node();
}

} // namespace

void readValue(Cursor const &c, Distribution &out)
{
   readGenericObject(c, out);
}

void readValue(Cursor const &c, Function &out)
{
   readGenericObject(c, out);
}

void describe(Reader &ar, Attributes &attributes)
{
   if (ar.has("dict")) {
      Cursor dict{ar.cursor().node()["dict"], ar.cursor(), "dict"};
      if (!dict.node().is_map()) {
         dict.error("must be a struct");
         return;
      }
      for (JSONNode const &item : dict.node().children()) {
         readValue(Cursor{item, dict, item.key().c_str()}, attributes.dict[item.key()]);
      }
   }
   ar.list("tags", attributes.tags);
   if (ar.has("is_const_var")) {
      int isConstVar = 0;
      ar("is_const_var", isConstVar);
      attributes.isConstVar = isConstVar == 1;
   }
}

void describe(Reader &ar, CombinedDistribution &info)
{
   ar("index_cat", info.indexCat);
   ar("labels", info.labels);
   ar("indices", info.indices);
   ar("distributions", info.distributions);
   if (info.labels.size() != info.indices.size() || info.labels.size() != info.distributions.size()) {
      ar.cursor().error("mismatch in number of labels, indices and distributions!");
   }
}

void describe(Reader &ar, CombinedDataset &info)
{
   ar("index_cat", info.indexCat);
   ar("labels", info.labels);
   ar("indices", info.indices);
   if (info.labels.size() != info.indices.size()) {
      ar.cursor().error("mismatch in number of indices and labels!");
   }
}

void describe(Reader &ar, ModelConfigInfo &info)
{
   ar("mcName", info.mcName);
   ar("pdfName", info.pdfName);
   ar("combined_data_name", info.combinedDataName);
}

namespace {

template <class Map_t>
void readMap(Reader &ar, const char *key, Map_t &out)
{
   if (!ar.has(key)) {
      return;
   }
   Cursor c{ar.cursor().node()[key], ar.cursor(), key};
   if (c.node().is_null()) {
      return;
   }
   if (!c.node().is_map()) {
      c.error("must be a struct");
      return;
   }
   for (JSONNode const &item : c.node().children()) {
      const std::string itemKey = item.key();
      readValue(Cursor{item, c, itemKey.c_str()}, out[itemKey]);
   }
}

// The content of "misc" is not supposed to change the result of evaluating the model, so problems with the step widths
// are only warnings.
void readStepWidths(Cursor const &misc, std::vector<StepWidth> &out)
{
   JSONNode const *node = misc.node().find("minimization", "parameter_stepwidths");
   if (!node) {
      return;
   }
   if (!node->is_seq()) {
      misc.warning("minimization.parameter_stepwidths is not a sequence, skipping.");
      return;
   }
   for (JSONNode const &item : node->children()) {
      if (!item.is_map() || !item.has_child("name") || !item.has_child("step_width") ||
          !item["step_width"].is_number() || item["name"].is_container() || item["name"].is_null()) {
         misc.warning("skipping malformed parameter_stepwidths entry.");
         continue;
      }
      out.push_back({item["name"].val(), item["step_width"].val_double()});
   }
}

void readMisc(Cursor const &root, RooFitInternal &out)
{
   if (!root.node().has_child("misc")) {
      return;
   }
   Cursor misc{root.node()["misc"], root, "misc"};
   if (!misc.node().is_map()) {
      return;
   }
   readStepWidths(misc, out.stepWidths);
   if (!misc.node().has_child("ROOT_internal")) {
      return;
   }
   Cursor internal{misc.node()["ROOT_internal"], misc, "ROOT_internal"};
   if (!internal.node().is_map()) {
      return;
   }
   Reader ar{internal};
   readMap(ar, "attributes", out.attributes);
   readMap(ar, "combined_distributions", out.combinedDistributions);
   readMap(ar, "combined_datasets", out.combinedDatasets);
   readMap(ar, "ModelConfigs", out.modelConfigs);
}

void readMetadata(Cursor const &root, Metadata &out, ParseOptions const &options)
{
   JSONNode const *metadata = root.node().find("metadata");
   JSONNode const *versionNode = metadata ? metadata->find("hs3_version") : nullptr;
   if (!versionNode) {
      if (options.requireMetadata) {
         std::stringstream ss;
         ss << "The HS3 version is missing in the JSON!\n"
            << "Please include the HS3 version in the metadata field, e.g.:\n"
            << "    \"metadata\" :\n"
            << "    {\n"
            << "        \"hs3_version\" : \"" << supportedVersion << "\"\n"
            << "    }";
         root.error(ss.str());
      }
      return;
   }
   Cursor metadataCursor{*metadata, root, "metadata"};
   Cursor versionCursor{*versionNode, metadataCursor, "hs3_version"};
   if (versionNode->is_container() || versionNode->is_null() || versionNode->is_number()) {
      versionCursor.error(std::string("The HS3 version in the metadata must be a string, e.g. \"") + supportedVersion +
                          "\"");
      return;
   }
   out.hs3Version = versionNode->val();

   Version supported;
   parseVersion(supportedVersion, supported);
   if (!parseVersion(out.hs3Version, out.version)) {
      versionCursor.warning("cannot parse hs3_version '" + out.hs3Version +
                            "', expected a string of the form \"MAJOR.MINOR\".");
   } else if (supported < Version{out.version.major, out.version.minor}) {
      versionCursor.warning("the document has hs3_version '" + out.hs3Version +
                            "', which is newer than the version supported by this RooFit ('" + supportedVersion +
                            "'). Some content might not be imported correctly.");
   }

   Reader ar{metadataCursor};
   ar.list("packages", out.packages);
   ar.list("authors", out.authors);
   ar.list("publications", out.publications);
   ar("description", out.description);
}

} // namespace

HS3Document parseDocument(JSONNode const &root, Diagnostics &diagnostics, ParseOptions const &options)
{
   HS3Document doc;
   Cursor cursor{root, diagnostics};
   readMetadata(cursor, doc.metadata, options);
   if (hasErrors(diagnostics)) {
      return doc;
   }

   Reader ar{cursor};
   ar.list("distributions", doc.distributions);
   ar.list("functions", doc.functions);
   ar.list("data", doc.data);
   ar.list("likelihoods", doc.likelihoods);
   ar.list("domains", doc.domains);
   ar.list("parameter_points", doc.parameterPoints);
   ar.list("analyses", doc.analyses);
   readMisc(cursor, doc.rooFit);
   return doc;
}

} // namespace HS3
} // namespace RooFit

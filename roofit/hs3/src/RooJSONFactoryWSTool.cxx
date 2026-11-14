/*
 * Project: RooFit
 * Authors:
 *   Carsten D. Burgard, DESY/ATLAS, Dec 2021
 *
 * Copyright (c) 2022, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include <RooFitHS3/JSONIO.h>
#include <RooFitHS3/RooJSONFactoryWSTool.h>

#include <RooConstVar.h>
#include <RooRealVar.h>
#include <RooBinning.h>
#include <RooRangeBinning.h>
#include <RooAbsCategory.h>
#include <RooArgProxy.h>
#include <RooAbsProxy.h>
#include <RooCategory.h>
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooSimultaneous.h>
#include <RooFit/ModelConfig.h>
#include <RooAbsCollection.h>

#include "JSONIOUtils.h"
#include "HS3Builder.h"
#include "HS3Schema.h"
#include "JSONReadRecorder.h"

#include <TROOT.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

/** \class RooJSONFactoryWSTool
\ingroup roofit_dev_docs_hs3

When using \ref Roofitmain, statistical models can be conveniently handled and
stored as a RooWorkspace. However, for the sake of interoperability
with other statistical frameworks, and also ease of manipulation, it
may be useful to store statistical models in text form.

The RooJSONFactoryWSTool is a helper class to achieve exactly this,
exporting to and importing from JSON.

In order to import a workspace from a JSON file, you can do

~~~ {.py}
ws = ROOT.RooWorkspace("ws")
tool = ROOT.RooJSONFactoryWSTool(ws)
tool.importJSON("myjson.json")
~~~

Similarly, in order to export a workspace to a JSON file, you can do

~~~ {.py}
tool = ROOT.RooJSONFactoryWSTool(ws)
tool.exportJSON("myjson.json")
~~~

Analogously, in C++, you can do

~~~ {.cxx}
#include "RooFitHS3/RooJSONFactoryWSTool.h"
// ...
RooWorkspace ws("ws");
RooJSONFactoryWSTool tool(ws);
tool.importJSON("myjson.json");
~~~

and

~~~ {.cxx}
#include "RooFitHS3/RooJSONFactoryWSTool.h"
// ...
RooJSONFactoryWSTool tool(ws);
tool.exportJSON("myjson.json");
~~~

For more details, consult the tutorial <a href="rf515__hfJSON_8py.html">rf515_hfJSON</a>.

The RooJSONFactoryWSTool only knows about a limited set of classes for
import and export. If import or export of a class you're interested in
fails, you might need to add your own importer or exporter. Please
consult the relevant section in the \ref roofit_dev_docs to learn how to do that (\ref roofit_dev_docs_hs3).

You can always get a list of all the available importers and exporters by calling the following functions:
~~~ {.py}
ROOT.RooFit.JSONIO.printImporters()
ROOT.RooFit.JSONIO.printExporters()
ROOT.RooFit.JSONIO.printFactoryExpressions()
ROOT.RooFit.JSONIO.printExportKeys()
~~~

Alternatively, you can generate a LaTeX version of the available importers and exporters by calling
~~~ {.py}
tool = ROOT.RooJSONFactoryWSTool(ws)
tool.writedoc("hs3.tex")
~~~
*/

constexpr auto hs3VersionTag = "0.3";

using RooFit::HS3Document;
using RooFit::Detail::JSONNode;
using RooFit::Detail::JSONTree;

namespace HS3 = RooFit::HS3;

/// What the tool needs to know about the document that it is importing.
struct RooFit::HS3::ToolImportState {
   explicit ToolImportState(HS3Document const &doc) : document{doc}
   {
      // Without these indices, every cross-reference resolved during import (e.g. dependencies of a
      // PiecewiseInterpolation, or factory-expression arguments) would need a linear scan over all sibling objects,
      // which becomes O(N^2) on workspaces with thousands of entries.
      for (auto const &function : doc.functions) {
         functions.emplace(function.name, &function);
      }
      for (auto const &distribution : doc.distributions) {
         distributions.emplace(distribution.name, &distribution);
      }
      for (auto const &point : doc.parameterPoints) {
         if (point.name == "default_values") {
            defaultValues = &point;
            for (auto const &p : point.parameters) {
               defaultValuesByName.emplace(p.name, &p);
            }
            break;
         }
      }
   }

   HS3Document const &document;
   std::unordered_map<std::string, HS3::Function const *> functions;
   std::unordered_map<std::string, HS3::Distribution const *> distributions;
   HS3::ParameterPoint const *defaultValues = nullptr;
   std::unordered_map<std::string, HS3::ParameterValue const *> defaultValuesByName;
   HS3::DomainRanges ranges{document.domains};
};

/// The document that the tool is exporting.
struct RooFit::HS3::ToolExportState {
   HS3Document doc;
   JSONNode *root = nullptr;
   /// The ranges and binnings of the exported variables, by domain and variable name.
   std::map<std::string, std::map<std::string, HS3::DomainAxis>> domains;
   /// Set by the exporter of an object that is interpreted by the document.
   std::optional<HistFactoryDist> typedObject;
   std::vector<DistributionRef> channelConstraints;
};

void RooFit::HS3::setExportedObject(RooJSONFactoryWSTool &tool, HistFactoryDist dist,
                                    std::vector<DistributionRef> channelConstraints)
{
   tool._export->typedObject = std::move(dist);
   tool._export->channelConstraints = std::move(channelConstraints);
}

namespace {

HS3::Binning toBinning(RooAbsBinning const &binning)
{
   if (binning.isUniform()) {
      return HS3::RegularBinning{binning.numBins()};
   }
   HS3::EdgesBinning edges;
   edges.edges.push_back(binning.binLow(0));
   for (int i = 0; i < binning.numBins(); ++i) {
      edges.edges.push_back(binning.binHigh(i));
   }
   return edges;
}

constexpr double inf = std::numeric_limits<double>::infinity();
constexpr int isInfinite(double x)
{
   return (x >= +inf) ? +1 : ((x <= -inf) ? -1 : 0);
}

// The axis of a domain for a range and binning. Infinite bounds are omitted, and only a bounded range has a binning.
// A RooRangeBinning is a plain named range without bin structure, fully described by its bounds.
HS3::DomainAxis toDomainAxis(std::string const &name, RooAbsBinning const &binning)
{
   HS3::DomainAxis axis;
   axis.name = name;
   if (!isInfinite(binning.lowBound())) {
      axis.min = binning.lowBound();
   }
   if (!isInfinite(binning.highBound())) {
      axis.max = binning.highBound();
   }
   const bool hasBins = binning.numBins() > 0 && !dynamic_cast<RooRangeBinning const *>(&binning);
   if (axis.min && axis.max && hasBins) {
      axis.binning = toBinning(binning);
   }
   return axis;
}

// Record the range and binning of a variable in the "default_domain", and its named ranges in the domains of their
// names.
void recordDomains(std::map<std::string, std::map<std::string, HS3::DomainAxis>> &domains, RooRealVar const &var)
{
   domains[HS3::defaultDomainName][var.GetName()] = toDomainAxis(var.GetName(), var.getBinning());
   for (auto const &name : var.getBinningNames()) {
      if (!name.empty()) {
         domains[name][var.GetName()] = toDomainAxis(var.GetName(), var.getBinning(name.c_str()));
      }
   }
}

void appendDomain(std::vector<HS3::Domain> &domains, std::string const &name,
                  std::map<std::string, HS3::DomainAxis> axes)
{
   HS3::Domain &domain = domains.emplace_back();
   domain.name = name;
   for (auto &item : axes) {
      domain.axes.emplace_back(std::move(item.second));
   }
}

// Set the binning of a variable, or the binning of one of its named ranges.
void applyBinning(RooRealVar &var, HS3::Binning const &binning, const char *name = nullptr)
{
   if (auto const *edges = std::get_if<HS3::EdgesBinning>(&binning)) {
      if (edges->edges.empty()) {
         return;
      }
      RooBinning bins(edges->edges.front(), edges->edges.back());
      for (double edge : edges->edges) {
         bins.addBoundary(edge);
      }
      var.setBinning(bins, name);
   } else if (int nbins = std::get<HS3::RegularBinning>(binning).nbins) {
      var.setBins(nbins, name);
   }
}

void applyRange(RooRealVar &var, HS3::DomainAxis const &axis)
{
   if (axis.min) {
      if (isInfinite(*axis.min)) {
         var.removeMin();
      } else {
         var.setMin(*axis.min);
      }
   }
   if (axis.max) {
      if (isInfinite(*axis.max)) {
         var.removeMax();
      } else {
         var.setMax(*axis.max);
      }
   }
   if (axis.binning) {
      applyBinning(var, *axis.binning);
   }
}

// Create the variables of the domains that don't exist yet, and the domains other than the "default_domain" as named
// ranges.
void createVariables(RooWorkspace &ws, std::vector<HS3::Domain> const &domains, HS3::DomainRanges const &ranges)
{
   auto forEachAxis = [&](auto &&fn) {
      for (HS3::Domain const &domain : domains) {
         for (auto const &axis : domain.axes) {
            if (auto const *continuous = std::get_if<HS3::DomainAxis>(&axis)) {
               fn(domain.name, *continuous);
            }
         }
      }
   };
   // Parameters with a single value are RooConstVars, whichever domain gives the value.
   forEachAxis([&](std::string const &, HS3::DomainAxis const &axis) {
      if (axis.fixedValue && !ws.arg(axis.name)) {
         ws.import(RooConstVar{axis.name.c_str(), axis.name.c_str(), *axis.fixedValue}, RooFit::Silence());
      }
   });
   forEachAxis([&](std::string const &, HS3::DomainAxis const &axis) {
      if (!axis.fixedValue && ranges.find(axis.name) == &axis && !ws.arg(axis.name)) {
         RooRealVar var{axis.name.c_str(), axis.name.c_str(), axis.min.value_or(-inf), axis.max.value_or(inf)};
         if (axis.binning) {
            applyBinning(var, *axis.binning);
         }
         ws.import(var, RooFit::Silence());
      }
   });
   forEachAxis([&](std::string const &domain, HS3::DomainAxis const &axis) {
      RooRealVar *var = ws.var(axis.name);
      if (var && domain != HS3::defaultDomainName) {
         var->setRange(domain.c_str(), axis.min.value_or(-inf), axis.max.value_or(inf));
         if (axis.binning) {
            applyBinning(*var, *axis.binning, domain.c_str());
         }
      }
   });
}

} // namespace

namespace {

// Keeps track of the objects that are being imported. Objects that depend on each other would otherwise recurse until
// the stack overflows.
class ImportGuard {
public:
   ImportGuard(std::vector<std::string> &stack, std::string const &name) : _stack{stack}
   {
      if (std::find(_stack.begin(), _stack.end(), name) != _stack.end()) {
         std::string chain;
         for (std::string const &ancestor : _stack) {
            chain += ancestor + " -> ";
         }
         RooJSONFactoryWSTool::error("circular dependency in the document: " + chain + name);
      }
      _stack.push_back(name);
   }
   ~ImportGuard() { _stack.pop_back(); }

private:
   std::vector<std::string> &_stack;
};

} // namespace

void RooFit::HS3::report(Diagnostics const &diagnostics)
{
   std::string errors;
   for (Diagnostic const &diagnostic : diagnostics) {
      const std::string text =
         diagnostic.path.empty() ? diagnostic.message : diagnostic.path + ": " + diagnostic.message;
      if (diagnostic.severity == Diagnostic::Severity::Error) {
         errors += (errors.empty() ? "" : "\n") + text;
      } else {
         RooJSONFactoryWSTool::warning("RooFitHS3: " + text);
      }
   }
   if (!errors.empty()) {
      RooJSONFactoryWSTool::error(errors);
   }
}

namespace {

// True if the number of components in `data` matches the number of categories in `pdf`.
bool matches(const RooJSONFactoryWSTool::CombinedData &data, const RooSimultaneous *pdf)
{
   return data.components.size() == pdf->indexCat().size();
}

// Configure `v` (value, error, binning, constness) from the parameter `p`.
void configureVariable(HS3::DomainRanges const &ranges, HS3::ParameterValue const &p, RooRealVar &v)
{
   if (p.value) {
      if (auto const *value = std::get_if<double>(&*p.value)) {
         v.setVal(*value);
      }
   }
   if (HS3::DomainAxis const *range = ranges.find(v.GetName())) {
      applyRange(v, *range);
   }
   if (p.nbins)
      v.setBins(*p.nbins);
   if (p.relError)
      v.setError(v.getVal() * *p.relError);
   if (p.error)
      v.setError(*p.error);
   v.setConstant(p.isConst);
}

std::string genPrefix(const JSONNode &p, bool trailing_underscore)
{
   std::string prefix;
   if (!p.is_map())
      return prefix;
   if (auto node = p.find("namespaces")) {
      for (const auto &ns : node->children()) {
         if (!prefix.empty())
            prefix += "_";
         prefix += ns.val();
      }
   }
   if (trailing_underscore && !prefix.empty())
      prefix += "_";
   return prefix;
}

// helpers for serializing / deserializing binned datasets
void genIndicesHelper(std::vector<std::vector<int>> &combinations, std::vector<int> &curr_comb,
                      const std::vector<int> &vars_numbins, size_t curridx)
{
   if (curridx == vars_numbins.size()) {
      // we have filled a combination. Copy it.
      combinations.emplace_back(curr_comb);
   } else {
      for (int i = 0; i < vars_numbins[curridx]; ++i) {
         curr_comb[curridx] = i;
         ::genIndicesHelper(combinations, curr_comb, vars_numbins, curridx + 1);
      }
   }
}

// Import string attributes ("dict") and boolean tags ("tags") onto `arg`.
void importAttributes(RooAbsArg *arg, HS3::Attributes const &attributes)
{
   for (auto const &attr : attributes.dict) {
      arg->setStringAttribute(attr.first.c_str(), attr.second.c_str());
   }
   for (auto const &tag : attributes.tags) {
      arg->setAttribute(tag.c_str());
   }
}

void addIfPresent(RooArgSet &out, RooArgSet const *args)
{
   if (args) {
      out.add(*args, true);
   }
}

void collectParameterStepWidthCandidatesFromModelConfigs(RooWorkspace const &workspace, RooArgSet &candidates,
                                                         RooArgSet &excluded)
{
   for (TObject *obj : workspace.allGenericObjects()) {
      auto const *mc = dynamic_cast<RooFit::ModelConfig const *>(obj);
      if (!mc) {
         continue;
      }

      addIfPresent(candidates, mc->GetParametersOfInterest());
      addIfPresent(candidates, mc->GetNuisanceParameters());

      addIfPresent(excluded, mc->GetObservables());
      addIfPresent(excluded, mc->GetGlobalObservables());
      addIfPresent(excluded, mc->GetConditionalObservables());
   }
}

void collectParameterStepWidthCandidatesFromPdfs(std::vector<RooAbsPdf *> const &pdfs,
                                                 std::vector<RooAbsData *> const &data, RooArgSet &candidates,
                                                 RooArgSet &excluded)
{
   for (RooAbsPdf const *pdf : pdfs) {
      RooArgSet observables;
      for (RooAbsData const *dataset : data) {
         std::unique_ptr<RooArgSet> pdfObs{pdf->getObservables(*dataset->get())};
         observables.add(*pdfObs, true);
      }

      if (observables.empty()) {
         continue;
      }

      RooArgSet params;
      pdf->getParameters(&observables, params);
      candidates.add(params, true);
      excluded.add(observables, true);
   }
}

void exportParameterStepWidths(RooWorkspace const &workspace, std::vector<RooAbsPdf *> const &pdfs,
                               std::vector<RooAbsData *> const &data, std::vector<HS3::StepWidth> &out)
{
   RooArgSet candidates;
   RooArgSet excluded;

   collectParameterStepWidthCandidatesFromModelConfigs(workspace, candidates, excluded);
   collectParameterStepWidthCandidatesFromPdfs(pdfs, data, candidates, excluded);

   candidates.sort();

   for (RooAbsArg *arg : candidates) {
      if (excluded.find(*arg)) {
         continue;
      }

      auto *var = dynamic_cast<RooRealVar *>(arg);
      if (!var || !var->hasError()) {
         continue;
      }

      out.push_back({var->GetName(), var->getError()});
   }
}

void importParameterStepWidths(RooWorkspace &workspace, HS3Document const &doc)
{
   for (HS3::StepWidth const &stepWidth : doc.rooFit.stepWidths) {
      auto *var = dynamic_cast<RooRealVar *>(workspace.arg(stepWidth.name));
      if (!var) {
         RooJSONFactoryWSTool::warning("RooFitHS3: skipping parameter_stepwidths entry for unknown or non-real "
                                       "variable '" +
                                       stepWidth.name + "'.");
         continue;
      }
      var->setError(stepWidth.width);
   }
}

// RooWSFactoryTool expression handling
std::string generate(const RooFit::JSONIO::ImportExpression &ex, const JSONNode &p, RooJSONFactoryWSTool *tool)
{
   std::stringstream expression;
   std::string classname(ex.tclass->GetName());
   size_t colon = classname.find_last_of(':');
   expression << (colon < classname.size() ? classname.substr(colon + 1) : classname);
   bool first = true;
   const auto &name = RooJSONFactoryWSTool::name(p);
   for (auto k : ex.arguments) {
      expression << (first ? "::" + name + "(" : ",");
      first = false;
      if (k == "true" || k == "false") {
         expression << (k == "true" ? "1" : "0");
      } else if (!p.has_child(k)) {
         RooJSONFactoryWSTool::error("node '" + name + "' is missing key '" + k + "'");
      } else if (p[k].is_seq()) {
         bool firstInner = true;
         expression << "{";
         for (RooAbsArg *arg : tool->requestArgList<RooAbsReal>(p, k)) {
            expression << (firstInner ? "" : ",") << arg->GetName();
            firstInner = false;
         }
         expression << "}";
      } else {
         tool->requestArg<RooAbsReal>(p, p[k].key());
         expression << p[k].val();
      }
   }
   expression << ")";
   return expression.str();
}

// Generate all combinations of bin indices for the RooRealVars in `vars`.
std::vector<std::vector<int>> generateBinIndices(const RooArgSet &vars)
{
   std::vector<std::vector<int>> combinations;
   std::vector<int> vars_numbins;
   vars_numbins.reserve(vars.size());
   for (const auto *absv : static_range_cast<RooRealVar *>(vars)) {
      vars_numbins.push_back(absv->getBins());
   }
   std::vector<int> curr_comb(vars.size());
   ::genIndicesHelper(combinations, curr_comb, vars_numbins, 0);
   return combinations;
}

template <typename... Keys_t>
JSONNode const *findRooFitInternal(JSONNode const &node, Keys_t const &...keys)
{
   return node.find("misc", "ROOT_internal", keys...);
}

// Export the string attributes and tags of `arg` into the ROOT-internal attributes.
void exportAttributes(const RooAbsArg *arg, std::map<std::string, HS3::Attributes> &attributes)
{
   // If this RooConst is a literal number, we don't need to export the attributes.
   if (isLiteralConstVar(*arg) || dynamic_cast<RooConstVar const *>(arg)) {
      return;
   }

   HS3::Attributes *out = nullptr;
   auto initialize = [&]() {
      if (!out) {
         out = &attributes[arg->GetName()];
      }
   };

   // export all string attributes of an object
   for (const auto &it : arg->stringAttributes()) {
      // Skip some RooFit internals
      if (it.first == "factory_tag" || it.first == "PROD_TERM_TYPE")
         continue;
      initialize();
      out->dict[it.first] = it.second;
   }
   for (auto const &attr : arg->attributes()) {
      // Skip some RooFit internals
      if (attr == "SnapShot_ExtRefClone" || attr == "RooRealConstant_Factory_Object")
         continue;
      initialize();
      out->tags.push_back(attr);
   }
}

} // namespace

HS3::BinnedAxis RooFit::HS3::toBinnedAxis(RooRealVar const &var)
{
   RooJSONFactoryWSTool::testValidName(var.GetName(), false);
   RooAbsBinning const &binning = var.getBinning();
   HS3::BinnedAxis axis;
   axis.name = var.GetName();
   axis.min = binning.lowBound();
   axis.max = binning.highBound();
   axis.binning = toBinning(binning);
   return axis;
}

HS3::NameOrNumber RooFit::HS3::toNameOrNumber(RooAbsArg const &arg)
{
   if (isLiteralConstVar(arg)) {
      return static_cast<RooConstVar const &>(arg).getVal();
   }
   return std::string{arg.GetName()};
}

namespace {

std::unique_ptr<RooRealVar> createObservable(HS3::BinnedAxis const &axis)
{
   auto obs = std::make_unique<RooRealVar>(axis.name.c_str(), axis.name.c_str(), axis.min, axis.max);
   applyBinning(*obs, axis.binning);
   return obs;
}

// Create the variables for the axes of binned data.
RooArgSet createBinnedObservables(std::vector<std::variant<HS3::BinnedAxis, HS3::CategoricalAxis>> const &axes)
{
   RooArgSet vars;
   for (auto const &axisVariant : axes) {
      auto const *axis = std::get_if<HS3::BinnedAxis>(&axisVariant);
      if (!axis) {
         RooJSONFactoryWSTool::error("categorical axes of binned data are not supported yet");
      }
      vars.addOwned(createObservable(*axis));
   }
   return vars;
}

// Make the range of an observable of unbinned data consistent with the "min" and "max" of the axis. Variables that
// didn't get their range from a domain take it from the axis. If `ranges` is null, all ranges are considered explicit.
void applyUnbinnedAxisRange(HS3::ContinuousAxis const &axis, std::string const &dataName, RooRealVar &var,
                            HS3::DomainRanges const *ranges)
{
   if (!axis.min || !axis.max) {
      return;
   }
   if (ranges && !ranges->contains(axis.name)) {
      var.setRange(*axis.min, *axis.max);
   } else if (var.getMin() != *axis.min || var.getMax() != *axis.max) {
      std::stringstream ss;
      ss << "RooFitHS3: the range [" << *axis.min << ", " << *axis.max << "] of the axis '" << axis.name
         << "' of the dataset '" << dataName << "' differs from the range [" << var.getMin() << ", " << var.getMax()
         << "] of the variable.";
      RooJSONFactoryWSTool::warning(ss.str());
   }
}

std::unique_ptr<RooAbsData> createUnbinnedData(std::string const &name, HS3::UnbinnedData const &input,
                                               RooWorkspace &workspace, HS3::DomainRanges const *ranges)
{
   RooArgList varlist;
   for (auto const &axisVariant : input.axes) {
      auto const *axis = std::get_if<HS3::ContinuousAxis>(&axisVariant);
      if (!axis) {
         RooJSONFactoryWSTool::error("categorical axes of unbinned data are not supported yet");
      }
      RooRealVar *var = workspace.var(axis->name);
      if (!var) {
         RooJSONFactoryWSTool::error("The observable \"" + axis->name + "\" could not be found in the workspace!");
      }
      varlist.add(*var);
      applyUnbinnedAxisRange(*axis, name, *var, ranges);
   }
   RooArgSet vars(varlist);
   auto data = std::make_unique<RooDataSet>(name, name, vars, RooFit::WeightVar());
   const std::size_t nAxes = input.axes.size();
   for (std::size_t i = 0; i < input.numEntries(); ++i) {
      for (std::size_t j = 0; j < nAxes; ++j) {
         static_cast<RooRealVar *>(varlist.at(j))->setVal(input.entries[i * nAxes + j]);
      }
      data->add(vars, input.weights ? (*input.weights)[i] : 1.);
   }
   return data;
}

// Create a RooAbsData (binned or unbinned) from its description.
std::unique_ptr<RooAbsData> loadData(HS3::Data const &input, RooWorkspace &workspace, HS3::DomainRanges const *ranges)
{
   RooJSONFactoryWSTool::testValidName(input.name, true);

   // What RooFit can't store is reported, instead of being dropped silently.
   if (auto const *binned = std::get_if<HS3::BinnedData>(&input.body)) {
      if (binned->uncertainty && !binned->uncertainty->correlation.empty()) {
         RooJSONFactoryWSTool::warning("RooFitHS3: the correlations of the uncertainties of the dataset '" +
                                       input.name + "' are ignored, because RooDataHist can't store them.");
      }
      return HS3::createDataHist(input.name, createBinnedObservables(binned->axes), binned->contents,
                                 binned->uncertainty ? &binned->uncertainty->sigma : nullptr);
   }
   if (auto const *unbinned = std::get_if<HS3::UnbinnedData>(&input.body)) {
      if (unbinned->hasEntriesUncertainties) {
         RooJSONFactoryWSTool::warning("RooFitHS3: the \"entries_uncertainties\" of the dataset '" + input.name +
                                       "' are ignored, because RooDataSet can't store them.");
      }
      return createUnbinnedData(input.name, *unbinned, workspace, ranges);
   }
   RooJSONFactoryWSTool::error("datasets of type \"point\" like '" + input.name + "' are not supported yet");
}

// The version of ROOT that writes the document, with dots only such that it can be read automatically.
std::string rootVersion()
{
   std::string versionName = gROOT->GetVersion();
   std::replace(versionName.begin(), versionName.end(), '/', '.');
   return versionName;
}

// A distribution of the document can be a function in RooFit, or it can be skipped.
void requirePdf(RooAbsArg const *arg, std::string const &what)
{
   if (!dynamic_cast<RooAbsPdf const *>(arg)) {
      RooJSONFactoryWSTool::error(what + " is " +
                                  (arg ? std::string("a ") + arg->ClassName() + ", not a pdf" : "not imported"));
   }
}

// Import an analysis (likelihood + domains) as one or more ModelConfig objects into the workspace.
void importAnalysis(HS3Document const &doc, HS3::Analysis const &analysis, RooWorkspace &workspace,
                    const std::vector<std::unique_ptr<RooAbsData>> &datasets)
{
   // if this is a toplevel pdf, also create a modelConfig for it
   std::string const &analysisName = analysis.name;
   auto foundInfo = doc.rooFit.modelConfigs.find(analysisName);
   HS3::ModelConfigInfo const *mcInfo = foundInfo != doc.rooFit.modelConfigs.end() ? &foundInfo->second : nullptr;

   std::string mcname = mcInfo && mcInfo->mcName ? *mcInfo->mcName : analysisName;
   if (workspace.obj(mcname))
      return;

   workspace.import(RooFit::ModelConfig{mcname.c_str(), mcname.c_str()});
   auto *mc = static_cast<RooFit::ModelConfig *>(workspace.obj(mcname));
   mc->SetWS(workspace);

   auto foundLikelihood = std::find_if(doc.likelihoods.begin(), doc.likelihoods.end(),
                                       [&](HS3::Likelihood const &l) { return l.name == analysis.likelihood.name; });
   if (foundLikelihood == doc.likelihoods.end()) {
      throw std::runtime_error("likelihood node not found!");
   }
   HS3::Likelihood const &likelihood = *foundLikelihood;

   auto checkPdf = [&](RooAbsArg const *arg, std::string const &name, std::string const &what) {
      requirePdf(arg, "the " + what + " '" + name + "' of analysis '" + analysisName + "'");
   };

   std::vector<std::string> nllDistNames;
   nllDistNames.reserve(likelihood.terms.size());
   for (auto const &term : likelihood.terms) {
      checkPdf(workspace.arg(term.distribution.name), term.distribution.name, "distribution");
      nllDistNames.push_back(term.distribution.name);
   }
   // The constraint terms of HistFactory distributions are listed as auxiliary distributions, but they are part of the
   // distributions in RooFit.
   std::set<std::string> channelConstraints;
   for (auto const &item : doc.rooFit.channelConstraints) {
      for (auto const &ref : item.second) {
         channelConstraints.insert(ref.name);
      }
   }
   RooArgSet extConstraints;
   for (auto const &aux : likelihood.auxDistributions) {
      RooAbsArg *extConstraint = workspace.arg(aux.name);
      if (!extConstraint) {
         continue;
      }
      checkPdf(extConstraint, aux.name, "auxiliary distribution");
      // Only if the constraint really ended up in a distribution of the likelihood.
      if (channelConstraints.count(aux.name) &&
          std::any_of(nllDistNames.begin(), nllDistNames.end(), [&](std::string const &distName) {
             RooAbsArg const *dist = workspace.arg(distName);
             return dist && dist->dependsOn(*extConstraint);
          })) {
         continue;
      }
      extConstraints.add(*extConstraint);
   }
   RooArgSet observables;
   for (auto const &term : likelihood.terms) {
      auto const *dataRef = std::get_if<HS3::DataRef>(&term.data);
      if (!dataRef) {
         continue;
      }
      bool found = false;
      for (const auto &d : datasets) {
         if (d->GetName() == dataRef->name) {
            found = true;
            observables.add(*d->get(), true);
         }
      }
      if (!found)
         throw std::runtime_error("dataset '" + dataRef->name + "' cannot be found!");
   }

   std::string const pdfName = mcInfo && mcInfo->pdfName ? *mcInfo->pdfName : "simPdf";

   RooAbsPdf *pdf = workspace.pdf(pdfName);

   if (!pdf) {
      // if there is no simultaneous pdf, we can check whether there is only one pdf in the list
      if (nllDistNames.size() == 1) {
         // if so, we can use that one to populate the ModelConfig
         pdf = workspace.pdf(nllDistNames[0]);
      } else {
         // otherwise, we have no choice but to build a simPdf by hand
         std::string simPdfName = analysisName + "_simPdf";
         std::string indexCatName = analysisName + "_categoryIndex";
         RooCategory indexCat{indexCatName.c_str(), indexCatName.c_str()};
         std::map<std::string, RooAbsPdf *> pdfMap;
         for (std::size_t i = 0; i < nllDistNames.size(); ++i) {
            indexCat.defineType(nllDistNames[i], i);
            pdfMap[nllDistNames[i]] = workspace.pdf(nllDistNames[i]);
         }
         RooSimultaneous simPdf{simPdfName.c_str(), simPdfName.c_str(), pdfMap, indexCat};
         workspace.import(simPdf, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
         pdf = workspace.pdf(simPdfName);
      }
   }

   mc->SetPdf(*pdf);

   if (!extConstraints.empty())
      mc->SetExternalConstraints(extConstraints);

   if (analysis.prior) {
      if (RooAbsArg *prior = workspace.arg(analysis.prior->name)) {
         checkPdf(prior, analysis.prior->name, "prior");
         mc->SetPriorPdf(static_cast<RooAbsPdf &>(*prior));
      }
   }

   RooArgSet parametersOfInterest;
   for (auto const &poi : analysis.parametersOfInterest) {
      RooAbsArg *arg = workspace.arg(poi.name);
      if (!arg) {
         RooJSONFactoryWSTool::error("the object '" + poi.name + "' in \"parameters_of_interest\" of analysis '" +
                                     analysisName + "' does not exist");
      }
      if (!dynamic_cast<RooRealVar *>(arg)) {
         RooJSONFactoryWSTool::error("the parameter of interest '" + poi.name + "' of analysis '" + analysisName +
                                     "' is a " + arg->ClassName() + ", not a variable" +
                                     (dynamic_cast<RooConstVar *>(arg) ? ", because no domain lists it" : ""));
      }
      parametersOfInterest.add(*arg);
   }

   mc->SetParametersOfInterest(parametersOfInterest);
   mc->SetObservables(observables);
   RooArgSet pars;
   pdf->getParameters(&observables, pars);

   // Figure out the set parameters that appear in the main measurement:
   // getAllConstraints() has the side effect to remove all parameters from
   // "mainPars" that are not part of any pdf over observables.
   RooArgSet mainPars{pars};
   pdf->getAllConstraints(observables, mainPars, /*stripDisconnected*/ true);

   RooArgSet domainPars;
   for (auto const &domainRef : analysis.domains) {
      auto thisDomain = std::find_if(doc.domains.begin(), doc.domains.end(),
                                     [&](HS3::Domain const &d) { return d.name == domainRef.name; });
      if (thisDomain == doc.domains.end())
         continue;
      for (auto const &axis : thisDomain->axes) {
         auto *wsvar = workspace.var(std::visit([](auto const &a) -> std::string const & { return a.name; }, axis));
         if (wsvar)
            domainPars.add(*wsvar);
      }
   }

   RooArgSet nps;
   RooArgSet globs;
   for (const auto &p : pars) {
      if (mc->GetParametersOfInterest()->find(*p))
         continue;
      if (p->isConstant() && !mainPars.find(*p) && domainPars.find(*p)) {
         globs.add(*p);
      } else if (domainPars.find(*p)) {
         nps.add(*p);
      }
   }

   mc->SetGlobalObservables(globs);
   mc->SetNuisanceParameters(nps);

   if (mcInfo && mcInfo->combinedDataName) {
      pdf->setStringAttribute("combined_data_name", mcInfo->combinedDataName->c_str());
   }

   // A missing parameter point was reported by HS3::validate().
   RooArgSet const *init = analysis.init ? workspace.getSnapshot(analysis.init->name.c_str()) : nullptr;
   if (init) {
      const std::string &initName = analysis.init->name;
      // ModelConfig::SetSnapshot() saves the snapshot as "<mc name>_<set name>_snapshot". If "init" already follows
      // this pattern, which is the case for files written by RooFit, pick the set name such that it is preserved.
      std::string setName = initName;
      const std::string prefix = mcname + "_";
      const std::string suffix = "_snapshot";
      if (initName.size() >= prefix.size() + suffix.size() && startsWith(initName, prefix) &&
          endsWith(initName, suffix)) {
         setName = initName.substr(prefix.size(), initName.size() - prefix.size() - suffix.size());
      }
      // Take an owning copy: SetSnapshot() might replace the workspace snapshot that `init` points to.
      RooArgSet initSet;
      init->snapshot(initSet);
      initSet.setName(setName.c_str());
      mc->SetSnapshot(initSet);
   }
}

void combinePdfs(HS3Document const &doc, RooWorkspace &ws)
{
   for (auto const &item : doc.rooFit.combinedDistributions) {
      std::string const &combinedName = item.first;
      HS3::CombinedDistribution const &info = item.second;

      RooCategory indexCat{info.indexCat.c_str(), info.indexCat.c_str()};
      std::map<std::string, RooAbsPdf *> pdfMap;

      for (std::size_t iChannel = 0; iChannel < info.labels.size(); ++iChannel) {
         indexCat.defineType(info.labels[iChannel], info.indices[iChannel]);
         std::string const &name = info.distributions[iChannel].name;
         requirePdf(ws.arg(name),
                    "the distribution '" + name + "' of the combined distribution '" + combinedName + "'");
         pdfMap[info.labels[iChannel]] = ws.pdf(name);
      }

      RooSimultaneous simPdf{combinedName.c_str(), combinedName.c_str(), pdfMap, indexCat};
      ws.import(simPdf, RooFit::RecycleConflictNodes(true), RooFit::Silence(true));
   }
}

void combineDatasets(HS3Document const &doc, std::vector<std::unique_ptr<RooAbsData>> &datasets)
{
   for (auto const &item : doc.rooFit.combinedDatasets) {
      std::string const &combinedName = item.first;
      HS3::CombinedDataset const &info = item.second;

      // Create the combined dataset for RooFit
      std::map<std::string, std::unique_ptr<RooAbsData>> dsMap;
      RooCategory indexCat{info.indexCat.c_str(), info.indexCat.c_str()};
      RooArgSet allVars{indexCat};
      for (std::size_t iChannel = 0; iChannel < info.labels.size(); ++iChannel) {
         auto componentName = combinedName + "_" + info.labels[iChannel];
         // We move the found channel data out of the "datasets" vector, such that
         // the data components don't get imported anymore.
         auto found =
            std::find_if(datasets.begin(), datasets.end(), [&](auto &d) { return d && d->GetName() == componentName; });
         if (found == datasets.end()) {
            RooJSONFactoryWSTool::error("unable to obtain component matching component name '" + componentName + "'");
         }
         allVars.add(*(*found)->get(), true);
         dsMap.insert({info.labels[iChannel], std::move(*found)});
         indexCat.defineType(info.labels[iChannel], info.indices[iChannel]);
      }

      auto combined = std::make_unique<RooDataSet>(combinedName, combinedName, allVars, RooFit::Import(dsMap),
                                                   RooFit::Index(indexCat));
      datasets.emplace_back(std::move(combined));
   }
}

template <class T>
void sortByName(T &coll)
{
   std::sort(coll.begin(), coll.end(), [](auto &l, auto &r) { return strcmp(l->GetName(), r->GetName()) < 0; });
}

/// Check whether an object is a top-level object of the workspace, i.e. not a
/// component of some other workspace object. Only clients that are part of the
/// workspace count: RooFit also registers evaluation artifacts as clients,
/// like the normalization integral that RooAbsPdf caches after
/// getVal(normSet), or the integral returned by createIntegral(). These live
/// outside the workspace and are not evidence that the object is a sub-node of
/// a bigger model (see https://github.com/root-project/root/issues/23221).
bool isTopLevel(RooAbsArg const &arg, RooWorkspace const &ws)
{
   for (RooAbsArg const *client : arg.clients()) {
      if (ws.components().containsInstance(*client)) {
         return false;
      }
   }
   return true;
}

/// Find the single category observable of a dataset, if any.
RooAbsCategory *findCategoryObservable(RooAbsData const &data)
{
   RooAbsCategory *cat = nullptr;
   for (RooAbsArg *obs : *data.get()) {
      if (auto *c = dynamic_cast<RooAbsCategory *>(obs)) {
         if (cat) {
            RooJSONFactoryWSTool::error("dataset '" + std::string(data.GetName()) +
                                        " has several category observables!");
         }
         cat = c;
      }
   }
   return cat;
}

} // namespace

RooArgSet RooFit::HS3::createObservables(std::vector<BinnedAxis> const &axes)
{
   RooArgSet vars;
   for (BinnedAxis const &axis : axes) {
      vars.addOwned(createObservable(axis));
   }
   return vars;
}

std::unique_ptr<RooDataHist> RooFit::HS3::createDataHist(std::string const &name, RooArgSet const &vars,
                                                         std::vector<double> const &contents,
                                                         std::vector<double> const *errors)
{
   if (errors && errors->size() != contents.size()) {
      RooJSONFactoryWSTool::error("inconsistent number of errors and contents in '" + name + "': errors=" +
                                  std::to_string(errors->size()) + ", contents=" + std::to_string(contents.size()));
   }
   auto bins = generateBinIndices(vars);
   if (contents.size() != bins.size()) {
      RooJSONFactoryWSTool::error("inconsistent bin numbers: contents=" + std::to_string(contents.size()) +
                                  ", bins=" + std::to_string(bins.size()));
   }
   auto dh = std::make_unique<RooDataHist>(name, name, vars);
   // RooDataHist only starts to store squared weights when it sees a positive error, and then initializes them from
   // the weights so far. Trigger that before setting any weight, so that errors of zero stay zero.
   if (errors && !errors->empty()) {
      dh->set(0, 0., 1.);
   }
   for (std::size_t ibin = 0; ibin < bins.size(); ++ibin) {
      dh->set(ibin, contents[ibin], errors ? (*errors)[ibin] : -1);
   }
   return dh;
}

RooJSONFactoryWSTool::RooJSONFactoryWSTool(RooWorkspace &ws) : _workspace{ws} {}

RooJSONFactoryWSTool::~RooJSONFactoryWSTool() = default;

void RooJSONFactoryWSTool::fillSeq(JSONNode &node, RooAbsCollection const &coll, size_t nMax)
{
   const size_t old_children = node.num_children();
   node.set_seq();
   size_t n = 0;
   for (RooAbsArg const *arg : coll) {
      if (n >= nMax)
         break;
      writeReference(node.append_child(), *arg);
      ++n;
   }
   if (node.num_children() != old_children + coll.size()) {
      error("unable to stream collection " + std::string(coll.GetName()) + " to " + node.key());
   }
}

JSONNode &RooJSONFactoryWSTool::appendNamedChild(JSONNode &node, std::string const &name)
{
   JSONNode &child = node.set_seq().append_child().set_map();
   child["name"] << name;
   return child;
}

JSONNode const *RooJSONFactoryWSTool::findNamedChild(JSONNode const &node, std::string const &name)
{
   if (!node.is_seq())
      return nullptr;
   for (JSONNode const &child : node.children()) {
      if (child["name"].val() == name)
         return &child;
   }

   return nullptr;
}

/**
 * @brief Check if a string is a valid name.
 *
 * A valid name should start with a letter or an underscore, followed by letters, digits, or underscores.
 * Only characters from the ASCII character set are allowed.
 *
 * @param str The string to be checked.
 * @return bool Returns true if the string is a valid name; otherwise, returns false.
 */
bool RooJSONFactoryWSTool::isValidName(const std::string &str)
{
   // Check if the string is empty or starts with a non-letter/non-underscore character
   if (str.empty() || !(std::isalpha(str[0]) || str[0] == '_')) {
      return false;
   }

   // Check the remaining characters in the string
   for (char c : str) {
      // Allow letters, digits, and underscore
      if (!(std::isalnum(c) || c == '_')) {
         return false;
      }
   }

   // If all characters are valid, the string is a valid name
   return true;
}

bool RooJSONFactoryWSTool::testValidName(const std::string &name, bool forceError)
{
   if (!RooJSONFactoryWSTool::isValidName(name)) {
      std::stringstream ss;
      ss << "RooJSONFactoryWSTool() name '" << name << "' is not valid!" << std::endl
         << "Sanitize names by setting RooJSONFactoryWSTool::config().allowSanitizeNames = true." << std::endl;
      if (RooJSONFactoryWSTool::config().allowExportInvalidNames && !forceError) {
         RooJSONFactoryWSTool::warning(ss.str());
         return false;
      } else {
         RooJSONFactoryWSTool::error(ss.str());
      }
   }
   return true;
}

std::string RooJSONFactoryWSTool::name(const JSONNode &n)
{
   return n["name"].val();
}

JSONNode &RooJSONFactoryWSTool::makeVariablesNode(JSONNode &rootNode)
{
   return appendNamedChild(rootNode["parameter_points"], "default_values")["parameters"];
}

template <>
RooRealVar *RooJSONFactoryWSTool::requestImpl<RooRealVar>(const std::string &objname)
{
   if (RooRealVar *retval = _workspace.var(objname))
      return retval;
   if (_import) {
      auto it = _import->defaultValuesByName.find(objname);
      if (it != _import->defaultValuesByName.end()) {
         this->importVariable(*it->second);
         if (RooRealVar *retval = _workspace.var(objname))
            return retval;
      }
   }
   return nullptr;
}

template <>
RooAbsPdf *RooJSONFactoryWSTool::requestImpl<RooAbsPdf>(const std::string &objname)
{
   if (RooAbsPdf *retval = _workspace.pdf(objname))
      return retval;
   if (_import) {
      auto it = _import->distributions.find(objname);
      if (it != _import->distributions.end()) {
         this->importDistribution(*it->second);
         if (RooAbsPdf *retval = _workspace.pdf(objname))
            return retval;
      }
      // HS3 lists some resolution models, which are pdfs in RooFit, as functions.
      auto function = _import->functions.find(objname);
      if (function != _import->functions.end()) {
         this->importFunction(*function->second->body.node, true);
         if (RooAbsPdf *retval = _workspace.pdf(objname))
            return retval;
      }
   }
   return nullptr;
}

template <>
RooAbsReal *RooJSONFactoryWSTool::requestImpl<RooAbsReal>(const std::string &objname)
{
   if (RooAbsReal *retval = _workspace.function(objname))
      return retval;
   auto toNumOutput = toNumber(objname);
   if (toNumOutput.valid)
      return &RooFit::RooConst(toNumOutput.value);
   if (RooAbsPdf *pdf = requestImpl<RooAbsPdf>(objname))
      return pdf;
   if (RooRealVar *var = requestImpl<RooRealVar>(objname))
      return var;
   if (RooAbsReal *retval = _workspace.function(objname))
      return retval;
   if (_import) {
      auto it = _import->functions.find(objname);
      if (it != _import->functions.end()) {
         this->importFunction(*it->second->body.node, true);
         if (RooAbsReal *retval = _workspace.function(objname))
            return retval;
      }
   }
   return nullptr;
}

// Export a single variable (RooRealVar or RooConstVar) `v` to the parameter point.
void RooJSONFactoryWSTool::exportVariable(const RooAbsArg *v, HS3::ParameterPoint &point, bool storeConstant)
{
   auto *cv = dynamic_cast<const RooConstVar *>(v);
   auto *rrv = dynamic_cast<const RooRealVar *>(v);
   if (!cv && !rrv)
      return;

   // for RooConstVar, if name and value are the same, we don't need to do anything
   if (cv && strcmp(cv->GetName(), TString::Format("%g", cv->getVal()).Data()) == 0) {
      return;
   }

   // A named RooConstVar only takes its value: it is written as a domain axis with that single value, not in the
   // parameter points.
   if (cv) {
      HS3::DomainAxis axis;
      axis.name = cv->GetName();
      axis.fixedValue = cv->getVal();
      _export->domains[HS3::defaultDomainName][axis.name] = std::move(axis);
      return;
   }

   HS3::ParameterValue var;
   var.name = v->GetName();
   var.value = rrv->getVal();
   var.isConst = storeConstant && (rrv->isConstant() || rrv->getMin() >= rrv->getMax());
   recordDomains(_export->domains, *rrv);
   point.parameters.push_back(std::move(var));
}

// Export all variables in `allElems` to the parameter point.
void RooJSONFactoryWSTool::exportVariables(const RooArgSet &allElems, HS3::ParameterPoint &point, bool storeConstant)
{
   for (RooAbsArg *arg : allElems) {
      exportVariable(arg, point, storeConstant);
   }
}

std::string RooJSONFactoryWSTool::exportTransformed(const RooAbsReal *original, const std::string &suffix,
                                                    const std::string &formula)
{
   std::string newname = std::string(original->GetName()) + suffix;
   // Another object that shares the parameter might have exported the helper already.
   auto &functions = _export->doc.functions;
   if (std::find_if(functions.begin(), functions.end(), [&](auto const &f) { return f.name == newname; }) !=
       functions.end()) {
      return newname;
   }
   RooFit::Detail::JSONNode &trafo_node = appendNamedChild((*_export->root)["functions"], newname);
   trafo_node["type"] << "generic";
   trafo_node["expression"] << TString::Format(formula.c_str(), original->GetName()).Data();
   functions.push_back({newname, {"generic", nullptr}});
   this->setAttribute(newname, "roofit_skip"); // this function should not be imported back in
   return newname;
}

namespace {

// Turn an arbitrary string into a valid variable name, but refuse to change the
// first character (which would silently rename the object).
std::string makeValidNameOrError(std::string const &in)
{
   if (!std::isalpha(in[0])) {
      RooJSONFactoryWSTool::error("refusing to change first character of string '" + in + "' to make a valid name!");
   }
   std::string out = in;
   std::transform(out.begin(), out.end(), out.begin(), [](char c) { return std::isalnum(c) ? c : '_'; });
   if (out != in) {
      oocoutW(nullptr, ObjectHandling) << "RooFitHS3: changed '" << in << "' to '" << out << "' to become a valid name";
   }
   return out;
}

} // namespace

namespace {

// Record an object that was written by an exporter in the document.
void recordGenericObject(HS3Document &doc, bool isPdf, std::string const &name, JSONNode const &elem)
{
   std::string type = elem.has_child("type") ? elem["type"].val() : "";
   if (isPdf) {
      doc.distributions.push_back({name, HS3::GenericObject{std::move(type), nullptr}});
   } else {
      doc.functions.push_back({name, {std::move(type), nullptr}});
   }
}

} // namespace

// Export a single object `func` (pdf, function, variable or category) to the output JSON, recording its name in
// `exportedObjectNames` to avoid exporting it twice.
void RooJSONFactoryWSTool::exportObject(RooAbsArg const &func, std::set<std::string> &exportedObjectNames)
{
   std::string name = func.GetName();

   // if this element was already exported, skip
   if (exportedObjectNames.find(name) != exportedObjectNames.end())
      return;

   exportedObjectNames.insert(name);

   if (auto simPdf = dynamic_cast<RooSimultaneous const *>(&func)) {
      // RooSimultaneous is not used in the HS3 standard, we only export the
      // dependents and some ROOT internal information.
      exportObjects(func.servers(), exportedObjectNames);

      HS3::CombinedDistribution &info = _export->doc.rooFit.combinedDistributions[simPdf->GetName()];
      info.indexCat = simPdf->indexCat().GetName();
      for (auto const &item : simPdf->indexCat()) {
         info.labels.push_back(makeValidNameOrError(item.first));
         info.indices.push_back(item.second);
         info.distributions.push_back({simPdf->getPdf(item.first.c_str())->GetName()});
      }

      return;
   } else if (dynamic_cast<RooAbsCategory const *>(&func)) {
      // categories are created by the respective RooSimultaneous, so we're skipping the export here
      return;
   } else if (dynamic_cast<RooRealVar const *>(&func) || dynamic_cast<RooConstVar const *>(&func)) {
      exportVariable(&func, _export->doc.parameterPoints.front(), true);
      return;
   }

   // HS3 lists some resolution models as functions.
   const bool isPdf = dynamic_cast<RooAbsPdf const *>(&func) && !func.InheritsFrom("RooGaussModel") &&
                      !func.InheritsFrom("RooTruthModel");
   auto &collectionNode = (*_export->root)[isPdf ? "distributions" : "functions"];

   auto const &exporters = RooFit::JSONIO::exporters();
   auto const &exportKeys = RooFit::JSONIO::exportKeys();

   TClass *cl = func.IsA();

   auto &elem = appendNamedChild(collectionNode, name);

   auto it = exporters.find(cl);
   if (it != exporters.end()) { // check if we have a specific exporter available
      for (auto &exp : it->second) {
         _serversToExport.clear();
         _serversToDelete.clear();
         if (!exp->exportObject(this, &func, elem)) {
            // The exporter might have messed with the content of the node
            // before failing. That's why we clear it and only reset the name.
            elem.clear();
            elem.set_map();
            elem["name"] << name;
            continue;
         }
         // Objects that are interpreted by the document are written here, the others were written by the exporter.
         if (_export->typedObject) {
            HS3::writeHistFactoryDist(elem, *_export->typedObject);
            elem["name"] << name;
            _export->doc.distributions.push_back({name, std::move(*_export->typedObject)});
            _export->typedObject.reset();
            if (!_export->channelConstraints.empty()) {
               _export->doc.rooFit.channelConstraints[name] = std::move(_export->channelConstraints);
            }
            _export->channelConstraints.clear();
         } else {
            recordGenericObject(_export->doc, isPdf, name, elem);
         }
         if (exp->autoExportDependants()) {
            exportObjects(func.servers(), exportedObjectNames);
         } else {
            exportObjects(_serversToExport, exportedObjectNames);
         }
         for (auto &s : _serversToDelete) {
            delete s;
         }
         return;
      }
   }

   // generic export using the factory expressions
   const auto &dict = exportKeys.find(cl);
   if (dict == exportKeys.end()) {
      std::cerr << "unable to export class '" << cl->GetName() << "' - no export keys available!\n"
                << "there are several possible reasons for this:\n"
                << " 1. " << cl->GetName() << " is a custom class that you or some package you are using added.\n"
                << " 2. " << cl->GetName()
                << " is a ROOT class that nobody ever bothered to write a serialization definition for.\n"
                << " 3. something is wrong with your setup, e.g. you might have called "
                   "RooFit::JSONIO::clearExportKeys() and/or never successfully read a file defining these "
                   "keys with RooFit::JSONIO::loadExportKeys(filename)\n"
                << "either way, please make sure that:\n"
                << " 3: you are reading a file with export keys - call RooFit::JSONIO::printExportKeys() to "
                   "see what is available\n"
                << " 2 & 1: you might need to write a serialization definition yourself. check "
                   "https://root.cern/doc/master/group__roofit__dev__docs__hs3.html to "
                   "see how to do this!\n";
      return;
   }

   elem["type"] << dict->second.type;
   recordGenericObject(_export->doc, isPdf, name, elem);

   size_t nprox = func.numProxies();

   for (size_t i = 0; i < nprox; ++i) {
      RooAbsProxy *p = func.getProxy(i);
      if (!p)
         continue;

      // some proxies start with a "!". This is a magic symbol that we don't want to stream
      std::string pname(p->name());
      if (pname[0] == '!')
         pname.erase(0, 1);

      auto k = dict->second.proxies.find(pname);
      if (k == dict->second.proxies.end()) {
         std::cerr << "failed to find key matching proxy '" << pname << "' for type '" << dict->second.type
                   << "', encountered in '" << func.GetName() << "', skipping" << std::endl;
         return;
      }

      // empty string is interpreted as an instruction to ignore this value
      if (k->second.empty())
         continue;

      if (auto l = dynamic_cast<RooAbsCollection *>(p)) {
         fillSeq(elem[k->second], *l);
      }
      if (auto r = dynamic_cast<RooArgProxy *>(p)) {
         writeReference(elem[k->second], *r->absArg());
      }
   }

   // export all the servers of a given RooAbsArg
   for (RooAbsArg *s : func.servers()) {
      if (!s) {
         std::cerr << "unable to locate server of " << func.GetName() << std::endl;
         continue;
      }
      this->exportObject(*s, exportedObjectNames);
   }
}

/**
 * @brief Import a function from the JSONNode into the workspace.
 *
 * This function imports a function from the given JSONNode into the workspace.
 * The function's information is read from the JSONNode and added to the workspace.
 *
 * @param p The JSONNode representing the function to be imported.
 * @param importAllDependants A boolean flag indicating whether to import all dependants (servers) of the function.
 * @return void
 */
void RooJSONFactoryWSTool::importFunction(const JSONNode &p, bool importAllDependants)
{
   // To report the components that no importer understands.
   const RooFit::Detail::JSONReadRecorder recorder{p};
   std::string name(RooJSONFactoryWSTool::name(p));

   // If this node if marked to be skipped by RooFit, exit
   if (hasAttribute(name, "roofit_skip")) {
      return;
   }

   auto const &importers = RooFit::JSONIO::importers();
   auto const &factoryExpressions = RooFit::JSONIO::importExpressions();

   // some preparations: what type of function are we dealing with here?
   RooJSONFactoryWSTool::testValidName(name, true);

   // if the RooAbsArg already exists, we don't need to do anything
   if (_workspace.arg(name)) {
      return;
   }
   ImportGuard importGuard{_importStack, name};
   // if the key we found is not a map, it's an error
   if (!p.is_map()) {
      RooJSONFactoryWSTool::error("RooJSONFactoryWSTool() function node " + name + " is not a map!");
   }
   std::string prefix = genPrefix(p, true);
   if (!prefix.empty())
      name = prefix + name;
   if (!p.has_child("type")) {
      RooJSONFactoryWSTool::error("RooJSONFactoryWSTool() no type given for function '" + name + "', skipping.");
   }

   std::string functype(p["type"].val());

   // import all dependents if importing a workspace, not for creating new objects
   if (!importAllDependants) {
      this->importDependants(p);
   }

   // check for specific implementations
   auto it = importers.find(functype);
   bool ok = false;
   if (it != importers.end()) {
      for (auto &imp : it->second) {
         try {
            // The built-in importer doesn't see the constraint terms that the document adds to the channel.
            if (_import && HS3::isBuiltinHistFactoryImporter(*imp)) {
               auto const &channelConstraints = _import->document.rooFit.channelConstraints;
               auto found = channelConstraints.find(name);
               HS3::importHistFactoryNode(*this, p, found != channelConstraints.end() ? &found->second : nullptr);
               ok = true;
            } else {
               ok = imp->importArg(this, p);
            }
         } catch (const std::exception &e) {
            std::stringstream ss;
            const auto *ptr = imp.get();
            ss << "RooJSONFactoryWSTool() failed. The importer " << typeid(*ptr).name()
               << " emitted and error: " << e.what() << std::endl;
            RooJSONFactoryWSTool::error(ss.str());
         }
         if (ok)
            break;
      }
   }
   if (!ok) { // generic import using the factory expressions
      auto expr = factoryExpressions.find(functype);
      if (expr != factoryExpressions.end()) {
         std::string expression = ::generate(expr->second, p, this);
         if (!_workspace.factory(expression)) {
            std::stringstream ss;
            ss << "RooJSONFactoryWSTool() failed to create " << expr->second.tclass->GetName() << " '" << name
               << "', skipping. expression was\n"
               << expression << std::endl;
            RooJSONFactoryWSTool::error(ss.str());
         }
      } else {
         std::stringstream ss;
         ss << "RooJSONFactoryWSTool() no handling for type '" << functype << "' implemented, skipping."
            << "\n"
            << "there are several possible reasons for this:\n"
            << " 1. " << functype << " is a custom type that is not available in RooFit.\n"
            << " 2. " << functype
            << " is a ROOT class that nobody ever bothered to write a deserialization definition for.\n"
            << " 3. something is wrong with your setup, e.g. you might have called "
               "RooFit::JSONIO::clearFactoryExpressions() and/or never successfully read a file defining "
               "these expressions with RooFit::JSONIO::loadFactoryExpressions(filename)\n"
            << "either way, please make sure that:\n"
            << " 3: you are reading a file with factory expressions - call "
               "RooFit::JSONIO::printFactoryExpressions() "
               "to see what is available\n"
            << " 2 & 1: you might need to write a deserialization definition yourself. check "
               "https://root.cern/doc/master/group__roofit__dev__docs__hs3.html to see "
               "how to do this!"
            << std::endl;
         RooJSONFactoryWSTool::error(ss.str());
         return;
      }
   }
   RooAbsReal *func = _workspace.function(name);
   if (!func) {
      RooJSONFactoryWSTool::error("something went wrong importing function '" + name + "'.");
   }
   for (std::string const &component : recorder.unreadComponents()) {
      warning("RooFitHS3: " + name + ": unknown component \"" + component + "\" is ignored");
   }
}

// Import a distribution of the document, which is interpreted by the document or by the registered importers.
void RooJSONFactoryWSTool::importDistribution(HS3::Distribution const &distribution)
{
   if (auto const *generic = std::get_if<HS3::GenericObject>(&distribution.body)) {
      this->importFunction(*generic->node, true);
      return;
   }
   std::string const &name = distribution.name;
   if (hasAttribute(name, "roofit_skip")) {
      return;
   }
   RooJSONFactoryWSTool::testValidName(name, true);
   if (_workspace.arg(name)) {
      return;
   }
   ImportGuard importGuard{_importStack, name};
   auto const &channelConstraints = _import->document.rooFit.channelConstraints;
   auto found = channelConstraints.find(name);
   HS3::buildHistFactoryDist(*this, name, std::get<HS3::HistFactoryDist>(distribution.body),
                             found != channelConstraints.end() ? &found->second : nullptr);
}

/**
 * @brief Import a function from a JSON string into the workspace.
 *
 * This function imports a function from the provided JSON string into the workspace.
 * The function's information is read from the JSON string and added to the workspace.
 *
 * @param jsonString The JSON string containing the function information.
 * @param importAllDependants A boolean flag indicating whether to import all dependants (servers) of the function.
 * @return void
 */
void RooJSONFactoryWSTool::importFunction(const std::string &jsonString, bool importAllDependants)
{
   this->importFunction((JSONTree::create(jsonString))->rootnode(), importAllDependants);
}

/**
 * @brief Export the name and binning of a RooRealVar to a JSONNode.
 *
 * @param obsNode The JSONNode to which the axis information will be exported.
 * @param var The RooRealVar representing the axis to be exported.
 * @return void
 */
void RooJSONFactoryWSTool::exportAxis(JSONNode &obsNode, RooRealVar const &var)
{
   std::string name = var.GetName();
   RooJSONFactoryWSTool::testValidName(name, false);
   obsNode["name"] << name;

   writeAxisBinning(obsNode, var.getBinning());
}

/**
 * @brief Export histogram data to a JSONNode.
 *
 * This function exports histogram data, represented by the provided variables and contents, to a JSONNode.
 * The histogram's axes information and bin contents are added as key-value pairs to the JSONNode.
 *
 * @param vars The RooArgSet representing the variables associated with the histogram.
 * @param n The number of bins in the histogram.
 * @param contents A pointer to the array containing the bin contents of the histogram.
 * @param output The JSONNode to which the histogram data will be exported.
 * @return void
 */
void RooJSONFactoryWSTool::exportHisto(RooArgSet const &vars, std::size_t n, double const *contents, JSONNode &output)
{
   auto &observablesNode = output["axes"].set_seq();
   // axes have to be ordered to get consistent bin indices
   for (auto *var : static_range_cast<RooRealVar *>(vars)) {
      exportAxis(observablesNode.append_child().set_map(), *var);
   }

   return exportArray(n, contents, output["contents"]);
}

/**
 * @brief Export an array of doubles to a JSONNode.
 *
 * This function exports an array of doubles, represented by the provided size and contents,
 * to a JSONNode. The array elements are added to the JSONNode as a sequence of values.
 *
 * @param n The size of the array.
 * @param contents A pointer to the array containing the double values.
 * @param output The JSONNode to which the array will be exported.
 * @return void
 */
void RooJSONFactoryWSTool::exportArray(std::size_t n, double const *contents, JSONNode &output)
{
   output.set_seq();
   for (std::size_t i = 0; i < n; ++i) {
      double w = contents[i];
      // To make sure there are no unnecessary floating points in the JSON
      if (int(w) == w) {
         output.append_child() << int(w);
      } else {
         output.append_child() << w;
      }
   }
}

/**
 * @brief Export a RooAbsCategory object to a JSONNode.
 *
 * This function exports a RooAbsCategory object, represented by the provided categories and indices,
 * to a JSONNode. The category labels and corresponding indices are added to the JSONNode as key-value pairs.
 *
 * @param cat The RooAbsCategory object to be exported.
 * @param node The JSONNode to which the category data will be exported.
 * @return void
 */
void RooJSONFactoryWSTool::exportCategory(RooAbsCategory const &cat, JSONNode &node)
{
   auto &labels = node["labels"].set_seq();
   auto &indices = node["indices"].set_seq();

   for (auto const &item : cat) {
      labels.append_child() << makeValidNameOrError(item.first);
      indices.append_child() << item.second;
   }
}

// Split `data` by its index category into per-channel datasets and export each, returning the resulting
// component-name map.
RooJSONFactoryWSTool::CombinedData RooJSONFactoryWSTool::exportCombinedData(RooAbsData const &data)
{
   RooAbsCategory *cat = findCategoryObservable(data);

   // prepare return value
   RooJSONFactoryWSTool::CombinedData datamap;

   if (!cat)
      return datamap;
   // this is a combined dataset

   datamap.name = data.GetName();

   // Write information necessary to reconstruct the combined dataset upon import
   HS3::CombinedDataset &info = _export->doc.rooFit.combinedDatasets[data.GetName()];
   info.indexCat = cat->GetName();
   for (auto const &item : *cat) {
      info.labels.push_back(makeValidNameOrError(item.first));
      info.indices.push_back(item.second);
   }

   // Find a RooSimultaneous model that would fit to this dataset
   RooSimultaneous const *simPdf = nullptr;
   for (auto const &item : _export->doc.rooFit.combinedDistributions) {
      if (item.second.indexCat == cat->GetName()) {
         simPdf = static_cast<RooSimultaneous const *>(_workspace.pdf(item.first));
      }
   }

   // If there is an associated simultaneous pdf for the index category, we
   // use the RooAbsData::split() overload that takes the RooSimultaneous.
   // Like this, the observables that are not relevant for a given channel
   // are automatically split from the component datasets.
   std::vector<std::unique_ptr<RooAbsData>> dataList{simPdf ? data.split(*simPdf, true) : data.split(*cat, true)};

   for (std::unique_ptr<RooAbsData> const &absData : dataList) {
      std::string catName(absData->GetName());
      std::string dataName = makeValidNameOrError(catName);
      absData->SetName((std::string(data.GetName()) + "_" + dataName).c_str());
      datamap.components[catName] = absData->GetName();
      this->exportData(*absData);
   }
   return datamap;
}

// Export a single dataset `data` (binned or unbinned) to the output document.
void RooJSONFactoryWSTool::exportData(RooAbsData const &data)
{
   RooAbsCategory *cat = findCategoryObservable(data);

   if (cat)
      return;

   HS3::Data output;
   output.name = data.GetName();

   // This works around a problem in RooStats/HistFactory that was only fixed
   // in ROOT 6.30: until then, the weight variable of the observed dataset,
   // called "weightVar", was added to the observables. Therefore, it also got
   // added to the Asimov dataset. But the Asimov has its own weight variable,
   // called "binWeightAsimov", making "weightVar" an actual observable in the
   // Asimov data. But this is only by accident and should be removed.
   RooArgSet variables = *data.get();
   if (auto weightVar = variables.find("weightVar")) {
      variables.remove(*weightVar);
   }

   auto exportBinned = [&](std::size_t n, double const *contents) {
      HS3::BinnedData binned;
      for (auto *var : static_range_cast<RooRealVar *>(variables)) {
         recordDomains(_export->domains, *var);
         binned.axes.emplace_back(HS3::toBinnedAxis(*var));
      }
      binned.contents.assign(contents, contents + n);
      output.body = std::move(binned);
      _export->doc.data.push_back(std::move(output));
   };

   // this is a regular binned dataset
   if (auto dh = dynamic_cast<RooDataHist const *>(&data)) {
      return exportBinned(dh->numEntries(), dh->weightArray());
   }

   // Check if this actually represents a binned dataset, and then import it
   // like a RooDataHist. This happens frequently when people create combined
   // RooDataSets from binned data to fit HistFactory models. In this case, it
   // doesn't make sense to export them like an unbinned dataset, because the
   // coordinates are redundant information with the binning. We only do this
   // for 1D data for now.
   if (data.isWeighted() && variables.size() == 1) {
      bool isBinnedData = false;
      auto &x = static_cast<RooRealVar const &>(*variables[0]);
      std::vector<double> contents;
      int i = 0;
      for (; i < data.numEntries(); ++i) {
         data.get(i);
         if (x.getBin() != i)
            break;
         contents.push_back(data.weight());
      }
      if (i == x.getBins())
         isBinnedData = true;
      if (isBinnedData) {
         return exportBinned(data.numEntries(), contents.data());
      }
   }

   // this really is an unbinned dataset
   HS3::UnbinnedData unbinned;
   for (auto *var : static_range_cast<RooRealVar *>(variables)) {
      recordDomains(_export->domains, *var);
      // HS3 forbids binning information on the axes of unbinned data.
      testValidName(var->GetName(), false);
      unbinned.axes.emplace_back(HS3::ContinuousAxis{var->GetName(), var->getMin(), var->getMax()});
   }
   std::vector<double> weightVals;
   bool hasNonUnityWeights = false;
   for (int i = 0; i < data.numEntries(); ++i) {
      data.get(i);
      for (auto *var : static_range_cast<RooRealVar *>(variables)) {
         unbinned.entries.push_back(var->getVal());
      }
      if (data.isWeighted()) {
         weightVals.push_back(data.weight());
         if (data.weight() != 1.)
            hasNonUnityWeights = true;
      }
   }
   if (data.isWeighted() && hasNonUnityWeights) {
      unbinned.weights = std::move(weightVals);
   }
   output.body = std::move(unbinned);
   _export->doc.data.push_back(std::move(output));
}

/**
 * @brief Read axes from the JSONNode and create a RooArgSet representing them.
 *
 * This function reads axes information from the given JSONNode and
 * creates a RooArgSet with variables representing these axes.
 *
 * @param topNode The JSONNode containing the axes information to be read.
 * @return RooArgSet A RooArgSet containing the variables created from the JSONNode.
 */
RooArgSet RooJSONFactoryWSTool::readAxes(const JSONNode &topNode)
{
   HS3::Diagnostics diagnostics;
   std::vector<HS3::BinnedAxis> axes = HS3::parseBinnedAxes(topNode["axes"], diagnostics);
   HS3::report(diagnostics);
   return HS3::createObservables(axes);
}

/**
 * @brief Read binned data from the JSONNode and create a RooDataHist object.
 *
 * This function reads binned data from the given JSONNode and creates a RooDataHist object.
 * The binned data is associated with the specified name and variables (RooArgSet) in the workspace.
 *
 * @param n The JSONNode representing the binned data to be read.
 * @param name The name to be associated with the created RooDataHist object.
 * @param vars The RooArgSet representing the variables associated with the binned data.
 * @return std::unique_ptr<RooDataHist> A unique pointer to the created RooDataHist object.
 */
std::unique_ptr<RooDataHist>
RooJSONFactoryWSTool::readBinnedData(const JSONNode &n, const std::string &name, RooArgSet const &vars)
{
   if (!n.has_child("contents"))
      RooJSONFactoryWSTool::error("no contents given");

   JSONNode const &contents = n["contents"];

   if (!contents.is_seq())
      RooJSONFactoryWSTool::error("contents are not in list form");

   std::vector<double> contentVals;
   contentVals << contents;

   if (!n.has_child("errors")) {
      return HS3::createDataHist(name, vars, contentVals, nullptr);
   }
   if (!n["errors"].is_seq())
      RooJSONFactoryWSTool::error("errors are not in list form");
   std::vector<double> errorVals;
   errorVals << n["errors"];
   return HS3::createDataHist(name, vars, contentVals, &errorVals);
}

// Import a single variable (RooRealVar or RooConstVar) from the parameter `p` into the workspace.
void RooJSONFactoryWSTool::importVariable(HS3::ParameterValue const &p)
{
   // import a RooRealVar object
   RooJSONFactoryWSTool::testValidName(p.name, true);

   if (_workspace.arg(p.name))
      return;
   // A string selects the state of a category, which is done together with the other parameter points.
   if (p.value && std::holds_alternative<std::string>(*p.value))
      return;
   if (!_import->ranges.contains(p.name)) {
      // ROOT versions before HS3 0.3 marked named RooConstVars with an attribute.
      auto const &attributes = _import->document.rooFit.attributes;
      auto found = attributes.find(p.name);
      if (found != attributes.end() && found->second.isConstVar) {
         if (!p.value) {
            RooJSONFactoryWSTool::error("cannot instantiate RooConstVar '" + p.name + "' without \"value\"!");
         }
         wsEmplace<RooConstVar>(p.name, std::get<double>(*p.value));
         return;
      }
   }
   RooRealVar &var = wsEmplace<RooRealVar>(p.name, 1.);
   configureVariable(_import->ranges, p, var);
   warnIfFreeWithoutDomain(var);
}

// HS3 requires every parameter to be listed in a domain or constant ("const": true) in the parameter point.
void RooJSONFactoryWSTool::warnIfFreeWithoutDomain(RooRealVar const &var)
{
   if (!var.isConstant() && !_import->ranges.contains(var.GetName())) {
      warning(std::string("RooFitHS3: the parameter '") + var.GetName() +
              "' is not listed in any domain and is not constant: HS3 requires a domain, or \"const\": true. It "
              "is read as a free parameter without range.");
   }
}

// Import the objects of a fragment of a document, for which the node `n` plays the role of the document.
void RooJSONFactoryWSTool::importDependants(const JSONNode &n)
{
   for (const char *section : {"functions", "distributions"}) {
      if (auto seq = n.find(section)) {
         for (const auto &p : seq->children()) {
            this->importFunction(p, true);
         }
      }
   }
}

void RooJSONFactoryWSTool::exportModelConfig(RooFit::ModelConfig const &mc,
                                             const std::vector<CombinedData> &combDataSets,
                                             const std::vector<RooAbsData *> &singleDataSets)
{
   auto pdf = mc.GetPdf();
   auto simpdf = dynamic_cast<RooSimultaneous const *>(pdf);
   if (simpdf) {
      for (std::size_t i = 0; i < std::max(combDataSets.size(), std::size_t(1)); ++i) {
         const bool hasdata = i < combDataSets.size();
         if (hasdata && !matches(combDataSets.at(i), simpdf))
            continue;

         std::string analysisName(mc.GetName());
         if (hasdata)
            analysisName += "_" + combDataSets[i].name;

         exportSingleModelConfig(mc, analysisName, hasdata ? &combDataSets[i].components : nullptr);
      }
   } else {
      RooArgSet observables(*mc.GetObservables());
      int founddata = 0;
      for (auto *data : singleDataSets) {
         if (observables.equals(*(data->get()))) {
            std::map<std::string, std::string> mapping;
            mapping[pdf->GetName()] = data->GetName();
            exportSingleModelConfig(mc, std::string(mc.GetName()) + "_" + data->GetName(), &mapping);
            ++founddata;
         }
      }
      if (founddata == 0) {
         exportSingleModelConfig(mc, mc.GetName(), nullptr);
      }
   }
}

void RooJSONFactoryWSTool::exportSingleModelConfig(RooFit::ModelConfig const &mc, std::string const &analysisName,
                                                   std::map<std::string, std::string> const *dataComponents)
{
   HS3Document &doc = _export->doc;
   auto pdf = mc.GetPdf();

   HS3::Analysis analysis;
   analysis.name = analysisName;

   // HS3 names must be unique across all top-level components, so the likelihood can't have the analysis name.
   HS3::Likelihood likelihood;
   likelihood.name = analysisName + "_likelihood";
   analysis.likelihood.name = likelihood.name;

   if (dataComponents) {
      auto simPdf = dynamic_cast<RooSimultaneous const *>(pdf);
      if (simPdf) {
         for (auto const &item : simPdf->indexCat()) {
            const auto &dataComp = dataComponents->find(item.first);
            likelihood.terms.push_back({{simPdf->getPdf(item.first)->GetName()}, HS3::DataRef{dataComp->second}});
         }
      } else {
         for (auto it : *dataComponents) {
            likelihood.terms.push_back({{it.first}, HS3::DataRef{it.second}});
         }
      }
   } else {
      likelihood.terms.push_back({{pdf->GetName()}, 0.0});
   }

   auto addAux = [&](std::string const &name) {
      auto &aux = likelihood.auxDistributions;
      if (std::find_if(aux.begin(), aux.end(), [&](auto const &ref) { return ref.name == name; }) == aux.end()) {
         aux.push_back({name});
      }
   };
   // The constraint terms of HistFactory channels that no modifier names are part of the likelihood.
   for (auto const &term : likelihood.terms) {
      auto found = doc.rooFit.channelConstraints.find(term.distribution.name);
      if (found != doc.rooFit.channelConstraints.end()) {
         for (auto const &ref : found->second) {
            addAux(ref.name);
         }
      }
   }
   if (mc.GetExternalConstraints()) {
      for (const auto &constr : *mc.GetExternalConstraints()) {
         addAux(constr->GetName());
      }
   }
   doc.likelihoods.push_back(std::move(likelihood));

   if (RooArgSet const *pois = mc.GetParametersOfInterest()) {
      std::vector<std::string> names;
      names.reserve(pois->size());
      for (RooAbsArg const *arg : *pois)
         names.push_back(arg->GetName());
      std::sort(names.begin(), names.end());
      for (auto const &name : names) {
         analysis.parametersOfInterest.push_back({name});
      }
   }

   auto writeProductDomain = [&](const char *suffix, RooArgSet const *args) {
      if (!args || args->empty())
         return;
      HS3::Domain &domain = doc.domains.emplace_back();
      domain.name = analysisName + suffix;
      analysis.domains.push_back({domain.name});
      RooArgList sorted{*args};
      sorted.sort();
      for (auto *var : static_range_cast<const RooRealVar *>(sorted)) {
         domain.axes.emplace_back(toDomainAxis(var->GetName(), var->getBinning()));
      }
   };

   writeProductDomain("_nuisance_parameters", mc.GetNuisanceParameters());
   writeProductDomain("_global_observables", mc.GetGlobalObservables());
   writeProductDomain("_parameters_of_interest", mc.GetParametersOfInterest());

   // The snapshots are exported as parameter points. Reference the one of the ModelConfig as the initial values.
   if (_workspace.getSnapshot(mc.GetSnapshotName().c_str())) {
      analysis.init = HS3::ParameterPointRef{mc.GetSnapshotName()};
   }
   // The prior is a top-level pdf of the workspace or a component of one, so it is exported already.
   if (RooAbsPdf const *prior = mc.GetPriorPdf()) {
      analysis.prior = HS3::DistributionRef{prior->GetName()};
   }
   doc.analyses.push_back(std::move(analysis));

   HS3::ModelConfigInfo &info = doc.rooFit.modelConfigs[analysisName];
   info.pdfName = pdf->GetName();
   info.mcName = mc.GetName();
}

// Export all top-level pdfs, functions, datasets and ModelConfigs of the workspace into `n`.
void RooJSONFactoryWSTool::exportAllObjects(JSONNode &n)
{
   _export = std::make_unique<HS3::ToolExportState>();
   _export->root = &n;
   // The state of the export must not survive it, not even if there is an exception.
   struct ExportScope {
      RooJSONFactoryWSTool &tool;
      ~ExportScope() { tool._export.reset(); }
   } exportScope{*this};

   HS3Document &doc = _export->doc;
   doc.metadata.hs3Version = hs3VersionTag;
   doc.metadata.packages.push_back({"ROOT", rootVersion()});
   doc.parameterPoints.push_back({"default_values", {}});

   // export all toplevel pdfs
   std::vector<RooAbsPdf *> allpdfs;
   for (auto &arg : _workspace.allPdfs()) {
      if (isTopLevel(*arg, _workspace)) {
         if (auto *pdf = dynamic_cast<RooAbsPdf *>(arg)) {
            allpdfs.push_back(pdf);
         }
      }
   }
   sortByName(allpdfs);
   std::set<std::string> exportedObjectNames;
   exportObjects(allpdfs, exportedObjectNames);

   // export all toplevel functions
   std::vector<RooAbsReal *> allfuncs;
   for (auto &arg : _workspace.allFunctions()) {
      if (isTopLevel(*arg, _workspace)) {
         if (auto *func = dynamic_cast<RooAbsReal *>(arg)) {
            allfuncs.push_back(func);
         }
      }
   }
   sortByName(allfuncs);
   exportObjects(allfuncs, exportedObjectNames);

   // export attributes of all objects
   for (RooAbsArg *arg : _workspace.components()) {
      exportAttributes(arg, doc.rooFit.attributes);
   }

   // collect all datasets
   std::vector<RooAbsData *> alldata;
   for (auto &d : _workspace.allData()) {
      alldata.push_back(d);
   }
   sortByName(alldata);
   // first, take care of combined datasets
   std::vector<RooAbsData *> singleData;
   std::vector<RooJSONFactoryWSTool::CombinedData> combData;
   for (auto &d : alldata) {
      auto data = this->exportCombinedData(*d);
      if (!data.components.empty()) {
         combData.push_back(data);
      } else {
         singleData.push_back(d);
      }
   }
   // next, take care datasets
   for (auto &d : alldata) {
      this->exportData(*d);
   }

   // export all ModelConfig objects and attached Pdfs
   for (TObject *obj : _workspace.allGenericObjects()) {
      if (auto mc = dynamic_cast<RooFit::ModelConfig *>(obj)) {
         exportModelConfig(*mc, combData, singleData);
      }
   }

   exportParameterStepWidths(_workspace, allpdfs, alldata, doc.rooFit.stepWidths);

   for (auto *snsh : static_range_cast<RooArgSet const *>(_workspace.getSnapshots())) {
      RooArgSet snapshotSorted;
      // We only want to add the variables that actually got exported and skip
      // the ones that the pdfs encoded implicitly (like in the case of
      // HistFactory).
      for (RooAbsArg *arg : *snsh) {
         bool do_export = false;
         for (const auto &pdf : allpdfs) {
            if (pdf->dependsOn(*arg)) {
               do_export = true;
            }
         }
         if (do_export) {
            RooJSONFactoryWSTool::testValidName(arg->GetName(), true);
            snapshotSorted.add(*arg);
         }
      }
      snapshotSorted.sort();
      std::string name(snsh->GetName());
      if (name != "default_values") {
         doc.parameterPoints.push_back({name, {}});
         this->exportVariables(snapshotSorted, doc.parameterPoints.back(), true);
      }
   }
   // The domains of the analyses take precedence.
   for (auto &item : _export->domains) {
      if (std::none_of(doc.domains.begin(), doc.domains.end(),
                       [&](HS3::Domain const &domain) { return domain.name == item.first; })) {
         appendDomain(doc.domains, item.first, std::move(item.second));
      }
   }

   // The document that RooFit writes has to pass the same checks as the ones that it reads. Until RooFit writes
   // documents that follow the current standard, problems are only reported.
   HS3::Diagnostics diagnostics;
   HS3::ValidateOptions validateOptions;
   validateOptions.isExternal = [](std::string const &) { return false; };
   doc.metadata.version = HS3::Version{};
   HS3::checkDocument(doc, diagnostics, validateOptions);
   for (auto &diagnostic : diagnostics) {
      diagnostic.severity = HS3::Diagnostic::Severity::Warning;
      diagnostic.message = "the exported document is not valid: " + diagnostic.message;
   }
   HS3::report(diagnostics);
   HS3::writeDocument(n, doc);
}

/**
 * @brief Import the workspace from a JSON string.
 *
 * @param s The JSON string containing the workspace data.
 * @return bool Returns true on successful import, false otherwise.
 */
bool RooJSONFactoryWSTool::importJSONfromString(const std::string &s)
{
   std::stringstream ss(s);
   return importJSON(ss);
}

/**
 * @brief Export the workspace to a JSON string.
 *
 * @return std::string The JSON string representing the exported workspace.
 */
std::string RooJSONFactoryWSTool::exportJSONtoString()
{
   std::stringstream ss;
   exportJSON(ss);
   return ss.str();
}

/**
 * @brief Create a new JSON tree with version information.
 *
 * @return std::unique_ptr<JSONTree> A unique pointer to the created JSON tree.
 */
std::unique_ptr<JSONTree> RooJSONFactoryWSTool::createNewJSONTree()
{
   std::unique_ptr<JSONTree> tree = JSONTree::create();
   JSONNode &n = tree->rootnode();
   n.set_map();
   auto &metadata = n["metadata"].set_map();

   // add the mandatory hs3 version number
   metadata["hs3_version"] << hs3VersionTag;

   appendNamedChild(metadata["packages"], "ROOT")["version"] << rootVersion();

   return tree;
}

/**
 * @brief Export the workspace to JSON format and write to the output stream.
 *
 * @param os The output stream to write the JSON data to.
 * @return bool Returns true on successful export, false otherwise.
 */
bool RooJSONFactoryWSTool::exportJSON(std::ostream &os)
{
   std::unique_ptr<JSONTree> tree = JSONTree::create();
   JSONNode &n = tree->rootnode().set_map();
   this->exportAllObjects(n);
   n.writeJSON(os);
   return true;
}

/**
 * @brief Export the workspace to JSON format and write to the specified file.
 *
 * @param filename The name of the JSON file to create and write the data to.
 * @return bool Returns true on successful export, false otherwise.
 */
bool RooJSONFactoryWSTool::exportJSON(std::string const &filename)
{
   std::ofstream out(filename.c_str());
   if (!out.is_open())
      RooJSONFactoryWSTool::error("RooJSONFactoryWSTool() invalid output file '" + filename + "'.");
   return this->exportJSON(out);
}

bool RooJSONFactoryWSTool::hasAttribute(const std::string &obj, const std::string &attrib)
{
   if (!_import)
      return false;
   auto const &attributes = _import->document.rooFit.attributes;
   auto found = attributes.find(obj);
   if (found == attributes.end())
      return false;
   auto const &tags = found->second.tags;
   return std::find(tags.begin(), tags.end(), attrib) != tags.end();
}
void RooJSONFactoryWSTool::setAttribute(const std::string &obj, const std::string &attrib)
{
   _export->doc.rooFit.attributes[obj].tags.push_back(attrib);
}

std::string RooJSONFactoryWSTool::getStringAttribute(const std::string &obj, const std::string &attrib)
{
   if (!_import)
      return "";
   auto const &attributes = _import->document.rooFit.attributes;
   auto found = attributes.find(obj);
   if (found == attributes.end())
      return "";
   auto value = found->second.dict.find(attrib);
   return value != found->second.dict.end() ? value->second : "";
}
void RooJSONFactoryWSTool::setStringAttribute(const std::string &obj, const std::string &attrib,
                                              const std::string &value)
{
   _export->doc.rooFit.attributes[obj].dict[attrib] = value;
}

// Import the JSON document rooted at `n` into the workspace.
void RooJSONFactoryWSTool::importAllNodes(JSONNode &n)
{
   HS3::ParseOptions options;
   // A user can replace how HistFactory distributions are imported by registering their own importer.
   auto const &importers = RooFit::JSONIO::importers();
   auto histFactoryImporters = importers.find("histfactory_dist");
   options.typedHistFactory = histFactoryImporters != importers.end() && !histFactoryImporters->second.empty() &&
                              HS3::isBuiltinHistFactoryImporter(*histFactoryImporters->second.front());

   HS3::Diagnostics diagnostics;
   HS3::upgradeDocument(n, diagnostics, options);
   HS3Document parsed = HS3::parseDocument(n, diagnostics, options);
   std::optional<RooFit::ValidatedHS3Document> validated;
   if (!HS3::hasErrors(diagnostics)) {
      HS3::ValidateOptions validateOptions;
      validateOptions.strict = !isHS3VersionBefore(parsed.metadata.hs3Version, 0, 3);
      validateOptions.isExternal = [this](std::string const &name) { return _workspace.obj(name) != nullptr; };
      validated = HS3::validate(std::move(parsed), diagnostics, validateOptions);
   }
   HS3::report(diagnostics);
   HS3Document const &doc = validated->doc();

   // The state of the import refers to the document, so it must not survive it, not even if there is an exception.
   struct ImportScope {
      RooJSONFactoryWSTool &tool;
      ~ImportScope()
      {
         tool._import.reset();
         tool._hs3VersionInput.clear();
      }
   } importScope{*this};

   _hs3VersionInput = doc.metadata.hs3Version;

   _import = std::make_unique<HS3::ToolImportState>(doc);

   createVariables(_workspace, _import->document.domains, _import->ranges);

   if (_import->defaultValues) {
      for (auto const &p : _import->defaultValues->parameters) {
         importVariable(p);
      }
   }
   for (auto const &function : doc.functions) {
      this->importFunction(*function.body.node, true);
   }
   for (auto const &distribution : doc.distributions) {
      this->importDistribution(distribution);
   }

   // This creates the index categories, which the parameter points can refer to.
   combinePdfs(doc, _workspace);

   for (auto const &point : doc.parameterPoints) {
      RooJSONFactoryWSTool::testValidName(point.name, true);

      RooArgSet vars;
      for (auto const &var : point.parameters) {
         std::string const *label = var.value ? std::get_if<std::string>(&*var.value) : nullptr;
         if (RooCategory *cat = _workspace.cat(var.name)) {
            if (label) {
               cat->setLabel(label->c_str());
            } else if (var.value) {
               cat->setIndex(static_cast<int>(std::get<double>(*var.value)));
            }
            vars.add(*cat);
            continue;
         }
         if (label) {
            warning("RooFitHS3: ignoring the string value of parameter '" + var.name + "' in parameter point '" +
                    point.name + "', because there is no category with that name.");
            continue;
         }
         RooRealVar *rrv = _workspace.var(var.name);
         if (!rrv && !_workspace.arg(var.name)) {
            // The parameter is not used by any other component: create it. "const": false is the same as no
            // "const" at all.
            testValidName(var.name, true);
            rrv = &wsEmplace<RooRealVar>(var.name, 1.);
            configureVariable(_import->ranges, var, *rrv);
            warnIfFreeWithoutDomain(*rrv);
         } else if (rrv) {
            configureVariable(_import->ranges, var, *rrv);
         }
         if (rrv) {
            vars.add(*rrv);
         }
      }
      _workspace.saveSnapshot(point.name, vars);
   }

   // Import attributes
   for (auto const &item : doc.rooFit.attributes) {
      if (RooAbsArg *arg = _workspace.arg(item.first))
         importAttributes(arg, item.second);
   }

   // We delay the import of the data to after combineDatasets(), because it
   // might be that some datasets are merged to combined datasets there. In
   // that case, we will remove the components from the "datasets" vector so they
   // don't get imported.
   std::vector<std::unique_ptr<RooAbsData>> datasets;
   datasets.reserve(doc.data.size());
   for (auto const &data : doc.data) {
      datasets.push_back(loadData(data, _workspace, &_import->ranges));
   }

   // Now, read in analyses and likelihoods if there are any

   for (auto const &analysis : doc.analyses) {
      importAnalysis(doc, analysis, _workspace, datasets);
   }

   combineDatasets(doc, datasets);

   for (auto const &d : datasets) {
      if (d) {
         _workspace.import(*d, RooFit::Silence());
         for (auto const &obs : *d->get()) {
            if (auto *rrv = dynamic_cast<RooRealVar *>(obs)) {
               _workspace.var(rrv->GetName())->setBinning(rrv->getBinning());
            }
         }
      }
   }

   if (_workspace.getSnapshot("default_values")) {
      _workspace.loadSnapshot("default_values");
   }
   importParameterStepWidths(_workspace, doc);
}

/**
 * @brief Imports a JSON file from the given input stream to the workspace.
 *
 * @param is The input stream containing the JSON data.
 * @return bool Returns true on successful import, false otherwise.
 */
bool RooJSONFactoryWSTool::importJSON(std::istream &is)
{
   // import a JSON file to the workspace
   std::unique_ptr<JSONTree> tree = JSONTree::create(is);
   this->importAllNodes(tree->rootnode());
   return true;
}

/**
 * @brief Imports a JSON file from the given filename to the workspace.
 *
 * @param filename The name of the JSON file to import.
 * @return bool Returns true on successful import, false otherwise.
 */
bool RooJSONFactoryWSTool::importJSON(std::string const &filename)
{
   // import a JSON file to the workspace
   std::ifstream infile(filename.c_str());
   if (!infile.is_open())
      RooJSONFactoryWSTool::error("RooJSONFactoryWSTool() invalid input file '" + filename + "'.");
   return this->importJSON(infile);
}

void RooJSONFactoryWSTool::importJSONElement(const std::string &name, const std::string &jsonString)
{
   // Create the JSON Tree from the string
   std::unique_ptr<RooFit::Detail::JSONTree> tree = RooFit::Detail::JSONTree::create(jsonString);
   JSONNode &n = tree->rootnode();

   // If the objects containts a parameter of interest, import it as a modelConfig
   if (n.find("poi")) {

      RooStats::ModelConfig modelConfig{"ModelConfig"};
      std::string poi = n.find("poi")->val();
      std::string pdname = n.find("pdfName")->val();
      modelConfig.SetWS(_workspace);
      modelConfig.SetPdf(pdname.c_str());
      modelConfig.SetParametersOfInterest(_workspace.argSet(poi));
      _workspace.import(modelConfig);

      return;
   }

   n["name"] << name;

   bool isVariable = true;
   bool isData = false;
   // Check for the type of object, if it doesn't contain a type, it must be a variable
   if (n.find("type")) {
      isVariable = false;
      std::string elementType = n.find("type")->val();
      if (elementType == "binned" || elementType == "unbinned") {
         isData = true;
      }
   }

   // Import the object to the workspace
   if (isVariable) {
      this->importVariableElement(n);
   } else if (isData) {
      HS3::Diagnostics diagnostics;
      HS3::Data data;
      readValue(HS3::Cursor{n, diagnostics}, data);
      HS3::report(diagnostics);
      _workspace.import(*loadData(data, _workspace, nullptr), RooFit::Silence());
   } else {
      this->importFunction(n, false);
   }
}

void RooJSONFactoryWSTool::importVariableElement(const JSONNode &elementNode)
{
   // A variable is given by its "value", by its range "min" and "max", or by both. With only a value, it is constant.
   JSONNode const *value = elementNode.find("value");
   JSONNode const *min = elementNode.find("min");
   JSONNode const *max = elementNode.find("max");
   if (!value && !(min && max)) {
      throw std::invalid_argument("Invalid Syntax: Please provide either 'value' or 'min' and 'max' or both");
   }
   HS3::DomainAxis axis;
   axis.name = elementNode["name"].val();
   HS3::ParameterValue parameter;
   parameter.name = axis.name;
   if (min && max) {
      axis.min = min->val_double();
      axis.max = max->val_double();
   }
   parameter.value = value ? value->val_double() : (*axis.max + *axis.min) / 2;
   parameter.isConst = value && !(min && max);
   HS3Document doc;
   doc.domains.push_back({HS3::defaultDomainName, {std::move(axis)}});
   doc.parameterPoints.push_back({"default_values", {parameter}});

   _import = std::make_unique<HS3::ToolImportState>(doc);
   struct ImportScope {
      RooJSONFactoryWSTool &tool;
      ~ImportScope() { tool._import.reset(); }
   } importScope{*this};

   importVariable(parameter);
   if (RooRealVar *rrv = _workspace.var(parameter.name)) {
      configureVariable(_import->ranges, parameter, *rrv);
   }
}

/**
 * @brief Writes a warning message to the RooFit message service.
 *
 * @param str The warning message to be logged.
 * @return std::ostream& A reference to the output stream.
 */
std::ostream &RooJSONFactoryWSTool::warning(std::string const &str)
{
   return RooMsgService::instance().log(nullptr, RooFit::MsgLevel::WARNING, RooFit::ObjectHandling) << str << std::endl;
}

/**
 * @brief Writes an error message to the RooFit message service and throws a runtime_error.
 *
 * @param s The error message to be logged and thrown.
 * @return void
 */
void RooJSONFactoryWSTool::error(const char *s)
{
   RooMsgService::instance().log(nullptr, RooFit::MsgLevel::ERROR, RooFit::IO) << s << std::endl;
   throw std::runtime_error(s);
}

/**
 * @brief Cleans up names to the HS3 standard
 *
 * @param str The string to be sanitized.
 * @return std::string
 */
std::string RooJSONFactoryWSTool::sanitizeName(const std::string str)
{
   std::string result;
   if (RooJSONFactoryWSTool::config().allowSanitizeNames) {
      for (char c : str) {
         switch (c) {
         case '[':
         case '|':
         case ',':
         case '(': result += '_'; break;
         case ']':
         case ')':
            // skip these characters entirely
            break;
         case '.': result += "_dot_"; break;
         case '@': result += "at"; break;
         case '-': result += "minus"; break;
         case '/': result += "_div_"; break;

         default: result += c; break;
         }
      }
      return result;
   }
   return str;
}

RooWorkspace RooJSONFactoryWSTool::cleanWS(const RooWorkspace &ws, bool onlyModelConfig)
{
   // Variables

   RooWorkspace tmpWS = RooWorkspace();
   if (onlyModelConfig) {
      for (auto *obj : ws.allGenericObjects()) {
         if (auto *mc = dynamic_cast<RooFit::ModelConfig *>(obj)) {
            tmpWS.import(*mc->GetPdf(), RooFit::RecycleConflictNodes(true));
         }
      }

   } else {

      for (auto *pdf : ws.allPdfs()) {
         if (isTopLevel(*pdf, ws)) {
            tmpWS.import(*pdf, RooFit::RecycleConflictNodes(true));
         }
      }

      for (auto *func : ws.allFunctions()) {
         if (isTopLevel(*func, ws)) {
            tmpWS.import(*func, RooFit::RecycleConflictNodes(true));
         }
      }
   }

   for (auto *data : ws.allData()) {
      tmpWS.import(*data);
   }

   for (auto *obj : ws.allGenericObjects()) {
      tmpWS.import(*obj);
   }

   for (auto *obj : ws.allResolutionModels()) {
      tmpWS.import(*obj);
   }

   for (auto *snsh : ws.getSnapshots()) {
      auto *snshSet = dynamic_cast<RooArgSet *>(snsh);
      if (snshSet) {
         tmpWS.saveSnapshot(snshSet->GetName(), *snshSet, true);
      }
   }

   return tmpWS;
}

RooJSONFactoryWSTool::Config &RooJSONFactoryWSTool::config()
{
   static Config conf;
   return conf;
}

// Sanitize all names in the workspace to be HS3 compliant
RooWorkspace RooJSONFactoryWSTool::sanitizeWS(const RooWorkspace &ws)
{
   // Variables

   RooWorkspace tmpWS = cleanWS(ws, false);

   auto sanitizeIfNeeded = [](auto const &list) {
      for (auto *obj : list) {
         if (!isValidName(obj->GetName())) {
            obj->SetName(sanitizeName(obj->GetName()).c_str());
         }
      }
   };
   sanitizeIfNeeded(tmpWS.allVars());
   sanitizeIfNeeded(tmpWS.allFunctions());
   sanitizeIfNeeded(tmpWS.allPdfs());
   sanitizeIfNeeded(tmpWS.allResolutionModels());
   // Datasets
   for (auto *data : tmpWS.allData()) {
      // Sanitize dataset name
      if (!isValidName(data->GetName())) {
         data->SetName(sanitizeName(data->GetName()).c_str());
      }
      for (auto *obj : *data->get()) {
         obj->SetName(sanitizeName(obj->GetName()).c_str());
      }
   }
   for (auto *data : tmpWS.allEmbeddedData()) {
      // Sanitize dataset name
      data->SetName(sanitizeName(data->GetName()).c_str());
      for (auto *obj : *data->get()) {
         obj->SetName(sanitizeName(obj->GetName()).c_str());
      }
   }
   for (auto *snshObj : tmpWS.getSnapshots()) {
      // Snapshots are stored as TObject*, but really they are RooArgSet*
      auto *snsh = dynamic_cast<RooArgSet *>(snshObj);
      if (!snsh) {
         std::cerr << "Warning: found snapshot that is not a RooArgSet, skipping\n";
         continue;
      }

      // Sanitize snapshot name
      if (!isValidName(snsh->GetName())) {
         snsh->setName(sanitizeName(snsh->GetName()).c_str());
      }

      // Sanitize the variables inside the snapshot
      for (auto *arg : *snsh) {
         if (!isValidName(arg->GetName())) {
            arg->SetName(sanitizeName(arg->GetName()).c_str());
         }
      }
   }

   // Generic objects (ModelConfigs, attributes, etc.)
   for (auto *obj : tmpWS.allGenericObjects()) {
      if (!isValidName(obj->GetName())) {
         if (auto *named = dynamic_cast<TNamed *>(obj)) {
            named->SetName(sanitizeName(named->GetName()).c_str());
         } else {
            std::cerr << "Warning: object " << obj->GetName() << " is not TNamed, cannot rename.\n";
         }
      }

      if (auto *mc = dynamic_cast<RooFit::ModelConfig *>(obj)) {
         // Sanitize ModelConfig name
         if (!isValidName(mc->GetName())) {
            mc->SetName(sanitizeName(mc->GetName()).c_str());
         }

         // Sanitize the sets inside ModelConfig
         for (auto *obs : mc->GetObservables()->get()) {
            if (obs) {
               obs->SetName(sanitizeName(obs->GetName()).c_str());
            }
         }
         for (auto *poi : mc->GetParametersOfInterest()->get()) {
            if (poi) {
               poi->SetName(sanitizeName(poi->GetName()).c_str());
            }
         }
         for (auto *nuis : mc->GetNuisanceParameters()->get()) {
            if (nuis) {
               nuis->SetName(sanitizeName(nuis->GetName()).c_str());
            }
         }
         for (auto *glob : mc->GetGlobalObservables()->get()) {
            if (glob) {
               glob->SetName(sanitizeName(glob->GetName()).c_str());
            }
         }
      }
   }
   std::string wsName = std::string{ws.GetName()} + "_sanitized";
   RooWorkspace newWS = cleanWS(tmpWS, false);
   newWS.SetName(wsName.c_str());

   return newWS;
}

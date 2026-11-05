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
#include <RooNumber.h>
#include <RooRealVar.h>
#include <RooBinning.h>
#include <RooAbsCategory.h>
#include <RooArgProxy.h>
#include <RooAbsProxy.h>
#include <RooAbsLValue.h>
#include <RooCategory.h>
#include <RooDataSet.h>
#include <RooDataHist.h>
#include <RooCustomizer.h>
#include <RooProdPdf.h>
#include <RooSimultaneous.h>
#include <RooSuperCategory.h>
#include <RooFit/ModelConfig.h>
#include <RooFitImplHelpers.h>
#include <RooAbsCollection.h>

#include "JSONIOUtils.h"
#include "HS3Builder.h"
#include "HS3Schema.h"
#include "JSONReadRecorder.h"

#include <TROOT.h>

#include <algorithm>
#include <iterator>
#include <fstream>
#include <iostream>
#include <numeric>
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
   /// The categories that are declared by categorical axes of the "default_domain", by name.
   std::map<std::string, HS3::CategoricalAxis> categories;
   /// Names of the objects written so far, to export each object once.
   std::set<std::string> exportedObjectNames;
   /// Array distributions to write at the end of the export: their components, and the array of each component.
   std::map<std::string, std::vector<RooAbsPdf const *>> arrayComponents;
   std::map<std::string, std::string> arrayOfComponent;
   /// The name of the top-level object whose export is in progress, and the depth of the export recursion.
   std::string rootName;
   int depth = 0;
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

// The axis of a domain for a range and binning. Infinite bounds are omitted, and only a bounded range has a binning.
HS3::DomainAxis toDomainAxis(std::string const &name, RooAbsBinning const &binning)
{
   HS3::DomainAxis axis;
   axis.name = name;
   if (!RooNumber::isInfinite(binning.lowBound())) {
      axis.min = binning.lowBound();
   }
   if (!RooNumber::isInfinite(binning.highBound())) {
      axis.max = binning.highBound();
   }
   if (axis.min && axis.max && binning.numBins() > 0) {
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

// Append a domain with the axes in the order of their names, continuous and categorical ones mixed.
void appendDomain(std::vector<HS3::Domain> &domains, std::string const &name,
                  std::map<std::string, HS3::DomainAxis> axes,
                  std::map<std::string, HS3::CategoricalAxis> categories = {})
{
   std::map<std::string, std::variant<HS3::DomainAxis, HS3::CategoricalAxis>> sorted;
   for (auto &item : axes) {
      sorted.emplace(item.first, std::move(item.second));
   }
   for (auto &item : categories) {
      sorted.emplace(item.first, std::move(item.second));
   }
   HS3::Domain &domain = domains.emplace_back();
   domain.name = name;
   for (auto &item : sorted) {
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
      if (RooNumber::isInfinite(*axis.min)) {
         var.removeMin();
      } else {
         var.setMin(*axis.min);
      }
   }
   if (axis.max) {
      if (RooNumber::isInfinite(*axis.max)) {
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
   const double inf = RooNumber::infinity();
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
   // A categorical axis is a RooCategory with one state per category, in the order of the document and with the
   // position as index, so that the bins of a RooDataHist follow the order of "categories". An existing RooCategory of
   // that name must have the same labels. That all domains agree on the categories is checked by the validation.
   for (HS3::Domain const &domain : domains) {
      for (auto const &axisVariant : domain.axes) {
         auto const *axis = std::get_if<HS3::CategoricalAxis>(&axisVariant);
         if (!axis) {
            continue;
         }
         if (RooAbsArg *existing = ws.arg(axis->name)) {
            auto *cat = dynamic_cast<RooCategory *>(existing);
            if (!cat) {
               RooJSONFactoryWSTool::error("the axis '" + axis->name + "' is categorical, but '" + axis->name +
                                           "' already exists and is not a category");
            }
            std::set<std::string> existingLabels;
            for (auto const &state : *cat) {
               existingLabels.insert(state.first);
            }
            if (existingLabels != std::set<std::string>(axis->categories.begin(), axis->categories.end())) {
               RooJSONFactoryWSTool::error("the categories of the axis '" + axis->name +
                                           "' differ from the states of the existing category '" + axis->name + "'");
            }
            continue;
         }
         RooCategory cat{axis->name.c_str(), axis->name.c_str()};
         for (std::size_t i = 0; i < axis->categories.size(); ++i) {
            cat.defineType(axis->categories[i], static_cast<int>(i));
         }
         ws.import(cat, RooFit::Silence());
      }
   }
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

// True if the entire string parses as a number (integer or floating-point).
bool isNumber(const std::string &str)
{
   // Parse with the same mechanism as toDouble() and require that the whole string is consumed, so that isNumber(s) is
   // true exactly when toDouble(s) can turn the entire string into a value. (std::from_chars for floating-point types
   // is not portably available on all platforms ROOT supports, so we rely on the stream extraction instead.)
   std::istringstream stream(str);
   double value = 0.0;
   return (stream >> value) && stream.eof();
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
   for (const RooAbsArg *arg : vars) {
      // Continuous variables and categories (one bin per state)
      vars_numbins.push_back(dynamic_cast<RooAbsLValue const &>(*arg).numBins());
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

// ---------------------------------------------------------------------------------------------------------------
// Distinct variates. HS3 requires the terms of a likelihood, and the components of a combined distribution or data
// set, to have distinct variates: a variate in two of them is one quantity modelled twice (e.g. a constraint that is
// a term of the likelihood and a factor of a channel), or two quantities with one name (e.g. the observable "mass"
// of two channels, which need their own names, "mass_SR" and "mass_CR"). All cases are checked with the same
// function, sharedVariates(), on the document, after the export and before the import.

bool isName(JSONNode const &node)
{
   return !node.is_container() && !node.is_null() && !node.is_number();
}

// Add the names in `node` (one name or a list) to `out`.
void addNames(JSONNode const &node, std::set<std::string> &out)
{
   if (node.is_seq()) {
      for (JSONNode const &child : node.children()) {
         addNames(child, out);
      }
   } else if (isName(node)) {
      out.insert(node.val());
   }
}

// The child `name` of the section `section` of the document, or nullptr.
JSONNode const *findInSection(JSONNode const &rootnode, const char *section, std::string const &name)
{
   JSONNode const *node = rootnode.find(section);
   return node ? RooJSONFactoryWSTool::findNamedChild(*node, name) : nullptr;
}

std::string typeOf(JSONNode const &node)
{
   return node.has_child("type") ? node["type"].val() : "";
}

// The names of the components of a combined distribution or data set, written as {"key": ..., <field>: <name>}.
std::vector<std::string> combinedComponents(JSONNode const &node, const char *field)
{
   std::vector<std::string> out;
   if (JSONNode const *components = node.find("components")) {
      for (JSONNode const &component : components->children()) {
         if (JSONNode const *name = component.find(field); name && isName(*name)) {
            out.push_back(name->val());
         }
      }
   }
   return out;
}

// The variates of the distribution `name` of the document: the "x" of a fundamental distribution (a name or a list)
// and the axes of a distribution with axes (e.g. histfactory_dist), through products, mixtures and combined
// distributions. The constraint terms named by a histfactory_dist are terms of their own, see likelihoodTerms().
void collectVariates(JSONNode const &rootnode, std::string const &name, std::set<std::string> &out,
                     std::set<std::string> &visited)
{
   if (!visited.insert(name).second) {
      return;
   }
   JSONNode const *node = findInSection(rootnode, "distributions", name);
   if (!node) {
      return;
   }
   std::string const type = typeOf(*node);
   auto recurse = [&](std::vector<std::string> const &parts) {
      for (std::string const &part : parts) {
         collectVariates(rootnode, part, out, visited);
      }
   };
   auto namesIn = [&](const char *key) {
      std::vector<std::string> parts;
      if (JSONNode const *list = node->find(key)) {
         for (JSONNode const &part : list->children()) {
            if (isName(part)) {
               parts.push_back(part.val());
            }
         }
      }
      return parts;
   };
   if (type == "product_dist") {
      recurse(namesIn("factors"));
   } else if (type == "mixture_dist") {
      recurse(namesIn("summands"));
   } else if (type == "combined_dist") {
      recurse(combinedComponents(*node, "dist"));
   } else {
      if (JSONNode const *x = node->find("x")) {
         addNames(*x, out);
      }
      if (JSONNode const *axes = node->find("axes")) {
         for (JSONNode const &axis : axes->children()) {
            if (JSONNode const *axisName = axis.find("name")) {
               addNames(*axisName, out);
            }
         }
      }
   }
}

std::set<std::string> variatesOf(JSONNode const &rootnode, std::string const &name)
{
   std::set<std::string> variates;
   std::set<std::string> visited;
   collectVariates(rootnode, name, variates, visited);
   return variates;
}

// The terms of the likelihood `likelihood` of the document, each once: its distributions (the components of a
// combined distribution, or of a RooSimultaneous written to ROOT_internal by older versions, are terms of their own),
// its "aux_distributions", and the constraint terms named by the modifiers of its histfactory_dist terms.
std::vector<std::string> likelihoodTerms(JSONNode const &rootnode, JSONNode const &likelihood)
{
   std::vector<std::string> terms;
   auto add = [&](std::string const &term) {
      if (std::find(terms.begin(), terms.end(), term) == terms.end()) {
         terms.push_back(term);
      }
   };
   for (const char *key : {"distributions", "aux_distributions"}) {
      JSONNode const *list = likelihood.find(key);
      if (!list) {
         continue;
      }
      for (JSONNode const &item : list->children()) {
         if (!isName(item)) {
            continue;
         }
         std::string const name = item.val();
         if (JSONNode const *node = findInSection(rootnode, "distributions", name)) {
            if (typeOf(*node) == "combined_dist") {
               for (std::string const &component : combinedComponents(*node, "dist")) {
                  add(component);
               }
               continue;
            }
         } else if (JSONNode const *legacy = findRooFitInternal(rootnode, "combined_distributions", name)) {
            if (JSONNode const *components = legacy->find("distributions")) {
               for (JSONNode const &component : components->children()) {
                  add(component.val());
               }
               continue;
            }
         }
         add(name);
      }
   }
   for (std::size_t i = 0; i < terms.size(); ++i) {
      JSONNode const *node = findInSection(rootnode, "distributions", terms[i]);
      if (!node || typeOf(*node) != "histfactory_dist" || !node->has_child("samples")) {
         continue;
      }
      for (JSONNode const &sample : (*node)["samples"].children()) {
         if (JSONNode const *modifiers = sample.find("modifiers")) {
            for (JSONNode const &modifier : modifiers->children()) {
               if (JSONNode const *constraints = modifier.find("constraints")) {
                  std::set<std::string> names;
                  addNames(*constraints, names);
                  for (std::string const &constraint : names) {
                     add(constraint);
                  }
               }
            }
         }
      }
   }
   return terms;
}

// The parts (terms or components) of something that must have distinct variates, with their variates.
using PartVariates = std::vector<std::pair<std::string, std::set<std::string>>>;

// The variates that several of `parts` share, one message each. `what` names the parts and `context` the whole,
// e.g. "terms" of "the likelihood 'lh'".
std::vector<std::string> sharedVariates(PartVariates const &parts, std::string const &what, std::string const &context)
{
   std::map<std::string, std::vector<std::string>> partsOfVariate;
   std::vector<std::string> order;
   for (auto const &[part, variates] : parts) {
      for (std::string const &variate : variates) {
         auto &list = partsOfVariate[variate];
         if (list.empty()) {
            order.push_back(variate);
         }
         if (std::find(list.begin(), list.end(), part) == list.end()) {
            list.push_back(part);
         }
      }
   }
   std::vector<std::string> messages;
   for (std::string const &variate : order) {
      auto const &list = partsOfVariate[variate];
      if (list.size() < 2) {
         continue;
      }
      std::string names;
      for (std::size_t i = 0; i < list.size(); ++i) {
         names += (i == 0 ? "'" : (i + 1 == list.size() ? "' and '" : "', '")) + list[i];
      }
      messages.push_back("the variate '" + variate + "' is shared by the " + what + " " + names + "' of " + context);
   }
   return messages;
}

// The axes of the components of a combined data set, by component.
PartVariates combinedDataAxes(JSONNode const &rootnode, JSONNode const &dataset)
{
   PartVariates parts;
   for (std::string const &component : combinedComponents(dataset, "dataset")) {
      std::set<std::string> axes;
      if (JSONNode const *node = findInSection(rootnode, "data", component)) {
         if (JSONNode const *axesNode = node->find("axes")) {
            for (JSONNode const &axis : axesNode->children()) {
               if (JSONNode const *axisName = axis.find("name")) {
                  addNames(*axisName, axes);
               }
            }
         }
      }
      parts.emplace_back(component, axes);
   }
   return parts;
}

// Bookkeeping columns (a RooFit extension, in misc.ROOT_internal.bookkeeping_columns as {"<combined data set>":
// [<axis>, ...]}): axes that several components of a combined data set share, but that are no variate of any
// distribution, e.g. an event number. They have no statistical meaning, and are exempt from the requirement of
// distinct axes. A declared column that is a variate of a distribution is not a bookkeeping column.
std::map<std::string, std::set<std::string>> bookkeepingColumnsOfDocument(JSONNode const &rootnode)
{
   std::map<std::string, std::set<std::string>> out;
   JSONNode const *declared = findRooFitInternal(rootnode, "bookkeeping_columns");
   if (!declared || !declared->is_map()) {
      return out;
   }
   std::set<std::string> variates;
   if (JSONNode const *distributions = rootnode.find("distributions")) {
      for (JSONNode const &dist : distributions->children()) {
         std::set<std::string> const v = variatesOf(rootnode, RooJSONFactoryWSTool::name(dist));
         variates.insert(v.begin(), v.end());
      }
   }
   for (JSONNode const &entry : declared->children()) {
      for (JSONNode const &column : entry.children()) {
         if (!variates.count(column.val())) {
            out[entry.key()].insert(column.val());
         }
      }
   }
   return out;
}

// All shared variates of the document: of the terms of each likelihood, of the components of the combined
// distributions that no likelihood lists (the others are checked through the likelihood), and the shared axes of the
// components of all combined data sets, except for bookkeeping columns.
std::vector<std::string> sharedVariatesOfDocument(JSONNode const &rootnode)
{
   std::vector<std::string> messages;
   auto append = [&](std::vector<std::string> const &more) {
      messages.insert(messages.end(), more.begin(), more.end());
   };
   std::set<std::string> inLikelihoods;
   if (JSONNode const *likelihoods = rootnode.find("likelihoods")) {
      for (JSONNode const &likelihood : likelihoods->children()) {
         for (const char *key : {"distributions", "data", "aux_distributions"}) {
            if (JSONNode const *list = likelihood.find(key)) {
               for (JSONNode const &item : list->children()) {
                  inLikelihoods.insert(item.val());
               }
            }
         }
         PartVariates parts;
         for (std::string const &term : likelihoodTerms(rootnode, likelihood)) {
            parts.emplace_back(term, variatesOf(rootnode, term));
         }
         append(sharedVariates(parts, "terms", "the likelihood '" + RooJSONFactoryWSTool::name(likelihood) + "'"));
      }
   }
   if (JSONNode const *distributions = rootnode.find("distributions")) {
      for (JSONNode const &dist : distributions->children()) {
         std::string const name = RooJSONFactoryWSTool::name(dist);
         if (typeOf(dist) != "combined_dist" || inLikelihoods.count(name)) {
            continue;
         }
         PartVariates parts;
         for (std::string const &component : combinedComponents(dist, "dist")) {
            parts.emplace_back(component, variatesOf(rootnode, component));
         }
         append(sharedVariates(parts, "components", "the combined_dist '" + name + "'"));
      }
   }
   // The axes of the components of every combined data set, also of those that a likelihood lists, except for the
   // bookkeeping columns.
   std::map<std::string, std::set<std::string>> const bookkeeping = bookkeepingColumnsOfDocument(rootnode);
   if (JSONNode const *data = rootnode.find("data")) {
      for (JSONNode const &dataset : data->children()) {
         std::string const name = RooJSONFactoryWSTool::name(dataset);
         if (typeOf(dataset) != "combined") {
            continue;
         }
         PartVariates parts = combinedDataAxes(rootnode, dataset);
         if (auto found = bookkeeping.find(name); found != bookkeeping.end()) {
            for (auto &part : parts) {
               for (std::string const &column : found->second) {
                  part.second.erase(column);
               }
            }
         }
         append(sharedVariates(parts, "components", "the combined data set '" + name + "'"));
      }
   }
   return messages;
}

// Declare the bookkeeping columns of the exported document `rootnode` (see bookkeepingColumnsOfDocument()): the axes
// that several components of a combined data set share, and whose variable in the workspace `ws` is used by nothing
// else, i.e. has no clients and is in no set of a ModelConfig. RooAbsData::split() keeps such a column (e.g. an event
// number) in every sub-data set.
void declareBookkeepingColumns(JSONNode &rootnode, RooWorkspace const &ws)
{
   JSONNode const *data = rootnode.find("data");
   if (!data) {
      return;
   }
   std::vector<RooArgSet const *> modelConfigSets;
   for (TObject *obj : ws.allGenericObjects()) {
      if (auto *mc = dynamic_cast<RooFit::ModelConfig *>(obj)) {
         for (RooArgSet const *set : {mc->GetObservables(), mc->GetParametersOfInterest(), mc->GetNuisanceParameters(),
                                      mc->GetGlobalObservables(), mc->GetConditionalObservables()}) {
            if (set) {
               modelConfigSets.push_back(set);
            }
         }
      }
   }
   auto usedByNothing = [&](std::string const &name) {
      RooAbsArg const *arg = ws.arg(name.c_str());
      if (!arg || !arg->clients().empty()) {
         return false;
      }
      return std::none_of(modelConfigSets.begin(), modelConfigSets.end(),
                          [&](RooArgSet const *set) { return set->find(name.c_str()) != nullptr; });
   };
   std::map<std::string, std::set<std::string>> columns;
   for (JSONNode const &dataset : data->children()) {
      if (typeOf(dataset) != "combined") {
         continue;
      }
      std::map<std::string, int> counts;
      for (auto const &[component, axes] : combinedDataAxes(rootnode, dataset)) {
         for (std::string const &axis : axes) {
            ++counts[axis];
         }
      }
      for (auto const &[axis, count] : counts) {
         if (count > 1 && usedByNothing(axis)) {
            columns[RooJSONFactoryWSTool::name(dataset)].insert(axis);
         }
      }
   }
   if (columns.empty()) {
      return;
   }
   JSONNode &node = rootnode["misc"]["ROOT_internal"]["bookkeeping_columns"].set_map();
   for (auto const &[dataset, names] : columns) {
      node[dataset].fill_seq(names);
   }
}

// Write each term of the likelihoods of the exported document `rootnode` once. A RooFit model often has a constraint
// both as a factor of the product of a channel and as a term of the likelihood (e.g. a constraint of a HistFactory
// modifier, or an external constraint of the ModelConfig); RooFit counts it once. In HS3 it would enter twice, so the
// factor is removed from the product. Only products that no other distribution uses are changed (being a component
// of a combined distribution is not a use), and only factors that are terms of every likelihood that has the product
// as a term.
void writeLikelihoodTermsOnce(JSONNode &rootnode)
{
   JSONNode const *likelihoods = rootnode.find("likelihoods");
   JSONNode const *distributions = rootnode.find("distributions");
   if (!likelihoods || !distributions) {
      return;
   }
   // The products that are terms, with the factors that are terms wherever the product is one.
   std::map<std::string, std::set<std::string>> removable;
   for (JSONNode const &likelihood : likelihoods->children()) {
      std::vector<std::string> const terms = likelihoodTerms(rootnode, likelihood);
      for (std::string const &term : terms) {
         JSONNode const *node = findInSection(rootnode, "distributions", term);
         if (!node || typeOf(*node) != "product_dist" || !node->has_child("factors")) {
            continue;
         }
         std::set<std::string> factorsThatAreTerms;
         for (JSONNode const &factor : (*node)["factors"].children()) {
            if (factor.val() != term && std::find(terms.begin(), terms.end(), factor.val()) != terms.end()) {
               factorsThatAreTerms.insert(factor.val());
            }
         }
         auto found = removable.find(term);
         if (found == removable.end()) {
            removable[term] = factorsThatAreTerms;
         } else {
            std::set<std::string> both;
            std::set_intersection(found->second.begin(), found->second.end(), factorsThatAreTerms.begin(),
                                  factorsThatAreTerms.end(), std::inserter(both, both.begin()));
            found->second = both;
         }
      }
   }
   // Products that another distribution uses are left alone.
   for (JSONNode const &dist : distributions->children()) {
      std::vector<std::string> used;
      for (const char *key : {"factors", "summands"}) {
         if (JSONNode const *parts = dist.find(key)) {
            for (JSONNode const &part : parts->children()) {
               if (isName(part)) {
                  used.push_back(part.val());
               }
            }
         }
      }
      if (typeOf(dist) != "combined_dist") {
         for (std::string const &component : combinedComponents(dist, "dist")) {
            used.push_back(component);
         }
      }
      for (std::string const &name : used) {
         removable.erase(name);
      }
   }
   for (auto const &[product, remove] : removable) {
      if (remove.empty()) {
         continue;
      }
      auto &node = const_cast<JSONNode &>(*findInSection(rootnode, "distributions", product));
      std::vector<std::string> keep;
      for (JSONNode const &factor : node["factors"].children()) {
         if (remove.find(factor.val()) == remove.end()) {
            keep.push_back(factor.val());
         }
      }
      if (keep.empty()) {
         continue;
      }
      for (std::string const &factor : remove) {
         oocoutI(nullptr, IO) << "RooFitHS3: '" << factor << "' is a factor of '" << product
                              << "' and a term of its likelihood; it is written once, as a term of the likelihood"
                              << std::endl;
      }
      node["factors"].clear();
      node["factors"].fill_seq(keep);
   }
}
// ---------------------------------------------------------------------------------------------------------------

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
      // Skip some RooFit internals. The array distribution tag is restored from the type on import.
      if (attr == "SnapShot_ExtRefClone" || attr == "RooRealConstant_Factory_Object" || attr == arrayDistTag)
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

// The labels of a category, as a set.
std::set<std::string> labelsOf(RooCategory const &cat)
{
   std::set<std::string> labels;
   for (auto const &state : cat) {
      labels.insert(state.first);
   }
   return labels;
}

// Create the variables for the axes of binned data. A categorical axis is one bin per category. HS3 identifies
// categories by name, so the data may list them in another order than the domain that declares the category.
// RooDataHist looks up the bin of a category by the order of the states of the category it is given, so the category
// of the data takes over the states and their order from the category of the workspace. `binOrder` is filled with the
// position in the document of each bin in RooFit's order (last axis fastest), or left empty if the orders agree.
RooArgSet createBinnedObservables(std::vector<std::variant<HS3::BinnedAxis, HS3::CategoricalAxis>> const &axes,
                                  RooWorkspace const &workspace, std::string const &dataName,
                                  std::vector<std::size_t> &binOrder)
{
   RooArgSet vars;
   std::vector<std::vector<std::size_t>> axisPositions;
   bool reordered = false;
   for (auto const &axisVariant : axes) {
      if (auto const *axis = std::get_if<HS3::BinnedAxis>(&axisVariant)) {
         auto obs = createObservable(*axis);
         std::vector<std::size_t> positions(obs->numBins());
         std::iota(positions.begin(), positions.end(), 0);
         axisPositions.push_back(std::move(positions));
         vars.addOwned(std::move(obs));
         continue;
      }
      auto const &axis = std::get<HS3::CategoricalAxis>(axisVariant);
      auto cat = std::make_unique<RooCategory>(axis.name.c_str(), axis.name.c_str());
      if (RooCategory const *wsCat = workspace.cat(axis.name)) {
         if (labelsOf(*wsCat) != std::set<std::string>(axis.categories.begin(), axis.categories.end())) {
            RooJSONFactoryWSTool::error("the categories of the axis '" + axis.name + "' of the dataset '" + dataName +
                                        "' differ from those of the category '" + axis.name + "'");
         }
         for (unsigned int i = 0; i < wsCat->size(); ++i) {
            auto const &state = wsCat->getOrdinal(i);
            cat->defineType(state.first, state.second);
         }
      } else {
         for (std::size_t i = 0; i < axis.categories.size(); ++i) {
            cat->defineType(axis.categories[i], static_cast<int>(i));
         }
      }
      std::vector<std::size_t> positions(axis.categories.size());
      for (std::size_t i = 0; i < positions.size(); ++i) {
         auto found = std::find(axis.categories.begin(), axis.categories.end(), cat->getOrdinal(i).first);
         positions[i] = std::distance(axis.categories.begin(), found);
         reordered |= positions[i] != i;
      }
      axisPositions.push_back(std::move(positions));
      vars.addOwned(std::move(cat));
   }
   binOrder.clear();
   if (!reordered) {
      return vars;
   }
   std::size_t total = 1;
   for (auto const &positions : axisPositions) {
      total *= positions.size();
   }
   binOrder.resize(total);
   for (std::size_t k = 0; k < total; ++k) {
      // Decompose RooFit's bin index (last axis fastest) and recompose it with the positions in the document.
      std::size_t rest = k;
      std::size_t docIndex = 0;
      std::size_t stride = 1;
      for (std::size_t j = axisPositions.size(); j-- > 0;) {
         const std::size_t nBins = axisPositions[j].size();
         docIndex += axisPositions[j][rest % nBins] * stride;
         rest /= nBins;
         stride *= nBins;
      }
      binOrder[k] = docIndex;
   }
   return vars;
}

// The values in RooFit's bin order, from the order of the document (see createBinnedObservables()).
std::vector<double> permute(std::vector<double> const &values, std::vector<std::size_t> const &order)
{
   if (order.empty() || values.size() != order.size()) {
      return values;
   }
   std::vector<double> out(values.size());
   for (std::size_t k = 0; k < order.size(); ++k) {
      out[k] = values[order[k]];
   }
   return out;
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
         // The category of a categorical axis is declared by a domain.
         auto const &catAxis = std::get<HS3::CategoricalAxis>(axisVariant);
         RooCategory *cat = workspace.cat(catAxis.name);
         if (!cat) {
            RooJSONFactoryWSTool::error("The categorical observable \"" + catAxis.name +
                                        "\" could not be found in the workspace! Categorical observables are "
                                        "declared by a categorical axis in a domain.");
         }
         if (labelsOf(*cat) != std::set<std::string>(catAxis.categories.begin(), catAxis.categories.end())) {
            RooJSONFactoryWSTool::error("The categories of the axis \"" + catAxis.name +
                                        "\" differ from the states of the category \"" + catAxis.name + "\"");
         }
         varlist.add(*cat);
         continue;
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
         const double value = input.entries[i * nAxes + j];
         if (auto const *catAxis = std::get_if<HS3::CategoricalAxis>(&input.axes[j])) {
            // The entry of a categorical axis is the position of the label in its "categories".
            static_cast<RooCategory *>(varlist.at(j))->setLabel(catAxis->categories[static_cast<std::size_t>(value)]);
         } else {
            static_cast<RooRealVar *>(varlist.at(j))->setVal(value);
         }
      }
      data->add(vars, input.weights ? (*input.weights)[i] : 1.);
   }
   return data;
}

// Create a RooAbsData (binned or unbinned) from its description.
std::unique_ptr<RooAbsData>
loadData(HS3::Data const &input, RooWorkspace &workspace, HS3::DomainRanges const *ranges)
{
   RooJSONFactoryWSTool::testValidName(input.name, true);

   // What RooFit can't store is reported, instead of being dropped silently.
   if (auto const *binned = std::get_if<HS3::BinnedData>(&input.body)) {
      if (binned->uncertainty && !binned->uncertainty->correlation.empty()) {
         RooJSONFactoryWSTool::warning("RooFitHS3: the correlations of the uncertainties of the dataset '" +
                                       input.name + "' are ignored, because RooDataHist can't store them.");
      }
      std::vector<std::size_t> binOrder;
      RooArgSet vars = createBinnedObservables(binned->axes, workspace, input.name, binOrder);
      std::vector<double> sigma;
      if (binned->uncertainty) {
         sigma = permute(binned->uncertainty->sigma, binOrder);
      }
      return HS3::createDataHist(input.name, vars, permute(binned->contents, binOrder),
                                 binned->uncertainty ? &sigma : nullptr);
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

// Check that a combined data set paired with a RooSimultaneous in a likelihood has a sub-data set for every component.
// RooFit has no notion of an unobserved component: a key without sub-data set is a channel without entries, which
// adds the expected number of events of an extended component to the likelihood. That changes the fit, so it is an
// error (https://github.com/root-project/root/issues/24004). For a non-extended component it changes nothing, and a
// sub-data set without component is ignored by RooFit; both are reported as warnings.
void checkCombinedPairing(RooSimultaneous const &simPdf, std::string const &dataName, HS3::CombinedData const &data)
{
   std::vector<RooCategory const *> columns = selectionColumns(simPdf.indexCat());
   std::vector<std::string> columnNames;
   for (RooCategory const *column : columns) {
      columnNames.push_back(column->GetName());
   }
   std::vector<std::string> dataColumns = data.columns;
   std::sort(dataColumns.begin(), dataColumns.end());
   if (dataColumns != columnNames) {
      RooJSONFactoryWSTool::error("the likelihood pairs '" + std::string(simPdf.GetName()) +
                                  "' with the combined data set '" + dataName + "', but their columns differ");
   }
   // The keys of the data, each as a map from the column to the label.
   std::set<std::map<std::string, std::string>> dataKeys;
   for (HS3::CombinedDataComponent const &component : data.components) {
      std::map<std::string, std::string> key;
      for (std::size_t i = 0; i < data.columns.size(); ++i) {
         key[data.columns[i]] = component.key[i];
      }
      dataKeys.insert(key);
   }
   auto keyToString = [&](std::map<std::string, std::string> const &key) {
      std::vector<std::string> labels;
      for (std::string const &column : data.columns) {
         labels.push_back(key.at(column));
      }
      return selectionKeyToString(labels);
   };
   std::set<std::map<std::string, std::string>> pdfKeys;
   for (std::vector<std::string> const &labels : selectionKeys(columns)) {
      RooAbsPdf const *component = simPdf.getPdf(selectionLabel(simPdf.indexCat(), columns, labels));
      if (!component) {
         continue;
      }
      std::map<std::string, std::string> key;
      for (std::size_t i = 0; i < columns.size(); ++i) {
         key[columnNames[i]] = labels[i];
      }
      pdfKeys.insert(key);
      if (dataKeys.count(key) > 0) {
         continue;
      }
      if (component->canBeExtended()) {
         RooJSONFactoryWSTool::error(
            "the combined data set '" + dataName + "' has no sub-data set for the key " + keyToString(key) +
            " of the extended component '" + component->GetName() + "' of '" + simPdf.GetName() +
            "'. RooFit cannot leave a component unobserved: it would add the expected number of events of the " +
            "component to the likelihood, see https://github.com/root-project/root/issues/24004");
      }
      RooJSONFactoryWSTool::warning("RooFitHS3: the combined data set '" + dataName +
                                    "' has no sub-data set for the key " + keyToString(key) + " of the component '" +
                                    component->GetName() + "' of '" + simPdf.GetName() +
                                    "', which RooFit treats as present and empty.");
   }
   for (auto const &key : dataKeys) {
      if (pdfKeys.count(key) == 0) {
         RooJSONFactoryWSTool::warning("RooFitHS3: the sub-data set for the key " + keyToString(key) +
                                       " of the combined data set '" + dataName + "' is ignored, because '" +
                                       simPdf.GetName() + "' has no component for it.");
      }
   }
}

// Import an analysis (likelihood + domains) as one or more ModelConfig objects into the workspace.
void importAnalysis(HS3Document const &doc, HS3::Analysis const &analysis, RooWorkspace &workspace,
                    const std::vector<std::unique_ptr<RooAbsData>> &datasets,
                    std::set<std::string> const &bookkeepingColumns)
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
         // Components of combined data sets have been moved out of the list.
         if (d && d->GetName() == dataRef->name) {
            found = true;
            observables.add(*d->get(), true);
         }
      }
      if (!found)
         throw std::runtime_error("dataset '" + dataRef->name + "' cannot be found!");
   }
   // Bookkeeping columns of the data sets are no observables of the model.
   for (std::string const &column : bookkeepingColumns) {
      if (RooAbsArg *arg = observables.find(column.c_str())) {
         observables.remove(*arg);
      }
   }

   // A combined distribution paired with a combined data set must have a sub-data set for every component.
   for (auto const &term : likelihood.terms) {
      auto const *dataRef = std::get_if<HS3::DataRef>(&term.data);
      auto *simPdf = dynamic_cast<RooSimultaneous const *>(workspace.pdf(term.distribution.name));
      if (!dataRef || !simPdf) {
         continue;
      }
      for (HS3::Data const &data : doc.data) {
         auto const *combined = std::get_if<HS3::CombinedData>(&data.body);
         if (combined && data.name == dataRef->name) {
            checkCombinedPairing(*simPdf, data.name, *combined);
         }
      }
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

// HistFactory modifiers pick the components that constrain their parameters out of an array distribution. By the
// standard the whole array enters the likelihood, so the components of a partially used array that nothing uses are
// reported.
void warnAboutUnusedArrayComponents(RooWorkspace const &ws)
{
   for (RooAbsArg *arg : ws.allPdfs()) {
      auto const *array = dynamic_cast<RooProdPdf const *>(arg);
      if (!array || !array->getAttribute(arrayDistTag) || array->hasClients()) {
         continue;
      }
      bool used = false;
      std::string unused;
      for (RooAbsArg const *component : array->pdfList()) {
         bool usedElsewhere = false;
         for (RooAbsArg const *client : component->clients()) {
            if (client != array) {
               usedElsewhere = true;
            }
         }
         if (usedElsewhere) {
            used = true;
         } else {
            unused += (unused.empty() ? "'" : ", '") + std::string(component->GetName()) + "'";
         }
      }
      if (used && !unused.empty()) {
         RooJSONFactoryWSTool::warning("RooFitHS3: the components " + unused + " of the array distribution '" +
                                       array->GetName() +
                                       "' are not used by any distribution and are not part of any likelihood");
      }
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

// Build the combined data sets of the document: a RooDataSet with one category column per entry of "columns" and the
// variables of all components, filled row by row from the components, which are then no longer imported on their own.
// The columns are categories declared by categorical axes of the domains. A key without a sub-data set is a state
// without entries: RooFit has no notion of an absent component. The validation has checked the columns, the keys and
// that the components are data sets of the document.
void assembleCombinedData(HS3Document const &doc, RooWorkspace const &workspace,
                          std::vector<std::unique_ptr<RooAbsData>> &datasets)
{
   for (HS3::Data const &input : doc.data) {
      auto const *info = std::get_if<HS3::CombinedData>(&input.body);
      if (!info) {
         continue;
      }
      std::string const &name = input.name;
      RooJSONFactoryWSTool::testValidName(name, true);
      std::string const context = "the combined data set '" + name + "'";
      std::vector<RooCategory *> columns;
      for (std::string const &column : info->columns) {
         RooCategory *cat = workspace.cat(column);
         if (!cat) {
            RooJSONFactoryWSTool::error("the column '" + column + "' of " + context +
                                        " is not declared by a categorical axis of a domain");
         }
         columns.push_back(cat);
      }

      struct Component {
         std::vector<std::string> const *key;
         std::unique_ptr<RooAbsData> data;
      };
      std::vector<Component> components;
      for (HS3::CombinedDataComponent const &c : info->components) {
         for (std::size_t i = 0; i < columns.size(); ++i) {
            if (!columns[i]->hasLabel(c.key[i])) {
               RooJSONFactoryWSTool::error("the label '" + c.key[i] + "' in position " + std::to_string(i) +
                                           " of a key of " + context + " is not one of the categories of '" +
                                           columns[i]->GetName() + "'");
            }
         }
         auto found =
            std::find_if(datasets.begin(), datasets.end(), [&](auto &d) { return d && d->GetName() == c.data.name; });
         if (found == datasets.end()) {
            RooJSONFactoryWSTool::error("the component '" + c.data.name + "' of " + context +
                                        " is not a data set of the document, or it is already a component of " +
                                        "another combined data set");
         }
         components.push_back({&c.key, std::move(*found)});
      }

      // The variables: the columns first, then those of the components. A weight variable if any component is
      // weighted, with errors if any component is binned, like RooDataSet does when importing a map of data sets.
      RooArgSet columnClones; // owns the clones; `vars` must not own anything, it also lists the components' variables
      for (RooCategory const *column : columns) {
         columnClones.addClone(*column);
      }
      RooArgSet vars{columnClones};
      bool weighted = false;
      bool storeErrors = false;
      std::string weightName = RooFit::WeightVar().getString(0);
      for (auto const &c : components) {
         vars.add(*c.data->get(), /*silent=*/true);
         weighted |= c.data->isWeighted();
         if (auto *ds = dynamic_cast<RooDataSet const *>(c.data.get()); ds && ds->isWeighted()) {
            weightName = ds->weightVar()->GetName();
         }
         storeErrors |= dynamic_cast<RooDataHist const *>(c.data.get()) != nullptr;
      }
      std::unique_ptr<RooDataSet> combined;
      if (weighted) {
         RooRealVar weightVar{weightName.c_str(), weightName.c_str(), 1.0};
         vars.add(weightVar);
         RooArgSet errorSet{weightVar};
         combined = std::make_unique<RooDataSet>(name.c_str(), name.c_str(), vars, RooFit::WeightVar(weightVar),
                                                 storeErrors ? RooFit::StoreError(errorSet) : RooCmdArg::none());
      } else {
         combined = std::make_unique<RooDataSet>(name.c_str(), name.c_str(), vars);
      }
      for (auto const &c : components) {
         for (std::size_t i = 0; i < columns.size(); ++i) {
            static_cast<RooCategory &>(vars[columns[i]->GetName()]).setLabel((*c.key)[i].c_str());
         }
         for (int i = 0; i < c.data->numEntries(); ++i) {
            vars.assign(*c.data->get(i));
            if (weighted) {
               combined->add(vars, c.data->weight(), storeErrors ? c.data->weightError(RooAbsData::SumW2) : 0.0);
            } else {
               combined->add(vars);
            }
         }
      }
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

bool isLiteralConstVar(RooAbsArg const &arg)
{
   bool isRooConstVar = dynamic_cast<RooConstVar const *>(&arg);
   return isRooConstVar && isNumber(arg.GetName());
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
      if (isLiteralConstVar(*arg)) {
         node.append_child() << static_cast<RooConstVar const *>(arg)->getVal();
      } else {
         node.append_child() << arg->GetName();
      }
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
   if (isNumber(objname))
      return &RooFit::RooConst(toDouble(objname));
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
   std::string out = RooFit::Detail::makeValidVarName(in);
   if (out != in) {
      oocoutW(nullptr, ObjectHandling) << "RooFitHS3: changed '" << in << "' to '" << out << "' to become a valid name";
   }
   return out;
}

// The labels of a selection key, made valid names and joined by "_". Used in the names of sub-data sets and of
// renamed observables.
std::string selectionKeyName(std::vector<std::string> const &key)
{
   std::string out;
   for (std::string const &label : key) {
      out += (out.empty() ? "" : "_") + makeValidNameOrError(label);
   }
   return out;
}

// The names of the category columns of `vars`.
std::vector<std::string> selectionColumnNames(RooAbsCollection const &vars)
{
   std::vector<std::string> names;
   for (RooCategory const *column : selectionColumns(vars)) {
      names.push_back(column->GetName());
   }
   return names;
}

// The RooSimultaneous of the workspace (the first by name) whose columns are `columns`, or nullptr.
RooSimultaneous const *findMatchingSimultaneous(RooWorkspace const &workspace, std::vector<std::string> const &columns)
{
   std::vector<RooSimultaneous const *> candidates;
   for (RooAbsArg *arg : workspace.allPdfs()) {
      if (auto *candidate = dynamic_cast<RooSimultaneous const *>(arg)) {
         candidates.push_back(candidate);
      }
   }
   sortByName(candidates);
   for (RooSimultaneous const *candidate : candidates) {
      std::vector<std::string> candidateColumns;
      for (RooCategory const *column : selectionColumns(candidate->indexCat())) {
         candidateColumns.push_back(column->GetName());
      }
      if (candidateColumns == columns) {
         return candidate;
      }
   }
   return nullptr;
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
// the exported object names to avoid exporting it twice.
void RooJSONFactoryWSTool::exportObject(RooAbsArg const &func)
{
   std::string name = func.GetName();
   std::set<std::string> &exportedObjectNames = _export->exportedObjectNames;

   // if this element was already exported, skip
   if (exportedObjectNames.find(name) != exportedObjectNames.end())
      return;

   exportedObjectNames.insert(name);

   // Exporters that group objects per top-level pdf (the HistFactory constraints) ask for the root of the recursion.
   if (_export->depth == 0) {
      _export->rootName = name;
   }
   struct DepthGuard {
      int &depth;
      DepthGuard(int &d) : depth{d} { ++depth; }
      ~DepthGuard() { --depth; }
   } depthGuard{_export->depth};

   if (auto simPdf = dynamic_cast<RooSimultaneous const *>(&func)) {
      exportSimultaneous(*simPdf);
      return;
   } else if (dynamic_cast<RooAbsCategory const *>(&func)) {
      // Categories are not objects of the document: the index category of a RooSimultaneous is declared by the
      // categorical axes of the domain that its "columns" name.
      return;
   } else if (dynamic_cast<RooRealVar const *>(&func) || dynamic_cast<RooConstVar const *>(&func)) {
      exportVariable(&func, _export->doc.parameterPoints.front(), true);
      return;
   } else if (config().exportArrayDist && func.getAttribute(arrayDistTag)) {
      // An array distribution (HS3 `array_dist`): its components are registered here, and the array is written at
      // the end of the export, see exportAsArrayComponent().
      auto const *prod = dynamic_cast<RooProdPdf const *>(&func);
      if (!prod) {
         error("'" + name + "' has the attribute '" + arrayDistTag +
               "' of an array distribution, but it is not a RooProdPdf");
      }
      for (RooAbsPdf *pdf : static_range_cast<RooAbsPdf *>(prod->pdfList())) {
         const std::string writtenIn = exportAsArrayComponent(*pdf, name);
         if (writtenIn != name) {
            error("cannot write '" + name + "' as an array distribution: its component '" + pdf->GetName() +
                  "' is already written as '" + writtenIn + "'");
         }
      }
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
            exportObjects(func.servers());
         } else {
            exportObjects(_serversToExport);
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
         if (isLiteralConstVar(*r->absArg())) {
            elem[k->second] << static_cast<RooConstVar *>(r->absArg())->getVal();
         } else {
            elem[k->second] << r->absArg()->GetName();
         }
      }
   }

   // export all the servers of a given RooAbsArg
   for (RooAbsArg *s : func.servers()) {
      if (!s) {
         std::cerr << "unable to locate server of " << func.GetName() << std::endl;
         continue;
      }
      this->exportObject(*s);
   }
}

std::string const &RooJSONFactoryWSTool::exportRootName() const
{
   return _export->rootName;
}

std::string RooJSONFactoryWSTool::exportAsArrayComponent(RooAbsPdf const &pdf, std::string const &arrayName)
{
   const std::string pdfName = pdf.GetName();
   auto &arrayOfComponent = _export->arrayOfComponent;
   auto &arrayComponents = _export->arrayComponents;
   auto &exportedObjectNames = _export->exportedObjectNames;
   if (auto it = arrayOfComponent.find(pdfName); it != arrayOfComponent.end()) {
      return it->second;
   }
   // Already written on its own: it is referenced by its name.
   if (exportedObjectNames.find(pdfName) != exportedObjectNames.end()) {
      return pdfName;
   }
   if (arrayComponents.find(arrayName) == arrayComponents.end()) {
      RooAbsArg const *existing = _workspace.arg(arrayName);
      if (existing && !existing->getAttribute(arrayDistTag)) {
         error("cannot write the array distribution '" + arrayName +
               "': the workspace has another object of that name");
      }
      exportedObjectNames.insert(arrayName);
   }
   arrayComponents[arrayName].push_back(&pdf);
   arrayOfComponent[pdfName] = arrayName;
   exportedObjectNames.insert(pdfName);
   return arrayName;
}

// Write the array distributions registered with exportAsArrayComponent(), in alphabetical order, each followed by the
// servers of its components. They are generic objects of the document.
void RooJSONFactoryWSTool::exportArrayDists()
{
   std::set<std::string> written;
   auto &arrayComponents = _export->arrayComponents;
   for (;;) {
      auto next = std::find_if(arrayComponents.begin(), arrayComponents.end(),
                               [&](auto const &item) { return written.find(item.first) == written.end(); });
      if (next == arrayComponents.end()) {
         break;
      }
      const std::string arrayName = next->first;
      written.insert(arrayName);
      // Exporting the servers may add to the registry, so the components are copied.
      const std::vector<RooAbsPdf const *> components = next->second;
      JSONNode &elem = appendNamedChild((*_export->root)["distributions"], arrayName);
      writeArrayDist(elem, arrayName, components);
      recordGenericObject(_export->doc, true, arrayName, elem);
      for (RooAbsPdf const *pdf : components) {
         exportObjects(pdf->servers());
      }
   }
   arrayComponents.clear();
   _export->arrayOfComponent.clear();
}

/**
 * @brief Export a RooSimultaneous as a "combined_dist".
 *
 * The components are selected by a key with one label per column. The columns are the index category of the
 * RooSimultaneous or, for a RooSuperCategory index, its input categories, sorted by name. They are declared as
 * categorical axes of the default domain. States without a pdf are not written: HS3 does not require a component
 * for every key.
 */
void RooJSONFactoryWSTool::exportSimultaneous(RooSimultaneous const &simPdf)
{
   RooAbsCategory const &indexCat = simPdf.indexCat();
   std::vector<RooCategory const *> columns = selectionColumns(indexCat);

   JSONNode &elem = appendNamedChild((*_export->root)["distributions"], simPdf.GetName());
   elem["type"] << "combined_dist";
   auto &columnsNode = elem["columns"].set_seq();
   for (RooCategory const *column : columns) {
      columnsNode.append_child() << column->GetName();
      recordCategory(*column);
   }
   auto &components = elem["components"].set_seq();
   // Components with renamed observables replace the original ones (Config::renameCombinedObservables).
   auto renamed = _renamedComponents.find(simPdf.GetName());
   std::set<RooAbsArg const *> replaced;
   std::vector<RooAbsArg const *> replacements;
   // In the order of the columns and of their states, which is the order in which the categories are written to
   // the domain. This does not depend on the order of the inputs of a RooSuperCategory, so the document is the same
   // after a round trip.
   for (std::vector<std::string> const &key : selectionKeys(columns)) {
      const std::string label = selectionLabel(indexCat, columns, key);
      RooAbsPdf const *pdf = simPdf.getPdf(label);
      if (!pdf) {
         continue;
      }
      if (renamed != _renamedComponents.end()) {
         auto found = renamed->second.find(label);
         if (found != renamed->second.end()) {
            replaced.insert(pdf);
            pdf = found->second;
            replacements.push_back(pdf);
         }
      }
      JSONNode &component = components.append_child().set_map();
      component["key"].fill_seq(key);
      component["dist"] << pdf->GetName();
   }
   recordGenericObject(_export->doc, true, simPdf.GetName(), elem);

   std::vector<RooAbsArg const *> servers;
   for (RooAbsArg const *server : simPdf.servers()) {
      if (replaced.count(server) == 0) {
         servers.push_back(server);
      }
   }
   servers.insert(servers.end(), replacements.begin(), replacements.end());
   exportObjects(servers);
}

// Declare a category as a categorical axis of the "default_domain". The states are written in their insertion order:
// this is the order of the bins of a RooDataHist over the category, and the position becomes the index when the
// document is read back.
void RooJSONFactoryWSTool::recordCategory(RooAbsCategory const &cat)
{
   HS3::CategoricalAxis axis;
   axis.name = cat.GetName();
   for (unsigned int i = 0; i < cat.size(); ++i) {
      axis.categories.push_back(cat.getOrdinal(i).first);
   }
   auto found = _export->categories.find(axis.name);
   if (found != _export->categories.end() && found->second.categories != axis.categories) {
      RooJSONFactoryWSTool::error("the category '" + axis.name +
                                  "' is written to the domain twice, with different states");
   }
   _export->categories[axis.name] = std::move(axis);
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
// A categorical axis: the labels in the order of the states of the category, which is the order of the bins of a
// RooDataHist.
void RooJSONFactoryWSTool::exportCategoricalAxis(JSONNode &obsNode, RooCategory const &cat)
{
   testValidName(cat.GetName(), false);
   obsNode["name"] << cat.GetName();
   auto &labels = obsNode["categories"].set_seq();
   for (unsigned int i = 0; i < cat.size(); ++i) {
      labels.append_child() << cat.getOrdinal(i).first;
   }
}

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
   for (RooAbsArg const *var : vars) {
      if (auto const *cat = dynamic_cast<RooCategory const *>(var)) {
         exportCategoricalAxis(observablesNode.append_child().set_map(), *cat);
      } else {
         exportAxis(observablesNode.append_child().set_map(), static_cast<RooRealVar const &>(*var));
      }
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

// Config::renameCombinedObservables: work out, before anything is exported, the new names of the observables of
// every combined data set, and the components of the matching RooSimultaneous with these observables. One
// observable per key is named "obs_<key>", several are named "obs_<key>_<observable>". Observables that already
// have their new name are left alone, so that a second export of an imported document gives the same result.
void RooJSONFactoryWSTool::prepareCombinedObservableRenaming()
{
   _combinedObservableRenames.clear();
   _renamedComponents.clear();
   _renamedObjects.clear();
   if (!config().renameCombinedObservables) {
      return;
   }
   std::vector<RooAbsData *> datasets;
   for (RooAbsData *data : _workspace.allData()) {
      datasets.push_back(data);
   }
   sortByName(datasets);
   for (RooAbsData const *data : datasets) {
      std::vector<RooCategory const *> columns = selectionColumns(*data->get());
      if (columns.empty()) {
         continue;
      }
      RooSimultaneous const *simPdf = findMatchingSimultaneous(_workspace, selectionColumnNames(*data->get()));
      for (std::vector<std::string> const &key : selectionKeys(columns)) {
         const std::string keyName = selectionKeyName(key);
         RooArgSet candidates;
         RooAbsPdf const *component = nullptr;
         std::string label;
         if (simPdf) {
            label = selectionLabel(simPdf->indexCat(), columns, key);
            component = simPdf->getPdf(label);
            if (!component) {
               continue;
            }
            std::unique_ptr<RooArgSet> observables{component->getObservables(*data->get())};
            candidates.add(*observables);
         } else {
            candidates.add(*data->get());
         }
         std::vector<RooAbsArg *> observables;
         for (RooAbsArg *arg : candidates) {
            if (!dynamic_cast<RooAbsCategory *>(arg)) {
               observables.push_back(arg);
            }
         }
         sortByName(observables);
         std::map<std::string, std::string> renames;
         for (RooAbsArg const *obs : observables) {
            std::string const newName =
               "obs_" + keyName + (observables.size() == 1 ? "" : "_" + std::string(obs->GetName()));
            if (newName == obs->GetName()) {
               continue;
            }
            if (_workspace.arg(newName.c_str())) {
               error("can't rename the observable '" + std::string(obs->GetName()) + "' of the data set '" +
                     data->GetName() + "' to '" + newName + "': an object of that name already exists");
            }
            renames[obs->GetName()] = newName;
         }
         if (renames.empty()) {
            continue;
         }
         _combinedObservableRenames[data->GetName()][keyName] = renames;
         if (!simPdf || _renamedComponents[simPdf->GetName()].count(label) > 0) {
            continue;
         }
         // The component with the renamed observables. Nodes that depend on them are cloned with the suffix
         // "_<key>", so that nodes shared by several components get one copy per component.
         RooCustomizer customizer{*component, keyName.c_str()};
         for (auto const &[from, to] : renames) {
            RooAbsArg *original = candidates.find(from.c_str());
            _renamedObjects.emplace_back(static_cast<RooAbsArg *>(original->clone(to.c_str())));
            customizer.replaceArg(*original, *_renamedObjects.back());
         }
         auto *renamedComponent = static_cast<RooAbsPdf *>(customizer.build());
         _renamedObjects.emplace_back(renamedComponent);
         _renamedComponents[simPdf->GetName()][label] = renamedComponent;
      }
   }
}

/**
 * @brief Export a data set with category observables as a "combined" data set.
 *
 * The categories are the columns, sorted by name, and are declared as categorical axes of the default domain. There
 * is one sub-data set per key, i.e. per combination of states, written like any other data set and named after the
 * data set and the labels of the key. A key is written also without entries: RooFit treats a channel without entries
 * as present and empty. If the workspace has a RooSimultaneous with the same columns, each sub-data set only carries
 * the observables of the corresponding component, and keys without entries for which it has no component are left
 * out; otherwise, every sub-data set carries all observables.
 */
RooJSONFactoryWSTool::CombinedData RooJSONFactoryWSTool::exportCombinedData(RooAbsData const &data)
{
   std::vector<RooCategory const *> columns = selectionColumns(*data.get());
   CombinedData info;
   info.name = data.GetName();
   for (RooCategory const *column : columns) {
      info.columns.push_back(column->GetName());
      recordCategory(*column);
   }

   // Split the data by the key: by the index category of a matching RooSimultaneous, or by the columns themselves.
   RooSimultaneous const *simPdf = findMatchingSimultaneous(_workspace, info.columns);
   std::unique_ptr<RooSuperCategory> superCat;
   RooAbsCategory const *indexCat = nullptr;
   std::vector<std::unique_ptr<RooAbsData>> parts;
   if (simPdf) {
      indexCat = &simPdf->indexCat();
      parts = data.split(*simPdf, true);
   } else if (columns.size() == 1) {
      indexCat = columns.front();
      parts = data.split(*indexCat, true);
   } else {
      RooArgSet inputs;
      for (RooCategory const *column : columns) {
         inputs.add(*column);
      }
      const std::string superName = info.name + "_key";
      superCat = std::make_unique<RooSuperCategory>(superName.c_str(), superName.c_str(), inputs);
      indexCat = superCat.get();
      parts = data.split(*indexCat, true);
   }

   // The combined data set is written before its sub-data sets.
   HS3::CombinedData combined;
   combined.columns = info.columns;
   std::vector<RooAbsData *> subDataSets;
   for (std::vector<std::string> const &key : selectionKeys(columns)) {
      std::string const label = selectionLabel(*indexCat, columns, key);
      auto found = std::find_if(parts.begin(), parts.end(), [&](auto &d) { return d->GetName() == label; });
      if (found == parts.end() || ((*found)->numEntries() == 0 && simPdf && !simPdf->getPdf(label))) {
         continue;
      }
      const std::string keyName = selectionKeyName(key);
      const std::string subName = info.name + "_" + keyName;
      (*found)->SetName(subName.c_str());
      // Config::renameCombinedObservables: give the observables of this key their own names.
      auto renamesOfData = _combinedObservableRenames.find(info.name);
      if (renamesOfData != _combinedObservableRenames.end()) {
         auto renames = renamesOfData->second.find(keyName);
         if (renames != renamesOfData->second.end()) {
            for (auto const &[from, to] : renames->second) {
               (*found)->changeObservableName(from.c_str(), to.c_str());
            }
         }
      }
      combined.components.push_back({key, {subName}});
      subDataSets.push_back(found->get());
   }
   _export->doc.data.push_back({info.name, std::move(combined)});
   for (RooAbsData *subData : subDataSets) {
      exportData(*subData);
   }
   return info;
}

// Export a single dataset `data` (binned or unbinned) to the output document.
void RooJSONFactoryWSTool::exportData(RooAbsData const &data)
{
   for (RooAbsArg const *obs : *data.get()) {
      if (dynamic_cast<RooAbsCategory const *>(obs) && !dynamic_cast<RooCategory const *>(obs)) {
         error("the data set '" + std::string(data.GetName()) + "' has the categorical observable '" + obs->GetName() +
               "', which is not a RooCategory, and can't be written as a combined data set");
      }
   }

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

   // Categories are categorical axes, declared by a categorical axis of the default domain.
   auto categoricalAxis = [&](RooCategory const &cat) {
      recordCategory(cat);
      HS3::CategoricalAxis axis;
      axis.name = cat.GetName();
      testValidName(axis.name, false);
      for (unsigned int i = 0; i < cat.size(); ++i) {
         axis.categories.push_back(cat.getOrdinal(i).first);
      }
      return axis;
   };

   auto exportBinned = [&](std::size_t n, double const *contents) {
      HS3::BinnedData binned;
      for (RooAbsArg const *arg : variables) {
         if (auto const *cat = dynamic_cast<RooCategory const *>(arg)) {
            binned.axes.emplace_back(categoricalAxis(*cat));
            continue;
         }
         auto const *var = static_cast<RooRealVar const *>(arg);
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
   // for 1D data for now. A variable without bins can't be binned data: unbinned data sets leave their axes without
   // binning, and would otherwise be written as binned once they have no entries.
   if (data.isWeighted() && variables.size() == 1 && dynamic_cast<RooRealVar const *>(variables[0]) &&
       static_cast<RooRealVar const &>(*variables[0]).getBins() > 0) {
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
      // Binned data has exactly one entry per bin, in the order of the bins. Further entries, e.g. a second one in
      // the last bin, make it unbinned data.
      if (i == x.getBins() && data.numEntries() == x.getBins())
         isBinnedData = true;
      if (isBinnedData) {
         return exportBinned(contents.size(), contents.data());
      }
   }

   // this really is an unbinned dataset
   HS3::UnbinnedData unbinned;
   for (RooAbsArg const *arg : variables) {
      if (auto const *cat = dynamic_cast<RooCategory const *>(arg)) {
         unbinned.axes.emplace_back(categoricalAxis(*cat));
         continue;
      }
      auto const *var = static_cast<RooRealVar const *>(arg);
      recordDomains(_export->domains, *var);
      // HS3 forbids binning information on the axes of unbinned data.
      testValidName(var->GetName(), false);
      unbinned.axes.emplace_back(HS3::ContinuousAxis{var->GetName(), var->getMin(), var->getMax()});
   }
   std::vector<double> weightVals;
   bool hasNonUnityWeights = false;
   for (int i = 0; i < data.numEntries(); ++i) {
      data.get(i);
      // The entries of a categorical axis are the positions of the labels in its "categories".
      for (RooAbsArg const *arg : variables) {
         if (auto const *cat = dynamic_cast<RooCategory const *>(arg)) {
            unbinned.entries.push_back(cat->getCurrentOrdinalNumber());
         } else {
            unbinned.entries.push_back(static_cast<RooRealVar const *>(arg)->getVal());
         }
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
   // The analyses are named after the ModelConfig, so that ModelConfigs sharing a pdf and a data set (like an S+B
   // and B-only pair) get separate analyses.
   auto pdf = mc.GetPdf();
   auto simpdf = dynamic_cast<RooSimultaneous const *>(pdf);
   int founddata = 0;
   if (simpdf) {
      // One analysis per combined data set with the same columns.
      std::vector<std::string> columns;
      for (RooCategory const *column : selectionColumns(simpdf->indexCat())) {
         columns.push_back(column->GetName());
      }
      for (auto const &data : combDataSets) {
         if (data.columns == columns) {
            exportSingleModelConfig(mc, std::string(mc.GetName()) + "_" + data.name, &data.name);
            ++founddata;
         }
      }
   } else {
      RooArgSet observables(*mc.GetObservables());
      for (auto *data : singleDataSets) {
         if (observables.equals(*(data->get()))) {
            const std::string dataName = data->GetName();
            exportSingleModelConfig(mc, std::string(mc.GetName()) + "_" + dataName, &dataName);
            ++founddata;
         }
      }
   }
   if (founddata == 0) {
      exportSingleModelConfig(mc, mc.GetName(), nullptr);
   }
}

void RooJSONFactoryWSTool::exportSingleModelConfig(RooFit::ModelConfig const &mc, std::string const &analysisName,
                                                   std::string const *dataName)
{
   HS3Document &doc = _export->doc;
   auto pdf = mc.GetPdf();

   HS3::Analysis analysis;
   analysis.name = analysisName;

   // HS3 names must be unique across all top-level components, so the likelihood can't have the analysis name.
   HS3::Likelihood likelihood;
   likelihood.name = analysisName + "_likelihood";
   analysis.likelihood.name = likelihood.name;

   if (dataName) {
      likelihood.terms.push_back({{pdf->GetName()}, HS3::DataRef{*dataName}});
   } else {
      likelihood.terms.push_back({{pdf->GetName()}, 0.0});
   }

   auto addAux = [&](std::string const &name) {
      auto &aux = likelihood.auxDistributions;
      if (std::find_if(aux.begin(), aux.end(), [&](auto const &ref) { return ref.name == name; }) == aux.end()) {
         aux.push_back({name});
      }
   };
   // The constraint terms of HistFactory channels that no modifier names are part of the likelihood. The channels
   // can be the components of a combined distribution.
   auto addChannelConstraints = [&](std::string const &channel) {
      auto found = doc.rooFit.channelConstraints.find(channel);
      if (found != doc.rooFit.channelConstraints.end()) {
         for (auto const &ref : found->second) {
            addAux(ref.name);
         }
      }
   };
   for (auto const &term : likelihood.terms) {
      addChannelConstraints(term.distribution.name);
      if (auto const *simPdf = dynamic_cast<RooSimultaneous const *>(_workspace.pdf(term.distribution.name))) {
         for (auto const &item : simPdf->indexCat()) {
            if (RooAbsPdf const *component = simPdf->getPdf(item.first)) {
               addChannelConstraints(component->GetName());
            }
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
      const std::string domainName = analysisName + suffix;
      analysis.domains.push_back({domainName});
      std::map<std::string, HS3::DomainAxis> axes;
      for (auto *var : static_range_cast<const RooRealVar *>(*args)) {
         axes[var->GetName()] = toDomainAxis(var->GetName(), var->getBinning());
      }
      appendDomain(doc.domains, domainName, std::move(axes));
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
   prepareCombinedObservableRenaming();

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
   exportObjects(allpdfs);

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
   exportObjects(allfuncs);

   // The array distributions are written after the objects that use them, see exportAsArrayComponent().
   exportArrayDists();

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
   // Data sets with category observables are combined data sets.
   std::vector<RooAbsData *> singleData;
   std::vector<RooJSONFactoryWSTool::CombinedData> combData;
   for (auto &d : alldata) {
      // Without a matching RooSimultaneous, the categories of a data set are ordinary observables, written as
      // categorical axes, unless its observables are renamed per key (Config::renameCombinedObservables).
      if (selectionColumns(*d->get()).empty() ||
          (!config().renameCombinedObservables &&
           !findMatchingSimultaneous(_workspace, selectionColumnNames(*d->get())))) {
         singleData.push_back(d);
         this->exportData(*d);
      } else {
         combData.push_back(this->exportCombinedData(*d));
      }
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
   // The categories are categorical axes of the "default_domain", which must not list them as continuous.
   auto &defaultAxes = _export->domains[HS3::defaultDomainName];
   for (auto const &item : _export->categories) {
      if (defaultAxes.count(item.first)) {
         RooJSONFactoryWSTool::error("the variable '" + item.first +
                                     "' is categorical, but the domain already lists it as continuous");
      }
   }
   // The domains of the analyses take precedence.
   for (auto &item : _export->domains) {
      if (std::none_of(doc.domains.begin(), doc.domains.end(),
                       [&](HS3::Domain const &domain) { return domain.name == item.first; })) {
         appendDomain(doc.domains, item.first, std::move(item.second),
                      item.first == HS3::defaultDomainName ? std::move(_export->categories)
                                                           : std::map<std::string, HS3::CategoricalAxis>{});
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

   // Each term of a likelihood once, and distinct variates (see sharedVariates()). Otherwise, the document is only
   // written if explicitly allowed, e.g. to rename the variates in the JSON by hand. This is checked on the written
   // document, which has the generic distributions.
   writeLikelihoodTermsOnce(n);
   declareBookkeepingColumns(n, _workspace);
   if (std::vector<std::string> const shared = sharedVariatesOfDocument(n); !shared.empty()) {
      if (!config().allowExportSharedVariates) {
         std::string text;
         for (std::string const &message : shared) {
            text += (text.empty() ? "" : "; ") + message;
         }
         error("cannot export: " + text +
               ". HS3 requires the terms of a likelihood and the components of a combined distribution to have "
               "distinct variates: give the variates their own names (e.g. 'mass_SR' and 'mass_CR' instead of "
               "'mass'; RooJSONFactoryWSTool::config().renameCombinedObservables does this for the observables of "
               "combined data sets), or set RooJSONFactoryWSTool::config().allowExportSharedVariates = true to "
               "write the document anyway, with a warning, and rename them in the JSON.");
      }
      for (std::string const &message : shared) {
         warning("RooFitHS3: " + message +
                 ". HS3 requires distinct variates: the document is not valid HS3, and can't be read back before the "
                 "variates are renamed.");
      }
   }
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

   // The components of combined data sets are moved out of the "datasets" vector when the combined data set is
   // assembled, so they don't get imported on their own. The ROOT-internal form of older files (combineDatasets())
   // is handled after the analyses, because their likelihoods refer to the components.
   std::vector<std::unique_ptr<RooAbsData>> datasets;
   for (auto const &data : doc.data) {
      // Combined data sets are assembled from the others once all of them are loaded.
      if (!std::holds_alternative<HS3::CombinedData>(data.body)) {
         datasets.push_back(loadData(data, _workspace, &_import->ranges));
      }
   }
   assembleCombinedData(doc, _workspace, datasets);

   // Now, read in analyses and likelihoods if there are any

   // Distinct variates (see sharedVariates()). Documents of versions before 0.3, which RooFit wrote with shared
   // variates itself, are read with a warning.
   std::set<std::string> bookkeepingColumns;
   for (auto const &[dataset, columns] : bookkeepingColumnsOfDocument(n)) {
      for (std::string const &column : columns) {
         warning("RooFitHS3: the axis '" + column + "' of the components of the combined data set '" + dataset +
                 "' is declared as a bookkeeping column without statistical meaning (misc.ROOT_internal."
                 "bookkeeping_columns), so it may be shared by the components, although HS3 requires distinct axes.");
         bookkeepingColumns.insert(column);
      }
   }
   for (std::string const &message : sharedVariatesOfDocument(n)) {
      if (!isHS3VersionBefore(_hs3VersionInput, 0, 3)) {
         error("invalid HS3: " + message +
               ". The terms of a likelihood and the components of a combined distribution or data set must have "
               "distinct variates.");
      }
      warning("RooFitHS3: " + message +
              ". HS3 0.3 requires distinct variates; the document is read anyway, since it declares an older "
              "version.");
   }

   for (auto const &analysis : doc.analyses) {
      importAnalysis(doc, analysis, _workspace, datasets, bookkeepingColumns);
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

   warnAboutUnusedArrayComponents(_workspace);

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

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
#include <RooAbsLValue.h>
#include <RooCategory.h>
#include <RooDataSet.h>
#include <RooGaussian.h>
#include <RooDataHist.h>
#include <RooCustomizer.h>
#include <RooProdPdf.h>
#include <RooSimultaneous.h>
#include <RooSuperCategory.h>
#include <RooFit/ModelConfig.h>
#include <RooAbsCollection.h>

#include "JSONIOUtils.h"
#include "HS3Builder.h"
#include "JSONReadRecorder.h"

#include <TROOT.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <iterator>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

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
   /// The constraint terms of the exported HistFactory channels, as written, and the data sets written for them.
   std::map<std::string, std::vector<std::string>> constraintTermsOfPdf;
   std::set<std::string> writtenConstraintData;
   /// The labels that the constraint terms that are written as components of a combined distribution add to its
   /// index category, by category. See exportSingleModelConfig().
   std::map<std::string, std::vector<std::string>> auxiliaryLabels;
   /// The observable of a constraint term, by the name of the term, see setConstraintObservable().
   std::map<std::string, std::string> constraintObservables;
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

// Append a name to a list that has every name once.
void addUnique(std::vector<std::string> &names, std::string const &name)
{
   if (std::find(names.begin(), names.end(), name) == names.end()) {
      names.push_back(name);
   }
}

template <class Coll_t>
std::vector<std::string> namesOf(Coll_t const &objects)
{
   std::vector<std::string> names;
   for (auto const *object : objects) {
      names.push_back(object->GetName());
   }
   return names;
}

// Whether the states of a category have these labels, in any order.
bool hasLabels(RooCategory const &cat, std::vector<std::string> const &labels)
{
   std::set<std::string> states;
   for (auto const &state : cat) {
      states.insert(state.first);
   }
   return states == std::set<std::string>(labels.begin(), labels.end());
}

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
            if (!hasLabels(*cat, axis->categories)) {
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

void RooFit::HS3::report(Diagnostics const &diagnostics, std::string const &object)
{
   std::string errors;
   for (Diagnostic const &diagnostic : diagnostics) {
      // A path starts with the name of a component, or with the index of an entry if the object is an array.
      const bool needsDot = !object.empty() && !diagnostic.path.empty() && diagnostic.path.front() != '[';
      const std::string path = object + (needsDot ? "." : "") + diagnostic.path;
      const std::string text = path.empty() ? diagnostic.message : path + ": " + diagnostic.message;
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

// The step widths of the minimization are the errors of the parameters of the ModelConfigs, and of the parameters of
// the top-level pdfs given the data sets. The observables are no parameters.
void exportParameterStepWidths(RooWorkspace const &workspace, std::vector<RooAbsPdf *> const &pdfs,
                               std::vector<RooAbsData *> const &data, std::vector<HS3::StepWidth> &out)
{
   RooArgSet candidates;
   RooArgSet excluded;
   auto addIfPresent = [](RooArgSet &set, RooArgSet const *args) {
      if (args) {
         set.add(*args, true);
      }
   };
   for (TObject *obj : workspace.allGenericObjects()) {
      if (auto const *mc = dynamic_cast<RooFit::ModelConfig const *>(obj)) {
         addIfPresent(candidates, mc->GetParametersOfInterest());
         addIfPresent(candidates, mc->GetNuisanceParameters());
         addIfPresent(excluded, mc->GetObservables());
         addIfPresent(excluded, mc->GetGlobalObservables());
         addIfPresent(excluded, mc->GetConditionalObservables());
      }
   }
   for (RooAbsPdf const *pdf : pdfs) {
      RooArgSet observables;
      for (RooAbsData const *dataset : data) {
         std::unique_ptr<RooArgSet> pdfObs{pdf->getObservables(*dataset->get())};
         observables.add(*pdfObs, true);
      }
      if (!observables.empty()) {
         RooArgSet params;
         pdf->getParameters(&observables, params);
         candidates.add(params, true);
         excluded.add(observables, true);
      }
   }
   candidates.sort();
   for (RooAbsArg *arg : candidates) {
      auto *var = dynamic_cast<RooRealVar *>(arg);
      if (var && var->hasError() && !excluded.find(*arg)) {
         out.push_back({var->GetName(), var->getError()});
      }
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

// RooFit writes an array of constraints with the data set of its observed values as a component of the combined
// distribution of the model (see exportSingleModelConfig()), and marks it in misc.ROOT_internal.auxiliary_channels.
// Unless these components are to stay components (Config::importConstraintsAsChannels), the document is changed
// here to what RooFit wrote before: the component and its data set become a term of every likelihood that has the
// combined distribution, which is then read as a constraint term with global observables. The label of the component
// is taken out of the index category. Returns the components that are no longer components of their combined
// distribution.
//
// The same is done for a document that lists a distribution twice: as a component of a combined distribution whose
// data set has no entries for it, and as a term of its own with data. It contributes once, as the term, but a
// component without data is not a constraint term for RooFit, which would leave it out of the likelihood. Such a
// document is read like the ones above, with a warning. If the components stay components (`asChannels`), the data
// of the term becomes the data of the component instead, and the term is removed; the components are returned in
// `componentsWithData`.
std::set<std::string> readAuxiliaryChannelsAsConstraints(HS3Document &doc, bool asChannels,
                                                         std::set<std::string> &dataOfConstraints,
                                                         std::set<std::string> &componentsWithData)
{
   // The components that are taken out of their combined distribution, by combined distribution, with their labels.
   std::map<std::string, std::map<std::string, std::string>> dropped;
   std::set<std::string> labels;

   auto findCombinedDist = [&](std::string const &name) -> HS3::CombinedDist * {
      for (HS3::Distribution &dist : doc.distributions) {
         if (dist.name == name) {
            return std::get_if<HS3::CombinedDist>(&dist.body);
         }
      }
      return nullptr;
   };
   auto findData = [&](std::string const &name) -> HS3::Data * {
      for (HS3::Data &data : doc.data) {
         if (data.name == name) {
            return &data;
         }
      }
      return nullptr;
   };
   auto hasEntries = [&](std::string const &name) {
      HS3::Data const *data = findData(name);
      auto const *unbinned = data ? std::get_if<HS3::UnbinnedData>(&data->body) : nullptr;
      return data && (!unbinned || !unbinned->entries.empty());
   };
   // Components without data that are also terms with data. A data set can only be a component of one combined data
   // set, so that the data of a term gets a copy for every further one; room for them is reserved, since the loop
   // keeps pointers into the data sets.
   doc.data.reserve(2 * doc.data.size() + 16);
   std::set<std::string> dataOfComponents;
   for (HS3::Likelihood &likelihood : doc.likelihoods) {
      std::set<std::string> termsToRemove;
      for (HS3::LikelihoodTerm const &term : likelihood.terms) {
         HS3::CombinedDist const *combinedDist = findCombinedDist(term.distribution.name);
         auto const *dataRef = std::get_if<HS3::DataRef>(&term.data);
         HS3::Data *data = dataRef ? findData(dataRef->name) : nullptr;
         auto *combined = data ? std::get_if<HS3::CombinedData>(&data->body) : nullptr;
         if (!combinedDist || !combined) {
            continue;
         }
         for (auto const &component : combinedDist->components) {
            if (component.key.size() != 1) {
               continue;
            }
            std::string const &label = component.key.front();
            std::string const &dist = component.ref.name;
            auto ownTerm =
               std::find_if(likelihood.terms.begin(), likelihood.terms.end(), [&](HS3::LikelihoodTerm const &other) {
                  return other.distribution.name == dist && std::holds_alternative<HS3::DataRef>(other.data);
               });
            if (ownTerm == likelihood.terms.end()) {
               continue;
            }
            auto dataComponent = std::find_if(combined->components.begin(), combined->components.end(),
                                              [&](auto const &c) { return c.key == component.key; });
            if (dataComponent != combined->components.end() && hasEntries(dataComponent->ref.name)) {
               continue; // two contributions, which is not for RooFit to resolve
            }
            RooJSONFactoryWSTool::warning(
               "RooFitHS3: the distribution '" + dist + "' is a component of '" + term.distribution.name +
               "' without data and a term of the likelihood '" + likelihood.name + "' with data. It is read as " +
               (asChannels ? "the component, with the data of the term." : "the term only."));
            if (asChannels) {
               HS3::DataRef termData = std::get<HS3::DataRef>(ownTerm->data);
               if (!dataOfComponents.insert(termData.name).second) {
                  HS3::Data const *original = findData(termData.name);
                  const std::string copyName = data->name + "_" + label;
                  if (original && !findData(copyName) && doc.data.size() < doc.data.capacity()) {
                     HS3::Data copy = *original;
                     copy.name = copyName;
                     doc.data.push_back(std::move(copy));
                     termData.name = copyName;
                  }
               }
               if (dataComponent != combined->components.end()) {
                  dataComponent->ref = termData;
               } else {
                  combined->components.push_back({{label}, termData});
               }
               termsToRemove.insert(dist);
               componentsWithData.insert(dist);
            } else {
               dropped[term.distribution.name][dist] = label;
               labels.insert(label);
            }
         }
      }
      likelihood.terms.erase(std::remove_if(likelihood.terms.begin(), likelihood.terms.end(),
                                            [&](HS3::LikelihoodTerm const &term) {
                                               return termsToRemove.count(term.distribution.name) > 0;
                                            }),
                             likelihood.terms.end());
   }
   if (asChannels) {
      return {};
   }

   for (auto const &item : doc.rooFit.auxiliaryChannels) {
      HS3::CombinedDist const *combinedDist = findCombinedDist(item.first);
      for (auto const &ref : item.second) {
         std::string label = ref.name;
         if (combinedDist) {
            for (auto const &component : combinedDist->components) {
               if (component.ref.name == ref.name && component.key.size() == 1) {
                  label = component.key.front();
               }
            }
         }
         dropped[item.first][ref.name] = label;
         labels.insert(label);
      }
   }
   auto isLabel = [&](std::vector<std::string> const &key) { return key.size() == 1 && labels.count(key.front()); };
   for (HS3::Likelihood &likelihood : doc.likelihoods) {
      std::vector<HS3::LikelihoodTerm> added;
      for (HS3::LikelihoodTerm const &term : likelihood.terms) {
         auto found = dropped.find(term.distribution.name);
         if (found == dropped.end()) {
            continue;
         }
         auto const *dataRef = std::get_if<HS3::DataRef>(&term.data);
         HS3::Data const *data = dataRef ? findData(dataRef->name) : nullptr;
         auto const *combined = data ? std::get_if<HS3::CombinedData>(&data->body) : nullptr;
         for (auto const &item : found->second) {
            std::string const &dist = item.first;
            std::string const &label = item.second;
            if (std::any_of(likelihood.terms.begin(), likelihood.terms.end(),
                            [&](auto const &other) { return other.distribution.name == dist; })) {
               continue; // a term of its own already
            }
            if (auto const *component = combined ? combined->find({label}) : nullptr) {
               added.push_back({{dist}, component->ref});
            } else if (std::none_of(likelihood.auxDistributions.begin(), likelihood.auxDistributions.end(),
                                    [&](auto const &ref) { return ref.name == dist; })) {
               // Without data, the observed values are the values of the constant observables.
               likelihood.auxDistributions.push_back({dist});
            }
         }
      }
      likelihood.terms.insert(likelihood.terms.end(), added.begin(), added.end());
   }
   for (HS3::Data &data : doc.data) {
      if (auto *combined = std::get_if<HS3::CombinedData>(&data.body)) {
         auto &components = combined->components;
         for (auto const &component : components) {
            // Not the data sets without entries: they hold no values, and stay data sets of the workspace.
            if (isLabel(component.key) && hasEntries(component.ref.name)) {
               dataOfConstraints.insert(component.ref.name);
            }
         }
         components.erase(std::remove_if(components.begin(), components.end(),
                                         [&](auto const &component) { return isLabel(component.key); }),
                          components.end());
      }
   }
   for (HS3::Domain &domain : doc.domains) {
      for (auto &axis : domain.axes) {
         if (auto *categorical = std::get_if<HS3::CategoricalAxis>(&axis)) {
            auto &categories = categorical->categories;
            categories.erase(std::remove_if(categories.begin(), categories.end(),
                                            [&](std::string const &label) { return labels.count(label) > 0; }),
                             categories.end());
         }
      }
   }
   std::set<std::string> out;
   for (auto const &droppedFrom : dropped) {
      std::map<std::string, std::string> const &components = droppedFrom.second;
      for (auto const &item : components) {
         out.insert(item.first);
      }
      if (HS3::CombinedDist *combinedDist = findCombinedDist(droppedFrom.first)) {
         auto &all = combinedDist->components;
         all.erase(std::remove_if(all.begin(), all.end(),
                                  [&](auto const &component) { return components.count(component.ref.name) > 0; }),
                   all.end());
      }
   }
   return out;
}

// Declare the bookkeeping columns of the exported document (see HS3::bookkeepingColumns()): the axes that several
// components of a combined data set share, and whose variable in the workspace `ws` is used by nothing else, i.e. has
// no clients and is in no set of a ModelConfig. RooAbsData::split() keeps such a column (e.g. an event number) in
// every sub-data set.
void declareBookkeepingColumns(HS3Document &doc, RooWorkspace const &ws)
{
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
   for (auto const &[dataset, axes] : HS3::sharedAxes(doc)) {
      for (std::string const &axis : axes) {
         if (usedByNothing(axis)) {
            doc.rooFit.bookkeepingColumns[dataset].push_back(axis);
         }
      }
   }
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
      // Skip some RooFit internals. The array distribution tag is restored from the type on import.
      if (attr == "SnapShot_ExtRefClone" || attr == "RooRealConstant_Factory_Object" || attr == arrayDistTag ||
          attr == helperObjectTag)
         continue;
      initialize();
      out->tags.push_back(attr);
   }
}

} // namespace

HS3::BinnedAxis RooFit::HS3::toBinnedAxis(RooRealVar const &var)
{
   RooJSONFactoryWSTool::testValidName(var.GetName(), false);
   return toBinnedAxis(var.GetName(), var.getBinning());
}

HS3::BinnedAxis RooFit::HS3::toBinnedAxis(std::string const &name, RooAbsBinning const &binning)
{
   return {name, binning.lowBound(), binning.highBound(), toBinning(binning)};
}

HS3::ContinuousAxis RooFit::HS3::toContinuousAxis(RooRealVar const &var)
{
   RooJSONFactoryWSTool::testValidName(var.GetName(), false);
   HS3::ContinuousAxis axis;
   axis.name = var.GetName();
   if (!RooNumber::isInfinite(var.getMin())) {
      axis.min = var.getMin();
   }
   if (!RooNumber::isInfinite(var.getMax())) {
      axis.max = var.getMax();
   }
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

// A categorical axis has the labels in the order of the states of the category, which is the order of the bins of a
// RooDataHist.
HS3::CategoricalAxis toCategoricalAxis(RooAbsCategory const &cat)
{
   HS3::CategoricalAxis axis;
   axis.name = cat.GetName();
   for (unsigned int i = 0; i < cat.size(); ++i) {
      axis.categories.push_back(cat.getOrdinal(i).first);
   }
   return axis;
}

// Binned data over variables that are RooRealVars or RooCategories.
HS3::BinnedData toBinnedData(RooArgSet const &vars, std::size_t n, double const *contents)
{
   HS3::BinnedData data;
   for (RooAbsArg const *var : vars) {
      if (auto const *cat = dynamic_cast<RooCategory const *>(var)) {
         RooJSONFactoryWSTool::testValidName(cat->GetName(), false);
         data.axes.emplace_back(toCategoricalAxis(*cat));
      } else {
         data.axes.emplace_back(HS3::toBinnedAxis(static_cast<RooRealVar const &>(*var)));
      }
   }
   data.contents.assign(contents, contents + n);
   return data;
}

std::unique_ptr<RooRealVar> createObservable(HS3::BinnedAxis const &axis)
{
   auto obs = std::make_unique<RooRealVar>(axis.name.c_str(), axis.name.c_str(), axis.min, axis.max);
   applyBinning(*obs, axis.binning);
   return obs;
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
         if (!hasLabels(*wsCat, axis.categories)) {
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
   // An absent bound means that the axis is unbounded on that side.
   const double min = axis.min.value_or(-RooNumber::infinity());
   const double max = axis.max.value_or(RooNumber::infinity());
   if (ranges && !ranges->contains(axis.name)) {
      var.setRange(min, max);
   } else if (var.getMin() != min || var.getMax() != max) {
      std::stringstream ss;
      ss << "RooFitHS3: the range [" << min << ", " << max << "] of the axis '" << axis.name << "' of the dataset '"
         << dataName << "' differs from the range [" << var.getMin() << ", " << var.getMax() << "] of the variable.";
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
         if (!hasLabels(*cat, catAxis.categories)) {
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
std::unique_ptr<RooAbsData> loadData(HS3::Data const &input, RooWorkspace &workspace, HS3::DomainRanges const *ranges,
                                     std::string const *soleVariate = nullptr)
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
   // Point data is the observation of a single variable. With its axis, it is the same as unbinned data with a single
   // entry, and becomes a RooDataSet with one entry.
   auto const &point = std::get<HS3::PointData>(input.body);
   if (point.uncertainty) {
      RooJSONFactoryWSTool::warning("RooFitHS3: the \"uncertainty\" of the dataset '" + input.name +
                                    "' is ignored: HS3 doesn't define how it enters a likelihood.");
   }
   HS3::UnbinnedData unbinned;
   if (point.axes.empty() && soleVariate) {
      // Without an axis, the value is the one of the single variate of the paired distribution (HS3 0.3).
      unbinned.axes.emplace_back(HS3::ContinuousAxis{*soleVariate, std::nullopt, std::nullopt});
   } else if (point.axes.size() == 1) {
      unbinned.axes.emplace_back(point.axes.front());
   } else if (point.axes.empty()) {
      RooJSONFactoryWSTool::error("the point data set '" + input.name +
                                  "' has no axis, and it is not paired with a distribution that has a single variate");
   } else {
      RooJSONFactoryWSTool::error("the point data set '" + input.name + "' must have exactly one axis");
   }
   unbinned.entries.push_back(point.value);
   return createUnbinnedData(input.name, unbinned, workspace, ranges);
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
   const std::vector<std::string> columnNames = namesOf(columns);
   std::vector<std::string> dataColumns = data.columns;
   std::sort(dataColumns.begin(), dataColumns.end());
   if (dataColumns != columnNames) {
      RooJSONFactoryWSTool::error("the likelihood pairs '" + std::string(simPdf.GetName()) +
                                  "' with the combined data set '" + dataName + "', but their columns differ");
   }
   // The keys of the data, each as a map from the column to the label.
   std::set<std::map<std::string, std::string>> dataKeys;
   for (auto const &component : data.components) {
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

// The observables of the analysis `analysis` if no data set of its likelihood gives them: the list that RooFit writes
// in misc.ROOT_internal.ModelConfigs for a ModelConfig without data, or else the variables that HS3 makes
// observables. By the standard, all parameters of the distributions of the likelihood are listed in a domain of the
// analysis or are constant, so a variate of the distributions of the likelihood (see HS3::variatesOf(), the columns
// of a combined distribution are observables as well) that is neither, is an observable. Requiring all three excludes
// the parameters, and the observed values of the constraint terms, which are constant parameters.
void observablesWithoutData(HS3Document const &doc, HS3::Analysis const &analysis, HS3::Likelihood const &likelihood,
                            HS3::ModelConfigInfo const *mcInfo, RooWorkspace &workspace, RooArgSet &observables)
{
   if (mcInfo && mcInfo->observables) {
      for (std::string const &name : *mcInfo->observables) {
         if (RooAbsArg *arg = workspace.arg(name)) {
            observables.add(*arg);
         } else {
            RooJSONFactoryWSTool::warning("RooFitHS3: the observable '" + name + "' of the analysis '" + analysis.name +
                                          "' (misc.ROOT_internal.ModelConfigs) does not exist");
         }
      }
      return;
   }
   std::set<std::string> inDomains;
   for (auto const &domainRef : analysis.domains) {
      for (HS3::Domain const &domain : doc.domains) {
         if (domain.name != domainRef.name) {
            continue;
         }
         for (auto const &axis : domain.axes) {
            inDomains.insert(HS3::nameOf(axis));
         }
      }
   }
   for (auto const &poi : analysis.parametersOfInterest) {
      inDomains.insert(poi.name);
   }
   std::string list;
   for (std::string const &variate : HS3::variatesOf(doc, likelihood)) {
      RooAbsArg *arg = workspace.arg(variate);
      if (!dynamic_cast<RooAbsRealLValue *>(arg) && !dynamic_cast<RooAbsCategoryLValue *>(arg)) {
         continue;
      }
      if (inDomains.count(variate) || arg->isConstant()) {
         continue;
      }
      observables.add(*arg);
      list += (list.empty() ? "'" : ", '") + variate + "'";
   }
   if (!list.empty()) {
      oocoutI(nullptr, IO) << "RooFitHS3: no data set of the likelihood of the analysis '" << analysis.name
                           << "' gives its observables; they are the variates of the likelihood that are neither in a "
                              "domain of the analysis nor constant: "
                           << list << std::endl;
   }
}

// Whether the term `term` of a likelihood is a constraint term of the channel `channel` of a model with the channels
// `channels`: it has a global observable (a variable that no channel depends on) and another variable that the
// channel depends on. An array of constraints is one term.
bool isConstraintOfChannel(RooAbsPdf const &term, RooAbsPdf const &channel, RooArgSet const &channels)
{
   if (&term == &channel || channels.containsInstance(term)) {
      return false;
   }
   RooArgSet leaves;
   term.leafNodeServerList(&leaves);
   RooArgSet parameters;
   bool hasGlobalObservable = false;
   for (RooAbsArg *leaf : leaves) {
      auto *var = dynamic_cast<RooRealVar *>(leaf);
      if (!var) {
         continue;
      }
      // Whether a global observable is constant yet depends on where the document has its value: as a parameter, or
      // in the data set of the constraint, which is read later.
      if (std::none_of(channels.begin(), channels.end(),
                       [&](RooAbsArg const *other) { return other->dependsOn(*var); })) {
         hasGlobalObservable = true;
      } else {
         parameters.add(*var);
      }
   }
   return hasGlobalObservable && !parameters.empty() && channel.dependsOn(parameters);
}

// The other terms of the likelihoods of the document that have the distribution `dist` as a term: their other
// distributions and their auxiliary distributions. These are the candidates for constraint terms of `dist`.
std::vector<std::string> otherLikelihoodTerms(HS3Document const &doc, std::string const &dist)
{
   std::vector<std::string> out;
   auto add = [&](std::string const &name) {
      if (name != dist) {
         addUnique(out, name);
      }
   };
   for (auto const &likelihood : doc.likelihoods) {
      if (std::none_of(likelihood.terms.begin(), likelihood.terms.end(),
                       [&](auto const &term) { return term.distribution.name == dist; })) {
         continue;
      }
      for (auto const &term : likelihood.terms) {
         add(term.distribution.name);
      }
      for (auto const &aux : likelihood.auxDistributions) {
         add(aux.name);
      }
   }
   return out;
}

// In HS3, the constraint terms of a model are terms of the likelihood; in RooFit, as HistFactory builds its models, a
// constraint is a factor of every channel whose parameters it constrains, which makes the pdf of the model the
// sampling distribution of the auxiliary measurements as well. The export takes the constraints out of the channels
// (writeLikelihoodTermsOnce()); this puts them back: a constraint term with a global observable (a constant variable
// that the model itself does not depend on) becomes a factor of every channel that is a product and depends on one
// of its other parameters. An array of constraints goes to its channels as a whole. The constraints that
// found a channel are removed from `constraints`; the others stay external constraints of the ModelConfig, e.g. a
// penalty term without an observation, like the regularization of an unfolding, or a constraint of a model whose
// channels are not products.
void attachConstraintsToChannels(RooAbsPdf &pdf, RooArgSet &constraints)
{
   RooArgSet channels;
   if (auto *simPdf = dynamic_cast<RooSimultaneous *>(&pdf)) {
      for (auto const &state : simPdf->indexCat()) {
         if (RooAbsPdf *channel = simPdf->getPdf(state.first)) {
            channels.add(*channel, /*silent=*/true);
         }
      }
   } else {
      channels.add(pdf);
   }
   RooArgSet attached;
   for (auto *constraint : static_range_cast<RooAbsPdf *>(constraints)) {
      // Decided for all channels before any is changed: a channel that has the constraint depends on its global
      // observable, which then would not look like one anymore.
      bool isFactor = false;
      std::vector<RooProdPdf *> products;
      for (auto *channel : static_range_cast<RooAbsPdf *>(channels)) {
         auto *product = dynamic_cast<RooProdPdf *>(channel);
         if (!product || channel == constraint) {
            continue;
         }
         if (product->pdfList().find(*constraint)) {
            isFactor = true;
         } else if (isConstraintOfChannel(*constraint, *channel, channels)) {
            products.push_back(product);
         }
      }
      for (RooProdPdf *product : products) {
         product->addPdfs(RooArgSet{*constraint});
      }
      if (isFactor || !products.empty()) {
         attached.add(*constraint);
      }
   }
   constraints.remove(attached);
}

// Import an analysis (likelihood + domains) as one or more ModelConfig objects into the workspace.
void importAnalysis(HS3Document const &doc, HS3::Analysis const &analysis, RooWorkspace &workspace,
                    std::vector<std::unique_ptr<RooAbsData>> &datasets, std::set<std::string> const &bookkeepingColumns,
                    std::map<std::pair<std::string, std::size_t>, std::string> const &inlineData,
                    std::set<std::string> const &auxiliaryMeasurements)
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

   // The data set of a term: the one it refers to, or the one made from a number given in-line.
   std::map<HS3::LikelihoodTerm const *, HS3::DataRef> inlineRefs;
   for (std::size_t i = 0; i < likelihood.terms.size(); ++i) {
      auto found = inlineData.find({likelihood.name, i});
      if (found != inlineData.end()) {
         inlineRefs[&likelihood.terms[i]] = HS3::DataRef{found->second};
      }
   }
   auto dataRefOf = [&](HS3::LikelihoodTerm const &term) -> HS3::DataRef const * {
      if (auto const *ref = std::get_if<HS3::DataRef>(&term.data)) {
         return ref;
      }
      auto found = inlineRefs.find(&term);
      return found != inlineRefs.end() ? &found->second : nullptr;
   };

   auto checkPdf = [&](RooAbsArg const *arg, std::string const &name, std::string const &what) {
      requirePdf(arg, "the " + what + " '" + name + "' of analysis '" + analysisName + "'");
   };

   for (auto const &term : likelihood.terms) {
      checkPdf(workspace.arg(term.distribution.name), term.distribution.name, "distribution");
   }

   // The constraint terms of the HistFactory distributions of the likelihood enter through those distributions.
   // Listed in "aux_distributions", they are not added again; paired with a data set in "distributions", they are
   // not channels, and the data set gives the observed values of their observables.
   std::vector<std::unique_ptr<RooArgSet>> trees;
   for (auto const &term : likelihood.terms) {
      RooAbsPdf *listed = workspace.pdf(term.distribution.name);
      trees.push_back(listed ? std::unique_ptr<RooArgSet>{listed->getComponents()} : nullptr);
   }
   auto isConstraintTermOfOthers = [&](RooAbsArg const &term, std::size_t self) {
      auto isPartOf = [&term](RooArgSet const &tree) {
         if (tree.find(term.GetName())) {
            return true;
         }
         if (term.getAttribute(arrayDistTag)) {
            if (auto const *array = dynamic_cast<RooProdPdf const *>(&term)) {
               for (RooAbsArg const *component : array->pdfList()) {
                  if (tree.find(component->GetName())) {
                     return true;
                  }
               }
            }
         }
         return false;
      };
      for (std::size_t i = 0; i < trees.size(); ++i) {
         if (i != self && trees[i] && isPartOf(*trees[i])) {
            return true;
         }
      }
      return false;
   };

   // RooFit writes the constraint terms of a model as terms of the likelihood, with config().exportConstraintData
   // paired with a data set of one entry. If the document names the pdf of the ModelConfig, and it is a term of the
   // likelihood, another term that is not part of it and is paired with such a data set is a constraint term: an
   // external constraint of the ModelConfig.
   std::string const modelPdfName = mcInfo && mcInfo->pdfName ? *mcInfo->pdfName : "";
   std::unique_ptr<RooArgSet> modelTree;
   if (RooAbsPdf const *modelPdf = workspace.pdf(modelPdfName);
       modelPdf && std::any_of(likelihood.terms.begin(), likelihood.terms.end(),
                               [&](auto const &term) { return term.distribution.name == modelPdfName; })) {
      modelTree.reset(modelPdf->getComponents());
   }
   // So is a component of a combined distribution that is read as a constraint term (`auxiliaryMeasurements`, see
   // readAuxiliaryChannelsAsConstraints()), whatever the document says about the ModelConfig.
   auto isConstraintTermOfModel = [&](RooAbsPdf const &listed, HS3::LikelihoodTerm const &term) {
      auto const *dataRef = dataRefOf(term);
      const bool auxiliary = auxiliaryMeasurements.count(listed.GetName()) > 0;
      if (!dataRef || (!auxiliary && (!modelTree || modelTree->find(listed.GetName())))) {
         return false;
      }
      return std::any_of(datasets.begin(), datasets.end(),
                         [&](auto const &d) { return d && d->GetName() == dataRef->name && d->numEntries() == 1; });
   };

   std::vector<std::string> nllDistNames;
   std::vector<HS3::LikelihoodTerm const *> nllTerms;
   RooArgSet constraintObservables;
   RooArgSet constraintsWithData;
   for (std::size_t i = 0; i < likelihood.terms.size(); ++i) {
      auto const &term = likelihood.terms[i];
      RooAbsPdf *listed = workspace.pdf(term.distribution.name);
      if (listed && !isConstraintTermOfOthers(*listed, i) && isConstraintTermOfModel(*listed, term)) {
         constraintsWithData.add(*listed);
      } else if (!listed || !isConstraintTermOfOthers(*listed, i)) {
         nllDistNames.push_back(term.distribution.name);
         nllTerms.push_back(&term);
         continue;
      }
      auto const *dataRef = dataRefOf(term);
      if (!dataRef) {
         continue;
      }
      for (auto &d : datasets) {
         if (!d || d->GetName() != dataRef->name) {
            continue;
         }
         if (d->numEntries() != 1) {
            throw std::runtime_error("the data set '" + dataRef->name + "' paired with the constraint term '" +
                                     term.distribution.name + "' must have exactly one entry");
         }
         for (RooAbsArg const *obs : *d->get(0)) {
            auto *var = workspace.var(obs->GetName());
            auto const *value = dynamic_cast<RooAbsReal const *>(obs);
            if (!var || !value) {
               RooJSONFactoryWSTool::warning("RooFitHS3: cannot set the observed value of '" +
                                             std::string(obs->GetName()) + "' from the data set '" + dataRef->name +
                                             "': it is not a variable of the workspace");
               continue;
            }
            var->setVal(value->getVal());
            var->setConstant(true);
            constraintObservables.add(*var);
         }
         // The values live in the variables now; the data set is not imported.
         d.reset();
      }
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
      if (isConstraintTermOfOthers(*extConstraint, likelihood.terms.size())) {
         continue;
      }
      extConstraints.add(*extConstraint);
   }
   RooArgSet observables;
   for (HS3::LikelihoodTerm const *termPtr : nllTerms) {
      HS3::LikelihoodTerm const &term = *termPtr;
      auto const *dataRef = dataRefOf(term);
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
   if (observables.empty()) {
      observablesWithoutData(doc, analysis, likelihood, mcInfo, workspace, observables);
   }

   // A combined distribution paired with a combined data set must have a sub-data set for every component.
   for (auto const &term : likelihood.terms) {
      auto const *dataRef = dataRefOf(term);
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

   extConstraints.add(constraintsWithData, /*silent=*/true);
   attachConstraintsToChannels(*pdf, extConstraints);
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

   // Parameters that only the external constraints depend on are parameters of the likelihood as well, e.g. the
   // global observable and the nuisance parameter of a constraint that is not part of the pdf. They are not part of
   // the main measurement.
   for (RooAbsArg const *extConstraint : extConstraints) {
      RooArgSet extPars;
      extConstraint->getParameters(&observables, extPars);
      pars.add(extPars, /*silent=*/true);
   }

   RooArgSet domainPars;
   for (auto const &domainRef : analysis.domains) {
      auto thisDomain = std::find_if(doc.domains.begin(), doc.domains.end(),
                                     [&](HS3::Domain const &d) { return d.name == domainRef.name; });
      if (thisDomain == doc.domains.end())
         continue;
      for (auto const &axis : thisDomain->axes) {
         auto *wsvar = workspace.var(HS3::nameOf(axis));
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

   globs.add(constraintObservables, /*silent=*/true);
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

// The categories of the workspace that are the columns of a combined data set or distribution: they are declared by
// categorical axes of the domains, or existed before. The validation has checked the columns and the keys against the
// domains, but not against the categories that existed before.
template <class Ref_t>
std::vector<RooCategory *>
columnCategories(RooWorkspace const &workspace, HS3::Combined<Ref_t> const &combined, std::string const &context)
{
   std::vector<RooCategory *> columns;
   for (std::string const &column : combined.columns) {
      RooCategory *cat = workspace.cat(column);
      if (!cat) {
         RooJSONFactoryWSTool::error("the column '" + column + "' of " + context +
                                     " is not declared by a categorical axis of a domain");
      }
      columns.push_back(cat);
   }
   for (auto const &component : combined.components) {
      for (std::size_t i = 0; i < columns.size(); ++i) {
         if (!columns[i]->hasLabel(component.key[i])) {
            RooJSONFactoryWSTool::error("the label '" + component.key[i] + "' in position " + std::to_string(i) +
                                        " of a key of " + context + " is not one of the categories of '" +
                                        columns[i]->GetName() + "'");
         }
      }
   }
   return columns;
}

// Build the combined data sets of the document: a RooDataSet with one category column per entry of "columns" and the
// variables of all components, filled row by row from the components, which are then no longer imported on their own.
// A key without a sub-data set is a state without entries: RooFit has no notion of an absent component. The validation
// has checked that the components are data sets of the document.
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
      std::vector<RooCategory *> columns = columnCategories(workspace, *info, context);

      struct Component {
         std::vector<std::string> const *key;
         std::unique_ptr<RooAbsData> data;
      };
      std::vector<Component> components;
      for (auto const &c : info->components) {
         auto found =
            std::find_if(datasets.begin(), datasets.end(), [&](auto &d) { return d && d->GetName() == c.ref.name; });
         if (found == datasets.end()) {
            RooJSONFactoryWSTool::error("the component '" + c.ref.name + "' of " + context +
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
// Whether no object of the workspace uses `arg`. Helpers that are not written don't count as users.
bool isTopLevel(RooAbsArg const &arg, RooWorkspace const &ws)
{
   for (RooAbsArg const *client : arg.clients()) {
      if (ws.components().containsInstance(*client) && !isSkippedHelper(*client, ws)) {
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
   // A continuous variable has the bins of its binning, and a category one bin per state.
   std::size_t nBins = 1;
   for (RooAbsArg const *var : vars) {
      nBins *= std::max(0, dynamic_cast<RooAbsLValue const &>(*var).numBins());
   }
   if (contents.size() != nBins) {
      RooJSONFactoryWSTool::error("inconsistent bin numbers: contents=" + std::to_string(contents.size()) +
                                  ", bins=" + std::to_string(nBins));
   }
   auto dh = std::make_unique<RooDataHist>(name, name, vars);
   // RooDataHist only starts to store squared weights when it sees a positive error, and then initializes them from
   // the weights so far. Trigger that before setting any weight, so that errors of zero stay zero.
   if (errors && !errors->empty()) {
      dh->set(0, 0., 1.);
   }
   for (std::size_t ibin = 0; ibin < nBins; ++ibin) {
      dh->set(ibin, contents[ibin], errors ? (*errors)[ibin] : -1);
   }
   return dh;
}

// A combined distribution is a RooSimultaneous. Its index category is the single column, or a RooSuperCategory over the
// columns in the order of the document.
void RooFit::HS3::buildCombinedDist(RooJSONFactoryWSTool &tool, std::string const &name, CombinedDist const &dist,
                                    std::vector<std::string> const &otherLikelihoodTerms)
{
   std::vector<RooCategory *> columns = columnCategories(*tool.workspace(), dist, "the combined_dist '" + name + "'");
   std::map<std::string, RooAbsPdf *> pdfMap;
   for (auto const &component : dist.components) {
      pdfMap[selectionLabel(component.key)] = tool.request<RooAbsPdf>(component.ref.name, name);
   }

   // In RooFit, the constraint terms of a channel are factors of it, which RooFit takes out of the product when it
   // builds a likelihood. A channel that is a product gets them when the likelihood is read; the other channels are
   // wrapped in a product "<channel>_with_constraints" here, since the components of a RooSimultaneous can't be
   // replaced later. The candidates are the other terms of the likelihoods that have this distribution.
   std::vector<RooAbsPdf *> terms;
   for (std::string const &term : otherLikelihoodTerms) {
      if (auto *pdf = tool.tryRequest<RooAbsPdf>(term)) {
         terms.push_back(pdf);
      }
   }
   RooArgSet channels;
   for (auto const &item : pdfMap) {
      channels.add(*item.second, /*silent=*/true);
   }
   for (auto &item : pdfMap) {
      RooAbsPdf &channel = *item.second;
      RooArgList factors;
      for (RooAbsPdf *term : terms) {
         if (!dynamic_cast<RooProdPdf *>(&channel) && isConstraintOfChannel(*term, channel, channels)) {
            factors.add(*term);
         }
      }
      if (!factors.empty()) {
         factors.add(channel);
         item.second = &tool.wsEmplace<RooProdPdf>(std::string(channel.GetName()) + "_with_constraints", factors);
      }
   }

   if (columns.size() == 1) {
      tool.wsImport(RooSimultaneous{name.c_str(), name.c_str(), pdfMap, *columns.front()});
      return;
   }
   RooArgSet inputs;
   for (RooCategory *column : columns) {
      inputs.add(*column);
   }
   std::string const indexName = name + "_index";
   RooSuperCategory indexCat{indexName.c_str(), indexName.c_str(), inputs};
   tool.wsImport(RooSimultaneous{name.c_str(), name.c_str(), pdfMap, indexCat});
}

bool isSkippedHelper(RooAbsArg const &arg, RooWorkspace const &ws)
{
   if (!arg.getAttribute(helperObjectTag)) {
      return false;
   }
   // The HS3 object that the helper stands for is written: an array distribution.
   if (arg.getAttribute(arrayDistTag) && RooJSONFactoryWSTool::config().exportArrayDist) {
      return false;
   }
   for (RooAbsArg const *client : arg.clients()) {
      if (ws.components().containsInstance(*client) && !isSkippedHelper(*client, ws)) {
         return false;
      }
   }
   for (TObject const *obj : ws.allGenericObjects()) {
      auto const *mc = dynamic_cast<RooFit::ModelConfig const *>(obj);
      if (!mc) {
         continue;
      }
      if (mc->GetPdf() == &arg || mc->GetPriorPdf() == &arg ||
          (mc->GetExternalConstraints() && mc->GetExternalConstraints()->containsInstance(arg))) {
         return false;
      }
   }
   return true;
}

RooJSONFactoryWSTool::RooJSONFactoryWSTool(RooWorkspace &ws) : _workspace{ws} {}

bool RooJSONFactoryWSTool::isConstraintReadAsChannel(RooAbsPdf const &pdf) const
{
   if (_componentsWithData.count(pdf.GetName())) {
      return true;
   }
   // A component of an array distribution, which is the product of its components in RooFit.
   for (RooAbsArg const *client : pdf.clients()) {
      if (client->getAttribute(arrayDistTag) && _componentsWithData.count(client->GetName())) {
         return true;
      }
   }
   return false;
}

RooJSONFactoryWSTool::~RooJSONFactoryWSTool() = default;

HS3::Version RooJSONFactoryWSTool::hs3Version() const
{
   return _import ? _import->document.metadata.version : HS3::Version{};
}

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

/// Whether a string is a valid name: a letter or an underscore, followed by letters, digits or underscores. Only
/// characters from the ASCII character set are allowed.
bool RooJSONFactoryWSTool::isValidName(const std::string &str)
{
   return !str.empty() && (std::isalpha(str[0]) || str[0] == '_') &&
          std::all_of(str.begin(), str.end(), [](char c) { return std::isalnum(c) || c == '_'; });
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
   // A literal constant is spelled as an identifier, which a parameter must not be mistaken for when reading.
   if (!isLiteralConstVar(*original) && isLiteralIdentifier(original->GetName())) {
      error("cannot write a function of '" + std::string(original->GetName()) +
            "': names of the form 'const_<number>' are reserved for functions of literal constants");
   }
   std::string newname = transformedBaseName(*original) + suffix;
   // HS3 0.3 (HS3 PR #154) only allows expressions to refer to parameters, variables and functions.
   if (dynamic_cast<RooAbsPdf const *>(original)) {
      error("the parameter '" + std::string(original->GetName()) +
            "' would have to be written as an expression that refers to a pdf, which HS3 doesn't allow");
   }
   // Another object that shares the parameter might have exported the helper already.
   auto &functions = _export->doc.functions;
   if (std::find_if(functions.begin(), functions.end(), [&](auto const &f) { return f.name == newname; }) !=
       functions.end()) {
      return newname;
   }
   RooFit::Detail::JSONNode &trafo_node = appendNamedChild((*_export->root)["functions"], newname);
   trafo_node["type"] << "generic";
   trafo_node["expression"] << TString::Format(formula.c_str(), transformedArgument(*original).c_str()).Data();
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
      if (namesOf(selectionColumns(candidate->indexCat())) == columns) {
         return candidate;
      }
   }
   return nullptr;
}

} // namespace

namespace {

// Record an object that was written by an exporter in the document. Its node moves when the array that it is in
// grows, so the distributions only get their nodes at the end of the export.
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
   if (exportedObjectNames.find(name) != exportedObjectNames.end()) {
      // A component of an array distribution is not written under its own name, so an object that refers to it by
      // that name would have a dangling reference.
      auto const &arrayOfComponent = _export->arrayOfComponent;
      if (auto found = arrayOfComponent.find(name); found != arrayOfComponent.end() && found->second != name) {
         error("'" + name + "' is written as a component of the array distribution '" + found->second +
               "', but another object refers to it by its own name. Set "
               "RooJSONFactoryWSTool::config().exportArrayDist = false to write it on its own.");
      }
      return;
   }

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

   std::cerr << "unable to export '" << name << "' of class '" << cl->GetName()
             << "': no exporter is registered for this class, or none of them could handle the object. Check "
                "https://root.cern/doc/master/group__roofit__dev__docs__hs3.html to see how to write one!\n";
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
      writeArrayDist(*this, elem, arrayName, components, _export->constraintObservables);
      recordGenericObject(_export->doc, true, arrayName, elem);
      for (RooAbsPdf const *pdf : components) {
         exportObjects(pdf->servers());
      }
   }
}

void RooJSONFactoryWSTool::setConstraintObservable(RooAbsPdf const &pdf, std::string const &observable)
{
   _export->constraintObservables[pdf.GetName()] = observable;
}

std::string const *RooJSONFactoryWSTool::constraintObservable(std::string const &pdfName) const
{
   if (!_export) {
      return nullptr;
   }
   auto found = _export->constraintObservables.find(pdfName);
   return found == _export->constraintObservables.end() ? nullptr : &found->second;
}

void RooJSONFactoryWSTool::recordConstraintTerms(std::string const &pdfName, std::vector<std::string> const &terms)
{
   auto &out = _export->constraintTermsOfPdf[pdfName];
   for (auto const &term : terms) {
      addUnique(out, term);
   }
}

// Write the data set `dataName` with the observed values of the observables of the constraint term `term` (an
// array distribution written in this export, or a single constraint pdf) and return true, or return false if the
// observables of the term are not all global observables of the model.
bool RooJSONFactoryWSTool::exportConstraintData(std::string const &term, RooArgSet const *globalObservables,
                                                std::string const &dataName)
{
   std::vector<RooAbsPdf const *> components;
   if (auto found = _export->arrayComponents.find(term); found != _export->arrayComponents.end()) {
      components = found->second;
   } else if (auto const *product = dynamic_cast<RooProdPdf const *>(_workspace.pdf(term))) {
      // A product of constraint terms: its factors.
      for (RooAbsArg const *factor : product->pdfList()) {
         components.push_back(static_cast<RooAbsPdf const *>(factor));
      }
   } else if (RooAbsPdf *pdf = _workspace.pdf(term)) {
      components.push_back(pdf);
   }
   std::vector<std::string> observables;
   for (RooAbsPdf const *pdf : components) {
      auto info = arrayComponentInfo(*pdf, constraintObservable(pdf->GetName()));
      if (!info || !globalObservables || !globalObservables->find(info->observable.c_str())) {
         return false;
      }
      observables.push_back(info->observable);
   }
   if (observables.empty()) {
      return false;
   }
   if (!_export->writtenConstraintData.insert(dataName).second) {
      return true;
   }
   if (_workspace.data(dataName)) {
      error("cannot write the observed values of the constraint term '" + term + "' as the data set '" + dataName +
            "': the workspace has a data set of that name");
   }
   HS3::UnbinnedData data;
   for (std::string const &name : observables) {
      auto const *var = dynamic_cast<RooAbsReal const *>(_workspace.arg(name));
      if (!var) {
         error("the observable '" + name + "' of the constraint term '" + term + "' is not in the workspace");
      }
      // A range without a bound leaves the bound out, which means that the axis is unbounded there.
      HS3::ContinuousAxis axis;
      axis.name = name;
      if (auto const *rrv = dynamic_cast<RooRealVar const *>(var)) {
         axis = HS3::toContinuousAxis(*rrv);
      }
      data.axes.emplace_back(std::move(axis));
      data.entries.push_back(var->getVal());
   }
   // The observed value of a single observable is point data.
   if (data.axes.size() == 1) {
      HS3::PointData point;
      point.value = data.entries.front();
      point.axes.push_back(std::get<HS3::ContinuousAxis>(data.axes.front()));
      _export->doc.data.push_back({dataName, std::move(point)});
      return true;
   }
   _export->doc.data.push_back({dataName, std::move(data)});
   return true;
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

   // The distribution takes its place among the distributions here, and is written at the end of the export.
   appendNamedChild((*_export->root)["distributions"], simPdf.GetName());
   HS3::CombinedDist combined;
   for (RooCategory const *column : columns) {
      combined.columns.push_back(column->GetName());
      recordCategory(*column);
   }
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
      combined.components.push_back({key, {pdf->GetName()}});
   }
   _export->doc.distributions.push_back({simPdf.GetName(), std::move(combined)});

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
   HS3::CategoricalAxis axis = toCategoricalAxis(cat);
   auto found = _export->categories.find(axis.name);
   if (found != _export->categories.end() && found->second.categories != axis.categories) {
      RooJSONFactoryWSTool::error("the category '" + axis.name +
                                  "' is written to the domain twice, with different states");
   }
   _export->categories[axis.name] = std::move(axis);
}

/// Import the function or distribution `p` into the workspace, with the importers that are registered for its type.
/// If `importAllDependants` is false, `p` is a fragment of a document, whose own "functions" and "distributions" are
/// imported first.
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
         // The built-in importer doesn't see the constraint terms that the document adds to the channel.
         if (_import && HS3::isBuiltinHistFactoryImporter(*imp)) {
            auto const &channelConstraints = _import->document.rooFit.channelConstraints;
            auto found = channelConstraints.find(name);
            HS3::importHistFactoryNode(*this, p, found != channelConstraints.end() ? &found->second : nullptr);
            ok = true;
         } else {
            ok = imp->importArg(this, p);
         }
         if (ok)
            break;
      }
   }
   if (!ok) {
      RooJSONFactoryWSTool::error(
         "RooJSONFactoryWSTool() no handling for type '" + functype +
         "' implemented. Check https://root.cern/doc/master/group__roofit__dev__docs__hs3.html "
         "to see how to write an importer for it!");
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
   if (auto const *combined = std::get_if<HS3::CombinedDist>(&distribution.body)) {
      HS3::buildCombinedDist(*this, name, *combined, otherLikelihoodTerms(_import->document, name));
      return;
   }
   auto const &channelConstraints = _import->document.rooFit.channelConstraints;
   auto found = channelConstraints.find(name);
   HS3::buildHistFactoryDist(*this, name, std::get<HS3::HistFactoryDist>(distribution.body),
                             found != channelConstraints.end() ? &found->second : nullptr);
}

/// Import a function or distribution that is given as a JSON string, see the overload for a JSONNode.
void RooJSONFactoryWSTool::importFunction(const std::string &jsonString, bool importAllDependants)
{
   this->importFunction((JSONTree::create(jsonString))->rootnode(), importAllDependants);
}

/// Write the axes and the `n` bin contents of a histogram over the variables `vars`, which are RooRealVars or
/// RooCategories, in the order of the variables.
void RooJSONFactoryWSTool::exportHisto(RooArgSet const &vars, std::size_t n, double const *contents, JSONNode &output)
{
   HS3::writeBinnedData(output, toBinnedData(vars, n, contents));
}

/// Write `n` numbers as an array, each as an integer if it has no fractional part. RooFit doesn't use this function
/// itself, but exporters outside of ROOT do.
void RooJSONFactoryWSTool::exportArray(std::size_t n, double const *contents, JSONNode &output)
{
   HS3::writeContents(output, std::vector<double>(contents, contents + n));
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
      RooSimultaneous const *simPdf = findMatchingSimultaneous(_workspace, namesOf(selectionColumns(*data->get())));
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

   // The ranges of the variables are written to the domains. Categories are declared by a categorical axis of the
   // default domain.
   for (RooAbsArg const *arg : variables) {
      if (auto const *cat = dynamic_cast<RooCategory const *>(arg)) {
         recordCategory(*cat);
      } else {
         recordDomains(_export->domains, static_cast<RooRealVar const &>(*arg));
      }
   }

   auto exportBinned = [&](std::size_t n, double const *contents) {
      output.body = toBinnedData(variables, n, contents);
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

   // A single unweighted entry of a single variable is point data.
   if (variables.size() == 1 && data.numEntries() == 1) {
      data.get(0);
      auto const *var = dynamic_cast<RooRealVar const *>(variables[0]);
      if (var && (!data.isWeighted() || data.weight() == 1.)) {
         recordDomains(_export->domains, *var);
         HS3::PointData point;
         point.value = var->getVal();
         point.axes.push_back(HS3::toContinuousAxis(*var));
         output.body = std::move(point);
         _export->doc.data.push_back(std::move(output));
         return;
      }
   }

   // this really is an unbinned dataset
   HS3::UnbinnedData unbinned;
   for (RooAbsArg const *arg : variables) {
      testValidName(arg->GetName(), false);
      if (auto const *cat = dynamic_cast<RooCategory const *>(arg)) {
         unbinned.axes.emplace_back(toCategoricalAxis(*cat));
      } else {
         // HS3 forbids binning information on the axes of unbinned data.
         auto const &var = static_cast<RooRealVar const &>(*arg);
         unbinned.axes.emplace_back(HS3::toContinuousAxis(var));
      }
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

/// Create the variables for the "axes" of `topNode`, which are owned by the returned set.
RooArgSet RooJSONFactoryWSTool::readAxes(const JSONNode &topNode)
{
   HS3::Diagnostics diagnostics;
   std::vector<HS3::BinnedAxis> axes = HS3::parseBinnedAxes(topNode["axes"], diagnostics);
   HS3::report(diagnostics);
   return HS3::createObservables(axes);
}

/// Create a RooDataHist with the name `name` over the variables `vars` from the "contents" of `n`, and its "errors"
/// if it has any.
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
      const std::vector<std::string> columns = namesOf(selectionColumns(simpdf->indexCat()));
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

namespace {

// A constraint term of a channel of the pdf of a ModelConfig, see constraintsOfChannels().
struct ChannelConstraint {
   std::string channel;
   RooAbsPdf const *constraint = nullptr;
   bool histFactory = false; ///< Whether the channel is written as a histfactory_dist.
   /// Whether the channel consists of constraint terms only; `constraint` is then the channel itself.
   bool wholeChannel = false;
};

// The constraint terms of the pdf `pdf` of a ModelConfig with the observables `observables`, which are written as
// terms of the likelihood: the constraints that RooFit finds (RooAbsPdf::getAllConstraints()) and that are factors
// of a channel, i.e. of `pdf` itself or of a component of the RooSimultaneous `pdf`. In RooFit, constraints are
// factors of the channels, and RooFit counts each once, also if several channels share it; in HS3, they are terms
// of the likelihood.
// - The channels that are written as histfactory_dist (`isHistFactoryChannel`) don't contain their constraints in
//   the document; their modifiers refer to them. Nested products of their constraint terms are expanded into the
//   factors, which the HistFactory exporter writes as terms of their own.
// - The constraints of a channel that is written as product_dist are taken out of the product by
//   writeLikelihoodTermsOnce(). This is only done if no other object of the workspace uses the product, so the
//   constraints of other products are not returned.
// Constraints elsewhere, e.g. in a nested product, stay where they are. The channels are taken as they are written:
// a component of the RooSimultaneous is replaced by the one in `renamed` (by label), if any
// (Config::renameCombinedObservables).
std::vector<ChannelConstraint>
constraintsOfChannels(RooWorkspace const &ws, RooAbsPdf const &pdf, RooArgSet const &observables,
                      std::function<bool(std::string const &)> const &isHistFactoryChannel,
                      std::map<std::string, RooAbsPdf const *> const *renamed)
{
   RooArgSet params;
   pdf.getParameters(&observables, params);
   std::unique_ptr<RooArgSet> constraints{pdf.getAllConstraints(observables, params, /*stripDisconnected=*/false)};
   std::vector<RooAbsPdf const *> channels;
   if (auto const *simPdf = dynamic_cast<RooSimultaneous const *>(&pdf)) {
      for (auto const &item : simPdf->indexCat()) {
         RooAbsPdf const *component = simPdf->getPdf(item.first);
         if (renamed) {
            if (auto found = renamed->find(item.first); found != renamed->end()) {
               component = found->second;
            }
         }
         if (component) {
            channels.push_back(component);
         }
      }
   } else {
      channels.push_back(&pdf);
   }
   std::vector<ChannelConstraint> out;
   for (RooAbsPdf const *channel : channels) {
      auto const *prod = dynamic_cast<RooProdPdf const *>(channel);
      if (!prod) {
         continue;
      }
      const bool histFactory = isHistFactoryChannel(prod->GetName());
      // A channel that consists of constraint terms only is an auxiliary measurement that is part of the model. It
      // is written as that channel, once, and not as constraint terms of the likelihood.
      if (!histFactory && !prod->pdfList().empty() &&
          std::all_of(prod->pdfList().begin(), prod->pdfList().end(),
                      [&](RooAbsArg const *factor) { return constraints->find(*factor) != nullptr; })) {
         out.push_back({prod->GetName(), prod, false, true});
         continue;
      }
      if (!histFactory) {
         // Being a component of a RooSimultaneous, i.e. of a combined distribution, is not a use.
         bool usedElsewhere = false;
         for (RooAbsArg const *client : prod->clients()) {
            if (ws.arg(client->GetName()) == client && !dynamic_cast<RooSimultaneous const *>(client)) {
               usedElsewhere = true;
            }
         }
         if (usedElsewhere) {
            continue;
         }
      }
      std::function<void(RooAbsPdf const &)> add = [&](RooAbsPdf const &term) {
         auto const *nested = dynamic_cast<RooProdPdf const *>(&term);
         if (!histFactory || !nested) {
            out.push_back({prod->GetName(), &term, histFactory});
            return;
         }
         for (RooAbsArg const *factor : nested->pdfList()) {
            add(static_cast<RooAbsPdf const &>(*factor));
         }
      };
      for (RooAbsArg const *factor : prod->pdfList()) {
         if (constraints->find(*factor)) {
            add(static_cast<RooAbsPdf const &>(*factor));
         }
      }
   }
   return out;
}

} // namespace

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
      // Without data, the likelihood is a pure model, which is written with "data": null.
      likelihood.terms.push_back({{pdf->GetName()}, std::monostate{}});
      likelihood.pureModel = true;
   }

   // The additional terms of the likelihood: the external constraints of the ModelConfig, and the constraint terms
   // of the channels of the model, which can be the components of a combined distribution (constraintsOfChannels()).
   // They are listed in "aux_distributions" with the observed values of their observables as constant parameters, or
   // (config().exportConstraintData) in "distributions", each paired with a data set that holds the observed values.
   // The channels as they are written: the components of a RooSimultaneous can be renamed clones
   // (Config::renameCombinedObservables).
   std::vector<std::string> channels;
   for (auto const &term : likelihood.terms) {
      channels.push_back(term.distribution.name);
      if (auto const *simPdf = dynamic_cast<RooSimultaneous const *>(_workspace.pdf(term.distribution.name))) {
         auto renamed = _renamedComponents.find(simPdf->GetName());
         for (auto const &item : simPdf->indexCat()) {
            RooAbsPdf const *component = simPdf->getPdf(item.first);
            if (renamed != _renamedComponents.end()) {
               if (auto found = renamed->second.find(item.first); found != renamed->second.end()) {
                  component = found->second;
               }
            }
            if (component) {
               channels.push_back(component->GetName());
            }
         }
      }
   }
   // The channels of the model that consist of constraint terms only, see constraintsOfChannels().
   std::vector<std::string> auxiliaryChannelsOfModel;
   std::vector<std::string> constraintTerms;
   if (mc.GetExternalConstraints()) {
      for (const auto &constr : *mc.GetExternalConstraints()) {
         addUnique(constraintTerms, constr->GetName());
      }
   }
   // The constraint terms that the HistFactory exporter wrote for the channel `channel`, in the order of the document:
   // the constraints that the modifiers name, as written (e.g. array distributions), and those that no modifier
   // names.
   auto writtenByHistFactory = [&](std::string const &channel) {
      std::vector<std::string> out;
      if (auto found = _export->constraintTermsOfPdf.find(channel); found != _export->constraintTermsOfPdf.end()) {
         for (std::string const &term : found->second) {
            addUnique(out, term);
         }
      }
      if (auto found = doc.rooFit.channelConstraints.find(channel); found != doc.rooFit.channelConstraints.end()) {
         for (auto const &ref : found->second) {
            addUnique(out, ref.name);
         }
      }
      return out;
   };
   RooArgSet const *observables = mc.GetObservables();
   if (!observables || observables->empty()) {
      // Without observables (e.g. a ModelConfig without data), RooFit can't tell constraints from other pdfs: only the
      // constraint terms of the HistFactory channels are listed.
      for (std::string const &channel : channels) {
         for (std::string const &term : writtenByHistFactory(channel)) {
            addUnique(constraintTerms, term);
         }
      }
   } else {
      auto isHistFactoryChannel = [&](std::string const &name) {
         return _export->constraintTermsOfPdf.count(name) ||
                std::any_of(doc.distributions.begin(), doc.distributions.end(), [&](HS3::Distribution const &dist) {
                   return dist.name == name && std::holds_alternative<HS3::HistFactoryDist>(dist.body);
                });
      };
      // The constraint terms of the channels as RooFit finds them, under the names under which they are written:
      // the array distribution that holds a constraint of a HistFactory channel, or the name of the constraint.
      std::map<std::string, std::vector<std::string>> termsOfChannel;
      auto renamedOfPdf = _renamedComponents.find(pdf->GetName());
      for (ChannelConstraint const &item :
           constraintsOfChannels(_workspace, *pdf, *observables, isHistFactoryChannel,
                                 renamedOfPdf != _renamedComponents.end() ? &renamedOfPdf->second : nullptr)) {
         if (item.wholeChannel) {
            auxiliaryChannelsOfModel.push_back(item.channel);
            continue;
         }
         // A component of an array distribution is never referred to by itself: the array is the term.
         std::string term = item.constraint->GetName();
         if (auto found = _export->arrayOfComponent.find(term); found != _export->arrayOfComponent.end()) {
            term = found->second;
         }
         addUnique(termsOfChannel[item.channel], term);
      }
      for (std::string const &channel : channels) {
         std::vector<std::string> const &found = termsOfChannel[channel];
         if (!isHistFactoryChannel(channel)) {
            for (std::string const &term : found) {
               addUnique(constraintTerms, term);
            }
            continue;
         }
         // The constraint terms that the HistFactory exporter wrote must be the ones that RooFit finds, or a
         // constraint would be missing from the likelihood, or would be in it without being written. They are listed
         // in the order of the document, which doesn't depend on how the workspace was built.
         std::vector<std::string> const written = writtenByHistFactory(channel);
         std::set<std::string> const writtenSet{written.begin(), written.end()};
         std::set<std::string> const foundSet{found.begin(), found.end()};
         if (writtenSet != foundSet) {
            auto join = [](std::set<std::string> const &names) {
               std::string out;
               for (std::string const &name : names) {
                  out += (out.empty() ? "'" : ", '") + name + "'";
               }
               return out.empty() ? std::string{"none"} : out;
            };
            error("cannot export the constraint terms of the HistFactory channel '" + channel +
                  "': the channel writes " + join(writtenSet) + ", but RooFit finds " + join(foundSet));
         }
         for (std::string const &term : written) {
            addUnique(constraintTerms, term);
         }
      }
   }
   // An array of constraints with its data set is a measurement like the others: if the model is a combined
   // distribution, it is written as a component of it, paired with its data set as a component of the combined data
   // set. The combined distribution is then the sampling distribution of all measurements, auxiliary or not. Its
   // label is the name of the array. Constraints that are written one by one stay terms of the likelihood: there
   // can be hundreds, which is not what an index category is for. So do the constraints of models whose index has
   // several columns.
   auto const *simPdf = dynamic_cast<RooSimultaneous const *>(pdf);
   // Looked up when needed: writing the data set of a constraint term adds to the data sets of the document.
   auto findCombinedData = [&]() -> HS3::CombinedData * {
      for (HS3::Data &data : doc.data) {
         if (dataName && data.name == *dataName) {
            return std::get_if<HS3::CombinedData>(&data.body);
         }
      }
      return nullptr;
   };
   RooCategory const *indexColumn = nullptr;
   if (simPdf && findCombinedData()) {
      std::vector<RooCategory const *> columns = selectionColumns(simPdf->indexCat());
      if (columns.size() == 1) {
         indexColumn = columns.front();
      }
   }
   auto addAuxiliaryChannel = [&](std::string const &term) {
      if (indexColumn->hasLabel(term)) {
         error("cannot write the constraint term '" + term + "' as a component of the combined distribution '" +
               simPdf->GetName() + "': its index category '" + indexColumn->GetName() + "' has a state of that name");
      }
      addUnique(_export->auxiliaryLabels[indexColumn->GetName()], term);
      auto written = std::find_if(doc.distributions.begin(), doc.distributions.end(),
                                  [&](HS3::Distribution const &dist) { return dist.name == simPdf->GetName(); });
      auto *combinedDist =
         written != doc.distributions.end() ? std::get_if<HS3::CombinedDist>(&written->body) : nullptr;
      if (!combinedDist) {
         error("cannot write the constraint terms of '" + std::string(simPdf->GetName()) +
               "' as its components: it was not written as a combined distribution");
      }
      if (!combinedDist->find({term})) {
         combinedDist->components.push_back({{term}, {term}});
         doc.rooFit.auxiliaryChannels[simPdf->GetName()].push_back({term});
      }
      HS3::CombinedData *combinedData = findCombinedData();
      if (!combinedData->find({term})) {
         combinedData->components.push_back({{term}, HS3::DataRef{*dataName + "_" + term}});
      }
   };
   // A channel of the model that consists of constraint terms is an auxiliary measurement, whose observed values are
   // the values of its global observables if the data set has no entries for it. It is written like the arrays
   // above: with a data set of the observed values, and marked for RooFit, which can read it as constraint terms.
   for (std::string const &channel : auxiliaryChannelsOfModel) {
      if (!indexColumn || likelihood.pureModel) {
         continue;
      }
      std::string label;
      for (auto const &state : simPdf->indexCat()) {
         RooAbsPdf const *component = simPdf->getPdf(state.first);
         if (component && channel == component->GetName()) {
            label = state.first;
         }
      }
      if (label.empty()) {
         continue;
      }
      HS3::CombinedData *combinedData = findCombinedData();
      auto component = std::find_if(combinedData->components.begin(), combinedData->components.end(),
                                    [&](auto const &c) { return c.key == std::vector<std::string>{label}; });
      const std::string observed =
         component != combinedData->components.end() ? component->ref.name : *dataName + "_" + label;
      // A data set without entries for the channel is replaced by the observed values.
      auto written = std::find_if(doc.data.begin(), doc.data.end(), [&](auto const &d) { return d.name == observed; });
      bool hasEntries = false;
      if (written != doc.data.end()) {
         auto const *unbinned = std::get_if<HS3::UnbinnedData>(&written->body);
         hasEntries = !unbinned || !unbinned->entries.empty();
      }
      if (!hasEntries) {
         std::optional<HS3::Data> empty;
         if (written != doc.data.end()) {
            empty = std::move(*written);
            doc.data.erase(written);
         }
         if (exportConstraintData(channel, mc.GetGlobalObservables(), observed)) {
            combinedData = findCombinedData();
            if (std::none_of(combinedData->components.begin(), combinedData->components.end(),
                             [&](auto const &c) { return c.key == std::vector<std::string>{label}; })) {
               combinedData->components.push_back({{label}, HS3::DataRef{observed}});
            }
         } else {
            if (empty) {
               doc.data.push_back(std::move(*empty));
            }
            warning("RooFitHS3: the channel '" + channel + "' of '" + simPdf->GetName() +
                    "' consists of constraint terms, but has no data, and its observables are not all global "
                    "observables of the ModelConfig: it is written without an observation.");
            continue;
         }
      }
      auto &marked = doc.rooFit.auxiliaryChannels[simPdf->GetName()];
      if (std::none_of(marked.begin(), marked.end(), [&](auto const &ref) { return ref.name == channel; })) {
         marked.push_back({channel});
      }
   }
   for (std::string const &term : constraintTerms) {
      // A pure model has no data at all, so the observed values of its constraint terms are not written as data sets
      // either: they are the constant parameters of auxiliary distributions.
      // A component of a combined data set belongs to that data set alone, and is named after it like the others.
      const bool asChannel = indexColumn && _export->arrayComponents.count(term);
      const std::string constraintData = asChannel ? *dataName + "_" + term : term + "_data";
      if (!likelihood.pureModel && config().exportConstraintData &&
          exportConstraintData(term, mc.GetGlobalObservables(), constraintData)) {
         if (asChannel) {
            addAuxiliaryChannel(term);
         } else {
            likelihood.terms.push_back({{term}, HS3::DataRef{constraintData}});
         }
      } else {
         likelihood.auxDistributions.push_back({term});
      }
   }
   doc.likelihoods.push_back(std::move(likelihood));

   if (RooArgSet const *pois = mc.GetParametersOfInterest()) {
      std::vector<std::string> names = namesOf(*pois);
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
   // Without a data set, the document doesn't say what the observables are.
   if (!dataName && mc.GetObservables() && !mc.GetObservables()->empty()) {
      std::vector<std::string> names = namesOf(*mc.GetObservables());
      std::sort(names.begin(), names.end());
      info.observables = std::move(names);
   }
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
         auto *pdf = dynamic_cast<RooAbsPdf *>(arg);
         if (pdf && !isSkippedHelper(*pdf, _workspace)) {
            allpdfs.push_back(pdf);
         }
      }
   }
   // A Gaussian constraint is written with its global observable as "x", also if that is its mean in RooFit (see
   // setConstraintObservable()). Which one it is must be known before any pdf is written: the pdfs are written in the
   // order of their names, and a constraint can come before the model that uses it.
   for (TObject *obj : _workspace.allGenericObjects()) {
      auto const *mc = dynamic_cast<RooFit::ModelConfig const *>(obj);
      RooArgSet const *globs = mc ? mc->GetGlobalObservables() : nullptr;
      if (!globs) {
         continue;
      }
      for (RooAbsArg const *arg : _workspace.components()) {
         auto const *gauss = dynamic_cast<RooGaussian const *>(arg);
         if (gauss && globs->find(gauss->getMean()) && !globs->find(gauss->getX()) &&
             !constraintObservable(gauss->GetName())) {
            setConstraintObservable(*gauss, gauss->getMean().GetName());
         }
      }
   }
   sortByName(allpdfs);
   exportObjects(allpdfs);

   // export all toplevel functions
   std::vector<RooAbsReal *> allfuncs;
   for (auto &arg : _workspace.allFunctions()) {
      if (isTopLevel(*arg, _workspace)) {
         auto *func = dynamic_cast<RooAbsReal *>(arg);
         if (func && !isSkippedHelper(*func, _workspace)) {
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
           !findMatchingSimultaneous(_workspace, namesOf(selectionColumns(*d->get()))))) {
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
   // The constraint terms that are components of a combined distribution have a label of their own.
   for (auto const &item : _export->auxiliaryLabels) {
      auto &categories = _export->categories[item.first].categories;
      for (std::string const &label : item.second) {
         if (std::find(categories.begin(), categories.end(), label) == categories.end()) {
            categories.push_back(label);
         }
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

   // The exporters have written the generic distributions to the JSON tree. Now that the tree has all of them, the
   // document refers to their nodes, like a document that was read. The combined distributions are written now, since
   // constraint terms can become their components until here, see exportSingleModelConfig().
   if (n.has_child("distributions")) {
      std::map<std::string, JSONNode *> nodes;
      for (JSONNode &node : n["distributions"].children()) {
         nodes.emplace(name(node), &node);
      }
      for (HS3::Distribution &dist : doc.distributions) {
         auto found = nodes.find(dist.name);
         if (found == nodes.end()) {
            continue;
         }
         if (auto *generic = std::get_if<HS3::GenericObject>(&dist.body)) {
            generic->node = found->second;
         } else if (auto const *combined = std::get_if<HS3::CombinedDist>(&dist.body)) {
            HS3::writeCombinedDist(*found->second, *combined);
         }
      }
   }

   // A RooFit model often has a constraint both as a factor of the product of a channel and as a term of the
   // likelihood (e.g. a constraint of a HistFactory modifier, an external constraint of the ModelConfig, or any
   // constraint of a channel, see constraintsOfChannels()); RooFit counts it once. In HS3 it would enter twice.
   for (auto const &[product, factors] : HS3::writeLikelihoodTermsOnce(doc, n)) {
      for (std::string const &factor : factors) {
         oocoutI(nullptr, IO) << "RooFitHS3: '" << factor << "' is a factor of '" << product
                              << "' and a term of its likelihood; it is written once, as a term of the likelihood"
                              << std::endl;
      }
   }
   declareBookkeepingColumns(doc, _workspace);

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

   // Without distinct variates (see HS3::sharedVariates()), the document is only written if explicitly allowed, e.g.
   // to rename the variates in the JSON by hand.
   if (std::vector<std::string> const shared = HS3::sharedVariates(doc); !shared.empty()) {
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

/// Import an HS3 document that is given as a JSON string into the workspace. Returns true; errors are exceptions.
bool RooJSONFactoryWSTool::importJSONfromString(const std::string &s)
{
   std::stringstream ss(s);
   return importJSON(ss);
}

/// Export the workspace as an HS3 document, and return it as a JSON string.
std::string RooJSONFactoryWSTool::exportJSONtoString()
{
   std::stringstream ss;
   exportJSON(ss);
   return ss.str();
}

/// Export the workspace as an HS3 document, and write it as JSON to the stream. Returns true; errors are exceptions.
bool RooJSONFactoryWSTool::exportJSON(std::ostream &os)
{
   std::unique_ptr<JSONTree> tree = JSONTree::create();
   JSONNode &n = tree->rootnode().set_map();
   this->exportAllObjects(n);
   n.writeJSON(os);
   return true;
}

/// Export the workspace as an HS3 document, and write it to a JSON file. Returns true; errors are exceptions.
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
   // The data sets of the components that are read as constraint terms: their values end up in the global
   // observables, and they are not data sets of the workspace.
   std::set<std::string> dataOfConstraints;
   std::set<std::string> componentsWithData;
   // As the document has them, before the components that are read as constraint terms are taken out of it.
   _componentsWithData = HS3::componentsWithData(parsed);
   std::vector<std::string> constraintNotes;
   std::vector<std::string> const unlistedConstraints = HS3::unlistedConstraints(parsed, &constraintNotes);
   std::set<std::string> const auxiliaryMeasurements = readAuxiliaryChannelsAsConstraints(
      parsed, config().importConstraintsAsChannels, dataOfConstraints, componentsWithData);
   _componentsWithData.insert(componentsWithData.begin(), componentsWithData.end());
   for (std::string const &component : auxiliaryMeasurements) {
      _componentsWithData.erase(component);
   }
   std::optional<RooFit::ValidatedHS3Document> validated;
   if (!HS3::hasErrors(diagnostics)) {
      HS3::ValidateOptions validateOptions;
      validateOptions.strict = !(parsed.metadata.version < HS3::Version{0, 3});
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
         tool._componentsWithData.clear();
      }
   } importScope{*this};

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
   // A number given in-line in the "data" of a likelihood, or a point data set without axes, is the value of the single
   // variate of the paired distribution (HS3 0.3). If that can't be determined, the document is rejected.
   auto soleVariateOf = [&](std::string const &distribution, std::string const &what) {
      std::set<std::string> const v = HS3::variatesOf(doc, distribution);
      if (v.size() != 1) {
         error("the variate of the distribution '" + distribution + "', which " + what +
               " is paired with, can't be determined, since it doesn't have a single variate");
      }
      return *v.begin();
   };
   std::map<std::string, std::string> pointDataVariates;
   std::map<std::pair<std::string, std::size_t>, std::string> inlineData;
   std::vector<HS3::Data> inlineDataSets;
   // Older versions of RooFit wrote "data": [0], with the number 0 as the only entry, for a ModelConfig without data.
   // In this exact form, it is no observation in documents before 0.3 (HS3 PR #154). RooFit now writes "data": null.
   auto isNoDataPlaceholder = [&](HS3::Likelihood const &likelihood, double value) {
      return value == 0. && likelihood.terms.size() == 1 && doc.metadata.version < HS3::Version{0, 3};
   };
   for (auto const &likelihood : doc.likelihoods) {
      for (std::size_t i = 0; i < likelihood.terms.size(); ++i) {
         auto const &term = likelihood.terms[i];
         if (auto const *value = std::get_if<double>(&term.data)) {
            if (isNoDataPlaceholder(likelihood, *value)) {
               continue;
            }
            std::string const what = "the number in the \"data\" of the likelihood '" + likelihood.name + "'";
            HS3::PointData point;
            point.value = *value;
            point.axes.push_back(
               HS3::ContinuousAxis{soleVariateOf(term.distribution.name, what), std::nullopt, std::nullopt});
            auto isTaken = [&](std::string const &candidate) {
               return std::any_of(doc.data.begin(), doc.data.end(),
                                  [&](HS3::Data const &d) { return d.name == candidate; }) ||
                      std::any_of(inlineDataSets.begin(), inlineDataSets.end(),
                                  [&](HS3::Data const &d) { return d.name == candidate; });
            };
            std::string name = term.distribution.name + "_data";
            for (std::size_t k = 1; isTaken(name); ++k) {
               name = term.distribution.name + "_data_" + std::to_string(k);
            }
            inlineData[{likelihood.name, i}] = name;
            inlineDataSets.push_back({name, std::move(point)});
            continue;
         }
         auto const *refPtr = std::get_if<HS3::DataRef>(&term.data);
         if (!refPtr) {
            continue; // no data in a pure model
         }
         auto const &ref = *refPtr;
         for (auto const &data : doc.data) {
            auto const *point = std::get_if<HS3::PointData>(&data.body);
            if (data.name != ref.name || !point) {
               continue;
            }
            if (!point->axes.empty()) {
               // The axis of point data is named like the variate of the paired distribution.
               std::set<std::string> const v = HS3::variatesOf(doc, term.distribution.name);
               if (v.size() == 1 && point->axes.front().name != *v.begin()) {
                  error("the axis '" + point->axes.front().name + "' of the point data set '" + data.name +
                        "' is not the variate '" + *v.begin() + "' of the distribution '" + term.distribution.name +
                        "' it is paired with");
               }
               continue;
            }
            std::string const variate = soleVariateOf(term.distribution.name, "the point data set '" + data.name + "'");
            auto inserted = pointDataVariates.emplace(data.name, variate);
            if (!inserted.second && inserted.first->second != variate) {
               error("the point data set '" + data.name +
                     "' without axes is paired with distributions of different variates, '" + inserted.first->second +
                     "' and '" + variate + "'");
            }
         }
      }
   }

   std::vector<std::unique_ptr<RooAbsData>> datasets;
   datasets.reserve(doc.data.size() + inlineDataSets.size());
   for (auto const &data : doc.data) {
      // Combined data sets are assembled from the others once all of them are loaded.
      if (!std::holds_alternative<HS3::CombinedData>(data.body)) {
         auto variate = pointDataVariates.find(data.name);
         datasets.push_back(loadData(data, _workspace, &_import->ranges,
                                     variate != pointDataVariates.end() ? &variate->second : nullptr));
      }
   }
   for (auto const &data : inlineDataSets) {
      datasets.push_back(loadData(data, _workspace, &_import->ranges));
   }
   assembleCombinedData(doc, _workspace, datasets);

   // Now, read in analyses and likelihoods if there are any

   // Distinct variates (see HS3::sharedVariates()). Documents of versions before 0.3, which RooFit wrote with shared
   // variates itself, are read with a warning.
   std::set<std::string> bookkeepingColumns;
   for (auto const &[dataset, columns] : HS3::bookkeepingColumns(doc)) {
      for (std::string const &column : columns) {
         warning("RooFitHS3: the axis '" + column + "' of the components of the combined data set '" + dataset +
                 "' is declared as a bookkeeping column without statistical meaning (misc.ROOT_internal."
                 "bookkeeping_columns), so it may be shared by the components, although HS3 requires distinct axes.");
         bookkeepingColumns.insert(column);
      }
   }
   for (std::string const &message : HS3::sharedVariates(doc)) {
      if (!(doc.metadata.version < HS3::Version{0, 3})) {
         error("invalid HS3: " + message +
               ". The terms of a likelihood and the components of a combined distribution or data set must have "
               "distinct variates.");
      }
      warning("RooFitHS3: " + message +
              ". HS3 0.3 requires distinct variates; the document is read anyway, since it declares an older "
              "version.");
   }
   if (!(doc.metadata.version < HS3::Version{0, 3})) {
      for (std::string const &message : unlistedConstraints) {
         error("invalid HS3: " + message +
               ". The \"constraints\" of a modifier add no term to the likelihood, which has to list them.");
      }
      for (std::string const &note : constraintNotes) {
         warning("RooFitHS3: " + note + ".");
      }
   }

   for (auto const &analysis : doc.analyses) {
      importAnalysis(doc, analysis, _workspace, datasets, bookkeepingColumns, inlineData, auxiliaryMeasurements);
   }

   combineDatasets(doc, datasets);
   // Several analyses can share a ModelConfig, of which only the first one consumes the data of its constraint
   // terms.
   for (auto &d : datasets) {
      if (d && dataOfConstraints.count(d->GetName())) {
         d.reset();
      }
   }

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

/// Import an HS3 document that is read as JSON from the stream into the workspace. Returns true; errors are exceptions.
bool RooJSONFactoryWSTool::importJSON(std::istream &is)
{
   std::unique_ptr<JSONTree> tree = JSONTree::create(is);
   this->importAllNodes(tree->rootnode());
   return true;
}

/// Import the HS3 document of a JSON file into the workspace. Returns true; errors are exceptions.
bool RooJSONFactoryWSTool::importJSON(std::string const &filename)
{
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
      HS3::Data data = HS3::parseData(n, diagnostics);
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

/// Write a warning to the RooFit message service, and return its stream.
std::ostream &RooJSONFactoryWSTool::warning(std::string const &str)
{
   return RooMsgService::instance().log(nullptr, RooFit::MsgLevel::WARNING, RooFit::ObjectHandling) << str << std::endl;
}

/// Write an error to the RooFit message service, and throw it as a std::runtime_error.
void RooJSONFactoryWSTool::error(const char *s)
{
   RooMsgService::instance().log(nullptr, RooFit::MsgLevel::ERROR, RooFit::IO) << s << std::endl;
   throw std::runtime_error(s);
}

/// Replace some characters that are common in the names of RooFit objects, but that the HS3 standard doesn't allow in
/// names, if config().allowSanitizeNames is set.
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
   RooWorkspace tmpWS = cleanWS(ws, false);

   // Valid names stay as they are.
   auto sanitize = [](auto const &objects) {
      for (auto *obj : objects) {
         obj->SetName(sanitizeName(obj->GetName()).c_str());
      }
   };
   sanitize(tmpWS.allVars());
   sanitize(tmpWS.allFunctions());
   sanitize(tmpWS.allPdfs());
   sanitize(tmpWS.allResolutionModels());
   for (auto const &datasets : {tmpWS.allData(), tmpWS.allEmbeddedData()}) {
      sanitize(datasets);
      for (RooAbsData *data : datasets) {
         sanitize(*data->get());
      }
   }
   for (auto *snshObj : tmpWS.getSnapshots()) {
      // Snapshots are stored as TObject*, but really they are RooArgSet*
      auto *snsh = dynamic_cast<RooArgSet *>(snshObj);
      if (!snsh) {
         std::cerr << "Warning: found snapshot that is not a RooArgSet, skipping\n";
         continue;
      }
      snsh->setName(sanitizeName(snsh->GetName()).c_str());
      sanitize(*snsh);
   }
   // Generic objects, like the ModelConfigs with their sets
   for (auto *obj : tmpWS.allGenericObjects()) {
      if (auto *named = dynamic_cast<TNamed *>(obj)) {
         named->SetName(sanitizeName(named->GetName()).c_str());
      } else if (!isValidName(obj->GetName())) {
         std::cerr << "Warning: object " << obj->GetName() << " is not TNamed, cannot rename.\n";
      }
      if (auto *mc = dynamic_cast<RooFit::ModelConfig *>(obj)) {
         sanitize(*mc->GetObservables());
         sanitize(*mc->GetParametersOfInterest());
         sanitize(*mc->GetNuisanceParameters());
         sanitize(*mc->GetGlobalObservables());
      }
   }
   RooWorkspace newWS = cleanWS(tmpWS, false);
   newWS.SetName((std::string{ws.GetName()} + "_sanitized").c_str());
   return newWS;
}

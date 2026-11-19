/*
 * Project: RooFit
 * Authors:
 *   Carsten D. Burgard, DESY/ATLAS, Nov 2026
 *   Jonas Rembser, CERN, Nov 2026
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

// HS3 requires the terms of a likelihood, and the components of a combined distribution or data set, to have distinct
// variates: a variate in two of them is one quantity modelled twice (e.g. a constraint that is a term of the likelihood
// and a factor of a channel), or two quantities with one name (e.g. the observable "mass" of two channels, which need
// their own names, "mass_SR" and "mass_CR"). All cases are checked with the same function, sharedVariates(), on the
// document, when it is exported and when it is imported. The generic distributions are read as far as the standard
// says what they consist of, through the nodes that the document refers to.
//
// This file also has what else follows from which distribution is paired with which data: how often a constraint
// contributes to a likelihood, and how a document is changed so that RooFit can read its auxiliary measurements and
// its point data.

#include "HS3Document.h"

#include <RooFit/Detail/JSONInterface.h>

#include <algorithm>
#include <functional>
#include <iterator>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>

using RooFit::Detail::JSONNode;

namespace RooFit {
namespace HS3 {
namespace {

// Append a name to a list that has every name once.
void addUnique(std::vector<std::string> &names, std::string const &name)
{
   if (std::find(names.begin(), names.end(), name) == names.end()) {
      names.push_back(name);
   }
}

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

// The names in the list `key` of a generic distribution. The entries of the list are the names, or components that
// give the name as `field`.
std::vector<std::string> namesIn(JSONNode const &node, const char *key, const char *field = nullptr)
{
   std::vector<std::string> out;
   if (JSONNode const *list = node.find(key)) {
      for (JSONNode const &entry : list->children()) {
         JSONNode const *name = field ? entry.find(field) : &entry;
         if (name && isName(*name)) {
            out.push_back(name->val());
         }
      }
   }
   return out;
}

// The constraint terms that the modifiers of a HistFactory distribution name in "constraints". The "constraint_name"
// of older documents is not counted: it predates the requirement of distinct variates.
std::vector<std::string> namedConstraints(HistFactoryDist const &dist)
{
   std::vector<std::string> out;
   auto add = overloaded{[&](ConstraintPdf const &constraint) {
                            if (!constraint.legacyName) {
                               out.push_back(constraint.pdf.name);
                            }
                         },
                         [&](GammaConstraintPdfs const &constraints) {
                            for (auto const &pdf : constraints.pdfs) {
                               if (pdf) {
                                  out.push_back(pdf->name);
                               }
                            }
                         },
                         [](auto const &) {}};
   for (Sample const &sample : dist.samples) {
      for (Modifier const &modifier : sample.modifiers) {
         std::visit(overloaded{[](CustomModifier const &) {}, [&](auto const &m) { std::visit(add, m.constraint); }},
                    modifier);
      }
   }
   return out;
}

// The same for a HistFactory distribution that the document doesn't interpret, because a user registered their own
// importer or exporter for it.
std::vector<std::string> namedConstraints(JSONNode const &dist)
{
   std::vector<std::string> out;
   if (JSONNode const *samples = dist.find("samples")) {
      for (JSONNode const &sample : samples->children()) {
         if (JSONNode const *modifiers = sample.find("modifiers")) {
            for (JSONNode const &modifier : modifiers->children()) {
               for (std::string const &constraint : namesIn(modifier, "constraints")) {
                  out.push_back(constraint);
               }
            }
         }
      }
   }
   return out;
}

// The names of the components of a combined distribution.
std::vector<std::string> componentsOf(CombinedDist const &dist)
{
   std::vector<std::string> out;
   for (auto const &component : dist.components) {
      out.push_back(component.ref.name);
   }
   return out;
}

// The node of a distribution that the document doesn't interpret, if it has one and, optionally, the type `type`.
JSONNode const *genericNode(Distribution const &dist, const char *type = nullptr)
{
   auto const *generic = std::get_if<GenericObject>(&dist.body);
   return generic && (!type || generic->type == type) ? generic->node : nullptr;
}

// The distributions of a document by name, and what they say about the variates. The document must outlive this
// object.
class Distributions {
public:
   explicit Distributions(HS3Document const &doc) : _doc{doc}
   {
      for (Distribution const &dist : doc.distributions) {
         _byName.emplace(dist.name, &dist);
      }
   }

   // The distribution `name` if the document represents it as a `Body_t`, or nullptr.
   template <class Body_t>
   Body_t const *find(std::string const &name) const
   {
      auto found = _byName.find(name);
      return found != _byName.end() ? std::get_if<Body_t>(&found->second->body) : nullptr;
   }

   // The node of the generic distribution `name` if it has the type `type`, or nullptr.
   JSONNode const *nodeOf(std::string const &name, const char *type) const
   {
      auto found = _byName.find(name);
      return found != _byName.end() ? genericNode(*found->second, type) : nullptr;
   }

   // Add the variates of the distribution `name` to `out`: the "x" of a fundamental distribution (a name or a list)
   // and the axes of a distribution with axes (e.g. histfactory_dist), through products, mixtures and combined
   // distributions, and the "x" of the components of an array distribution. The constraint terms named by a
   // histfactory_dist are terms of their own, see likelihoodTerms(). Distributions in `visited` are skipped.
   void addVariates(std::string const &name, std::set<std::string> &out, std::set<std::string> &visited) const
   {
      if (!visited.insert(name).second) {
         return;
      }
      if (auto const *histFactory = find<HistFactoryDist>(name)) {
         for (BinnedAxis const &axis : histFactory->axes) {
            out.insert(axis.name);
         }
      }
      auto addVariatesOf = [&](std::vector<std::string> const &parts) {
         for (std::string const &part : parts) {
            addVariates(part, out, visited);
         }
      };
      if (auto const *combined = find<CombinedDist>(name)) {
         addVariatesOf(componentsOf(*combined));
      }
      auto const *generic = find<GenericObject>(name);
      if (!generic || !generic->node) {
         return;
      }
      JSONNode const &node = *generic->node;
      if (generic->type == "product_dist") {
         addVariatesOf(namesIn(node, "factors"));
      } else if (generic->type == "mixture_dist") {
         addVariatesOf(namesIn(node, "summands"));
      } else if (generic->type == "array_dist") {
         if (JSONNode const *components = node.find("components")) {
            for (JSONNode const &component : components->children()) {
               if (JSONNode const *x = component.find("x")) {
                  addNames(*x, out);
               }
            }
         }
      } else {
         if (JSONNode const *x = node.find("x")) {
            addNames(*x, out);
         }
         for (std::string const &axis : namesIn(node, "axes", "name")) {
            out.insert(axis);
         }
      }
   }

   // The terms of a likelihood, each once: its distributions (the components of a combined distribution, or of a
   // RooSimultaneous written to ROOT_internal by older versions, are terms of their own), its "aux_distributions",
   // and the constraint terms named by the modifiers of its histfactory_dist terms.
   std::vector<std::string> likelihoodTerms(Likelihood const &likelihood) const
   {
      std::vector<std::string> terms;
      auto add = [&](std::string const &name) {
         auto legacy = _doc.rooFit.combinedDistributions.find(name);
         if (auto const *combined = find<CombinedDist>(name)) {
            for (std::string const &component : componentsOf(*combined)) {
               addUnique(terms, component);
            }
         } else if (!_byName.count(name) && legacy != _doc.rooFit.combinedDistributions.end()) {
            for (DistributionRef const &component : legacy->second.distributions) {
               addUnique(terms, component.name);
            }
         } else {
            addUnique(terms, name);
         }
      };
      for (LikelihoodTerm const &term : likelihood.terms) {
         add(term.distribution.name);
      }
      for (DistributionRef const &aux : likelihood.auxDistributions) {
         add(aux.name);
      }
      // The list grows by the constraint terms while it is read.
      for (std::size_t i = 0; i < terms.size(); ++i) {
         std::vector<std::string> constraints;
         if (auto const *histFactory = find<HistFactoryDist>(terms[i])) {
            constraints = namedConstraints(*histFactory);
         } else if (JSONNode const *generic = nodeOf(terms[i], "histfactory_dist")) {
            constraints = namedConstraints(*generic);
         }
         for (std::string const &constraint : constraints) {
            addUnique(terms, constraint);
         }
      }
      return terms;
   }

private:
   HS3Document const &_doc;
   std::map<std::string, Distribution const *> _byName;
};

// The parts (terms or components) of something that must have distinct variates, with their variates.
using PartVariates = std::vector<std::pair<std::string, std::set<std::string>>>;

// Append a message for each variate that several of `parts` share. `what` names the parts and `context` the whole,
// e.g. "terms" of "the likelihood 'lh'".
void addShared(PartVariates const &parts, std::string const &what, std::string const &context,
               std::vector<std::string> &messages)
{
   std::map<std::string, std::vector<std::string>> partsOfVariate;
   std::vector<std::string> order;
   for (auto const &[part, variates] : parts) {
      for (std::string const &variate : variates) {
         auto &list = partsOfVariate[variate];
         if (list.empty()) {
            order.push_back(variate);
         }
         addUnique(list, part);
      }
   }
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
}

// The axes of the components of a combined data set, by component.
PartVariates componentAxes(HS3Document const &doc, CombinedData const &combined)
{
   PartVariates parts;
   for (auto const &component : combined.components) {
      std::set<std::string> axes;
      if (Data const *found = findByName(doc.data, component.ref.name)) {
         std::visit(overloaded{[](CombinedData const &) {},
                               [&](PointData const &data) {
                                  for (ContinuousAxis const &axis : data.axes) {
                                     axes.insert(axis.name);
                                  }
                               },
                               [&](auto const &data) {
                                  for (auto const &axis : data.axes) {
                                     axes.insert(nameOf(axis));
                                  }
                               }},
                    found->body);
      }
      parts.emplace_back(component.ref.name, axes);
   }
   return parts;
}

// The number of entries of the data set `name` of the document (see Data::numEntries()). Without such a data set, the
// document is not valid, which is reported elsewhere: one is assumed.
int numDataEntries(HS3Document const &doc, std::string const &name)
{
   Data const *data = findByName(doc.data, name);
   return data ? static_cast<int>(data->numEntries()) : 1;
}

} // namespace

std::set<std::string> variatesOf(HS3Document const &doc, Likelihood const &likelihood)
{
   Distributions distributions{doc};
   std::set<std::string> variates;
   std::set<std::string> visited;
   for (LikelihoodTerm const &term : likelihood.terms) {
      distributions.addVariates(term.distribution.name, variates, visited);
      if (auto const *combined = distributions.find<CombinedDist>(term.distribution.name)) {
         variates.insert(combined->columns.begin(), combined->columns.end());
      }
   }
   return variates;
}

std::set<std::string> variatesOf(HS3Document const &doc, std::string const &distribution)
{
   std::set<std::string> variates;
   std::set<std::string> visited;
   Distributions{doc}.addVariates(distribution, variates, visited);
   return variates;
}

std::map<std::string, std::set<std::string>> sharedAxes(HS3Document const &doc)
{
   std::map<std::string, std::set<std::string>> out;
   for (Data const &data : doc.data) {
      if (auto const *combined = std::get_if<CombinedData>(&data.body)) {
         std::map<std::string, int> counts;
         for (auto const &[component, axes] : componentAxes(doc, *combined)) {
            for (std::string const &axis : axes) {
               if (++counts[axis] == 2) {
                  out[data.name].insert(axis);
               }
            }
         }
      }
   }
   return out;
}

std::map<std::string, std::set<std::string>> bookkeepingColumns(HS3Document const &doc)
{
   std::map<std::string, std::set<std::string>> out;
   if (doc.rooFit.bookkeepingColumns.empty()) {
      return out;
   }
   Distributions distributions{doc};
   std::set<std::string> variates;
   std::set<std::string> visited;
   for (Distribution const &dist : doc.distributions) {
      distributions.addVariates(dist.name, variates, visited);
   }
   for (auto const &[dataset, columns] : doc.rooFit.bookkeepingColumns) {
      for (std::string const &column : columns) {
         if (!variates.count(column)) {
            out[dataset].insert(column);
         }
      }
   }
   return out;
}

std::vector<std::string> unlistedConstraints(HS3Document const &doc, std::vector<std::string> *warnings)
{
   Distributions distributions{doc};
   std::vector<std::string> messages;
   for (Likelihood const &likelihood : doc.likelihoods) {
      std::map<std::string, int> contributions;
      // The combined distributions in which a distribution is a component without data.
      std::map<std::string, std::set<std::string>> withoutData;
      std::set<std::string> named;
      // The products that are being expanded, since a document that was not imported yet can have circular ones.
      std::set<std::string> expanding;
      // `combined` is the combined distribution whose component without data the distribution is reached through.
      std::function<void(std::string const &, int, std::string const &)> contribute =
         [&](std::string const &name, int entries, std::string const &combined) {
            contributions[name] += entries;
            if (entries == 0) {
               withoutData[name].insert(combined);
            }
            if (auto const *histFactory = distributions.find<HistFactoryDist>(name)) {
               for (std::string const &constraint : namedConstraints(*histFactory)) {
                  named.insert(constraint);
               }
            } else if (JSONNode const *generic = distributions.nodeOf(name, "histfactory_dist")) {
               for (std::string const &constraint : namedConstraints(*generic)) {
                  named.insert(constraint);
               }
            } else if (JSONNode const *product = distributions.nodeOf(name, "product_dist");
                       product && expanding.insert(name).second) {
               for (std::string const &factor : namesIn(*product, "factors")) {
                  contribute(factor, entries, combined);
               }
               expanding.erase(name);
            }
         };
      // The components of a combined distribution, each with the entries of the component of the paired combined data
      // set that has the same key. Without paired data (a pure model), every component contributes once.
      auto contributeCombined = [&](std::string const &distName, CombinedDist const &dist, Data const *data) {
         auto const *combinedData = data ? std::get_if<CombinedData>(&data->body) : nullptr;
         for (auto const &component : dist.components) {
            int entries = 1;
            if (combinedData) {
               auto const *dataComponent = combinedData->find(component.key);
               entries = dataComponent ? numDataEntries(doc, dataComponent->ref.name) : 0;
               if (!dataComponent && warnings) {
                  warnings->push_back("the component '" + component.ref.name + "' of the combined distribution '" +
                                      distName + "' has no component in the data set '" + data->name +
                                      "' that the likelihood '" + likelihood.name +
                                      "' pairs it with, so it does not contribute to the likelihood");
               }
            }
            contribute(component.ref.name, entries, distName);
         }
      };
      for (LikelihoodTerm const &term : likelihood.terms) {
         // The paired data: a data set, a value given in-line, or nothing (a pure model).
         auto const *dataRef = std::get_if<DataRef>(&term.data);
         Data const *data = dataRef ? findByName(doc.data, dataRef->name) : nullptr;
         std::string const &name = term.distribution.name;
         if (auto const *combined = distributions.find<CombinedDist>(name)) {
            ++contributions[name];
            contributeCombined(name, *combined, data);
         } else {
            contribute(name, data ? numDataEntries(doc, data->name) : 1, "");
         }
      }
      for (DistributionRef const &aux : likelihood.auxDistributions) {
         contribute(aux.name, 1, "");
      }
      for (std::string const &constraint : named) {
         const int n = contributions[constraint];
         auto const &regions = withoutData[constraint];
         std::string inRegions;
         for (std::string const &region : regions) {
            inRegions += (inRegions.empty() ? "'" : ", '") + region + "'";
         }
         if (n != 1) {
            messages.push_back("the constraint '" + constraint +
                               "' named by a modifier of a HistFactory channel contributes " + std::to_string(n) +
                               " times to the likelihood '" + likelihood.name +
                               "', but it must contribute exactly once" +
                               (n == 0 && !regions.empty() ? " (it is a component of the combined distribution " +
                                                                inRegions + " without data, which is no observation)"
                                                           : ""));
         } else if (!regions.empty() && warnings) {
            warnings->push_back("the constraint '" + constraint + "' is a term of the likelihood '" + likelihood.name +
                                "' and also a component of the combined distribution " + inRegions +
                                " without data. It contributes once, as the term");
         }
      }
   }
   return messages;
}

std::set<std::string> componentsWithData(HS3Document const &doc)
{
   Distributions distributions{doc};
   std::set<std::string> out;
   for (Likelihood const &likelihood : doc.likelihoods) {
      for (LikelihoodTerm const &term : likelihood.terms) {
         auto const *dataRef = std::get_if<DataRef>(&term.data);
         auto const *combinedData = dataRef ? find<CombinedData>(doc.data, dataRef->name) : nullptr;
         auto const *dist = distributions.find<CombinedDist>(term.distribution.name);
         if (!dist || !combinedData) {
            continue;
         }
         for (auto const &component : dist->components) {
            auto const *dataComponent = combinedData->find(component.key);
            if (dataComponent && numDataEntries(doc, dataComponent->ref.name) > 0) {
               out.insert(component.ref.name);
            }
         }
      }
   }
   return out;
}

std::vector<std::string> sharedVariates(HS3Document const &doc)
{
   Distributions distributions{doc};
   std::vector<std::string> messages;
   auto withVariates = [&](std::vector<std::string> const &names) {
      PartVariates parts;
      for (std::string const &name : names) {
         std::set<std::string> visited;
         distributions.addVariates(name, parts.emplace_back(name, std::set<std::string>{}).second, visited);
      }
      return parts;
   };
   std::set<std::string> inLikelihoods;
   for (Likelihood const &likelihood : doc.likelihoods) {
      for (LikelihoodTerm const &term : likelihood.terms) {
         inLikelihoods.insert(term.distribution.name);
      }
      for (DistributionRef const &aux : likelihood.auxDistributions) {
         inLikelihoods.insert(aux.name);
      }
      addShared(withVariates(distributions.likelihoodTerms(likelihood)), "terms",
                "the likelihood '" + likelihood.name + "'", messages);
   }
   for (Distribution const &dist : doc.distributions) {
      auto const *combined = std::get_if<CombinedDist>(&dist.body);
      if (combined && !inLikelihoods.count(dist.name)) {
         addShared(withVariates(componentsOf(*combined)), "components", "the combined_dist '" + dist.name + "'",
                   messages);
      }
   }
   // The axes of the components of every combined data set, also of those that a likelihood lists.
   std::map<std::string, std::set<std::string>> const bookkeeping = bookkeepingColumns(doc);
   for (Data const &data : doc.data) {
      auto const *combined = std::get_if<CombinedData>(&data.body);
      if (!combined) {
         continue;
      }
      PartVariates parts = componentAxes(doc, *combined);
      if (auto found = bookkeeping.find(data.name); found != bookkeeping.end()) {
         for (auto &part : parts) {
            for (std::string const &column : found->second) {
               part.second.erase(column);
            }
         }
      }
      addShared(parts, "components", "the combined data set '" + data.name + "'", messages);
   }
   return messages;
}

AuxiliaryConstraints readAuxiliaryChannelsAsConstraints(ValidatedHS3Document &validated, bool asChannels)
{
   HS3Document &doc = validated._doc;
   AuxiliaryConstraints out;
   // The components that are taken out of their combined distribution, by combined distribution, with their labels.
   std::map<std::string, std::map<std::string, std::string>> dropped;
   std::set<std::string> labels;

   // Components without data that are also terms with data. A data set can only be a component of one combined data
   // set, so that the data of a term gets a copy for every further one. The copies are added to the document after
   // the loop, which keeps pointers into the data sets.
   std::vector<Data> copies;
   auto hasEntries = [&](std::string const &name) {
      Data const *data = findByName(doc.data, name);
      data = data ? data : findByName(copies, name);
      return data && data->numEntries() > 0;
   };
   std::set<std::string> dataOfComponents;
   for (Likelihood &likelihood : doc.likelihoods) {
      std::set<std::string> termsToRemove;
      for (LikelihoodTerm const &term : likelihood.terms) {
         auto const *combinedDist = find<CombinedDist>(doc.distributions, term.distribution.name);
         auto const *dataRef = std::get_if<DataRef>(&term.data);
         auto *combined = dataRef ? find<CombinedData>(doc.data, dataRef->name) : nullptr;
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
               std::find_if(likelihood.terms.begin(), likelihood.terms.end(), [&](LikelihoodTerm const &other) {
                  return other.distribution.name == dist && std::holds_alternative<DataRef>(other.data);
               });
            if (ownTerm == likelihood.terms.end()) {
               continue;
            }
            auto dataComponent = std::find_if(combined->components.begin(), combined->components.end(),
                                              [&](auto const &c) { return c.key == component.key; });
            if (dataComponent != combined->components.end() && hasEntries(dataComponent->ref.name)) {
               continue; // two contributions, which is not for RooFit to resolve
            }
            out.warnings.push_back("the distribution '" + dist + "' is a component of '" + term.distribution.name +
                                   "' without data and a term of the likelihood '" + likelihood.name +
                                   "' with data. It is read as " +
                                   (asChannels ? "the component, with the data of the term." : "the term only."));
            if (asChannels) {
               DataRef termData = std::get<DataRef>(ownTerm->data);
               if (!dataOfComponents.insert(termData.name).second) {
                  Data const *original = findByName(doc.data, termData.name);
                  const std::string copyName = dataRef->name + "_" + label;
                  if (original && !findByName(doc.data, copyName) && !findByName(copies, copyName)) {
                     copies.push_back(*original);
                     copies.back().name = copyName;
                     termData.name = copyName;
                  }
               }
               if (dataComponent != combined->components.end()) {
                  dataComponent->ref = termData;
               } else {
                  combined->components.push_back({{label}, termData});
               }
               termsToRemove.insert(dist);
            } else {
               dropped[term.distribution.name][dist] = label;
               labels.insert(label);
            }
         }
      }
      likelihood.terms.erase(
         std::remove_if(likelihood.terms.begin(), likelihood.terms.end(),
                        [&](LikelihoodTerm const &term) { return termsToRemove.count(term.distribution.name) > 0; }),
         likelihood.terms.end());
   }
   doc.data.insert(doc.data.end(), copies.begin(), copies.end());
   if (asChannels) {
      return out;
   }

   for (auto const &item : doc.rooFit.auxiliaryChannels) {
      auto const *combinedDist = find<CombinedDist>(doc.distributions, item.first);
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
   for (Likelihood &likelihood : doc.likelihoods) {
      std::vector<LikelihoodTerm> added;
      for (LikelihoodTerm const &term : likelihood.terms) {
         auto found = dropped.find(term.distribution.name);
         if (found == dropped.end()) {
            continue;
         }
         auto const *dataRef = std::get_if<DataRef>(&term.data);
         auto const *combined = dataRef ? find<CombinedData>(doc.data, dataRef->name) : nullptr;
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
   for (Data &data : doc.data) {
      if (auto *combined = std::get_if<CombinedData>(&data.body)) {
         auto &components = combined->components;
         for (auto const &component : components) {
            // Not the data sets without entries: they hold no values, and stay data sets of the workspace.
            if (isLabel(component.key) && hasEntries(component.ref.name)) {
               out.data.insert(component.ref.name);
            }
         }
         components.erase(std::remove_if(components.begin(), components.end(),
                                         [&](auto const &component) { return isLabel(component.key); }),
                          components.end());
      }
   }
   for (Domain &domain : doc.domains) {
      for (auto &axis : domain.axes) {
         if (auto *categorical = std::get_if<CategoricalAxis>(&axis)) {
            auto &categories = categorical->categories;
            categories.erase(std::remove_if(categories.begin(), categories.end(),
                                            [&](std::string const &label) { return labels.count(label) > 0; }),
                             categories.end());
         }
      }
   }
   for (auto const &droppedFrom : dropped) {
      std::map<std::string, std::string> const &components = droppedFrom.second;
      for (auto const &item : components) {
         out.terms.insert(item.first);
      }
      if (auto *combinedDist = find<CombinedDist>(doc.distributions, droppedFrom.first)) {
         auto &all = combinedDist->components;
         all.erase(std::remove_if(all.begin(), all.end(),
                                  [&](auto const &component) { return components.count(component.ref.name) > 0; }),
                   all.end());
      }
   }
   return out;
}

std::vector<std::string> resolvePointData(ValidatedHS3Document &validated)
{
   HS3Document &doc = validated._doc;
   std::vector<std::string> errors;
   auto soleVariateOf = [&](std::string const &distribution, std::string const &what) {
      std::set<std::string> const variates = variatesOf(doc, distribution);
      if (variates.size() == 1) {
         return *variates.begin();
      }
      errors.push_back("the variate of the distribution '" + distribution + "', which " + what +
                       " is paired with, can't be determined, since it doesn't have a single variate");
      return std::string{};
   };
   std::vector<Data> inlineData;
   std::map<std::string, std::string> variateOfPointData; // of the point data sets without axes
   for (Likelihood &likelihood : doc.likelihoods) {
      for (LikelihoodTerm &term : likelihood.terms) {
         std::string const &distribution = term.distribution.name;
         if (auto const *value = std::get_if<double>(&term.data)) {
            // Older versions of RooFit wrote "data": [0], with the number 0 as the only entry, for a ModelConfig
            // without data. In this exact form, it is no observation in documents before 0.3 (HS3 PR #154). RooFit
            // now writes "data": null.
            if (*value == 0. && likelihood.terms.size() == 1 && doc.metadata.version < Version{0, 3}) {
               term.data = std::monostate{};
               continue;
            }
            std::string name = distribution + "_data";
            for (std::size_t k = 1; findByName(doc.data, name) || findByName(inlineData, name); ++k) {
               name = distribution + "_data_" + std::to_string(k);
            }
            PointData point;
            point.value = *value;
            point.axes.push_back(
               {soleVariateOf(distribution, "the number in the \"data\" of the likelihood '" + likelihood.name + "'"),
                std::nullopt, std::nullopt});
            inlineData.push_back({name, std::move(point)});
            term.data = DataRef{name};
            continue;
         }
         auto const *ref = std::get_if<DataRef>(&term.data);
         auto const *point = ref ? find<PointData>(doc.data, ref->name) : nullptr;
         if (!point) {
            continue;
         }
         if (!point->axes.empty()) {
            // The axis of point data is named like the variate of the paired distribution.
            std::set<std::string> const variates = variatesOf(doc, distribution);
            if (variates.size() == 1 && point->axes.front().name != *variates.begin()) {
               errors.push_back("the axis '" + point->axes.front().name + "' of the point data set '" + ref->name +
                                "' is not the variate '" + *variates.begin() + "' of the distribution '" +
                                distribution + "' it is paired with");
            }
            continue;
         }
         std::string const variate = soleVariateOf(distribution, "the point data set '" + ref->name + "'");
         auto inserted = variateOfPointData.emplace(ref->name, variate);
         if (!inserted.second && inserted.first->second != variate) {
            errors.push_back("the point data set '" + ref->name +
                             "' without axes is paired with distributions of different variates, '" +
                             inserted.first->second + "' and '" + variate + "'");
         }
      }
   }
   for (auto const &[name, variate] : variateOfPointData) {
      find<PointData>(doc.data, name)->axes.push_back({variate, std::nullopt, std::nullopt});
   }
   doc.data.insert(doc.data.end(), std::make_move_iterator(inlineData.begin()),
                   std::make_move_iterator(inlineData.end()));
   return errors;
}

std::map<std::string, std::set<std::string>> writeLikelihoodTermsOnce(HS3Document const &doc, JSONNode &root)
{
   std::map<std::string, std::set<std::string>> removed;
   if (!root.has_child("distributions")) {
      return removed;
   }
   Distributions distributions{doc};
   // The products that are terms, with the factors that are terms wherever the product is one.
   std::map<std::string, std::set<std::string>> removable;
   for (Likelihood const &likelihood : doc.likelihoods) {
      std::vector<std::string> const terms = distributions.likelihoodTerms(likelihood);
      for (std::string const &term : terms) {
         JSONNode const *product = distributions.nodeOf(term, "product_dist");
         if (!product) {
            continue;
         }
         std::set<std::string> factorsThatAreTerms;
         for (std::string const &factor : namesIn(*product, "factors")) {
            if (factor != term && std::find(terms.begin(), terms.end(), factor) != terms.end()) {
               factorsThatAreTerms.insert(factor);
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
   for (Distribution const &dist : doc.distributions) {
      if (JSONNode const *node = genericNode(dist)) {
         for (const char *key : {"factors", "summands"}) {
            for (std::string const &part : namesIn(*node, key)) {
               removable.erase(part);
            }
         }
      }
   }
   for (JSONNode &dist : root["distributions"].children()) {
      auto found = removable.find(dist["name"].val());
      if (found == removable.end() || found->second.empty()) {
         continue;
      }
      std::vector<std::string> keep;
      for (JSONNode const &factor : dist["factors"].children()) {
         if (!found->second.count(factor.val())) {
            keep.push_back(factor.val());
         }
      }
      if (!keep.empty()) {
         dist["factors"].clear();
         dist["factors"].fill_seq(keep);
         removed.insert(*found);
      }
   }
   return removed;
}

} // namespace HS3
} // namespace RooFit

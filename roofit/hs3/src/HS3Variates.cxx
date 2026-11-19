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

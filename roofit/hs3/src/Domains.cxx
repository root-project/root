/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN, Jan 2023
 *
 * Copyright (c) 2023, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

#include "Domains.h"
#include "JSONIOUtils.h"

#include <RooAbsBinning.h>
#include <RooBinning.h>
#include <RooCategory.h>
#include <RooFitHS3/RooJSONFactoryWSTool.h>
#include <RooNumber.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <RooFit/Detail/JSONInterface.h>

#include <set>

namespace RooFit {
namespace JSONIO {
namespace Detail {

constexpr static auto defaultDomainName = "default_domain";

namespace {

// JSON can't represent infinity, so an infinite bound is expressed by omitting it. An absent bound is read as
// infinite by readBound() in JSONIOUtils.
void writeBound(RooFit::Detail::JSONNode &node, const char *key, double value)
{
   if (!RooNumber::isInfinite(value)) {
      node[key] << value;
   }
}

// Read the "categories" of a categorical axis. The labels must be unique non-empty strings, and the axis must not have
// the fields of a continuous axis.
std::vector<std::string> readCategories(RooFit::Detail::JSONNode const &axis, std::string const &context)
{
   for (const char *key : {"min", "max", "nbins", "edges"}) {
      if (axis.has_child(key)) {
         RooJSONFactoryWSTool::error(context + " is categorical and must not have \"" + key + "\"");
      }
   }
   auto const &node = axis["categories"];
   if (!node.is_seq() || node.num_children() == 0) {
      RooJSONFactoryWSTool::error(context + ": \"categories\" must be a non-empty list of strings");
   }
   std::vector<std::string> labels;
   std::set<std::string> seen;
   for (auto const &label : node.children()) {
      if (label.is_seq() || label.is_map() || label.is_number() || label.val().empty()) {
         RooJSONFactoryWSTool::error(context + ": the entries of \"categories\" must be non-empty strings");
      }
      if (!seen.insert(label.val()).second) {
         RooJSONFactoryWSTool::error(context + ": the category \"" + label.val() + "\" is listed twice");
      }
      labels.push_back(label.val());
   }
   return labels;
}

} // namespace

std::vector<std::string> const *Domains::categories(const char *name) const
{
   for (auto const &domain : _map) {
      if (auto const *labels = domain.second.categories(name)) {
         return labels;
      }
   }
   return nullptr;
}

std::vector<std::string> const *Domains::ProductDomain::categories(const char *name) const
{
   auto found = _map.find(name);
   if (found == _map.end() || found->second.categories.empty()) {
      return nullptr;
   }
   return &found->second.categories;
}

void Domains::populate(RooWorkspace &ws) const
{
   auto default_domain = _map.find(defaultDomainName);
   if (default_domain != _map.end()) {
      default_domain->second.populate(ws);
   } else {
      // Without the RooFit-specific "default_domain", the variables are created with the range of the domain that
      // mentions them.
      for (const auto &domain : _map) {
         domain.second.populate(ws);
      }
   }
   for (const auto &domain : _map) {
      if (domain.first == defaultDomainName)
         continue;
      domain.second.registerBinnings(domain.first.c_str(), ws);
   }
}

void Domains::readVariable(RooRealVar const &var)
{
   _map[defaultDomainName].readVariable(var.GetName(), var.getBinning());
   for (const auto &bname : var.getBinningNames()) {
      if (bname.empty())
         continue;
      auto &binning = var.getBinning(bname.c_str());
      _map[bname].readVariable(var.GetName(), binning);
   }
}

void Domains::writeVariable(RooRealVar &var) const
{
   auto default_domain = _map.find(defaultDomainName);
   if (default_domain != _map.end()) {
      default_domain->second.writeVariable(var);
      return;
   }
   // Without the RooFit-specific "default_domain", use the first domain that mentions the variable.
   for (const auto &domain : _map) {
      if (domain.second.hasVariable(var.GetName())) {
         domain.second.writeVariable(var);
         return;
      }
   }
}

void Domains::readJSON(RooFit::Detail::JSONNode const &node)
{
   // The "default_domain" is written by RooFit, but it is not required by the HS3 standard.
   for (auto &domain : node.children()) {
      if (!domain.has_child("name")) {
         RooJSONFactoryWSTool::error("encountered domain without \"name\"");
      }
      auto &name = domain["name"];
      _map[name.val()].readJSON(domain);
   }
}

void Domains::writeJSON(RooFit::Detail::JSONNode &node) const
{
   for (auto const &domain : _map) {
      // Avoid writing a domain that was already written
      if (!RooJSONFactoryWSTool::findNamedChild(node, domain.first)) {
         domain.second.writeJSON(RooJSONFactoryWSTool::appendNamedChild(node, domain.first));
      }
   }
}

bool Domains::hasVariable(const char *name) const
{
   for (auto const &domain : _map) {
      if (domain.second.hasVariable(name)) {
         return true;
      }
   }
   return false;
}

void Domains::ProductDomain::readVariable(RooRealVar const &var)
{
   readVariable(var.GetName(), var.getBinning());
}

void Domains::ProductDomain::readBinning(ProductDomainElement &elem, RooAbsBinning const &binning)
{
   elem.hasNBins = false;
   elem.nBins = 0;
   elem.edges.clear();

   const int nBins = binning.numBins();
   if (nBins <= 0) {
      return;
   }

   if (binning.isUniform()) {
      elem.hasNBins = true;
      elem.nBins = nBins;
   } else {
      elem.edges.push_back(binning.binLow(0));
      for (int i = 0; i < nBins; ++i) {
         elem.edges.push_back(binning.binHigh(i));
      }
   }
}

void Domains::ProductDomain::readVariable(const char *name, RooAbsBinning const &binning)
{
   auto &elem = _map[name];

   elem.hasMin = true;
   elem.min = binning.lowBound();
   elem.hasMax = true;
   elem.max = binning.highBound();
   readBinning(elem, binning);
}

void Domains::ProductDomain::applyBinning(RooRealVar &var, ProductDomainElement const &elem, const char *name)
{
   if (!elem.edges.empty()) {
      RooBinning binning(elem.edges.front(), elem.edges.back());
      for (double edge : elem.edges) {
         binning.addBoundary(edge);
      }
      var.setBinning(binning, name);
   } else if (elem.hasNBins && elem.nBins != 0) {
      var.setBins(elem.nBins, name);
   }
}

void Domains::ProductDomain::writeBinning(RooFit::Detail::JSONNode &node, ProductDomainElement const &elem)
{
   if (!elem.edges.empty()) {
      auto &edges = node["edges"].set_seq();
      for (double edge : elem.edges) {
         edges.append_child() << edge;
      }
   } else if (elem.hasNBins && elem.nBins != 0) {
      node["nbins"] << elem.nBins;
   }
}
void Domains::ProductDomain::writeVariable(RooRealVar &var) const
{
   auto found = _map.find(var.GetName());
   if (found != _map.end()) {
      auto const &elem = found->second;
      if (!elem.categories.empty()) {
         RooJSONFactoryWSTool::error(std::string("the variable '") + var.GetName() +
                                     "' is continuous, but a domain lists it as categorical");
      }
      if (elem.hasMin) {
         if (RooNumber::isInfinite(elem.min)) {
            var.removeMin();
         } else {
            var.setMin(elem.min);
         }
      }
      if (elem.hasMax) {
         if (RooNumber::isInfinite(elem.max)) {
            var.removeMax();
         } else {
            var.setMax(elem.max);
         }
      }
      applyBinning(var, elem);
   }
}

bool Domains::ProductDomain::hasVariable(const char *name) const
{
   return _map.find(name) != _map.end();
}

void Domains::ProductDomain::readJSON(RooFit::Detail::JSONNode const &node)
{
   if (!node.has_child("type") || node["type"].val() != "product_domain") {
      RooJSONFactoryWSTool::error("only domains of type \"product_domain\" are currently supported!");
   }
   for (auto const &varNode : node["axes"].children()) {
      const std::string varName = RooJSONFactoryWSTool::name(varNode);
      auto &elem = _map[varName];

      if (varNode.has_child("categories")) {
         std::vector<std::string> labels = readCategories(varNode, "the axis '" + varName + "' of a domain");
         if (!elem.categories.empty() && elem.categories != labels) {
            RooJSONFactoryWSTool::error("the categorical axis '" + varName +
                                        "' is listed twice in a domain, with different categories");
         }
         elem.categories = std::move(labels);
         continue;
      }
      if (!elem.categories.empty()) {
         RooJSONFactoryWSTool::error("the axis '" + varName +
                                     "' is listed in a domain as categorical and as continuous");
      }

      if (varNode.has_child("min")) {
         elem.min = readBound(varNode, "min", -RooNumber::infinity());
         elem.hasMin = true;
      }
      if (varNode.has_child("max")) {
         elem.max = readBound(varNode, "max", RooNumber::infinity());
         elem.hasMax = true;
      }
      if (varNode.has_child("edges")) {
         elem.hasNBins = false;
         elem.edges.clear();
         for (auto const &edge : varNode["edges"].children()) {
            elem.edges.push_back(edge.val_double());
         }
         if (!elem.edges.empty()) {
            if (!elem.hasMin) {
               elem.min = elem.edges.front();
               elem.hasMin = true;
            }
            if (!elem.hasMax) {
               elem.max = elem.edges.back();
               elem.hasMax = true;
            }
         }
      } else if (varNode.has_child("nbins")) {
         elem.nBins = varNode["nbins"].val_int();
         elem.hasNBins = elem.nBins != 0;
         elem.edges.clear();
      }
   }
}
void Domains::ProductDomain::writeJSON(RooFit::Detail::JSONNode &node) const

{
   node.set_map();
   node["type"] << "product_domain";

   auto &variablesNode = node["axes"];

   for (auto const &item : _map) {
      auto const &elem = item.second;
      RooFit::Detail::JSONNode &varnode = RooJSONFactoryWSTool::appendNamedChild(variablesNode, item.first);
      if (!elem.categories.empty()) {
         varnode["categories"].fill_seq(elem.categories);
         continue;
      }
      const double min = elem.hasMin ? elem.min : -RooNumber::infinity();
      const double max = elem.hasMax ? elem.max : RooNumber::infinity();
      writeBound(varnode, "min", min);
      writeBound(varnode, "max", max);
      // A binning requires a bounded range.
      if (!RooNumber::isInfinite(min) && !RooNumber::isInfinite(max)) {
         writeBinning(varnode, elem);
      }
   }
}
void Domains::ProductDomain::populate(RooWorkspace &ws) const
{
   for (auto const &item : _map) {
      const auto &name = item.first;
      const auto &elem = item.second;
      if (!elem.categories.empty()) {
         populateCategory(name, elem.categories, ws);
         continue;
      }
      if (!ws.arg(name)) {
         const double vMin = elem.hasMin ? elem.min : -RooNumber::infinity();
         const double vMax = elem.hasMax ? elem.max : RooNumber::infinity();
         RooRealVar var{name.c_str(), name.c_str(), vMin, vMax};
         applyBinning(var, elem);
         ws.import(var);
      }
   }
}
// Create a RooCategory for a categorical axis, with one state per category. The states are defined in the order of the
// document, with the position as index, so that the bins of a RooDataHist follow the order of "categories". An
// existing RooCategory of that name must have the same labels.
void Domains::ProductDomain::populateCategory(std::string const &name, std::vector<std::string> const &labels,
                                              RooWorkspace &ws)
{
   if (RooAbsArg *existing = ws.arg(name)) {
      auto *cat = dynamic_cast<RooCategory *>(existing);
      if (!cat) {
         RooJSONFactoryWSTool::error("the axis '" + name + "' is categorical, but '" + name +
                                     "' already exists and is not a category");
      }
      std::set<std::string> existingLabels;
      for (auto const &state : *cat) {
         existingLabels.insert(state.first);
      }
      if (existingLabels != std::set<std::string>(labels.begin(), labels.end())) {
         RooJSONFactoryWSTool::error("the categories of the axis '" + name +
                                     "' differ from the states of the existing category '" + name + "'");
      }
      return;
   }
   RooCategory cat{name.c_str(), name.c_str()};
   for (std::size_t i = 0; i < labels.size(); ++i) {
      cat.defineType(labels[i], static_cast<int>(i));
   }
   ws.import(cat, RooFit::Silence());
}

void Domains::ProductDomain::registerBinnings(const char *name, RooWorkspace &ws) const
{
   for (auto const &item : _map) {
      if (!item.second.categories.empty())
         continue;
      auto *var = ws.var(item.first);
      if (!var)
         continue;
      const double vMin = item.second.hasMin ? item.second.min : -RooNumber::infinity();
      const double vMax = item.second.hasMax ? item.second.max : RooNumber::infinity();
      var->setRange(name, vMin, vMax);
      applyBinning(*var, item.second, name);
   }
}

} // namespace Detail
} // namespace JSONIO
} // namespace RooFit

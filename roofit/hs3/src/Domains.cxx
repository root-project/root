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

#include <RooAbsBinning.h>
#include <RooBinning.h>
#include <RooFitHS3/RooJSONFactoryWSTool.h>
#include <RooNumber.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <RooFit/Detail/JSONInterface.h>

namespace RooFit {
namespace JSONIO {
namespace Detail {

constexpr static auto defaultDomainName = "default_domain";

namespace {

double readBound(RooFit::Detail::JSONNode const &node, const char *key, double defaultValue)
{
   if (!node.has_child(key)) {
      return defaultValue;
   }
   auto const &bound = node[key];
   return bound.is_null() ? defaultValue : bound.val_double();
}

void writeBound(RooFit::Detail::JSONNode &node, double value)
{
   if (RooNumber::isInfinite(value)) {
      node.set_null();
   } else {
      node << value;
   }
}

} // namespace

void Domains::populate(RooWorkspace &ws) const
{
   auto default_domain = _map.find(defaultDomainName);
   if (default_domain != _map.end()) {
      default_domain->second.populate(ws);
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
   }
}

void Domains::readJSON(RooFit::Detail::JSONNode const &node)
{
   auto default_domain_element = RooJSONFactoryWSTool::findNamedChild(node, defaultDomainName);
   if (!default_domain_element) {
      RooJSONFactoryWSTool::error("\"domains\" do not contain \"" + std::string{defaultDomainName} + "\"");
   }
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
      auto &elem = _map[RooJSONFactoryWSTool::name(varNode)];

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
      writeBound(varnode["min"], elem.hasMin ? elem.min : -RooNumber::infinity());
      writeBound(varnode["max"], elem.hasMax ? elem.max : RooNumber::infinity());
      writeBinning(varnode, elem);
   }
}
void Domains::ProductDomain::populate(RooWorkspace &ws) const
{
   for (auto const &item : _map) {
      const auto &name = item.first;
      if (!ws.arg(name)) {
         const auto &elem = item.second;
         const double vMin = elem.hasMin ? elem.min : -RooNumber::infinity();
         const double vMax = elem.hasMax ? elem.max : RooNumber::infinity();
         RooRealVar var{name.c_str(), name.c_str(), vMin, vMax};
         applyBinning(var, elem);
         ws.import(var);
      }
   }
}
void Domains::ProductDomain::registerBinnings(const char *name, RooWorkspace &ws) const
{
   for (auto const &item : _map) {
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

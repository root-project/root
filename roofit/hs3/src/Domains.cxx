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

#include <RooFitHS3/RooJSONFactoryWSTool.h>
#include <RooNumber.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <RooFit/Detail/JSONInterface.h>

namespace RooFit {
namespace JSONIO {
namespace Detail {

constexpr static auto defaultDomainName = "default_domain";

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

void Domains::readVariable(const char *name, double min, double max)
{
   readVariable(name, min, max, defaultDomainName);
}
void Domains::readVariable(const char *name, double min, double max, const char *domain)
{
   _map[domain].readVariable(name, min, max);
}

void Domains::readVariable(RooRealVar const &var)
{
   readVariable(var.GetName(), var.getMin(), var.getMax(), defaultDomainName);
   for (const auto &bname : var.getBinningNames()) {
      if (bname.empty())
         continue;
      auto &binning = var.getBinning(bname.c_str());
      readVariable(var.GetName(), binning.lowBound(), binning.highBound(), bname.c_str());
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

void Domains::ProductDomain::readVariable(RooRealVar const &var)
{
   readVariable(var.GetName(), var.getMin(), var.getMax());
}

void Domains::ProductDomain::readVariable(const char *name, double min, double max)
{
   if (RooNumber::isInfinite(min) && RooNumber::isInfinite(max))
      return;

   auto &elem = _map[name];

   if (!RooNumber::isInfinite(min)) {
      elem.hasMin = true;
      elem.min = min;
   }
   if (!RooNumber::isInfinite(max)) {
      elem.hasMax = true;
      elem.max = max;
   }
}
void Domains::ProductDomain::writeVariable(RooRealVar &var) const
{
   auto found = _map.find(var.GetName());
   if (found != _map.end()) {
      auto const &elem = found->second;
      if (elem.hasMin)
         var.setMin(elem.min);
      if (elem.hasMax)
         var.setMax(elem.max);
   }
}
void Domains::ProductDomain::readJSON(RooFit::Detail::JSONNode const &node)
{
   if (!node.has_child("type") || node["type"].val() != "product_domain") {
      RooJSONFactoryWSTool::error("only domains of type \"product_domain\" are currently supported!");
   }
   for (auto const &varNode : node["axes"].children()) {
      auto &elem = _map[RooJSONFactoryWSTool::name(varNode)];

      if (varNode.has_child("min")) {
         elem.min = varNode["min"].val_double();
         elem.hasMin = true;
      }
      if (varNode.has_child("max")) {
         elem.max = varNode["max"].val_double();
         elem.hasMax = true;
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
      if (elem.hasMin)
         varnode["min"] << elem.min;
      if (elem.hasMax)
         varnode["max"] << elem.max;
   }
}
void Domains::ProductDomain::populate(RooWorkspace &ws) const
{
   for (auto const &item : _map) {
      const auto &name = item.first;
      if (!ws.var(name)) {
         const auto &elem = item.second;
         const double vMin = elem.hasMin ? elem.min : -RooNumber::infinity();
         const double vMax = elem.hasMax ? elem.max : RooNumber::infinity();
         ws.import(RooRealVar{name.c_str(), name.c_str(), vMin, vMax});
      }
   }
}
void Domains::ProductDomain::registerBinnings(const char *name, RooWorkspace &ws) const
{
   for (auto const &item : _map) {
      auto *var = ws.var(item.first);
      if (!var)
         continue;
      var->setRange(name, item.second.min, item.second.max);
   }
}

} // namespace Detail
} // namespace JSONIO
} // namespace RooFit

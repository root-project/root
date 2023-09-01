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

#include <RooFit/Detail/JSONInterface.h>

namespace RooFit {
namespace JSONIO {
namespace Detail {

void Domains::readVariable(const char *name, double min, double max)
{
   _map["default_domain"].readVariable(name, min, max);
}
void Domains::readVariable(RooRealVar const &var)
{
   readVariable(var.GetName(), var.getMin(), var.getMax());
}
void Domains::writeVariable(RooRealVar &var) const
{
   _map.at("default_domain").writeVariable(var);
}

void Domains::readJSON(RooFit::Detail::JSONNode const &node)
{
   _map["default_domain"].readJSON(*RooJSONFactoryWSTool::findNamedChild(node, "default_domain"));
}
void Domains::writeJSON(RooFit::Detail::JSONNode &node) const
{
   for (auto const &domain : _map) {
      domain.second.writeJSON(RooJSONFactoryWSTool::appendNamedChild(node, domain.first));
   }
}
void Domains::ProductDomain::readVariable(const char *name, double min, double max)
{
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
   // In the future, throw an exception if the type is not product domain
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

} // namespace Detail
} // namespace JSONIO
} // namespace RooFit

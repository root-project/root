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
#include "HS3Document.h"
#include "JSONIOUtils.h"

#include <RooAbsBinning.h>
#include <RooBinning.h>
#include <RooConstVar.h>
#include <RooGlobalFunc.h>
#include <RooFitHS3/RooJSONFactoryWSTool.h>
#include <RooNumber.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include <RooFit/Detail/JSONInterface.h>

#include <algorithm>

namespace RooFit {
namespace JSONIO {
namespace Detail {

constexpr static auto defaultDomainName = "default_domain";

void Domains::populate(RooWorkspace &ws) const
{
   // Parameters with a single value are RooConstVars, whichever domain gives the value.
   for (const auto &domain : _map) {
      domain.second.populateFixedValues(ws);
   }
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

void Domains::read(std::vector<RooFit::HS3::Domain> const &domains)
{
   // The "default_domain" is written by RooFit, but it is not required by the HS3 standard.
   for (auto const &domain : domains) {
      _map[domain.name].read(domain);
   }
   // A parameter with a single value has no other domain: every domain that lists it must give it the same value.
   for (auto const &[domainName, domain] : _map) {
      for (auto const &[varName, elem] : domain._map) {
         if (!elem.hasFixedValue) {
            continue;
         }
         for (auto const &[otherName, other] : _map) {
            auto found = other._map.find(varName);
            if (found == other._map.end()) {
               continue;
            }
            if (!found->second.hasFixedValue) {
               RooJSONFactoryWSTool::error("the parameter '" + varName + "' has the single value " +
                                           std::to_string(elem.fixedValue) + " in the domain '" + domainName +
                                           "', but a range in the domain '" + otherName + "'");
            }
            if (found->second.fixedValue != elem.fixedValue) {
               RooJSONFactoryWSTool::error("the parameter '" + varName +
                                           "' has different single values in the "
                                           "domains '" +
                                           domainName + "' and '" + otherName + "'");
            }
         }
      }
   }
}

void Domains::readFixedValue(const char *name, double value)
{
   _map[defaultDomainName].readFixedValue(name, value);
}

double const *Domains::fixedValue(const char *name) const
{
   for (auto const &domain : _map) {
      if (double const *value = domain.second.fixedValue(name)) {
         return value;
      }
   }
   return nullptr;
}

void Domains::write(std::vector<RooFit::HS3::Domain> &domains) const
{
   for (auto const &domain : _map) {
      // Avoid writing a domain that was already written
      auto found = std::find_if(domains.begin(), domains.end(), [&](auto const &d) { return d.name == domain.first; });
      if (found == domains.end()) {
         domains.emplace_back();
         domains.back().name = domain.first;
         domain.second.write(domains.back());
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

void Domains::ProductDomain::writeVariable(RooRealVar &var) const
{
   auto found = _map.find(var.GetName());
   if (found != _map.end() && !found->second.hasFixedValue) {
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

double const *Domains::ProductDomain::fixedValue(const char *name) const
{
   auto found = _map.find(name);
   return found != _map.end() && found->second.hasFixedValue ? &found->second.fixedValue : nullptr;
}

void Domains::ProductDomain::readFixedValue(const char *name, double value)
{
   auto &elem = _map[name];
   elem = ProductDomainElement{};
   elem.hasFixedValue = true;
   elem.fixedValue = value;
}

void Domains::ProductDomain::read(RooFit::HS3::Domain const &domain)
{
   for (auto const &axisVariant : domain.axes) {
      auto const *axis = std::get_if<RooFit::HS3::DomainAxis>(&axisVariant);
      if (!axis) {
         continue;
      }
      if (axis->fixedValue) {
         readFixedValue(axis->name.c_str(), *axis->fixedValue);
         continue;
      }
      auto &elem = _map[axis->name];

      if (axis->min) {
         elem.min = *axis->min;
         elem.hasMin = true;
      }
      if (axis->max) {
         elem.max = *axis->max;
         elem.hasMax = true;
      }
      if (!axis->binning) {
         continue;
      }
      if (auto const *edges = std::get_if<RooFit::HS3::EdgesBinning>(&*axis->binning)) {
         elem.hasNBins = false;
         elem.edges = edges->edges;
      } else {
         elem.nBins = std::get<RooFit::HS3::RegularBinning>(*axis->binning).nbins;
         elem.hasNBins = elem.nBins != 0;
         elem.edges.clear();
      }
   }
}
void Domains::ProductDomain::write(RooFit::HS3::Domain &domain) const
{
   for (auto const &item : _map) {
      auto const &elem = item.second;
      RooFit::HS3::DomainAxis axis;
      axis.name = item.first;
      if (elem.hasFixedValue) {
         axis.fixedValue = elem.fixedValue;
         domain.axes.emplace_back(std::move(axis));
         continue;
      }
      // An infinite bound is expressed by omitting it.
      if (elem.hasMin && !RooNumber::isInfinite(elem.min)) {
         axis.min = elem.min;
      }
      if (elem.hasMax && !RooNumber::isInfinite(elem.max)) {
         axis.max = elem.max;
      }
      // A binning requires a bounded range.
      if (axis.min && axis.max) {
         if (!elem.edges.empty()) {
            axis.binning = RooFit::HS3::EdgesBinning{elem.edges};
         } else if (elem.hasNBins && elem.nBins != 0) {
            axis.binning = RooFit::HS3::RegularBinning{elem.nBins};
         }
      }
      domain.axes.emplace_back(std::move(axis));
   }
}
void Domains::ProductDomain::populate(RooWorkspace &ws) const
{
   for (auto const &item : _map) {
      const auto &name = item.first;
      if (item.second.hasFixedValue) {
         continue; // see populateFixedValues()
      }
      if (!ws.arg(name)) {
         const auto &elem = item.second;
         const double vMin = elem.hasMin ? elem.min : -RooNumber::infinity();
         const double vMax = elem.hasMax ? elem.max : RooNumber::infinity();
         RooRealVar var{name.c_str(), name.c_str(), vMin, vMax};
         applyBinning(var, elem);
         ws.import(var, RooFit::Silence());
      }
   }
}
void Domains::ProductDomain::populateFixedValues(RooWorkspace &ws) const
{
   for (auto const &[name, elem] : _map) {
      if (elem.hasFixedValue && !ws.arg(name)) {
         ws.import(RooConstVar{name.c_str(), name.c_str(), elem.fixedValue}, RooFit::Silence());
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

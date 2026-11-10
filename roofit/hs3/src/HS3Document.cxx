/*
 * Project: RooFit
 * Authors:
 *   Jonas Rembser, CERN, Nov 2026
 *
 * Copyright (c) 2026, CERN
 *
 * Redistribution and use in source and binary forms,
 * with or without modification, are permitted according to the terms
 * listed in LICENSE (http://roofit.sourceforge.net/license.txt)
 */

// Operations on documents that RooFit needs besides reading, upgrading, validating and writing them.

#include "HS3Document.h"

#include <algorithm>

namespace RooFit {
namespace HS3 {

DomainRanges::DomainRanges(std::vector<Domain> const &domains)
{
   std::vector<Domain const *> sorted;
   for (Domain const &domain : domains) {
      sorted.push_back(&domain);
   }
   std::stable_sort(sorted.begin(), sorted.end(), [](Domain const *l, Domain const *r) { return l->name < r->name; });
   const bool hasDefault =
      std::any_of(domains.begin(), domains.end(), [](Domain const &d) { return d.name == defaultDomainName; });

   for (Domain const *domain : sorted) {
      // Of several axes of a variable in one domain, the last one counts.
      std::map<std::string, DomainAxis const *> axes;
      for (auto const &axis : domain->axes) {
         if (auto const *continuous = std::get_if<DomainAxis>(&axis)) {
            axes[continuous->name] = continuous;
            _names.insert(continuous->name);
         }
      }
      if (!hasDefault || domain->name == defaultDomainName) {
         _ranges.insert(axes.begin(), axes.end());
      }
   }
}

DomainAxis const *DomainRanges::find(std::string const &name) const
{
   auto found = _ranges.find(name);
   return found != _ranges.end() ? found->second : nullptr;
}

namespace {

std::string const &nameOf(Modifier const &modifier)
{
   return std::visit([](auto const &m) -> std::string const & { return m.name; }, modifier);
}

char const *typeOf(Modifier const &modifier)
{
   return std::visit(
      overloaded{[](NormFactor const &) { return "normfactor"; }, [](NormSys const &) { return "normsys"; },
                 [](HistoSys const &) { return "histosys"; }, [](ShapeSys const &) { return "shapefactor"; },
                 [](StatError const &) { return "staterror"; }, [](CustomModifier const &) { return "custom"; }},
      modifier);
}

// What duplicates must have in common to be combined.
template <class Modifier_t>
char const *differences(Modifier_t const &l, Modifier_t const &r)
{
   if (!(l.constraint == r.constraint)) {
      return "constraint metadata differs";
   }
   if (l.parameter.name != r.parameter.name) {
      return "parameter metadata differs";
   }
   if (l.interpolation != r.interpolation) {
      return "interpolation behaviours differ";
   }
   return nullptr;
}

// Combines `modifier` into `merged`, or returns why that's not possible.
char const *combine(NormSys &merged, NormSys const &modifier, std::vector<double> const &)
{
   if (char const *reason = differences(merged, modifier)) {
      return reason;
   }
   // Multiplicatively combining two normsys is only faithful when the interpolation is done in log-space, so that
   // f1(alpha) * f2(alpha) is again representable by a single normsys with the multiplied lo/hi factors. This holds
   // for the piecewise exponential (exact everywhere) and for the polynomial exponential (exact at the +-1 sigma
   // anchors and in the exponential extrapolation region). The linear-space interpolations would turn the product into
   // a shape that cannot be represented by a single normsys.
   if (merged.interpolation != multiplicativePiecewiseExponential &&
       merged.interpolation != multiplicativePolynomialExponential) {
      return "multiplicative combination is only valid for log-space interpolation";
   }
   merged.lo *= modifier.lo;
   merged.hi *= modifier.hi;
   return nullptr;
}

char const *combine(HistoSys &merged, HistoSys const &modifier, std::vector<double> const &nominal)
{
   if (char const *reason = differences(merged, modifier)) {
      return reason;
   }
   if (merged.interpolation != additivePolynomialLinear) {
      return "this interpolation cannot currently be combined for duplicate histosys modifiers";
   }
   const std::size_t nBins = nominal.size();
   auto fits = [&](HistoSysVariation const &variation) { return variation.contents.size() == nBins; };
   if (!fits(merged.lo) || !fits(merged.hi) || !fits(modifier.lo) || !fits(modifier.hi)) {
      return "histogram binning differs";
   }
   for (std::size_t bin = 0; bin < nBins; ++bin) {
      merged.lo.contents[bin] += modifier.lo.contents[bin] - nominal[bin];
      merged.hi.contents[bin] += modifier.hi.contents[bin] - nominal[bin];
   }
   return nullptr;
}

} // namespace

void mergeDuplicateModifiers(HistFactoryDist &dist, std::string const &channel, Diagnostics &diagnostics)
{
   for (Sample &sample : dist.samples) {
      auto error = [&](Modifier const &modifier, std::string const &reason) {
         diagnostics.push_back({Diagnostic::Severity::Error, "",
                                "cannot combine duplicate modifier '" + nameOf(modifier) + "' of type '" +
                                   typeOf(modifier) + "' in sample '" + sample.name + "' of channel '" + channel +
                                   "': " + reason});
      };

      // The duplicates of a type are next to each other, because the modifiers of a type are sorted by name.
      std::vector<Modifier> merged;
      std::size_t count = 0; // of the modifiers that the last merged one combines
      bool failed = false;
      auto warnIfCombined = [&]() {
         if (count > 1 && !failed) {
            diagnostics.push_back({Diagnostic::Severity::Warning, "",
                                   "combined " + std::to_string(count) + " duplicate modifiers named '" +
                                      nameOf(merged.back()) + "' of type '" + typeOf(merged.back()) + "' in sample '" +
                                      sample.name + "' of channel '" + channel + "'"});
         }
      };
      for (Modifier &modifier : sample.modifiers) {
         Modifier *last = merged.empty() ? nullptr : &merged.back();
         const bool combinable =
            std::holds_alternative<NormSys>(modifier) || std::holds_alternative<HistoSys>(modifier);
         if (combinable && last && last->index() == modifier.index() && nameOf(*last) == nameOf(modifier)) {
            char const *reason = nullptr;
            if (auto *normSys = std::get_if<NormSys>(last)) {
               reason = combine(*normSys, std::get<NormSys>(modifier), sample.contents);
            } else {
               reason = combine(std::get<HistoSys>(*last), std::get<HistoSys>(modifier), sample.contents);
            }
            if (reason) {
               error(modifier, reason);
               failed = true;
            }
            ++count;
            continue;
         }
         warnIfCombined();
         merged.push_back(std::move(modifier));
         count = 1;
         failed = false;
      }
      warnIfCombined();
      sample.modifiers = std::move(merged);

      std::set<std::pair<std::string, std::string>> seen;
      for (Modifier const &modifier : sample.modifiers) {
         if (!seen.emplace(typeOf(modifier), nameOf(modifier)).second) {
            error(modifier, "this modifier type cannot be combined without changing its meaning");
         }
      }
   }
}

void factorOutDefaultInterpolation(HistFactoryDist &dist)
{
   auto forEachInterpolation = [&](auto &&fn) {
      for (Sample &sample : dist.samples) {
         for (Modifier &modifier : sample.modifiers) {
            if (auto *normSys = std::get_if<NormSys>(&modifier)) {
               fn(normSys->interpolation);
            } else if (auto *histoSys = std::get_if<HistoSys>(&modifier)) {
               fn(histoSys->interpolation);
            }
         }
      }
   };

   std::map<Interpolation, std::size_t> counts;
   forEachInterpolation([&](std::optional<Interpolation> &interpolation) {
      if (interpolation) {
         ++counts[*interpolation];
      }
   });
   if (counts.empty()) {
      return;
   }
   // A tie is decided in favor of the default of HistFactory.
   auto best = counts.begin();
   for (auto current = std::next(counts.begin()); current != counts.end(); ++current) {
      if (current->second > best->second ||
          (current->second == best->second && current->first == multiplicativePolynomialExponential &&
           best->first != multiplicativePolynomialExponential)) {
         best = current;
      }
   }
   dist.defaultInterpolation = best->first;
   forEachInterpolation([&](std::optional<Interpolation> &interpolation) {
      if (interpolation == dist.defaultInterpolation) {
         interpolation.reset();
      }
   });
}

} // namespace HS3
} // namespace RooFit

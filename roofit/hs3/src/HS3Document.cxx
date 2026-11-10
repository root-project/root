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

} // namespace HS3
} // namespace RooFit

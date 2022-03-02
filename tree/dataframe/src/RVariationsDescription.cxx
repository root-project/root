// Author: Enrico Guiraud CERN  02/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RVariationsDescription.hxx"

#include <iostream>
#include <unordered_set>

namespace {
static std::string GetStringRepr(
   const std::unordered_multimap<std::string, std::shared_ptr<ROOT::Internal::RDF::RVariationBase>> &variationsMap)
{
   std::unordered_set<ROOT::Internal::RDF::RVariationBase*> uniqueVariations;
   std::string s;

   for (const auto &e : variationsMap) {
      const auto it = uniqueVariations.insert(e.second.get());
      if (!it.second) // we have already seen this variation
         continue;

      const auto &variation = *e.second;

      s += "Variations {";
      for (const auto &tag : variation.GetVariationNames())
         s += tag + ", ";
      s.erase(s.size() - 2);
      s += "} affect column";
      const auto &columns = variation.GetColumnNames();
      if (columns.size() == 1)
         s += " " + columns[0];
      else {
         s += "s {";
         for (const auto &col : columns)
            s += col + ", ";
         s.erase(s.size() - 2);
         s += "}";
      }
      s += '\n';
   }
   return s;
}
} // namespace

namespace ROOT {
namespace RDF {

RVariationsDescription::RVariationsDescription(const Variations_t &variations) : fStringRepr(GetStringRepr(variations))
{
}

void RVariationsDescription::Print() const
{
   std::cout << AsString();
}

} // namespace RDF
} // namespace ROOT

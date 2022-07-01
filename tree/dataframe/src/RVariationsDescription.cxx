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

namespace {
static std::string GetStringRepr(const std::vector<const ROOT::Internal::RDF::RVariationBase *> &variations)
{
   std::string s;

   for (const auto *varPtr : variations) {
      const auto &variation = *varPtr;

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
} // anonymous namespace

namespace ROOT {
namespace RDF {

// Pre-condition: elements in variations are expected to be unique.
RVariationsDescription::RVariationsDescription(const Variations_t &variations) : fStringRepr(GetStringRepr(variations))
{
}

void RVariationsDescription::Print() const
{
   std::cout << AsString();
}

} // namespace RDF
} // namespace ROOT

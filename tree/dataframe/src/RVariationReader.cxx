/**
 \author Vincenzo Eduardo Padulano
 \date 2024-04
*/

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#include <cassert>

#include "ROOT/RDF/RVariationReader.hxx"
#include "ROOT/RDF/Utils.hxx" // IsStrInVec

ROOT::Internal::RDF::RVariationsWithReaders::RVariationsWithReaders(
   std::shared_ptr<ROOT::Internal::RDF::RVariationBase> variation, unsigned int nSlots)
   : fVariation(std::move(variation)), fReadersPerVariation(nSlots)
{
   assert(fVariation != nullptr);
}

////////////////////////////////////////////////////////////////////////////
/// Return a column reader for the given slot, column and variation.
ROOT::Internal::RDF::RVariationReader &
ROOT::Internal::RDF::RVariationsWithReaders::GetReader(unsigned int slot, const std::string &colName,
                                                       const std::string &variationName)
{
   assert(ROOT::Internal::RDF::IsStrInVec(variationName, fVariation->GetVariationNames()));
   assert(ROOT::Internal::RDF::IsStrInVec(colName, fVariation->GetColumnNames()));

   auto &varReaders = fReadersPerVariation[slot];

   auto it = varReaders.find(variationName);
   if (it != varReaders.end())
      return *it->second;

#if !defined(__clang__) && __GNUC__ >= 7 && __GNUC_MINOR__ >= 3
   const auto insertion = varReaders.insert({variationName, std::make_unique<ROOT::Internal::RDF::RVariationReader>(
                                                               slot, colName, variationName, *fVariation)});
   return *insertion.first->second;
#else
   // gcc < 7.3 has issues with passing the non-movable std::pair temporary into the insert call
   auto reader = std::make_unique<ROOT::Internal::RDF::RVariationReader>(slot, colName, variationName, *fVariation);
   auto &ret = *reader;
   varReaders[variationName] = std::move(reader);
   return ret;
#endif
}

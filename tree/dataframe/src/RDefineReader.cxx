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

#include "ROOT/RDF/RDefineReader.hxx"

ROOT::Internal::RDF::RDefinesWithReaders::RDefinesWithReaders(std::shared_ptr<ROOT::Detail::RDF::RDefineBase> define,
                                                              unsigned int nSlots,
                                                              ROOT::Internal::RDF::RStringCache &cachedColNames)
   : fDefine(std::move(define)), fReadersPerVariation(nSlots), fCachedColNames(cachedColNames)
{
   assert(fDefine != nullptr);
}

ROOT::Internal::RDF::RDefineReader &
ROOT::Internal::RDF::RDefinesWithReaders::GetReader(unsigned int slot, std::string_view variationName)
{
   auto nameIt = fCachedColNames.Insert(std::string(variationName));
   auto &defineReaders = fReadersPerVariation[slot];

   auto it = defineReaders.find(*nameIt);
   if (it != defineReaders.end())
      return *it->second;

   auto *define = fDefine.get();
   if (*nameIt != "nominal")
      define = &define->GetVariedDefine(std::string(variationName));

#if !defined(__clang__) && __GNUC__ >= 7 && __GNUC_MINOR__ >= 3
   const auto insertion =
      defineReaders.insert({*nameIt, std::make_unique<ROOT::Internal::RDF::RDefineReader>(slot, *define)});
   return *insertion.first->second;
#else
   // gcc < 7.3 has issues with passing the non-movable std::pair temporary into the insert call
   auto reader = std::make_unique<ROOT::Internal::RDF::RDefineReader>(slot, *define);
   auto &ret = *reader;
   defineReaders[*nameIt] = std::move(reader);
   return ret;
#endif
}

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

   std::shared_ptr<RDefineReader> readerToReturn;
   auto *define = fDefine.get();
   if (*nameIt == "nominal") {
      readerToReturn = std::make_shared<RDefineReader>(slot, *define);
   } else {
      auto *variedDefine = &define->GetVariedDefine(std::string(variationName));
      if (variedDefine == define) {
         // The column in not affected by variations. We can return the same reader as for nominal
         if (auto nominalReaderIt = defineReaders.find("nominal"); nominalReaderIt != defineReaders.end()) {
            readerToReturn = nominalReaderIt->second;
         } else {
            // The nominal reader doesn't exist yet
            readerToReturn = std::make_shared<RDefineReader>(slot, *define);
            auto nominalNameIt = fCachedColNames.Insert("nominal");
            defineReaders.insert({*nominalNameIt, readerToReturn});
         }
      } else {
         readerToReturn = std::make_shared<RDefineReader>(slot, *variedDefine);
      }
   }

   defineReaders.insert({*nameIt, readerToReturn});

   return *readerToReturn;
}

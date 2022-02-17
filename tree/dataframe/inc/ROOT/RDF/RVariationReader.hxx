// Author: Enrico Guiraud CERN 11/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RVARIATIONREADER
#define ROOT_RDF_RVARIATIONREADER

#include "RColumnReaderBase.hxx"
#include "RVariationBase.hxx"
#include <Rtypes.h> // Long64_t, R__CLING_PTRCHECK

#include <limits>
#include <type_traits>

namespace ROOT {
namespace Internal {
namespace RDF {

/// Column reader that reads the value for a specific column, variation and slot.
class R__CLING_PTRCHECK(off) RVariationReader final : public ROOT::Detail::RDF::RColumnReaderBase {
   RVariationBase *fVariation;

   /// Non-owning ptr to the value of the variation.
   void *fValuePtr = nullptr;

   /// The slot this value belongs to.
   unsigned int fSlot = std::numeric_limits<unsigned int>::max();

   void *GetImpl(Long64_t entry) final
   {
      fVariation->Update(fSlot, entry);
      return fValuePtr;
   }

public:
   RVariationReader(unsigned int slot, const std::string &colName, const std::string &variationName,
                    RVariationBase &variation, const std::type_info &tid)
      : fVariation(&variation), fValuePtr(variation.GetValuePtr(slot, colName, variationName)), fSlot(slot)
   {
      CheckReaderTypeMatches(variation.GetTypeId(), tid, colName, "RVariationReader");
   }
};

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif

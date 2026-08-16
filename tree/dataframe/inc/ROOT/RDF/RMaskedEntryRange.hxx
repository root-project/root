// Author: Enrico Guiraud, Vincenzo Eduardo Padulano

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RMASKEDENTRYRANGE
#define ROOT_RDF_RMASKEDENTRYRANGE

#include <ROOT/RVec.hxx>

#include <cstdint>
#include <cstddef>

namespace ROOT::Internal::RDF {

/// RDataFrame's internal representation of an entry range with a boolean mask.
/// The mask has static size but depending on the dynamic bulk size fewer elements could be in use:
/// do not take the size of the mask as the size of the bulk.
class RMaskedEntryRange {
   ROOT::RVec<bool> fMask{}; ///< Boolean mask. Its size is set at construction time.
   std::uint64_t fBegin{};   ///< Entry number of the first entry in the range this mask corresponds to.

public:
   RMaskedEntryRange(std::size_t size, bool set = true, std::uint64_t entry = 0) : fMask(size, set), fBegin(entry) {}
   RMaskedEntryRange(const ROOT::RVec<bool> &mask, std::uint64_t begin) : fMask(mask), fBegin(begin) {}
   const bool &operator[](std::size_t idx) const { return fMask.at(idx); }
   bool &operator[](std::size_t idx) { return fMask.at(idx); }
   std::uint64_t GetFirstEntry() const { return fBegin; }
   void SetFirstEntry(std::uint64_t e) { fBegin = e; }
   ROOT::RVec<std::size_t> GetValidIndices() const
   {
      ROOT::RVec<std::size_t> validIndices{};
      for (std::size_t i = 0; i < fMask.size(); ++i) {
         if (fMask[i])
            validIndices.push_back(i);
      }
      return validIndices;
   }
};

} // namespace ROOT::Internal::RDF
#endif

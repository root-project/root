// Author: Enrico Guiraud 09/2022

/*************************************************************************
 * Copyright (C) 1995-2022, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RDF_RMASKEDENTRYRANGE
#define ROOT_RDF_RMASKEDENTRYRANGE

#include <Rtypes.h>
#include <ROOT/RVec.hxx>

#include <cassert>
#include <limits>

namespace ROOT {
namespace Internal {
namespace RDF {

class RMaskedEntryRange {
   ROOT::RVec<bool> fMask; ///< Boolean mask. Its size is set at construction time.
   Long64_t fBegin;        ///< Entry number of the first entry in the range this mask corresponds to.

public:
   RMaskedEntryRange(std::size_t size) : fMask(size, true), fBegin(-1ll) {}
   Long64_t FirstEntry() const { return fBegin; }
   const bool &operator[](std::size_t idx) const { return fMask[idx]; }
   bool &operator[](std::size_t idx) { return fMask[idx]; }
   void SetAll(bool to) { fMask.assign(fMask.size(), to); }
   void SetFirstEntry(Long64_t e) { fBegin = e; }
   void Union(const RMaskedEntryRange &other)
   {
      for (std::size_t i = 0u; i < fMask.size(); ++i)
         fMask[i] |= other[i];
   }
   std::size_t Count(std::size_t until) { return std::accumulate(fMask.begin(), fMask.begin() + until, 0ul); }
};

} // namespace RDF
} // namespace Internal
} // namespace ROOT

#endif

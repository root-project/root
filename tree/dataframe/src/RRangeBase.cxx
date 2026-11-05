// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RRangeBase.hxx"
#include <ROOT/RDF/RLoopManager.hxx>

using ROOT::Detail::RDF::RRangeBase;

RRangeBase::RRangeBase(RLoopManager *implPtr, unsigned int start, unsigned int stop, unsigned int stride,
                       const unsigned int nSlots, const std::vector<std::string> &prevVariations)
   : RNodeBase(prevVariations, implPtr, nSlots),
     fStart(start),
     fStop(stop),
     fStride(stride),
     fNSlots(nSlots),
     fMaskPerSlot(fNSlots * RDFInternal::CacheLineStep<ROOT::Internal::RDF::RMaskedEntryRange>(),
                  ROOT::Internal::RDF::RMaskedEntryRange{implPtr->GetCurrentBulkSize(), false,
                                                         std::numeric_limits<std::uint64_t>::max()})
{
}

void RRangeBase::InitNode()
{
   fNProcessedEntries = 0;
   fHasStopped = false;
   for (unsigned slot = 0; slot < fNSlots; slot++)
      fMaskPerSlot[slot * RDFInternal::CacheLineStep<RDFInternal::RMaskedEntryRange>()].SetFirstEntry(
         std::numeric_limits<std::uint64_t>::max());
}

// outlined to pin virtual table
ROOT::Detail::RDF::RRangeBase::~RRangeBase() = default;

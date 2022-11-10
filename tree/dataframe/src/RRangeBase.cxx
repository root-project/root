// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RDF/RRangeBase.hxx"

using ROOT::Detail::RDF::RRangeBase;

RRangeBase::RRangeBase(RLoopManager *implPtr, unsigned int start, unsigned int stop, unsigned int stride,
                       const unsigned int nSlots, const std::vector<std::string> &prevVariations)
   : RNodeBase(prevVariations, implPtr), fStart(start), fStop(stop), fStride(stride), fMask(1ul), fNSlots(nSlots)
{
}

void RRangeBase::InitNode()
{
   fNProcessedEntries = 0;
}

// outlined to pin virtual table
RRangeBase::~RRangeBase() { }

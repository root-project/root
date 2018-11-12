// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/TSeq.hxx>
#include <ROOT/RDF/RSlotStack.hxx>
#include <TError.h> // R__ASSERT

ROOT::Internal::RDF::RSlotStack::RSlotStack(unsigned int size) : fSize(size)
{
   for (auto i : ROOT::TSeqU(size)) fStack.push(i);
}

void ROOT::Internal::RDF::RSlotStack::ReturnSlot(unsigned int slot)
{
   ROOT::TRWSpinLockWriteGuard guard(fRWLock);
   R__ASSERT(fStack.size() < fSize && "Trying to put back a slot to a full stack!");
   fStack.push(slot);
}

unsigned int ROOT::Internal::RDF::RSlotStack::GetSlot()
{
   ROOT::TRWSpinLockWriteGuard guard(fRWLock);
   R__ASSERT(!fStack.empty() && "Trying to pop a slot from an empty stack!");
   const auto slot = fStack.top();
   fStack.pop();
   return slot;
}

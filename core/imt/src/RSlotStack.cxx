// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/TSeq.hxx>
#include <ROOT/RSlotStack.hxx>

#include <cassert>
#include <mutex> // std::lock_guard

ROOT::Internal::RSlotStack::RSlotStack(unsigned int size) : fSize(size)
{
   for (auto i : ROOT::TSeqU(size))
      fStack.push(i);
}

void ROOT::Internal::RSlotStack::ReturnSlot(unsigned int slot)
{
   std::lock_guard<ROOT::TSpinMutex> guard(fMutex);
   assert(fStack.size() < fSize && "Trying to put back a slot to a full stack!");
   (void)fSize;
   fStack.push(slot);
}

unsigned int ROOT::Internal::RSlotStack::GetSlot()
{
   std::lock_guard<ROOT::TSpinMutex> guard(fMutex);
   assert(!fStack.empty() && "Trying to pop a slot from an empty stack!");
   const auto slot = fStack.top();
   fStack.pop();
   return slot;
}

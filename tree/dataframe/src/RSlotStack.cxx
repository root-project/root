// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RDF/RSlotStack.hxx>
#include <TError.h> // R__ASSERT

#include <limits>
#include <numeric>

ROOT::Internal::RDF::RSlotStack::RSlotStack(unsigned int size) : fCursor(size), fBuf(size)
{
   std::iota(fBuf.begin(), fBuf.end(), 0U);
}

unsigned int &ROOT::Internal::RDF::RSlotStack::GetCount()
{
   const auto tid = std::this_thread::get_id();
   {
      ROOT::TRWSpinLockReadGuard rg(fRWLock);
      auto it = fCountMap.find(tid);
      if (fCountMap.end() != it)
         return it->second;
   }

   {
      ROOT::TRWSpinLockWriteGuard rg(fRWLock);
      return (fCountMap[tid] = 0U);
   }
}
unsigned int &ROOT::Internal::RDF::RSlotStack::GetIndex()
{
   const auto tid = std::this_thread::get_id();

   {
      ROOT::TRWSpinLockReadGuard rg(fRWLock);
      if (fIndexMap.end() != fIndexMap.find(tid))
         return fIndexMap[tid];
   }

   {
      ROOT::TRWSpinLockWriteGuard rg(fRWLock);
      return (fIndexMap[tid] = std::numeric_limits<unsigned int>::max());
   }
}

void ROOT::Internal::RDF::RSlotStack::ReturnSlot(unsigned int slotNumber)
{
   auto &index = GetIndex();
   auto &count = GetCount();
   R__ASSERT(count > 0U && "RSlotStack has a reference count relative to an index which will become negative.");
   count--;
   if (0U == count) {
      index = std::numeric_limits<unsigned int>::max();
      ROOT::TRWSpinLockWriteGuard guard(fRWLock);
      fBuf[fCursor++] = slotNumber;
      R__ASSERT(fCursor <= fBuf.size() &&
                "RSlotStack assumes that at most a fixed number of values can be present in the "
                "stack. fCursor is greater than the size of the internal buffer. This violates "
                "such assumption.");
   }
}

unsigned int ROOT::Internal::RDF::RSlotStack::GetSlot()
{
   auto &index = GetIndex();
   auto &count = GetCount();
   count++;
   if (std::numeric_limits<unsigned int>::max() != index)
      return index;
   ROOT::TRWSpinLockWriteGuard guard(fRWLock);
   R__ASSERT(fCursor > 0 &&
             "RSlotStack assumes that a value can be always obtained. In this case fCursor is <=0 and this "
             "violates such assumption.");
   index = fBuf[--fCursor];
   return index;
}

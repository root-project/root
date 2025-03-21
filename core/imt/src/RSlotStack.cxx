// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RSlotStack.hxx>

#include <stdexcept>
#include <string>

ROOT::Internal::RSlotStack::RSlotStack(unsigned int size) : fSlots(size) {}

void ROOT::Internal::RSlotStack::ReturnSlot(unsigned int slot)
{
   if (slot >= fSlots.size())
      throw std::invalid_argument("RSlotStack: A slot that is larger than the number of slots was returned :" +
                                  std::to_string(slot));
   bool expected = true;
   if (!fSlots[slot].fAtomic.compare_exchange_strong(expected, false))
      throw std::logic_error("RSlotStack: A slot that is not assigned was returned: " + std::to_string(slot));
}

unsigned int ROOT::Internal::RSlotStack::GetSlot()
{
   while (true) {
      for (unsigned int i = 0; i < fSlots.size(); ++i) {
         // test if a slot is available (assigned == false)
         bool expectedState = false;
         if (fSlots[i].fAtomic.compare_exchange_strong(expectedState, true)) {
            return i;
         }
      }
   }
}

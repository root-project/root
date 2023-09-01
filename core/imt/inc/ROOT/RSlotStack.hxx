// Author: Enrico Guiraud, Danilo Piparo CERN  03/2017

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RSLOTSTACK
#define ROOT_RSLOTSTACK

#include <ROOT/TSpinMutex.hxx>

#include <stack>

namespace ROOT {
namespace Internal {

/// A thread-safe stack of N indexes (0 to size - 1).
/// RSlotStack can be used to safely assign a "processing slot" number to
/// each thread in multi-thread applications.
/// In release builds, pop and push operations are unchecked, potentially
/// resulting in undefined behavior if more slot numbers than available are
/// requested.
/// An important design assumption is that a slot will almost always be available
/// when a thread asks for it, and if it is not available it will be very soon,
/// therefore a spinlock is used for synchronization.
class RSlotStack {
private:
   const unsigned int fSize;
   std::stack<unsigned int> fStack;
   ROOT::TSpinMutex fMutex;

public:
   RSlotStack() = delete;
   RSlotStack(unsigned int size);
   void ReturnSlot(unsigned int slotNumber);
   unsigned int GetSlot();
};

/// A RAII object to pop and push slot numbers from a RSlotStack object.
/// After construction the slot number is available as the data member fSlot.
struct RSlotStackRAII {
   ROOT::Internal::RSlotStack &fSlotStack;
   unsigned int fSlot;
   RSlotStackRAII(ROOT::Internal::RSlotStack &slotStack) : fSlotStack(slotStack), fSlot(slotStack.GetSlot()) {}
   ~RSlotStackRAII() { fSlotStack.ReturnSlot(fSlot); }
};

} // namespace Internal
} // namespace ROOT

#endif

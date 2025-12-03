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

#include <atomic>
#include <vector>

namespace ROOT {
namespace Internal {

/// A thread-safe list of N indexes (0 to size - 1).
/// RSlotStack can be used to atomically assign a "processing slot" number to
/// each thread in multi-threaded applications.
/// When there are no more slots available, the thread busy-waits for a slot.
/// This case should be avoided by the scheduler.
class RSlotStack {
   struct alignas(8) AtomicWrapper {
      std::atomic_bool fAtomic{false};
      AtomicWrapper() = default;
      ~AtomicWrapper() = default;
      AtomicWrapper(const AtomicWrapper &) = delete;
      AtomicWrapper &operator=(const AtomicWrapper &) = delete;
      AtomicWrapper(AtomicWrapper &&other) { fAtomic = other.fAtomic.load(); }
      AtomicWrapper &operator=(AtomicWrapper &&other)
      {
         fAtomic = other.fAtomic.load();
         return *this;
      }
   };
   std::vector<AtomicWrapper> fSlots;

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
   const unsigned int fSlot;
   RSlotStackRAII(ROOT::Internal::RSlotStack &slotStack) : fSlotStack(slotStack), fSlot(slotStack.GetSlot()) {}
   ~RSlotStackRAII() { fSlotStack.ReturnSlot(fSlot); }
};

} // namespace Internal
} // namespace ROOT

#endif

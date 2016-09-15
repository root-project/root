// @(#)root/thread
// Author: Danilo Piparo, 2016

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSpinMutex
#define ROOT_TSpinMutex

#include <atomic>

namespace ROOT {
   /// A spin mutex class which respects the STL interface.
   class TSpinMutex {

   private:
      std::atomic_flag fAFlag = ATOMIC_FLAG_INIT;

   public:
      TSpinMutex() = default;
      TSpinMutex(const TSpinMutex&) = delete;
      ~TSpinMutex() = default;
      TSpinMutex& operator=(const TSpinMutex&) = delete;

      void lock() { while (fAFlag.test_and_set(std::memory_order_acquire)); }
      void unlock() { fAFlag.clear(std::memory_order_release); }
      bool try_lock() { return !fAFlag.test_and_set(std::memory_order_acquire); }

   };
}

#endif

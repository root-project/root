// @(#)root/meta:$Id$
// Author: Rene Brun   07/01/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSpinLockGuard
#define ROOT_TSpinLockGuard

#include <atomic>

namespace ROOT {
namespace Internal {

/**
* \class ROOT::Internal::TSpinLockGuard
* \brief A spin mutex-as-code-guard class.
* \ingroup Foundation
* This class allows to acquire spin locks in combination with a std::atomic_flag variable.
* For example:
* ~~~{.cpp}
* mutable std::atomic_flag fSpinLock;
* [...]
* ROOT::Internal::TSpinLockGuard slg(fSpinLock);
* // do something important
* [...]
* ~~~{.cpp}
*/

class TSpinLockGuard {
   // Trivial spin lock guard
public:
   TSpinLockGuard(std::atomic_flag& aflag) : fAFlag(aflag)
   {
      while (fAFlag.test_and_set(std::memory_order_acquire));
   }
   ~TSpinLockGuard() {
      fAFlag.clear(std::memory_order_release);
   }

private:
   std::atomic_flag& fAFlag;
};

} // namespace Internal
} // namespace ROOT

#endif // ROOT_TSpinLockGuard

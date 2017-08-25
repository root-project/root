// @(#)root/thread:$Id$
// Author: Danilo Piparo August 2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTaskGroup
#define ROOT_TTaskGroup

#include "RConfigure.h"

#include <atomic>
#include <functional>

// exclude in case ROOT does not have IMT support
#ifndef R__USE_IMT
// No need to error out for dictionaries.
#if !defined(__ROOTCLING__) && !defined(G__DICTIONARY)
#error "Cannot use ROOT::Experimental::TTaskGroup without defining R__USE_IMT."
#endif
#else

namespace ROOT {
namespace Experimental {

class TTaskGroup {
   /**
   \class ROOT::Experimental::TTaskGroup
   \ingroup Parallelism
   \brief A class to manage the asynchronous execution of work items.

   A TTaskGroup represents concurrent execution of a group of tasks. Tasks may be dynamically added to the group as it
   is executing.
   */
private:
   using TaskContainerPtr_t = void *; /// Shield completely from implementation
   TaskContainerPtr_t fTaskContainer{nullptr};
   std::atomic<bool> fCanRun{true};

public:
   TTaskGroup();
   TTaskGroup(TTaskGroup &&other);
   TTaskGroup(const TTaskGroup &) = delete;
   TTaskGroup &operator=(TTaskGroup &&other);
   ~TTaskGroup();

   void Run(const std::function<void(void)> &closure);

   void Wait();
};
}
}

#endif

#endif

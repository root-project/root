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

#include <atomic>
#include <functional>

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
   void *fTaskContainer{nullptr};
   std::atomic<bool> fCanRun{true};
   void ExecuteInIsolation(const std::function<void(void)> &operation);

public:
   TTaskGroup();
   TTaskGroup(TTaskGroup &&other);
   TTaskGroup(const TTaskGroup &) = delete;
   TTaskGroup &operator=(TTaskGroup &&other);
   ~TTaskGroup();

   void Cancel();
   void Run(const std::function<void(void)> &closure);
   void Wait();
};
} // namespace Experimental
} // namespace ROOT

#endif

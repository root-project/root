// @(#)root/thread:$Id$
// Author: Danilo Piparo August 2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RConfigure.h"

#include "ROOT/TTaskGroup.hxx"

#ifdef R__USE_IMT
#include "TROOT.h"
#include "tbb/task_group.h"
#endif

#include <type_traits>

/**
\class ROOT::Experimental::TTaskGroup
\ingroup Parallelism
\brief A class to manage the asynchronous execution of work items.

A TTaskGroup represents concurrent execution of a group of tasks. Tasks may be dynamically added to the group as it is
executing.
*/

namespace ROOT {

namespace Experimental {

// in the constructor and destructor the casts are present in order to be able
// to be independent from the runtime used.
// This leaves the door open for other TTaskGroup implementations.

TTaskGroup::TTaskGroup()
{
#ifdef R__USE_IMT
   if (!ROOT::IsImplicitMTEnabled()) {
      throw std::runtime_error("Implicit parallelism not enabled. Cannot instantiate a TTaskGroup.");
   }
   fTaskContainer = ((TaskContainerPtr_t *)new tbb::task_group());
#endif
}

TTaskGroup::TTaskGroup(TTaskGroup &&other)
{
   *this = std::move(other);
}

TTaskGroup &TTaskGroup::operator=(TTaskGroup &&other)
{
   fTaskContainer = other.fTaskContainer;
   other.fTaskContainer = nullptr;
   fCanRun.store(other.fCanRun);
   return *this;
}

TTaskGroup::~TTaskGroup()
{
#ifdef R__USE_IMT
   if (!fTaskContainer)
      return;
   Wait();
   delete ((tbb::task_group *)fTaskContainer);
#endif
}

/////////////////////////////////////////////////////////////////////////////
/// Add to the group an item of work which will be ran asynchronously.
/// Adding many small items of work to the TTaskGroup is not efficient,
/// unless they run for long enough. If the work to be done is little, look
/// try to express nested parallelism or resort to other constructs such as
/// the TThreadExecutor.
/// Trying to add a work item to the group while it is in waiting state
/// makes the method block.
void TTaskGroup::Run(const std::function<void(void)> &closure)
{
#ifdef R__USE_IMT
   while (!fCanRun)
      /* empty */;

   ((tbb::task_group *)fTaskContainer)->run(closure);
#else
   closure();
#endif
}

/////////////////////////////////////////////////////////////////////////////
/// Wait until all submitted items of work are completed. This method
/// is blocking.
void TTaskGroup::Wait()
{
#ifdef R__USE_IMT
   fCanRun = false;
   ((tbb::task_group *)fTaskContainer)->wait();
   fCanRun = true;
#endif
}
}
}

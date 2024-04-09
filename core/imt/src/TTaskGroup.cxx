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
#define TBB_PREVIEW_ISOLATED_TASK_GROUP 1
#include "tbb/task_group.h"
#include "tbb/task_arena.h"
#include "ROOT/RTaskArena.hxx"
#include "ROpaqueTaskArena.hxx"
#endif

#include <type_traits>
#include <stdexcept>

/**
\class ROOT::Experimental::TTaskGroup
\ingroup Parallelism
\brief A class to manage the asynchronous execution of work items.

A TTaskGroup represents concurrent execution of a group of tasks.
Tasks may be dynamically added to the group as it is executing.
Nesting TTaskGroup instances may result in a runtime overhead.
*/

namespace ROOT {

namespace Internal {

#ifdef R__USE_IMT
tbb::isolated_task_group *CastToTG(void *p)
{
   return (tbb::isolated_task_group *)p;
}
#endif

} // namespace Internal

namespace Experimental {

using namespace ROOT::Internal;

// in the constructor and destructor the casts are present in order to be able
// to be independent from the runtime used.
// This leaves the door open for other TTaskGroup implementations.

TTaskGroup::TTaskGroup()
{
#ifdef R__USE_IMT
   if (!ROOT::IsImplicitMTEnabled()) {
      throw std::runtime_error("Implicit parallelism not enabled. Cannot instantiate a TTaskGroup.");
   }
   fTaskArenaW = ROOT::Internal::GetGlobalTaskArena();
   fTaskContainer = ((void *)new tbb::isolated_task_group());
#endif
}

TTaskGroup::TTaskGroup(TTaskGroup &&other)
{
   *this = std::move(other);
}

TTaskGroup &TTaskGroup::operator=(TTaskGroup &&other)
{
   fTaskArenaW = other.fTaskArenaW;
   fTaskContainer = other.fTaskContainer;
   other.fTaskContainer = nullptr;
   return *this;
}

TTaskGroup::~TTaskGroup()
{
#ifdef R__USE_IMT
   if (!fTaskContainer)
      return;
   Wait();
   delete CastToTG(fTaskContainer);
#endif
}

/////////////////////////////////////////////////////////////////////////////
/// Run operation in the internal task arena to implement work isolation, i.e.
/// prevent stealing of work items spawned by ancestors.
void TTaskGroup::ExecuteInIsolation(const std::function<void(void)> &operation)
{
#ifdef R__USE_IMT
   fTaskArenaW->Access().execute([&] { operation(); });
#else
   operation();
#endif
}

/////////////////////////////////////////////////////////////////////////////
/// Cancel all submitted tasks immediately.
void TTaskGroup::Cancel()
{
#ifdef R__USE_IMT
   ExecuteInIsolation([&] { CastToTG(fTaskContainer)->cancel(); });
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
   ExecuteInIsolation([&] { CastToTG(fTaskContainer)->run(closure); });
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
   ExecuteInIsolation([&] { CastToTG(fTaskContainer)->wait(); });
#endif
}
} // namespace Experimental
} // namespace ROOT

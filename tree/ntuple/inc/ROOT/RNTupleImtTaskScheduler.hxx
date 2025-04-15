/// \file ROOT/RNTupleImtTaskScheduler.hxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2024-02-19
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_RNTupleImtTaskScheduler
#define ROOT_RNTupleImtTaskScheduler

#ifdef R__USE_IMT

#include <ROOT/RPageStorage.hxx>
#include <ROOT/TTaskGroup.hxx>

#include <functional>
#include <utility>

namespace ROOT {
namespace Experimental {

namespace Internal {

class RNTupleImtTaskScheduler : public ROOT::Internal::RPageStorage::RTaskScheduler {
private:
   TTaskGroup fTaskGroup;

public:
   RNTupleImtTaskScheduler() = default;
   ~RNTupleImtTaskScheduler() override = default;
   void AddTask(const std::function<void(void)> &taskFunc) final { fTaskGroup.Run(taskFunc); }
   void Wait() final { fTaskGroup.Wait(); }
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif // R__USE_IMT
#endif // ROOT_RNTupleImtTaskScheduler

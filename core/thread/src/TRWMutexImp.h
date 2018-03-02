// Author: Philippe Canal, 2017

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRWMutexImp
#define ROOT_TRWMutexImp

#include "TVirtualRWMutex.h"
#include "ROOT/TSpinMutex.hxx"
#include "ROOT/TReentrantRWLock.hxx"

#include "TBuffer.h" // Needed by ClassDefInlineOverride

namespace ROOT {
template <typename MutexT, typename RecurseCountsT = ROOT::Internal::RecurseCounts>
class TRWMutexImp : public TVirtualRWMutex {
   ROOT::TReentrantRWLock<MutexT, RecurseCountsT> fMutexImp;

public:
   Hint_t * ReadLock() override;
   void ReadUnLock(Hint_t *) override;
   Hint_t * WriteLock() override;
   void WriteUnLock(Hint_t *) override;

   TVirtualRWMutex *Factory(Bool_t /*recursive*/ = kFALSE) override;
   std::unique_ptr<State> GetStateBefore() override;
   std::unique_ptr<StateDelta> Rewind(const State &earlierState) override;
   void Apply(std::unique_ptr<StateDelta> &&delta) override;

   ClassDefInlineOverride(TRWMutexImp,0)  // Concrete RW mutex lock class
};

} // namespace ROOT.

#endif
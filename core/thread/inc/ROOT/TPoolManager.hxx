// @(#)root/thread:$Id$
// Author: Xavier Valls January 2017

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TPoolManager
#define ROOT_TPoolManager

#include "RConfigure.h"
#include "Rtypes.h"

// exclude in case ROOT does not have IMT support
#ifndef R__USE_IMT
// No need to error out for dictionaries.
# if !defined(__ROOTCLING__) && !defined(G__DICTIONARY)
#  error "Cannot use ROOT::TPoolManager without defining R__USE_IMT."
# endif
#else

#include<memory>

namespace tbb {
   class task_scheduler_init;
}

namespace ROOT {
   class TPoolManager {
   public:
      TPoolManager(UInt_t nThreads = 0);
      ~TPoolManager();
      static UInt_t GetNThreads();
   private:
      static UInt_t fgPoolSize;
      bool mustDelete = true;
      tbb::task_scheduler_init *fSched = nullptr;
   };
}


std::weak_ptr<ROOT::TPoolManager> &GetWP();


inline std::shared_ptr<ROOT::TPoolManager> GetPoolManager(UInt_t nThreads = 0)
{

   if (GetWP().lock() == nullptr) {
      auto shared = std::make_shared<ROOT::TPoolManager>(nThreads);
      GetWP() = shared;
      return GetWP().lock();
   }
   return GetWP().lock();
}


#endif   // R__USE_IMT

#endif

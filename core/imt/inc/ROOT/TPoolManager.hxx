// @(#)root/thread:$Id$
// Author: Xavier Valls January 2017

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPoolManager                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


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
   namespace Internal {
      /**
      \class ROOT::TPoolManager
      \ingroup TPoolManager
      \brief A manager for the scheduler behind ROOT multithreading operations.

      A manager for the multithreading scheduler that solves undefined behaviours and interferences between
      classes and functions that made direct use of the scheduler, such as EnableImplicitMT() ot TThreadExecutor.
      */

      class TPoolManager {
      public:
         friend std::shared_ptr<TPoolManager> GetPoolManager(UInt_t nThreads);
         /// Returns the number of threads running when the scheduler has been instantiated within ROOT.
         static UInt_t GetPoolSize();
         /// Terminates the scheduler instantiated within ROOT.
         ~TPoolManager();
      private:
         ///Initializes the scheduler within ROOT. If the scheduler has already been initialized by the
         /// user before invoking the constructor it won't change its behaviour and it won't terminate it,
         /// but it will still keep record of the number of threads passed as a parameter.
         TPoolManager(UInt_t nThreads = 0);
         static UInt_t fgPoolSize;
         bool mustDelete = true;
         tbb::task_scheduler_init *fSched = nullptr;
      };
      /// Get a shared pointer to the manager. Initialize the manager with nThreads if not active. If active,
      /// the number of threads, even if specified otherwise, will remain the same.
      ///
      /// The number of threads will be able to change calling the factory function again after the last
      /// remaining shared_ptr owning the object is destroyed or reasigned, which will trigger the destructor of the manager.
      std::shared_ptr<TPoolManager> GetPoolManager(UInt_t nThreads = 0);
   }
}

#endif   // R__USE_IMT

#endif

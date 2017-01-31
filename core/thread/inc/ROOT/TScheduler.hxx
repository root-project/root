// @(#)root/thread:$Id$
// Author: Xavier Valls January 2017

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef ROOT_TScheduler
#define ROOT_TScheduler

#include "RConfigure.h"

// exclude in case ROOT does not have IMT support
#ifndef R__USE_IMT
// No need to error out for dictionaries.
# if !defined(__ROOTCLING__) && !defined(G__DICTIONARY)
#  error "Cannot use ROOT::TScheduler without defining R__USE_IMT."
# endif
#else

namespace ROOT{
   namespace Internal{
     class TScheduler{
     public:
         TScheduler();
         ~TScheduler();
         static unsigned GetNThreads();
         static unsigned GetPoolSize();
         static unsigned GetNSubscribers();
         void Subscribe();
         void Subscribe(unsigned nThreads);
         void Unsubscribe();
     private:
         static unsigned fgSubscriptionsCount;
         static unsigned fgPoolSize;
         bool fSubscribed = false;
     };
  }
}

#endif   // R__USE_IMT

#endif

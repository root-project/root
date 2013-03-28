// @(#)root/thread:$Id$
// Author: Fons Rademakers   01/07/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TPosixThreadFactory                                                  //
//                                                                      //
// This is a factory for Posix thread components.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TPosixThreadFactory.h"
#include "TPosixMutex.h"
#include "TPosixCondition.h"
#include "TPosixThread.h"

// Force creation of TPosixThreadFactory when shared library will be loaded
// (don't explicitly create a TPosixThreadFactory).
static TPosixThreadFactory gPosixThreadFactoryCreator;

ClassImp(TPosixThreadFactory)

//______________________________________________________________________________
TPosixThreadFactory::TPosixThreadFactory(const char *name, const char *title) :
                     TThreadFactory(name, title)
{
   // Create Posix thread factory. Also sets global gThreadFactory to this.

   gThreadFactory = this;
}

//______________________________________________________________________________
TMutexImp *TPosixThreadFactory::CreateMutexImp(Bool_t recursive=kFALSE)
{
   // Return a Posix Mutex.

   return new TPosixMutex(recursive);
}

//______________________________________________________________________________
TThreadImp *TPosixThreadFactory::CreateThreadImp()
{
   // Return a Posix thread.

   return new TPosixThread;
}

//______________________________________________________________________________
TConditionImp *TPosixThreadFactory::CreateConditionImp(TMutexImp *m)
{
   // Return a Posix condition variable.

   return new TPosixCondition(m);
}

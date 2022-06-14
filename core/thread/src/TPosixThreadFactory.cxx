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

ClassImp(TPosixThreadFactory);

////////////////////////////////////////////////////////////////////////////////
/// Create Posix thread factory. Also sets global gThreadFactory to this.

TPosixThreadFactory::TPosixThreadFactory(const char *name, const char *title) :
                     TThreadFactory(name, title)
{
   gThreadFactory = this;
}

////////////////////////////////////////////////////////////////////////////////
/// Return a Posix Mutex.

TMutexImp *TPosixThreadFactory::CreateMutexImp(Bool_t recursive=kFALSE)
{
   return new TPosixMutex(recursive);
}

////////////////////////////////////////////////////////////////////////////////
/// Return a Posix thread.

TThreadImp *TPosixThreadFactory::CreateThreadImp()
{
   return new TPosixThread;
}

////////////////////////////////////////////////////////////////////////////////
/// Return a Posix condition variable.

TConditionImp *TPosixThreadFactory::CreateConditionImp(TMutexImp *m)
{
   return new TPosixCondition(m);
}

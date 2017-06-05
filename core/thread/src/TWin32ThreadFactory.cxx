// @(#)root/thread:$Id$
// Author: Bertrand Bellenot  20/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWin32ThreadFactory                                                  //
//                                                                      //
// This is a factory for Win32 thread components.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TWin32ThreadFactory.h"
#include "TWin32Mutex.h"
#include "TWin32Condition.h"
#include "TWin32Thread.h"

// Force creation of TWin32ThreadFactory when shared library will be loaded
// (don't explicitly create a TWin32ThreadFactory).
static TWin32ThreadFactory gWin32ThreadFactoryCreator;

ClassImp(TWin32ThreadFactory);

////////////////////////////////////////////////////////////////////////////////
/// Create Win32 thread factory. Also sets global gThreadFactory to this.

TWin32ThreadFactory::TWin32ThreadFactory(const char *name, const char *title) :
                     TThreadFactory(name, title)
{
   gThreadFactory = this;
}

////////////////////////////////////////////////////////////////////////////////
/// Return a Win32 Mutex.

TMutexImp *TWin32ThreadFactory::CreateMutexImp(Bool_t recursive)
{
   return new TWin32Mutex(recursive);
}

////////////////////////////////////////////////////////////////////////////////
/// Return a Win32 thread.

TThreadImp *TWin32ThreadFactory::CreateThreadImp()
{
   return new TWin32Thread;
}

////////////////////////////////////////////////////////////////////////////////
/// Return a Win32 condition variable.

TConditionImp *TWin32ThreadFactory::CreateConditionImp(TMutexImp *m)
{
   return new TWin32Condition(m);
}

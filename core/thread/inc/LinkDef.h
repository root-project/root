/* @(#)root/thread:$Id$ */

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifdef __CINT__

#pragma link off all globals;
#pragma link off all classes;
#pragma link off all functions;

#pragma link C++ class TThread;
#pragma link C++ class TConditionImp;
#pragma link C++ class TCondition;
#pragma link C++ class TMutex;
#pragma link C++ class ROOT::TSpinMutex;
#pragma link C++ class TMutexImp;
#ifndef _WIN32
#pragma link C++ class TPosixCondition;
#pragma link C++ class TPosixMutex;
#pragma link C++ class TPosixThread;
#pragma link C++ class TPosixThreadFactory;
#else
#pragma link C++ class TWin32Condition;
#pragma link C++ class TWin32Mutex;
#pragma link C++ class TWin32Thread;
#pragma link C++ class TWin32ThreadFactory;
#endif
#pragma link C++ class TSemaphore;
#pragma link C++ class TThreadFactory;
#pragma link C++ class TThreadImp;
#pragma link C++ class TRWLock;
#pragma link C++ class TAtomicCount;

#endif

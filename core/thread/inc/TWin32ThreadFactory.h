// @(#)root/thread:$Id$
// Author: Bertrand Bellenot  20/10/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TWin32ThreadFactory
#define ROOT_TWin32ThreadFactory

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TWin32ThreadFactory                                                  //
//                                                                      //
// This is a factory for Win32 thread components.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TThreadFactory
#include "TThreadFactory.h"
#endif

class TMutexImp;
class TConditionImp;
class TThreadImp;


class TWin32ThreadFactory : public TThreadFactory {

public:
   TWin32ThreadFactory(const char *name = "Win32", const char *title = "Win32 Thread Factory");
   virtual ~TWin32ThreadFactory() { }

   virtual TMutexImp      *CreateMutexImp(Bool_t recursive);
   virtual TConditionImp  *CreateConditionImp(TMutexImp *m);
   virtual TThreadImp     *CreateThreadImp();

   ClassDef(TWin32ThreadFactory,0)  // Win32 thread factory
};

#endif

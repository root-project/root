#ifndef ROOT_TWaitCondition
#define ROOT_TWaitCondition

// @(#)root/qt:$Id: TWaitCondition.h,v 1.3 2005/03/04 07:11:54 brun Exp $
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <limits.h>
#include "TQtRConfig.h"
#ifdef R__QTGUITHREAD
#include "TWin32Semaphore.h"

class TWaitCondition : public TWin32Semaphore 
{
   public:
     TWaitCondition() : TWin32Semaphore() {}
     ~TWaitCondition() {}
     bool wait (unsigned long time= ULONG_MAX) { Wait(); return TRUE;}
     void wakeOne () { Release(); }
};
#else
// An dummy version for the "non-thread" implementations
class TWaitCondition
{
   public:
     TWaitCondition()  {}
     ~TWaitCondition() {}
	  bool wait (unsigned long time=ULONG_MAX ) { if (time) {/* Wait() */}  return TRUE;}
     void wakeOne () { /* Release();*/  }
};

#endif
#endif

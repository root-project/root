// @(#)root/qt:$Name:$:$Id:$
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRootMainThread
#define ROOT_TRootMainThread

#include "TQtRootThread.h"

class TRootMainThread : public TQtRootThread {
  public:
   TRootMainThread();
   ~TRootMainThread(){;}

  protected:
   virtual void Run();
   virtual bool ProcessThreadMessage(void *message,bool synch);
};

#endif

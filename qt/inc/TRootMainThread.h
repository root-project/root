#ifndef ROOT_TROOTMAINTHREAD
#define ROOT_TROOTMAINTHREAD

// Author: Valeri Fine   21/01/2002
/****************************************************************************
** $Id: TRootMainThread.h,v 1.1.1.1 2002/03/27 18:17:02 fine Exp $
**
** Copyright (C) 2002 by Valeri Fine.  All rights reserved.
**
** This file may be distributed under the terms of the Q Public License
** as defined by Trolltech AS of Norway and appearing in the file
** LICENSE.QPL included in the packaging of this file.
*****************************************************************************/

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

// @(#)root/qt:$Name:  $:$Id: TQtTimer.cxx,v 1.1 2004/08/13 06:21:09 brun Exp $
// Author: Valery Fine  09/08/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <qapplication.h>
#include "TQtTimer.h"
#include "TSystem.h"


TQtTimer *TQtTimer::fgQTimer=0;

//______________________________________________________________________________
void TQtTimer::AwakeRootEvent(){
   gSystem->DispatchOneEvent(kTRUE);
}

//______________________________________________________________________________
TQtTimer * TQtTimer::Create(QObject *parent, const char *name)
{
   qApp->lock();
   if (!fgQTimer) {
      fgQTimer = new  TQtTimer(parent,name);
      qApp->unlock();
      connect(fgQTimer,SIGNAL(timeout()),fgQTimer,SLOT(AwakeRootEvent()) );
   } else {
      qApp->unlock();
   }
   return fgQTimer;
}

// @(#)root/qt:$Name:  $:$Id: TQtWidget.cxx,v 1.10 2004/09/12 11:00:22 brun Exp $
// Author: Valeri Fine   23/01/2003

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2003 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <qapplication.h>
#include "TQtTimer.h"
#include "TSystem.h"

////////////////////////////////////////////////////////////////////////////////
//
//
////////////////////////////////////////////////////////////////////////////////

TQtTimer *TQtTimer::fgQTimer=0;
//______________________________________________________________________________
void TQtTimer::AwakeRootEvent(){
     // proceess the ROOT events inside of Qt event loop
     gSystem->DispatchOneEvent(kFALSE);
   start(300,TRUE);
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

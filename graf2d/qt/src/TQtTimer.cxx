// @(#)root/qt:$Id$
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
// TQtTimer is a singelton singleshot QTimer to awake the ROOT event loop from Qt event loop
//
////////////////////////////////////////////////////////////////////////////////

ClassImp(TQtTimer)

TQtTimer *TQtTimer::fgQTimer=0;
////////////////////////////////////////////////////////////////////////////////
/// proceess the ROOT events inside of Qt event loop

void TQtTimer::AwakeRootEvent(){
     gSystem->DispatchOneEvent(kFALSE);
     start(5);
}
////////////////////////////////////////////////////////////////////////////////
/// Create a singelton object TQtTimer

TQtTimer * TQtTimer::Create(QObject *parent)
{
   if (!fgQTimer) {
      fgQTimer = new  TQtTimer(parent);
      fgQTimer->setSingleShot(true);
      connect(fgQTimer,SIGNAL(timeout()),fgQTimer,SLOT(AwakeRootEvent()) );
   }
   return fgQTimer;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the singelton TQtTimer object

TQtTimer *TQtTimer::QtTimer()
{
   return fgQTimer;
}


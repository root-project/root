// @(#)root/qtgsi:$Name:$:$Id:$
// Author: Denis Bertini, M. AL-Turany  01/11/2000

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TQRootApplication
#define ROOT_TQRootApplication

//////////////////////////////////////////////////////////////////////
//
//  TQRootApplication
//
//  This class creates Qt environement that will
//  interface with the ROOT windowing system eventloop and eventhandlers,
//  via a polling mechanism.
//
///////////////////////////////////////////////////////////////////////

#ifndef ROOT_Riostream
#include "Riostream.h"
#endif
#include "qapplication.h"
#include "qobject.h"
#include "qtimer.h"
#ifndef ROOT_TTimer
#include "TTimer.h"
#endif

class TQRootApplication : public QApplication {
   Q_OBJECT
protected:
   QTimer *fQTimer;                    // Qt timer that poll the event loop of ROOT
   TTimer *fRTimer;                    // Root timer
public:
   static Bool_t fgDebug, fgWarning;   // debug and warning flags

   TQRootApplication(int argc, char **argv,int poll=0);
   ~TQRootApplication();
   void SetDebugOn(){ fgDebug=kTRUE; }
   void SetWarningOn(){ fgWarning=kTRUE;}
public slots:
   void Execute();
   void Quit();
};

#endif

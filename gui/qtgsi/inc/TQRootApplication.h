// @(#)root/qtgsi:$Id$
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
//  This class creates Qt environment that will
//  interface with the ROOT windowing system eventloop and eventhandlers,
//  via a polling mechanism.
//
///////////////////////////////////////////////////////////////////////

#ifndef __CINT__
#include "qapplication.h"
#include "qobject.h"
#include "qtimer.h"
#endif

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

class TTimer;
class QApplication;
class QTimer;

class TQRootApplication : public QApplication {
#ifndef __CINT__
   Q_OBJECT
#endif
private:
   TQRootApplication(const TQRootApplication &);
   TQRootApplication& operator=(const TQRootApplication &);
protected:
   QTimer *fQTimer;                    // Qt timer that poll the event loop of ROOT
   TTimer *fRTimer;                    // Root timer
public:
   static Bool_t fgDebug, fgWarning;   // debug and warning flags

   TQRootApplication(int &argc, char **argv,int poll=0);
   ~TQRootApplication();
   void SetDebugOn(){ fgDebug=kTRUE; }
   void SetWarningOn(){ fgWarning=kTRUE;}
public slots:
   void Execute();
   void Quit();
   
public:
   ClassDef(TQRootApplication,1)  //creates Qt environment interface with the ROOT windowing system
};

#endif

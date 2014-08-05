// @(#)root/gt:$Id$
// Author: Valery Fine      18/01/2007

/****************************************************************************
** $Id$
**
** Copyright (C) 2007 by Valeri Fine. Brookhaven National Laboratory.
**                                    All rights reserved.
**
**
*****************************************************************************/
///////////////////////////////////////////////////////////////////////////
//
// The TQRootSlot singleton class introduces the global SLOT to invoke
// the  ROOT command line from the GUI signals
// Optionally one can execute TApplication::Terminate method directly
//
// It provides a Qt slot to attach the the CINT C++ interpreter
// to any Qt signal
// To execute any C++ statement from the GUI one should connect
// one's Qt signal with the Qt slot of the global instance of this class
//
//  connect(GUI object, SIGNAL(const char *editedLine),TQtRootSlot::CintSlot(),SLOT(ProcessLine(const char*)))
//
//  To terminate the ROOT from Qt GUI element connect the signal with
//  the Terminate  or TerminateAndQuite slot.
//  For example to terminate ROOT and Qt smoothly do
//
//  connect(qApp,SIGNAL(lastWindowClosed()),TQtRootSlot::CintSlot(),SLOT(TerminateAndQuit())
//
//  To terminate just ROOT (in case the Qt is terminated by the other means)
//  connect(qApp,SIGNAL(lastWindowClosed()),TQtRootSlot::CintSlot(),SLOT(Terminate())
//
///////////////////////////////////////////////////////////////////////////

#include "TQtRootSlot.h"
#include "TROOT.h"
#include "TApplication.h"
#include "TInterpreter.h"
#include <qapplication.h>
#include <QString>

TQtRootSlot *TQtRootSlot::fgTQtRootSlot = 0;
//____________________________________________________
TQtRootSlot *TQtRootSlot::CintSlot()
{
   // create and return the singleton
   if (!fgTQtRootSlot) fgTQtRootSlot = new TQtRootSlot();
   return fgTQtRootSlot;
}
//____________________________________________________
void TQtRootSlot::EndOfLine()
{
   // slot to perform the standard "EndOfLine" ROOT action
   // it used to update the current gPad
   if (gInterpreter)  gInterpreter->EndOfLineAction();
}

//____________________________________________________
void TQtRootSlot::ProcessLine(const QString &command)
{
     // execute the arbitrary ROOT /CINt command via
     // CINT C++ interpreter and emit the result
   std::string cmd = command.toStdString();
   ProcessLine(cmd.c_str());
}

//____________________________________________________
void TQtRootSlot::ProcessLine(const char *command)
{
     // execute the arbitrary ROOT /CINt command via
     // CINT C++ interpreter and emit the result
     int error;
     gROOT->ProcessLine(command,&error);
     emit Error(error);
}
//____________________________________________________
void TQtRootSlot::Terminate(int status)const
{
   // the dedicated slot to terminate the ROOT application
   // with "status"
   if (gApplication) gApplication->Terminate(status);
}

//____________________________________________________
void TQtRootSlot::Terminate()const
{
   // the dedicated slot to terminate the ROOT application
   // and return the "0" status
   Terminate(0);
}

//____________________________________________________
void TQtRootSlot::TerminateAndQuit() const
{
    // the dedicated  slot to terminate the ROOT application
    // and quit the Qt Application if any

   Bool_t rtrm = kTRUE;
   if (gApplication) {
      rtrm = gApplication->ReturnFromRun();
      gApplication->SetReturnFromRun(kTRUE);
      gApplication->Terminate(0);
   }
   if (qApp) qApp->quit();
   else if (!rtrm && gApplication ) {
      gApplication->SetReturnFromRun(rtrm);
      // to make sure the ROOT event loop is terminated
      gROOT->ProcessLine(".q");
   }
}

//__________________________________________________________________
bool QConnectCint(const QObject * sender, const char * signal)
{
   // Connect the Qt signal to the "execute C++ statement" via CINT SLOT
   // The first parameter of the Qt signal must be "const char*"
   return
   QObject::connect(sender,signal
      ,TQtRootSlot::CintSlot(),SLOT(ProcessLine(const char*)));
}

//__________________________________________________________________
bool QConnectTerminate(const QObject * sender, const char * signal)
{
   // Connect the Qt signal to the "TApplication::Terminate" method
   // Any extra parameters of the Qt signal are discarded
   return
   QObject::connect(sender,signal
      ,TQtRootSlot::CintSlot(),SLOT(Terminate()));
}

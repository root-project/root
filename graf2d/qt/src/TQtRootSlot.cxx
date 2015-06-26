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
////////////////////////////////////////////////////////////////////////////////
/// create and return the singleton

TQtRootSlot *TQtRootSlot::CintSlot()
{
   if (!fgTQtRootSlot) fgTQtRootSlot = new TQtRootSlot();
   return fgTQtRootSlot;
}
////////////////////////////////////////////////////////////////////////////////
/// slot to perform the standard "EndOfLine" ROOT action
/// it used to update the current gPad

void TQtRootSlot::EndOfLine()
{
   if (gInterpreter)  gInterpreter->EndOfLineAction();
}

////////////////////////////////////////////////////////////////////////////////
/// execute the arbitrary ROOT /CINt command via
/// CINT C++ interpreter and emit the result

void TQtRootSlot::ProcessLine(const QString &command)
{
   std::string cmd = command.toStdString();
   ProcessLine(cmd.c_str());
}

////////////////////////////////////////////////////////////////////////////////
/// execute the arbitrary ROOT /CINt command via
/// CINT C++ interpreter and emit the result

void TQtRootSlot::ProcessLine(const char *command)
{
     int error;
     gROOT->ProcessLine(command,&error);
     emit Error(error);
}
////////////////////////////////////////////////////////////////////////////////
/// the dedicated slot to terminate the ROOT application
/// with "status"

void TQtRootSlot::Terminate(int status)const
{
   if (gApplication) gApplication->Terminate(status);
}

////////////////////////////////////////////////////////////////////////////////
/// the dedicated slot to terminate the ROOT application
/// and return the "0" status

void TQtRootSlot::Terminate()const
{
   Terminate(0);
}

////////////////////////////////////////////////////////////////////////////////
/// the dedicated  slot to terminate the ROOT application
/// and quit the Qt Application if any

void TQtRootSlot::TerminateAndQuit() const
{
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

////////////////////////////////////////////////////////////////////////////////
/// Connect the Qt signal to the "execute C++ statement" via CINT SLOT
/// The first parameter of the Qt signal must be "const char*"

bool QConnectCint(const QObject * sender, const char * signal)
{
   return
   QObject::connect(sender,signal
      ,TQtRootSlot::CintSlot(),SLOT(ProcessLine(const char*)));
}

////////////////////////////////////////////////////////////////////////////////
/// Connect the Qt signal to the "TApplication::Terminate" method
/// Any extra parameters of the Qt signal are discarded

bool QConnectTerminate(const QObject * sender, const char * signal)
{
   return
   QObject::connect(sender,signal
      ,TQtRootSlot::CintSlot(),SLOT(Terminate()));
}

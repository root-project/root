// @(#)root/qt:$Name:  $:$Id: TQtApplication.cxx,v 1.5 2005/03/01 07:24:01 brun Exp $
// Author: Valeri Fine   21/01/2002

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * Copyright (C) 2002 by Valeri Fine.                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TQtApplication -  Instantiate the Qt system within ROOT environment  //
//                                                                      //
// Instantiate the Qt package by createing Qapplication object if any   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include <assert.h>
#include "qapplication.h"

#include "TQtApplication.h"
#include "TSystem.h"

#ifdef R__QTGUITHREAD
# include "TQtApplicationThread.h"
# include "TWaitCondition.h"
#endif

#include "TROOT.h"
#include "TEnv.h"
#include "TGClient.h"

TQtApplication *TQtApplication::fgQtApplication = 0;

ClassImp(TQtApplication)
//
//______________________________________________________________________________
TQtApplication::TQtApplication(const char * /*appClassName*/, int argc,char **argv)
                : fGUIThread(0)
{
   assert(!fgQtApplication);
   fgQtApplication  = this;
   CreateGUIThread(argc,argv);
}
//______________________________________________________________________________
TQtApplication::~TQtApplication()
{
#ifdef R__QTGUITHREAD
    // Send WN_QUIT message to GUI thread
    if (fGUIThread)
        PostThreadMessage((DWORD)fGUIThread->GetThreadId(),WM_QUIT,0,0);
#endif
}
//______________________________________________________________________________
void TQtApplication::CreateQApplication(int argc, char ** argv, bool GUIenabled)
{
  //  Initialize the Qt package
  //  Check the QT_BATCH environment variable to disactivate Qt GUI mode
   
  // QApplication must be created in the proper "GUI" thread
  // It may be called from TQtApplicationThread::Run
   if (!qApp) {
      // QApplication::setColorSpec( QApplication::NormalColor );
       QApplication::setColorSpec( QApplication::ManyColor );
       QString display = gSystem->Getenv("DISPLAY");
       // check the QT_BATCH option
       if (display.contains("QT_BATCH")) GUIenabled = false;
       qApp = new QApplication (argc, argv, GUIenabled );
       // The string must be one of the QStyleFactory::keys(),
       // typically one of
       //      "windows", "motif",     "cde",    "motifplus", "platinum", "sgi"
       //  and "compact", "windowsxp", "aqua" or "macintosh"
      QString fromConfig = "native";
      if (gEnv)
         fromConfig = gEnv->GetValue("Gui.Style","native");
      if (fromConfig != "native" ) QApplication::setStyle(fromConfig);
#ifdef Q_WS_MACX
      // create a timer to force the event loop with no X-server
      TTimer *idle = new TTimer(240); idle->TurnOn();
#endif
   }
   // Add Qt plugin path if  present (it is the case for Windows binary ROOT distribution)
   char *qtPluginPath = gSystem->ConcatFileName(gSystem->Getenv("ROOTSYS"),"/Qt/plugins");
   if (!gSystem->AccessPathName(qtPluginPath))
       qApp->addLibraryPath(qtPluginPath);
   delete [] qtPluginPath;
}
//______________________________________________________________________________
void TQtApplication::CreateGUIThread(int argc, char **argv)
{
  // Create GUI thread to Qt event loop
   if (gROOT->IsBatch()) {
     CreateQApplication(argc,argv,kFALSE);
   } else {
#ifdef R__QTGUITHREAD
     TWaitCondition ThrSem;
     fGUIThread = new TQtApplicationThread(argc,argv);
     fGUIThread->SetWait(ThrSem);
     fGUIThread->start();
     // Wait untill the thread initilized
     fGUIThread->Wait();
#else
     CreateQApplication(argc,argv, TRUE);
#endif
   }
}
//______________________________________________________________________________
TQtApplication *TQtApplication::GetQtApplication(){return fgQtApplication;}
//______________________________________________________________________________
bool TQtApplication::Terminate()
{
  // Terminate GUI thread
  if (fgQtApplication) {
    TQtApplication *app = fgQtApplication;
    fgQtApplication = 0;
    delete  app;
  }
  return TRUE;
}
//______________________________________________________________________________
bool TQtApplication::IsThisGuiThread()
{
   // Check whether the current thread belongs the GUI
#ifdef R__QTGUITHREAD
 TQtApplication *app = GetQtApplication();
   if (!app) return true;
   if (app->fGUIThread)
      return app->fGUIThread->IsThisThread();
#endif
  return true;
}

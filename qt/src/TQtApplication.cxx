// @(#)root/qt:$Name:  $:$Id: TQtApplication.cxx,v 1.2 2004/07/28 00:12:41 rdm Exp $
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
// TQtApplication                                                       //
//                                                                      //
// Interface to low level Qt package. This class gives access to basic  //
// Qt graphics, pixmap, text and font handling routines.                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <assert.h>
#include "qapplication.h"

#include "TQtRConfig.h"
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

// ClassImp(TQtApplication)
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
  // QApplication must be created in the proper "GUI" thread
  // It may be called from TQtApplicationThread::Run
   if (!qApp) {
      // QApplication::setColorSpec( QApplication::NormalColor );
       QApplication::setColorSpec( QApplication::ManyColor );
       qApp = new QApplication (argc, argv, GUIenabled );
       // The string must be one of the QStyleFactory::keys(),
       // typically one of
       //      "windows", "motif",     "cde",    "motifplus", "platinum", "sgi"
       //  and "compact", "windowsxp", "aqua" or "macintosh"
      QString fromConfig = "native";
      if (gEnv)
         fromConfig = gEnv->GetValue("Gui.Style","native");
      if (fromConfig != "native" ) QApplication::setStyle(fromConfig);
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

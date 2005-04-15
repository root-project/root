// @(#)root/qt:$Name:  $:$Id: TQtApplication.cxx,v 1.6 2005/03/07 07:44:12 brun Exp $
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

#include "qmessagebox.h"

//________________________________________________________________
static int QVersion(const char *ver) {
   // convert the Qversion string into the interger
    QString version = QString::fromLatin1(ver);
    return   (version.section('.',0,0).toInt()<<16)
          +  (version.section('.',1,1).toInt()<<8 )
          +  (version.section('.',2,2).toInt()    );
}

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
       qApp = new QApplication(argc,argv,GUIenabled);
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
   // Check the compatibility
   Int_t validQtVersion = QVersion(ROOT_VALID_QT_VERSION);
   Int_t thisQtVersion  = QVersion(qVersion());
   if (thisQtVersion < validQtVersion) {
       QString s = QApplication::tr("Executable '%1' was compiled with Qt %2 and requires Qt %3 at least, found Qt %4.")
            .arg(QString::fromLatin1(qAppName()))
            .arg(QString::fromLatin1(QT_VERSION_STR))
            .arg(QString::fromLatin1(ROOT_VALID_QT_VERSION))
            .arg(QString::fromLatin1(qVersion()) ); 
      QMessageBox::critical( 0, QApplication::tr("Incompatible Qt Library Error" ), s, QMessageBox::Abort,0 );
      qFatal(s.ascii());
   } else if (thisQtVersion < QtVersion()) {
       QString s = QApplication::tr("Executable '%1' was compiled with Qt %2, found Qt %3.")
            .arg(QString::fromLatin1(qAppName()))
            .arg(QString::fromLatin1(QT_VERSION_STR))
            .arg(QString::fromLatin1(qVersion()) ); 
      QMessageBox::warning( 0, QApplication::tr("Upgrade Qt Library Warning" ), s, QMessageBox::Abort,0 );
      qWarning(s.ascii());
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
Int_t TQtApplication::QtVersion(){
     // The Qt version the package was compiled with
   return  QVersion(QT_VERSION_STR);
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

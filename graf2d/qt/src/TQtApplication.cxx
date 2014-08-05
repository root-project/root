// @(#)root/qt:$Id$
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
#include "TQtApplication.h"

#include <QApplication>
#include <QCoreApplication>
#include <QDebug>

#include "TSystem.h"

#include "TROOT.h"
#include "TEnv.h"

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
TQtApplication::TQtApplication(const char * /*appClassName*/, int &argc,char **argv)
                : fGUIThread(0)
{
   assert(!fgQtApplication);
   fgQtApplication  = this;
   CreateGUIThread(argc,argv);
}
//______________________________________________________________________________
TQtApplication::~TQtApplication()
{ }
//______________________________________________________________________________
void TQtApplication::CreateQApplication(int &argc, char ** argv, bool GUIenabled)
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
#ifndef R__WIN32
       QCoreApplication::setAttribute(Qt::AA_ImmediateWidgetCreation);
#endif
       // Check whether we want to debug the Qt X11
       QString fatalWarnings = gSystem->Getenv("QT_FATAL_WARNINGS");
       if (fatalWarnings.contains("1")) {
          int argC = 2;
          static const char *argV[] = {"root.exe", "-sync" };
          qDebug() << "TQtApplication::CreateQApplication: "
                   << "ATTENTION !!! "
                   << "The env variable \"QT_FATAL_WARNIGNS\" was defined. The special debug option has  been turned on."
                   << " argc = " << argc << " argv = " << argv[0] << argv[1];
          qDebug() << " You may want to restart ROOT with " << argC << " parameters :"
                   << " like this: \"" << argV[0] << " " << argV[1];
//          new QApplication(argC,argV,GUIenabled);
          new QApplication(argc,argv,GUIenabled);
       } else {
          new QApplication(argc,argv,GUIenabled);
       }
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
            .arg(qAppName())
            .arg(QT_VERSION_STR)
            .arg(QString::fromLatin1(ROOT_VALID_QT_VERSION))
            .arg(QString::fromLatin1(qVersion()) );
      QMessageBox::critical( 0, QApplication::tr("Incompatible Qt Library Error" ), s, QMessageBox::Abort,0 );
      qFatal("%s",s.toAscii().data());
   } else if (thisQtVersion < QtVersion()) {
       QString s = QApplication::tr("Executable '%1' was compiled with Qt %2, found Qt %3.")
            .arg(qAppName())
            .arg(QT_VERSION_STR)
            .arg(QString::fromLatin1(qVersion()) );
      QMessageBox::warning( 0, QApplication::tr("Upgrade Qt Library Warning" ), s, QMessageBox::Abort,0 );
      qWarning("%s",s.toAscii().data());
   }

   // Add Qt plugin path if  present (it is the case for Windows binary ROOT distribution)
   char *qtPluginPath = gSystem->ConcatFileName(gSystem->Getenv("ROOTSYS"),"/Qt/plugins");
   if (!gSystem->AccessPathName(qtPluginPath))
       qApp->addLibraryPath(qtPluginPath);
   delete [] qtPluginPath;
}
//______________________________________________________________________________
void TQtApplication::CreateGUIThread(int &argc, char **argv)
{
  // Create GUI thread to Qt event loop
   if (gROOT->IsBatch()) {
     CreateQApplication(argc,argv,kFALSE);
   } else {
     CreateQApplication(argc,argv, TRUE);
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
  return true;
}

// @(#)root/qtgsi:$Id$
// Author: Denis Bertini, M. Al-Turany  01/11/2000

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TQRootApplication.h"
#include "TSystem.h"
#include <stdlib.h>

bool TQRootApplication::fgDebug=kFALSE;
bool TQRootApplication::fgWarning=kFALSE;

ClassImp(TQRootApplication);

////////////////////////////////////////////////////////////////////////////////

void qMessageOutput( QtMsgType type, const char *msg )
{
   switch ( type ) {
      case QtDebugMsg:
         if(TQRootApplication::fgDebug)
            fprintf( stderr, "QtRoot-Debug: \n %s\n", msg );
         break;
      case QtWarningMsg:
         if(TQRootApplication::fgWarning)
            fprintf( stderr, "QtRoot-Warning: \n %s\n", msg );
         break;
      case QtFatalMsg:
         fprintf( stderr, "QtRoot-Fatal: \n %s\n", msg );
         abort();         // dump core on purpose
         break;
      case QtCriticalMsg:
         fprintf( stderr, "QtRoot-Fatal: \n %s\n", msg );
         abort();         // dump core on purpose
         break;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Connect ROOT via Timer call back.

TQRootApplication::TQRootApplication(int &argc, char **argv, int poll) :
      QApplication(argc,argv), fQTimer(0), fRTimer(0)
{
   if (poll == 0) {
      fQTimer = new QTimer( this );
      QObject::connect( fQTimer, SIGNAL(timeout()),this, SLOT(Execute()) );
      fQTimer->start( 20, FALSE );
      fRTimer = new TTimer(20);
      fRTimer->Start(20, kFALSE);
   }

   // install a msg-handler
   fgWarning = fgDebug = kFALSE;
   qInstallMsgHandler( qMessageOutput );
}

////////////////////////////////////////////////////////////////////////////////
/// dtor

TQRootApplication::~TQRootApplication()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Call the inner loop of ROOT.

void TQRootApplication::Execute()
{
   gSystem->InnerLoop();
}

////////////////////////////////////////////////////////////////////////////////
/// Set a Qt-Specific error handler.

void TQRootApplication::Quit()
{
   gSystem->Exit( 0 );
}

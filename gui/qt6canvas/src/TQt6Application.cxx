// Author: Sergey Linev   30/07/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TQt6Application.h"

#include "TError.h"
#include "TTimer.h"
#include "TApplication.h"

#include <QApplication>

using namespace ROOT::Experimental;

class TQt6EventsTimer : public TTimer {
public:
   TQt6EventsTimer(Long_t milliSec, Bool_t mode) :
      TTimer(milliSec, mode) {}

   /// used to send control messages to clients
   void Timeout() override
   {
      QApplication::sendPostedEvents();
      QApplication::processEvents();
   }
};


/** \class TQt6Application
    \ingroup qt6canvas

Provides Qt6-specific methods for TApplication.
Main purpose - runs Qt event loop
*/


////////////////////////////////////////////////////////////////////////////////
/// Create Qt6 application environment.

TQt6Application::TQt6Application(const char *appClassName,
                                 Int_t *argc, char **argv) : TApplicationImp(appClassName, argc, argv)
{
    CreateQApplication(*argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
/// Delete Qt6 application environment.

TQt6Application::~TQt6Application()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Create QApplication and timer to handle Qt events

void TQt6Application::CreateQApplication([[maybe_unused]] int argc, char **argv)
{
   static QApplication *qapp = nullptr;
   static int qargc = 1;
   static char *qargv[2];

   if (!qapp && !QApplication::instance()) {

      if (argv)
         qargv[0] = argv[0];
      else if (gApplication)
         qargv[0] = gApplication->Argv(0);
      else {
         ::Error("TQt6Application::CreateQApplication", "Not found gApplication to create QApplication");
         return;
      }
      qargv[1] = nullptr;

      qapp = new QApplication(qargc, qargv);
   }

   static TQt6EventsTimer *timer = nullptr;

   if (!timer) {
      timer = new TQt6EventsTimer(10, kTRUE);
      timer->TurnOn();
   }

}

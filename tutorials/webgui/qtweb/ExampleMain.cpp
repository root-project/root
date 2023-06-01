// Author: Sergey Linev, GSI  13/01/2021

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <QTimer>
#include <QApplication>

#include "TApplication.h"
#include "TSystem.h"

#include "ExampleWidget.h"

#if QT_VERSION < QT_VERSION_CHECK(6,0,0)
#include <QtWebEngine>
#endif

int main(int argc, char **argv)
{
#if QT_VERSION < QT_VERSION_CHECK(6,0,0)
   // must be called before creating QApplication, from Qt 5.13, not needed for Qt6
   QtWebEngine::initialize();
#endif

   argc = 1; // hide all additional parameters from ROOT and Qt
   TApplication app("uno", &argc, argv); // ROOT application

   char* argv2[3];
   argv2[0] = argv[0];
   argv2[1] = 0;

   QApplication myapp(argc, argv2); // Qt application

   // let run ROOT event loop
   QTimer timer;
   QObject::connect( &timer, &QTimer::timeout, []() { gSystem->ProcessEvents();  });
   timer.setSingleShot(false);
   timer.start(4);

   // top widget
   ExampleWidget* widget = new ExampleWidget();

   widget->setWindowTitle(QString("QtWeb application, build with qt ") + QT_VERSION_STR);

   QObject::connect(&myapp, &QApplication::lastWindowClosed, &myapp, &QApplication::quit);

   widget->show();

   int res = myapp.exec();
   return res;
}

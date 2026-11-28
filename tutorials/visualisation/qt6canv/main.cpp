#include "mainwidget.h"

#include <QApplication>
#include <QLabel>
#include <QTimer>

#include "TSystem.h"
#include "TEnv.h"
#include "TQt6GuiFactory.h"

int main(int argc, char *argv[])
{
   QApplication myapp(argc, argv);

   gEnv->SetValue("Gui.Factory", "qt6");
   gEnv->SetValue("Root.PadEditor", "qt6");

   gGuiFactory = new ROOT::Experimental::TQt6GuiFactory();

   // let run ROOT event loop
   QTimer timer;
   QObject::connect(&timer, &QTimer::timeout, []() {
      gSystem->ProcessEvents();
   });
   timer.setSingleShot(false);
   timer.start(4);

   MainWidget w;
   w.setWindowTitle("Main Widget");
   w.resize(800, 500);

   QObject::connect(&myapp, &QApplication::lastWindowClosed, &myapp, &QApplication::quit);

   w.show();
   return myapp.exec();
}


/**
*  Main program 
*
*  Initialising both Root/Qt environment    
*
*  Updated 10/10/01 
*  @authors Denis Bertini <d.bertini@gsi.de> 
*  @version 2.3
*/

#include "guitest.h"
#include "qtroot.h"

#include "TBrowser.h"

#include "TQRootApplication.h"
#include "TQApplication.h"
#include "qapplication.h"

#include "stdlib.h"

int main( int argc, char **argv )
{
   // This is just a example of a main program using
   // the QtROOT interface. Here above a variable "mode"
   // defines different programs .ie.
   // mode 0 : Qtroot alone
   // mode 1 QtROOT + TBrowser
   // mode 2 QtROOT + TBrowser + Guitest (ROOT GUI"S examples)

   int mode = 0;

   TQApplication app("uno",&argc,argv);

   // Define a QRootApplication with polling mechanism on.
   // The ROOT events are then enabled .ie. the use of
   // TTimer TThread and Gui classes TGxx together
   // with Qt events handling is possible.

   TQRootApplication a( argc, argv, 0);

   // Define a QRootApplication without polling mechanism.

   //> QRootApplication a( argc, argv, 1);
   //> with debug info
   //> a.setDebugOn();

   if (argc>1 )  mode = atoi(argv[1]);

   // if no polling done, the user need to create a
   // Qt customized factory
   // app.setCustomized();

   if ( mode > 1 )
      TestMainFrame* mainWindow = new TestMainFrame( gClient->GetRoot(), 400, 220);

   if ( mode > 0  )
      TBrowser *b = new TBrowser();

   ApplicationWindow * mw = new ApplicationWindow();
   mw->setCaption( "Qt & ROOT" );
   mw->show();

   a.connect( &a, SIGNAL( lastWindowClosed() ), &a, SLOT( quit() ) );

   int res = a.exec();
   return res;
}

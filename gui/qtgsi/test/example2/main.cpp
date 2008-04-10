#include "TQApplication.h"
#include "TQRootApplication.h"
#include "qtrootexample1.h"
#include "TBrowser.h"

int main( int argc, char ** argv )
{
   TQApplication app("uno",&argc,argv);
   TQRootApplication a( argc, argv, 0);
   qtrootexample1 *w = new qtrootexample1;
   w->show();
   new TBrowser();
   a.connect( &a, SIGNAL( lastWindowClosed() ), &a, SLOT( quit() ) );
   a.exec();
   return 0;
}
 

/*

  Unit testing for the TGInputDialog class. It easily shows how to use
  it.

  To compile and run :
  g++ -g `root-config --cflags --libs` -lGui testInputDialog.C -o testInputDialog
  ./testInputDialog

  It can also be loaded with root:
  root testInputDialog.C

*/

#include "TGInputDialog.h"
#include "TGClient.h"
#include "Riostream.h"
#include <TApplication.h>

void testDefaultConstructor() {
   new TGInputDialog();
}

void testMinimumConstructor() {
   new TGInputDialog( gClient->GetRoot() );
}

void testFourConstructor() {
   new TGInputDialog( gClient->GetRoot(), 0, "Prompt:", "Default value" );
}

void testDefaultReturn() {
   char* retstr = 0;
   new TGInputDialog( gClient->GetRoot(), 0, "Prompt:", "Default value", retstr );
   if ( retstr ) cout << "Returned: " << retstr << endl;
   if ( retstr ) delete retstr;
}

void testReturn() {
   char retstr[256];
   new TGInputDialog( gClient->GetRoot(), 0, "Prompt:", "Default value", retstr );
   cout << "Returned: " << retstr << endl;
}

void testInputDialog ()
{
   testDefaultConstructor();
   testMinimumConstructor();
   testFourConstructor();
   testDefaultReturn();
   testReturn();
}

int main(int argc, char** argv)
{

   TApplication* theApp = new TApplication("App",&argc,argv);

   testInputDialog();

   theApp->Run();
   delete theApp;

   return 0;
}


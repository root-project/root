// @(#)root/test:$Id$
// Author: Fons Rademakers   04/04/97

// This small demo shows the traditional "Hello World". Its main use is
// to show how to use ROOT graphics and how to enter the eventloop to
// be able to interact with the graphics.

#include "TApplication.h"
#include "TCanvas.h"
#include "TLine.h"
#include "TPaveLabel.h"

int main(int argc, char **argv)
{
   TApplication theApp("App", &argc, argv);

   TCanvas *c = new TCanvas("c", "The Hello Canvas", 400, 400);

   TPaveLabel *hello = new TPaveLabel(0.2,0.4,0.8,0.6,"Hello World");
   hello->Draw();
   TPaveLabel *quit = new TPaveLabel(0.2,0.2,0.8,0.3,"Close via menu File/Quit");
   quit->Draw();
   c->Update();

   // Enter event loop, one can now interact with the objects in
   // the canvas. Select "Exit ROOT" from Canvas "File" menu to exit
   // the event loop and execute the next statements.
   theApp.Run(kTRUE);

   TLine *l = new TLine(0.1,0.2,0.5,0.9);
   l->Draw();
   quit->SetLabel("Select File/Quit again to exit app");
   c->Update();

   // Here we don't return from the eventloop. "Exit ROOT" will quit the app.
   theApp.Run();

   return 0;
}

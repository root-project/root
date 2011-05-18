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
   c->Connect("Closed()", "TApplication", &theApp, "Terminate()");

   TPaveLabel *hello = new TPaveLabel(0.2,0.4,0.8,0.6,"Hello World");
   hello->Draw();
   TPaveLabel *quit = new TPaveLabel(0.2,0.2,0.8,0.3,"Close via menu File/Quit");
   quit->Draw();
   c->Update();
   theApp.Run();
   return 0;
}

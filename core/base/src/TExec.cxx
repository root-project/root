// @(#)root/base:$Id$
// Author: Rene Brun   29/12/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TROOT.h"
#include "TExec.h"

ClassImp(TExec)

//______________________________________________________________________________
//
//   TExec is a utility class that can be used to execute a CINT command
//   when some event happens in a pad.
//   The command in turn can invoke a CINT macro to paint graphics objects
//   at positions depending on the histogram or graph contents.
//
// Case 1:
//
// The TExec object is in the list of pad primitives (after exec.Draw()).
// When the pad is drawn, the TExec::Paint function is called. This function
// will execute the specified command.
// The following example uses the services of the class Aclock created
// in $ROOTSYS/test/Aclock.cxx.
// This examples uses a TTimer to redraw a pad at regular intervals (clock).
// When the clock is updated, a string with the current date&time is drawn.
//{
//   gSystem->Load("$ROOTSYS/test/Aclock");
//   Aclock ck(400);
//   gPad->SetFillColor(5);
//   TDatime dt;
//   TText t(.5,.3,"t");
//   t.SetTextAlign(22);
//   t.SetTextSize(.07);
//   t.SetTextColor(4);
//   t.Draw();
//   TExec ex("ex","dt.Set();t.SetTitle(dt.AsString())");
//   ex.Draw();
//}
//
// Case 2:
//
// The TExec object may be added to the list of functions of a TH1 or TGraph
// object via hist->GetListOfFunctions()->Add(exec).
// When the histogram (or graph) is drawn, the TExec will be executed.
// If the histogram is made persistent on a file, the TExec object
// is also saved with the histogram. When redrawing the histogram in a
// new session, the TExec will be executed.
// Example:
//     Assume an histogram TH1F *h already filled.
//     TExec *ex1 = new TExec("ex1","DoSomething()");
//     TExec *ex2 = new TExec("ex2",".x macro.C");
//     h->GetListOfFunctions()->Add(ex1);
//     h->GetListOfFunctions()->Add(ex2);
//     h->Draw();
//  When the Paint function for the histogram will be called, the "DoSomething"
//  function will be called (interpreted or compiled) and also the macro.C.
//
// Case 3:
//
// A TExec object is automatically generated when invoking TPad::AddExec.
// Each pad contains a TList of TExecs (0, 1 or more). When a mouse event
// (motion, click, etc) happens, the pad object executes sequentially
// this list of TExecs. In the code (interpreted or compiled) executed
// by the TExec referenced command, one can call the pad service functions
// such as TPad::GetEvent, TPad::GetEventX, TPad::GetEventY to find
// which type of event and the X,Y position of the mouse.
// By default, the list of TExecs is executed. This can be disabled
// via the canvas menu "Option".
// See $ROOTSYS/tutorials/hist/exec2.C for an example.
//    Root > TFile f("hsimple.root");
//    Root > hpxpy.Draw();
//    Root > c1.AddExec("ex2",".x exec2.C");
//    When moving the mouse in the canvas, a second canvas shows the
//    projection along X of the bin corresponding to the Y position
//    of the mouse. The resulting histogram is fitted with a gaussian.
//    A "dynamic" line shows the current bin position in Y.
//    This more elaborated example can be used as a starting point
//    to develop more powerful interactive applications exploiting CINT
//    as a development engine.
//
// The 3 options above can be combined.


//______________________________________________________________________________
TExec::TExec(): TNamed()
{
   // Exec default constructor.
}


//______________________________________________________________________________
TExec::TExec(const char *name, const char *command) : TNamed(name,command)
{
   // Exec normal constructor.
}


//______________________________________________________________________________
TExec::~TExec()
{
   // Exec default destructor.
}


//______________________________________________________________________________
TExec::TExec(const TExec &e) : TNamed(e)
{
   // Copy constructor.

   TNamed::Copy(*this);
}


//______________________________________________________________________________
void TExec::Exec(const char *command)
{
   // Execute the command referenced by this object.
   //
   //  if command is given, this command is executed
   // otherwise the default command of the object is executed
   //
   // if the default command (in the exec title) is empty, an attemp is made
   // to execute the exec name if it contains a "." or a "(", otherwise
   // the command ".x execname.C" is executed.
   // The function returns the result of the user function/script.

   if (command && (strlen(command) > 1))  gROOT->ProcessLine(command);
   else  {
      if (strlen(GetTitle()) > 0)         gROOT->ProcessLine(GetTitle());
      else  {
         if (strchr(GetName(),'('))      {gROOT->ProcessLine(GetName()); return;}
         if (strchr(GetName(),'.'))      {gROOT->ProcessLine(GetName()); return;}
         char action[512];
         snprintf(action, sizeof(action), ".x %s.C", GetName());
         gROOT->ProcessLine(action);
      }
   }
}


//______________________________________________________________________________
void TExec::Paint(Option_t *)
{
   // Execute the command referenced by this object.

   Exec();
}


//______________________________________________________________________________
void TExec::SavePrimitive(ostream &out, Option_t * /*= ""*/)
{
   // Save primitive as a C++ statement(s) on output stream out.

   char quote = '"';
   if (gROOT->ClassSaved(TExec::Class())) {
      out<<"   ";
   } else {
      out<<"   TExec *";
   }
   out<<"exec = new TExec("<<quote<<GetName()<<quote<<","<<quote<<GetTitle()<<quote<<");"<<endl;

   out<<"   exec->Draw();"<<endl;
}

// @(#)root/meta:$Id$
// Author: Fons Rademakers   01/03/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TInterpreter                                                         //
//                                                                      //
// This class defines an abstract interface to a generic command line   //
// interpreter.                                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TInterpreter.h"

#include "TROOT.h"
#include "TError.h"

TInterpreter*   (*gPtr2Interpreter)() = 0; // returns pointer to global object
TInterpreter*   gCling = 0; // returns pointer to global TCling object
static TInterpreter *gInterpreterLocal = 0; // The real holder of the pointer.

ClassImp(TInterpreter)

//______________________________________________________________________________
TInterpreter::TInterpreter(const char *name, const char *title)
    : TNamed(name, title)
{
   // TInterpreter ctor only called by derived classes.

   gInterpreterLocal = this;
   gCling            = this;
}

//______________________________________________________________________________
TInterpreter *TInterpreter::Instance()
{
   // returns gInterpreter global

   if (gInterpreterLocal == 0) {
      static TROOT *getROOT = ROOT::GetROOT(); // Make sure gInterpreterLocal is set
      if (!getROOT) {
         ::Fatal("TInterpreter::Instance","TROOT object is required before accessing a TInterpreter");
      }
   }
   if (gPtr2Interpreter) return gPtr2Interpreter();
   return gInterpreterLocal;
}

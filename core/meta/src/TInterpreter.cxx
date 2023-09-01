// @(#)root/meta:$Id$
// Author: Fons Rademakers   01/03/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TInterpreter
This class defines an abstract interface to a generic command line
interpreter.
*/

#include "TInterpreter.h"

#include "TROOT.h"
#include "TError.h"
#include "TGlobal.h"


TInterpreter*   gCling = nullptr; // returns pointer to global TCling object
static TInterpreter *gInterpreterLocal = nullptr; // The real holder of the pointer.


namespace {
static struct AddPseudoGlobals {
AddPseudoGlobals() {

   // use special functor to extract pointer on gInterpreterLocal variable
   TGlobalMappedFunction::MakeFunctor("gInterpreter", "TInterpreter*", TInterpreter::Instance, [] {
      TInterpreter::Instance();
      return (void *) &gInterpreterLocal;
   });

}
} gAddPseudoGlobals;
}


ClassImp(TInterpreter);

////////////////////////////////////////////////////////////////////////////////
/// TInterpreter ctor only called by derived classes.

TInterpreter::TInterpreter(const char *name, const char *title)
    : TNamed(name, title)
{
   gInterpreterLocal = this;
   gCling            = this;
}

////////////////////////////////////////////////////////////////////////////////
/// returns gInterpreter global

TInterpreter *TInterpreter::Instance()
{
   if (gInterpreterLocal == nullptr) {
      static TROOT *getROOT = ROOT::GetROOT(); // Make sure gInterpreterLocal is set
      if (!getROOT) {
         ::Fatal("TInterpreter::Instance","TROOT object is required before accessing a TInterpreter");
      }
   }
   return gInterpreterLocal;
}

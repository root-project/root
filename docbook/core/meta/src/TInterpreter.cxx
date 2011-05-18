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

TInterpreter*   (*gPtr2Interpreter)() = 0; // returns pointer to global object
TInterpreter*   gCint = 0; // returns pointer to global TCint object

ClassImp(TInterpreter)

//______________________________________________________________________________
TInterpreter::TInterpreter(const char *name, const char *title)
    : TNamed(name, title)
{
   // TInterpreter ctor only called by derived classes.

   gInterpreter = this;
   gCint        = this;
}

//______________________________________________________________________________
TInterpreter *&TInterpreter::Instance()
{
   // returns gInterpreter global

   static TInterpreter *instance = 0;
   if (gPtr2Interpreter) instance = gPtr2Interpreter();
   return instance;
}

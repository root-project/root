// @(#)root/meta:$Name$:$Id$
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

TInterpreter *gInterpreter = 0;

ClassImp(TInterpreter)

//______________________________________________________________________________
TInterpreter::TInterpreter(const char *name, const char *title)
    : TNamed(name, title)
{
   // TInterpreter ctor only called by derived classes.

   gInterpreter = this;
}

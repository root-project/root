// @(#)root/geom:$Name:$:$Id:$
// Author: Andrei Gheata   15/01/02
   
/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////////
// TGeoBoolCombinator
//
//
//
//
//Begin_Html
/*
<img src=".gif">
*/
//End_Html

#include "TNamed.h"
#include "TGeoBoolCombinator.h"

// statics and globals

ClassImp(TGeoBoolCombinator)

//-----------------------------------------------------------------------------
TGeoBoolCombinator::TGeoBoolCombinator()
{
// Default constructor
}
//-----------------------------------------------------------------------------
TGeoBoolCombinator::TGeoBoolCombinator(const char *name, const char *formula)
                   :TNamed(name, formula)
{
// constructor
   if (!Compile()) {
      Error("ctor", "invalid formula");
   }
}
//-----------------------------------------------------------------------------
TGeoBoolCombinator::~TGeoBoolCombinator()
{
// Destructor
}
//-----------------------------------------------------------------------------
Bool_t TGeoBoolCombinator::Compile()
{
// compiles the formula and returns true if it is OK
   return kFALSE; 
}


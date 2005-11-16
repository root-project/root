// @(#)root/base:$Name:  $:$Id: TVirtualUtilPad.cxx,v 1.1 2002/09/15 19:41:52 brun Exp $
// Author: Rene Brun   14/09/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Abstract interface to the pad/canvas utilities                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualUtilPad.h"
#include "TROOT.h"

ClassImp(TVirtualUtilPad)

//______________________________________________________________________________
TVirtualUtilPad::TVirtualUtilPad()
{
   // Constructor.
   SetName("R__TVirtualUtilPad");
   TVirtualUtilPad *u = (TVirtualUtilPad*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtilPad");
   if (!u) gROOT->GetListOfSpecials()->Add(this);
}

//______________________________________________________________________________
TVirtualUtilPad::~TVirtualUtilPad()
{
   // Destructor.
}

// @(#)root/base:$Id$
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
// Abstract interface to the histogram utilities                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualUtilHist.h"
#include "TROOT.h"

ClassImp(TVirtualUtilHist)

//______________________________________________________________________________
TVirtualUtilHist::TVirtualUtilHist()
{
   // Constructor.
   SetName("R__TVirtualUtilHist");
   TVirtualUtilHist *u = (TVirtualUtilHist*)gROOT->GetListOfSpecials()->FindObject("R__TVirtualUtilHist");
   if (!u) gROOT->GetListOfSpecials()->Add(this);
}

//______________________________________________________________________________
TVirtualUtilHist::~TVirtualUtilHist()
{
   // Destructor.
}

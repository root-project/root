// @(#)root/base:$Name:  $:$Id: TTime.cxx,v 1.1.1.1 2000/05/16 17:00:39 rdm Exp $
// Author: Fons Rademakers   28/11/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTime                                                                //
//                                                                      //
// Basic time type with millisecond precision.                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TTime.h"
#include "TString.h"


ClassImp(TTime)

//______________________________________________________________________________
const char *TTime::AsString() const
{
   // Return the time as a string.

   return Form("%lu", (ULong_t)fMilliSec);
}

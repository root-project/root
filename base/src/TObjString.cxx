// @(#)root/base:$Name$:$Id$
// Author: Fons Rademakers   12/11/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjString                                                           //
//                                                                      //
// Collectable string class. This is a TObject containing a TString.    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TObjString.h"

ClassImp(TObjString)

//______________________________________________________________________________
Int_t TObjString::Compare(TObject *obj)
{
   if (this == obj) return 0;
   if (TObjString::Class() != obj->IsA()) return -1;
   return fString.CompareTo(((TObjString*)obj)->fString);
}

//______________________________________________________________________________
Bool_t TObjString::IsEqual(TObject *obj)
{
   if (this == obj) return kTRUE;
   if (TObjString::Class() != obj->IsA()) return kFALSE;
   return fString == ((TObjString*)obj)->fString;
}

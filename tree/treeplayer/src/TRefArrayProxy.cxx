// @(#)root/meta:$Id$
// Author: Markus Frank  10/02/2006

/*************************************************************************
* Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#include "TRefArray.h"
#include "TRefArrayProxy.h"
#include "TFormLeafInfoReference.h"

//______________________________________________________________________________
//
// TRefArrayProxy is a container proxy, which allows to access references stored
// in a TRefArray from TTree::Draw

//______________________________________________________________________________
void* TRefArrayProxy::GetObject(TFormLeafInfoReference* /*info*/, void* data, Int_t instance)  {
   // Access referenced object(-data)

   TRefArray* ref = (TRefArray*)data;//((char*)data + info->GetOffset());
   return ref->At(instance);
}

//______________________________________________________________________________
Int_t  TRefArrayProxy::GetCounterValue(TFormLeafInfoReference* /*info*/, void* data)   {
   // TVirtualRefProxy overload: Access to container size (if container reference (ie TRefArray) etc)

   TRefArray* ref = (TRefArray*)data;
   return ref->GetEntriesFast();
}

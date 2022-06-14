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

/** \class TRefArrayProxy
A container proxy, which allows to access references stored
in a TRefArray from TTree::Draw
*/

////////////////////////////////////////////////////////////////////////////////
/// Access referenced object(-data)

void* TRefArrayProxy::GetObject(TFormLeafInfoReference* /*info*/, void* data, Int_t instance)  {
   TRefArray* ref = (TRefArray*)data;//((char*)data + info->GetOffset());
   return ref->At(instance);
}

////////////////////////////////////////////////////////////////////////////////
/// TVirtualRefProxy overload: Access to container size (if container reference (ie TRefArray) etc)

Int_t  TRefArrayProxy::GetCounterValue(TFormLeafInfoReference* /*info*/, void* data)   {
   TRefArray* ref = (TRefArray*)data;
   return ref->GetEntriesFast();
}

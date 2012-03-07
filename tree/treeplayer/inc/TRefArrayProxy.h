// @(#)root/treeplayer:$Id$
// Author: Markus Frank 20/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRefArrayProxy
#define ROOT_TRefArrayProxy
#ifndef ROOT_TRefProxy
#include "TRefProxy.h"
#endif

//______________________________________________________________________________
//
// TRefArrayProxy is a container proxy, which allows to access references stored
// in a TRefArray from TTree::Draw
//
//______________________________________________________________________________
class TRefArrayProxy : public TRefProxy  {
public:
   // The implicit's constructor and destructor have the correct implementation.

   // TVirtualRefProxy overload: Clone the reference proxy (virtual constructor)
   virtual TVirtualRefProxy* Clone() const        { return new TRefArrayProxy(*this);}
   // TVirtualRefProxy overload: Flag to indicate if this is a container reference
   virtual Bool_t HasCounter()  const             { return kTRUE;                    }
   // TVirtualRefProxy overload: Access referenced object(-data)
   virtual void* GetObject(TFormLeafInfoReference* info, void* data, Int_t instance);
   // TVirtualRefProxy overload: Access to container size (if container reference (ie TRefArray) etc)
   virtual Int_t  GetCounterValue(TFormLeafInfoReference* info, void *data);
};
#endif // ROOT_TRefArrayProxy

// @(#)root/meta:$Id$
// Author: Markus Frank 20/05/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRefProxy
#define ROOT_TRefProxy

#include "TVirtualRefProxy.h"

#include "TClassRef.h"

// Forward declarations
class TFormLeafInfoReference;

//______________________________________________________________________________
//
// TRefProxy is a reference proxy, which allows to access ROOT references (TRef)
// stored contained in other objects from TTree::Draw
//______________________________________________________________________________

class TRefProxy : public TVirtualRefProxy  {
protected:
   TClassRef fClass;    //! Pointer to the reference class (TRef::Class())

public:
   /// Default constructor
   TRefProxy() : fClass("TRef") {}
   /// Copy constructor
   TRefProxy(const TRefProxy& c) : TVirtualRefProxy(), fClass(c.fClass) {}
   /// Assignement operator
   TRefProxy &operator=(const TRefProxy& c) { fClass =c.fClass; return *this; }

   /// TVirtualRefProxy overload: Release the reference proxy (virtual destructor)
   void Release() override                         { delete this;                }
   /// TVirtualRefProxy overload: Clone the reference proxy (virtual constructor)
   TVirtualRefProxy* Clone() const override        { return new TRefProxy(*this);}
   /// TVirtualRefProxy overload: Setter of reference class (executed when the proxy is adopted)
   void SetClass(TClass *cl) override              { fClass = cl;                }
   /// TVirtualRefProxy overload: Getter of reference class (executed when the proxy is adopted)
   TClass * GetClass() const override              { return fClass;              }
   /// TVirtualRefProxy overload: Access to value class
   TClass* GetValueClass(void* data) const override;
   /// TVirtualRefProxy overload: Prepare reused reference object (e.g. ZERO data pointers)
   void* GetPreparedReference(void* data) override {  return data;               }
   /// TVirtualRefProxy overload: Update (and propagate) cached information
   Bool_t Update() override;
   /// TVirtualRefProxy overload: Flag to indicate if this is a container reference
   Bool_t HasCounter()  const override             { return kFALSE;              }
   /// TVirtualRefProxy overload: Access to container size (if container reference (ie TRefArray) etc)
   Int_t  GetCounterValue(TFormLeafInfoReference* /* info */, void* /* data */) override
   {  return 0;                                                                 }
   /// TVirtualRefProxy overload: Access referenced object(-data)
   void* GetObject(TFormLeafInfoReference* info, void* data, int instance) override;
};
#endif // ROOT_TRefProxy

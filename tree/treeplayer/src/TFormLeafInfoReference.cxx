// @(#)root/treeplayer:$Id$
// Author: Markus Frank 01/02/2006

/*************************************************************************
* Copyright (C) 1995-2000, Rene Brun and Fons Rademakers and al.        *
* All rights reserved.                                                  *
*                                                                       *
* For the licensing terms see $ROOTSYS/LICENSE.                         *
* For the list of contributors see $ROOTSYS/README/CREDITS.             *
*************************************************************************/

#include "TError.h"
#include "TLeafObject.h"
#include "TInterpreter.h"
#include "TVirtualRefProxy.h"
#include "TFormLeafInfoReference.h"

//______________________________________________________________________________
//
// TFormLeafInfoReference is a small helper class to implement the following
// of reference objects stored in a TTree

//______________________________________________________________________________
TFormLeafInfoReference::TFormLeafInfoReference(TClass* cl, TStreamerElement* e, int off)
: TFormLeafInfo(cl,off,e), fProxy(0), fBranch(0)
{
   // Constructor.

   TVirtualRefProxy* p = cl->GetReferenceProxy();
   if ( !p )  {
      ::Error("TFormLeafInfoReference","No reference proxy for class %s availible",cl->GetName());
      return;
   }
   fProxy = p->Clone();
}

//______________________________________________________________________________
TFormLeafInfoReference::TFormLeafInfoReference(const TFormLeafInfoReference& org)
: TFormLeafInfo(org), fProxy(0), fBranch(org.fBranch)
{
   // Copy constructor.

   TVirtualRefProxy* p = org.fProxy;
   if ( !p )  {
      ::Error("TFormLeafInfoReference","No reference proxy for class %s availible",fClass->GetName());
      return;
   }
   fProxy = p->Clone();
}

//______________________________________________________________________________
TFormLeafInfoReference::~TFormLeafInfoReference()
{
   // Destructor.

   if ( fProxy ) fProxy->Release();
}

//______________________________________________________________________________
void TFormLeafInfoReference::Swap(TFormLeafInfoReference &other)
{
   // Exception safe swap.
   TFormLeafInfo::Swap(other);
   std::swap(fProxy,other.fProxy);
   std::swap(fBranch,other.fBranch);
}

//______________________________________________________________________________
TFormLeafInfoReference &TFormLeafInfoReference::operator=(const TFormLeafInfoReference &other)
{
   // Exception safe assignment operator.
   TFormLeafInfoReference tmp(other);
   Swap(tmp);
   return *this;
}

//______________________________________________________________________________
TFormLeafInfo* TFormLeafInfoReference::DeepCopy() const
{
   // Virtual copy constructor.

   return new TFormLeafInfoReference(*this);
}

//______________________________________________________________________________
TClass* TFormLeafInfoReference::GetClass() const
{
   // Access to target class pointer (if available)

   return fNext ? fNext->GetClass() : 0;
}

//______________________________________________________________________________
Bool_t TFormLeafInfoReference::HasCounter() const
{
   // Return true if any of underlying data has a array size counter

   Bool_t result = fProxy ? fProxy->HasCounter() : false;
   if (fNext) result |= fNext->HasCounter();
   return fCounter!=0 || result;
}
//______________________________________________________________________________
Int_t TFormLeafInfoReference::ReadCounterValue(char *where)
{
   // Return the size of the underlying array for the current entry in the TTree.

   Int_t result = 0;
   if ( where && HasCounter() )  {
      where = (char*)fProxy->GetPreparedReference(where);
      if ( where )  {
         return fProxy->GetCounterValue(this, where);
      }
   }
   gInterpreter->ClearStack();
   // Get rid of temporary return object.
   return result;
}

//______________________________________________________________________________
Int_t TFormLeafInfoReference::GetCounterValue(TLeaf* leaf)  {
   // Return the current size of the array container

   if ( HasCounter() )  {
      char *thisobj = 0;
      Int_t instance = 0;
      if (leaf->InheritsFrom(TLeafObject::Class()) ) {
         thisobj = (char*)((TLeafObject*)leaf)->GetObject();
      } else {
         thisobj = GetObjectAddress((TLeafElement*)leaf, instance); // instance might be modified
      }
      return ReadCounterValue(thisobj);
   }
   return 0;
}

//______________________________________________________________________________
TClass* TFormLeafInfoReference::GetValueClass(TLeaf* leaf)
{
   // Access to the value class of the reference proxy

   return this->GetValueClass(this->GetLocalValuePointer(leaf,0));
}

//______________________________________________________________________________
TClass* TFormLeafInfoReference::GetValueClass(void* obj)
{
   // Access to the value class of the reference proxy

   return fProxy ? fProxy->GetValueClass(obj) : 0;
}

//______________________________________________________________________________
Bool_t TFormLeafInfoReference::Update()
{
   // TFormLeafInfo overload: Update (and propagate) cached information

   Bool_t res = this->TFormLeafInfo::Update();
   if ( fProxy ) fProxy->Update();
   return res;
}

//______________________________________________________________________________
template <typename T>
T TFormLeafInfoReference::GetValueImpl(TLeaf *leaf, Int_t instance)
{
   // Return result of a leafobject method
   fBranch = leaf->GetBranch();
   return TFormLeafInfo::GetValueImpl<T>(leaf, instance);
}

template Double_t TFormLeafInfoReference::GetValueImpl<Double_t>(TLeaf*, Int_t);
template Long64_t TFormLeafInfoReference::GetValueImpl<Long64_t>(TLeaf*, Int_t);
template LongDouble_t TFormLeafInfoReference::GetValueImpl<LongDouble_t>(TLeaf*, Int_t);

//______________________________________________________________________________
void *TFormLeafInfoReference::GetLocalValuePointer( TLeaf *from, Int_t instance)
{
   // This is implemented here because some compiler want ALL the
   // signature of an overloaded function to be re-implemented.

   fBranch = from->GetBranch();
   return TFormLeafInfo::GetLocalValuePointer(from, instance);
}

//______________________________________________________________________________
void *TFormLeafInfoReference::GetLocalValuePointer(char *where, Int_t instance)
{
   // Access value of referenced object

   if (where) {
      where = (char*)fProxy->GetPreparedReference(where);
      if (where) {
         void* result = fProxy->GetObject(this, where, instance);
         gInterpreter->ClearStack();
         return result;
      }
   }
   gInterpreter->ClearStack();
   return 0;
}

//______________________________________________________________________________
template <typename T>
T  TFormLeafInfoReference::ReadValueImpl(char *where, Int_t instance)
{
   // Execute the method on the given address

   T result = 0;
   if ( where )  {
      where = (char*)fProxy->GetPreparedReference(where);
      if ( where )  {
         void* res = fProxy->GetObject(this, where, instance);
         if ( res )  {
            result = (fNext) ? fNext->ReadTypedValue<T>((char*)res,instance) : *(Double_t*)res;
         }
      }
   }
   gInterpreter->ClearStack();
   // Get rid of temporary return object.
   return result;
}

template Double_t TFormLeafInfoReference::ReadValueImpl<Double_t>(char*, Int_t);
template Long64_t TFormLeafInfoReference::ReadValueImpl<Long64_t>(char*, Int_t);
template LongDouble_t TFormLeafInfoReference::ReadValueImpl<LongDouble_t>(char*, Int_t);


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

/** \class TFormLeafInfoReference
A small helper class to implement the following
of reference objects stored in a TTree
*/

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TFormLeafInfoReference::TFormLeafInfoReference(TClass* cl, TStreamerElement* e, int off)
: TFormLeafInfo(cl,off,e), fProxy(0), fBranch(0)
{
   TVirtualRefProxy* p = cl->GetReferenceProxy();
   if ( !p )  {
      ::Error("TFormLeafInfoReference","No reference proxy for class %s available",cl->GetName());
      return;
   }
   fProxy = p->Clone();
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TFormLeafInfoReference::TFormLeafInfoReference(const TFormLeafInfoReference& org)
: TFormLeafInfo(org), fProxy(0), fBranch(org.fBranch)
{
   TVirtualRefProxy* p = org.fProxy;
   if ( !p )  {
      ::Error("TFormLeafInfoReference","No reference proxy for class %s available",fClass->GetName());
      return;
   }
   fProxy = p->Clone();
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TFormLeafInfoReference::~TFormLeafInfoReference()
{
   if ( fProxy ) fProxy->Release();
}

////////////////////////////////////////////////////////////////////////////////
/// Exception safe swap.

void TFormLeafInfoReference::Swap(TFormLeafInfoReference &other)
{
   TFormLeafInfo::Swap(other);
   std::swap(fProxy,other.fProxy);
   std::swap(fBranch,other.fBranch);
}

////////////////////////////////////////////////////////////////////////////////
/// Exception safe assignment operator.

TFormLeafInfoReference &TFormLeafInfoReference::operator=(const TFormLeafInfoReference &other)
{
   TFormLeafInfoReference tmp(other);
   Swap(tmp);
   return *this;
}

////////////////////////////////////////////////////////////////////////////////
/// Virtual copy constructor.

TFormLeafInfo* TFormLeafInfoReference::DeepCopy() const
{
   return new TFormLeafInfoReference(*this);
}

////////////////////////////////////////////////////////////////////////////////
/// Access to target class pointer (if available)

TClass* TFormLeafInfoReference::GetClass() const
{
   return fNext ? fNext->GetClass() : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// Return true if any of underlying data has a array size counter

Bool_t TFormLeafInfoReference::HasCounter() const
{
   Bool_t result = fProxy ? fProxy->HasCounter() : false;
   if (fNext) result |= fNext->HasCounter();
   return fCounter!=0 || result;
}
////////////////////////////////////////////////////////////////////////////////
/// Return the size of the underlying array for the current entry in the TTree.

Int_t TFormLeafInfoReference::ReadCounterValue(char *where)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Return the current size of the array container

Int_t TFormLeafInfoReference::GetCounterValue(TLeaf* leaf)  {
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

////////////////////////////////////////////////////////////////////////////////
/// Access to the value class of the reference proxy

TClass* TFormLeafInfoReference::GetValueClass(TLeaf* leaf)
{
   return this->GetValueClass(this->GetLocalValuePointer(leaf,0));
}

////////////////////////////////////////////////////////////////////////////////
/// Access to the value class of the reference proxy

TClass* TFormLeafInfoReference::GetValueClass(void* obj)
{
   return fProxy ? fProxy->GetValueClass(obj) : 0;
}

////////////////////////////////////////////////////////////////////////////////
/// TFormLeafInfo overload: Update (and propagate) cached information

Bool_t TFormLeafInfoReference::Update()
{
   Bool_t res = this->TFormLeafInfo::Update();
   if ( fProxy ) fProxy->Update();
   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// Return result of a leafobject method

template <typename T>
T TFormLeafInfoReference::GetValueImpl(TLeaf *leaf, Int_t instance)
{
   fBranch = leaf->GetBranch();
   return TFormLeafInfo::GetValueImpl<T>(leaf, instance);
}

template Double_t TFormLeafInfoReference::GetValueImpl<Double_t>(TLeaf*, Int_t);
template Long64_t TFormLeafInfoReference::GetValueImpl<Long64_t>(TLeaf*, Int_t);
template LongDouble_t TFormLeafInfoReference::GetValueImpl<LongDouble_t>(TLeaf*, Int_t);

////////////////////////////////////////////////////////////////////////////////
/// This is implemented here because some compiler want ALL the
/// signature of an overloaded function to be re-implemented.

void *TFormLeafInfoReference::GetLocalValuePointer( TLeaf *from, Int_t instance)
{
   fBranch = from->GetBranch();
   return TFormLeafInfo::GetLocalValuePointer(from, instance);
}

////////////////////////////////////////////////////////////////////////////////
/// Access value of referenced object

void *TFormLeafInfoReference::GetLocalValuePointer(char *where, Int_t instance)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Execute the method on the given address

template <typename T>
T  TFormLeafInfoReference::ReadValueImpl(char *where, Int_t instance)
{
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


// @(#)root/meta:$Id$
// Author: Philippe Canal 15/03/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClassRef is used to implement a permanent reference to a TClass     //
// object.  In particular this reference will change if and when the    //
// TClass object is regenerated.  This regeneration usually happens     //
// when a library containing the described class is loaded after a      //
// file containing an instance of this class has been opened.           //
//                                                                      //
// The references kept track of using an intrusive double linked list.  //
// The intrusive list is maintained by TClass::AddRef and               //
// TClass::RemoveRef.  The 'start' of the list is held in               //
// TClass::fRefStart.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClassRef.h"

//______________________________________________________________________________
TClassRef::TClassRef(const TClassRef &org) :
   fClassName(org.fClassName), fClassPtr(org.fClassPtr), fPrevious(0), fNext(0)
{
   // Copy ctor, increases reference count to original TClass object.

   if (fClassPtr) fClassPtr->AddRef(this);
}

//______________________________________________________________________________
TClassRef::TClassRef(const char *classname) :
    fClassName(classname), fClassPtr(0), fPrevious(0), fNext(0)
{
   // Create reference to specified class name, but don't set referenced
   // class object.
}

//______________________________________________________________________________
TClassRef::TClassRef(TClass *cl) : fClassPtr(cl), fPrevious(0), fNext(0)
{
   // Add reference to specified class object.
   
   if (fClassPtr) {
      fClassName = cl->GetName();
      fClassPtr->AddRef(this);
   }
}

//______________________________________________________________________________
void TClassRef::Assign(const TClassRef &rhs)
{
   // Assignment operator implementation, increases reference count to original class object.
   // This routines assumes that the copy actually need to be done.

   if (fClassPtr) fClassPtr->RemoveRef(this);
   fClassName = rhs.fClassName;
   fClassPtr  = rhs.fClassPtr;
   if (fClassPtr) fClassPtr->AddRef(this);
}

//______________________________________________________________________________
void TClassRef::Assign(TClass* rhs)
{
   // Assignment operator, increases reference count to original class object.
   // This routines assumes that the copy actually need to be done.

   if (fClassPtr) fClassPtr->RemoveRef(this);
   fClassPtr  = rhs;
   if (fClassPtr) {
      fClassName = fClassPtr->GetName();
      fClassPtr->AddRef(this);
   } else {
      fClassName.clear();
   }
}

//______________________________________________________________________________
TClass *TClassRef::InternalGetClass() const
{
   // Return the current TClass object corresponding to fClassName.

   if (fClassPtr) return fClassPtr;
   if (fClassName.size()==0) return 0;

   (const_cast<TClassRef*>(this))->fClassPtr = TClass::GetClass(fClassName.c_str());
   if (fClassPtr) fClassPtr->AddRef(const_cast<TClassRef*>(this));

   return fClassPtr;
}

//______________________________________________________________________________
void TClassRef::ListReset() 
{
   // Reset this object and all the objects in the list.
   // We assume that the TClass has also reset its fRefStart data member.

   for (TClassRef *ref = this; ref != 0; /* nothing */ ) {
      TClassRef *next = ref->fNext;
      ref->fNext = ref->fPrevious = 0;
      ref->fClassPtr = 0;
      ref = next;
   }
}

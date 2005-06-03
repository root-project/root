// @(#)root/meta:$Name:  $:$Id: TClassRef.cxx,v 1.4 2005/06/01 15:41:19 pcanal Exp $
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
//////////////////////////////////////////////////////////////////////////

#include "TClassRef.h"

//______________________________________________________________________________
TClassRef::TClassRef() :
   fClassName("unknown"), fClassPtr(0)
{
   // Default ctor.
}

//______________________________________________________________________________
TClassRef::TClassRef(const TClassRef &org) :
   fClassName(org.fClassName), fClassPtr(org.fClassPtr)
{
   // Copy ctor, increases reference count to original TClass object.

   if (fClassPtr) fClassPtr->AddRef(this);
}

//______________________________________________________________________________
TClassRef &TClassRef::operator=(const TClassRef &rhs)
{
   // Assignment operator, increases reference count to original class object.

   if (this != &rhs) {      
      Bool_t swap = fClassPtr != rhs.fClassPtr;
      if (fClassPtr && swap) fClassPtr->RemoveRef(this);
      fClassName = rhs.fClassName;
      fClassPtr  = rhs.fClassPtr;
      if (fClassPtr && swap) fClassPtr->AddRef(this);
   }
   return *this;
}

//______________________________________________________________________________
TClassRef &TClassRef::operator=(TClass* rhs)
{
   // Assignment operator, increases reference count to original class object.

   if (this->fClassPtr != rhs) {      
      if (fClassPtr) fClassPtr->RemoveRef(this);
      fClassPtr  = rhs;
      if (fClassPtr) fClassName = rhs->GetName();
      if (fClassPtr) fClassPtr->AddRef(this);
   }
   return *this;
}

//______________________________________________________________________________
TClassRef::TClassRef(const char *classname) :
    fClassName(classname), fClassPtr(0)
{
   // Create reference to specified class name, but don't set referenced
   // class object.
}

//______________________________________________________________________________
TClassRef::TClassRef(TClass *cl) :
    fClassName(cl?cl->GetName():"unknown"), fClassPtr(cl)
{
   // Add reference to specified class object.

   if (fClassPtr) fClassPtr->AddRef(this);
}

//______________________________________________________________________________
TClassRef::~TClassRef()
{
   // Dtor, decreases reference count of TClass object.

   if (fClassPtr) fClassPtr->RemoveRef(this);
}

//______________________________________________________________________________
TClass *TClassRef::InternalGetClass()  const
{
   // Return the current TClass object corresponding to fClassName.
   if (fClassPtr) return fClassPtr;
   (const_cast<TClassRef*>(this))->fClassPtr = TClass::GetClass(fClassName.Data());
   if (fClassPtr) fClassPtr->AddRef(const_cast<TClassRef*>(this));
   return fClassPtr;
}

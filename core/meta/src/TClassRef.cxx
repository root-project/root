// @(#)root/meta:$Id$
// Author: Philippe Canal 15/03/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TClassRef
TClassRef is used to implement a permanent reference to a TClass
object.  In particular this reference will change if and when the
TClass object is regenerated.  This regeneration usually happens
when a library containing the described class is loaded after a
file containing an instance of this class has been opened.

The references kept track of using an intrusive double linked list.
The intrusive list is maintained by TClass::AddRef and
TClass::RemoveRef.  The 'start' of the list is held in
TClass::fRefStart.
*/

#include "TClassRef.h"

////////////////////////////////////////////////////////////////////////////////
/// Copy ctor, increases reference count to original TClass object.

TClassRef::TClassRef(const TClassRef &org) :
   fClassName(org.fClassName), fClassPtr(org.fClassPtr)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Create reference to specified class name, but don't set referenced
/// class object.

TClassRef::TClassRef(const char *classname) :
    fClassName(classname), fClassPtr(nullptr)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Add reference to specified class object.

TClassRef::TClassRef(TClass *cl) : fClassPtr(nullptr)
{
   if (cl) {
      fClassName = cl->GetName();
      fClassPtr = cl->GetPersistentRef();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator implementation, increases reference count to original class object.
/// This routines assumes that the copy actually need to be done.

void TClassRef::Assign(const TClassRef &rhs)
{
   fClassName = rhs.fClassName;
   fClassPtr  = rhs.fClassPtr;
}

////////////////////////////////////////////////////////////////////////////////
/// Assignment operator, increases reference count to original class object.
/// This routines assumes that the copy actually need to be done.

void TClassRef::Assign(TClass* rhs)
{
   if (rhs) {
      fClassPtr  = rhs->GetPersistentRef();
      fClassName = rhs->GetName();
   } else {
      fClassPtr  = nullptr;
      fClassName.clear();
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return the current TClass object corresponding to fClassName.

TClass *TClassRef::InternalGetClass() const
{
   if (fClassPtr && *fClassPtr) return *fClassPtr;
   if (fClassName.size()==0) return nullptr;

   TClass *cl = TClass::GetClass(fClassName.c_str());
   if (cl) {
      (const_cast<TClassRef*>(this))->fClassPtr = cl->GetPersistentRef();
      return cl;
   } else {
      return nullptr;
   }
}


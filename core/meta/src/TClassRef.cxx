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
   fClassName(org.fClassName), fClassPtr(org.fClassPtr)
{
   // Copy ctor, increases reference count to original TClass object.

}

//______________________________________________________________________________
TClassRef::TClassRef(const char *classname) :
    fClassName(classname), fClassPtr(0)
{
   // Create reference to specified class name, but don't set referenced
   // class object.
}

//______________________________________________________________________________
TClassRef::TClassRef(TClass *cl) : fClassPtr(0)
{
   // Add reference to specified class object.

   if (cl) {
      fClassName = cl->GetName();
      fClassPtr = cl->GetPersistentRef();
   }
}

//______________________________________________________________________________
void TClassRef::Assign(const TClassRef &rhs)
{
   // Assignment operator implementation, increases reference count to original class object.
   // This routines assumes that the copy actually need to be done.

   fClassName = rhs.fClassName;
   fClassPtr  = rhs.fClassPtr;
}

//______________________________________________________________________________
void TClassRef::Assign(TClass* rhs)
{
   // Assignment operator, increases reference count to original class object.
   // This routines assumes that the copy actually need to be done.

   if (rhs) {
      fClassPtr  = rhs->GetPersistentRef();
      fClassName = rhs->GetName();
   } else {
      fClassPtr  = 0;
      fClassName.clear();
   }
}

//______________________________________________________________________________
TClass *TClassRef::InternalGetClass() const
{
   // Return the current TClass object corresponding to fClassName.

   if (fClassPtr && *fClassPtr) return *fClassPtr;
   if (fClassName.size()==0) return 0;

   TClass *cl = TClass::GetClass(fClassName.c_str());
   if (cl) {
      (const_cast<TClassRef*>(this))->fClassPtr = cl->GetPersistentRef();
      return cl;
   } else {
      return 0;
   }
}


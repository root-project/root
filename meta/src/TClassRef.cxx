// @(#)root/meta:$Name:  $:$Id: TClass.cxx,v 1.163 2005/02/25 17:06:34 brun Exp $
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
// TClassRef is used to implement a permanent references to a TClass    //
// object.  In particular this reference will change if and when the    //
// TClass object is regenerated.  This regeneration usually happens     //
// when a library containing the described class is loaded after a      //
// file containing instance of this class has been openeed.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TClassRef.h"

//////////////////////////////////////////////////////////////////////////
TClassRef::TClassRef() :
   fClassName("unknonw"), fClassPtr(0)
{
}

//////////////////////////////////////////////////////////////////////////
TClassRef::TClassRef(const TClassRef &rhs) :
   fClassName(rhs.fClassName), fClassPtr(rhs.fClassPtr)
{
   if (fClassPtr) fClassPtr->AddRef(this);
}

//////////////////////////////////////////////////////////////////////////
TClassRef &TClassRef::operator=(const TClassRef &rhs) 
{
   if (this != &rhs) {
      fClassName = rhs.fClassName;
      fClassPtr = rhs.fClassPtr;
      if (fClassPtr) fClassPtr->AddRef(this);
   }
   return *this;
}

//////////////////////////////////////////////////////////////////////////
TClassRef::TClassRef(const char *classname) : 
    fClassName(classname), fClassPtr(0)
{
}

//////////////////////////////////////////////////////////////////////////
TClassRef::TClassRef(TClass *cl) : 
    fClassName(cl?cl->GetName():"unknown"), fClassPtr(cl)
{
   if (fClassPtr) fClassPtr->AddRef(this);
}

//////////////////////////////////////////////////////////////////////////
TClassRef::~TClassRef() 
{
   if (fClassPtr) fClassPtr->RemoveRef(this);
}

//////////////////////////////////////////////////////////////////////////
TClass *TClassRef::GetClass() 
{
   // Return the current TClass object corresponding to
   // fClassName.

   if (fClassPtr) return fClassPtr;
   fClassPtr = TClass::GetClass(fClassName.Data());
   if (fClassPtr) fClassPtr->AddRef(this);
   return fClassPtr;
}

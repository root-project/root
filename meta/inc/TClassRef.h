// @(#)root/meta:$Name:  $:$Id: TClassRef.h,v 1.5 2005/06/01 15:41:19 pcanal Exp $
// Author: Philippe Canal 15/03/2005

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TClassRef
#define ROOT_TClassRef

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TClassRef                                                            //
//                                                                      //
// Reference to a TClass object.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TClass
#include "TClass.h"
#endif
#ifndef ROOT_TRef
#include "TRef.h"
#endif

class TClassRef {

private:
   TString   fClassName; //Name of referenced class
   TClass   *fClassPtr;  //! Ptr to the TClass object

   TClass   *InternalGetClass()  const;
public:
   TClassRef();
   TClassRef(TClass *cl);
   TClassRef(const char *classname);
   TClassRef(const TClassRef&);
   TClassRef& operator=(const TClassRef&);
   TClassRef& operator=(TClass*);

   ~TClassRef();

   void SetName(const char* new_name) { 
      if ( fClassPtr && fClassName != new_name ) *this = (TClass*)0; 
      fClassName = new_name; 
   }
   TClass *GetClass()  const { return fClassPtr ? fClassPtr : InternalGetClass(); }
   void Reset() { fClassPtr = 0; }

   TClass* operator->() const { return fClassPtr ? fClassPtr : InternalGetClass(); }
   operator TClass*() const { return fClassPtr ? fClassPtr : InternalGetClass(); }

};

#endif

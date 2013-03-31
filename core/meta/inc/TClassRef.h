// @(#)root/meta:$Id$
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
// Reference to a TClass object and intrusive list of other             //
// to thise same TClass object references                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TClass
#include "TClass.h"
#endif
#ifndef ROOT_TRef
#include "TRef.h"
#endif

#include <string>

class TClassRef {

private:
   std::string  fClassName; //Name of referenced class
   TClass      *fClassPtr;  //! Ptr to the TClass object
   TClassRef   *fPrevious;  //! link to the previous refs
   TClassRef   *fNext;      //! link to the next refs 

   friend class TClass;

   void Assign(const TClassRef &);
   void Assign(TClass *);
   TClass   *InternalGetClass() const;
   void      ListReset();
public:
   TClassRef() : fClassName(), fClassPtr(0), fPrevious(0), fNext(0) {}
   TClassRef(TClass *cl);
   TClassRef(const char *classname);
   TClassRef(const TClassRef&);
   inline TClassRef &operator=(const TClassRef &rhs) {
      // Inline implementation of operator= to speed the no-op case.
      if (this != &rhs && (fClassPtr == 0 || fClassPtr != rhs.fClassPtr)) {
         this->Assign(rhs);
      }
      return *this;
   }
   inline TClassRef &operator=(TClass *rhs) {
      // Inline implementation of operator= to speed the no-op case.
      if (this->fClassPtr != rhs) {
         this->Assign(rhs);
      }
      return *this;
   }      

   ~TClassRef() { if (fClassPtr) fClassPtr->RemoveRef(this); };

   void SetName(const char* new_name) { 
      if ( fClassPtr && fClassName != new_name ) Reset(); 
      fClassName = new_name; 
   }
   const char *GetClassName() { return fClassName.c_str(); }
   TClass *GetClass()  const { return fClassPtr ? fClassPtr : InternalGetClass(); }
   void Reset() { if (fClassPtr) fClassPtr->RemoveRef(this); fClassPtr = 0; }

   TClass* operator->() const { return fClassPtr ? fClassPtr : InternalGetClass(); }
   operator TClass*() const { return fClassPtr ? fClassPtr : InternalGetClass(); }

};

#endif

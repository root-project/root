// @(#)root/meta:$Name:  $:$Id: TClassRef.h,v 1.2 2005/03/20 21:25:12 brun Exp $
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

public:
   TClassRef();
   TClassRef(TClass *cl);
   TClassRef(const char *classname);
   TClassRef(const TClassRef&);
   TClassRef& operator=(const TClassRef&);

   ~TClassRef();

   TClass *GetClass();
   void Reset() { fClassPtr = 0; }

   TClass* operator->() { return GetClass(); }
   operator TClass*() { return GetClass(); }

};

#endif

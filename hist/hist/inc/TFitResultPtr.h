// @(#)root/mathcore:$Id$
// Author: David Gonzalez Maline Tue Nov 10 15:01:24 2009

/*************************************************************************
 * Copyright (C) 1995-2009, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFitResultPtr
#define ROOT_TFitResultPtr

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFitResultPtr                                                        //
//                                                                      //
// Provides an indirection to TFitResult class and with a semantics     //
// identical to a TFitResult pointer                                    //
//                                                                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

class TFitResult;

class TFitResultPtr {
public:

   TFitResultPtr(int status = 0): fStatus(status), fPointer(0) {};

   TFitResultPtr(TFitResult* p): fStatus(0), fPointer(p) {};

   TFitResultPtr(const TFitResultPtr& rhs); 

   operator int() const;
   
   TFitResult& operator*() const;

   TFitResult* operator->() const;

   TFitResult* Get() const { return fPointer; }

   TFitResultPtr& operator= (const TFitResultPtr& rhs); 

   virtual ~TFitResultPtr();

private:
   
   int fStatus;          // fit status code
   TFitResult* fPointer; // Smart Pointer to TFitResult class  

   ClassDef(TFitResultPtr,1)
};

#endif

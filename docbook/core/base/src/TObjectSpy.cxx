// @(#)root/base:$Id$
// Author: Matevz Tadel   16/08/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TObjectSpy.h"
#include "TROOT.h"


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TObjectSpy, TObjectRefSpy                                            //
//                                                                      //
// Monitors objects for deletion and reflects the deletion by reverting //
// the internal pointer to zero. When this pointer is zero we know the  //
// object has been deleted. This avoids the unsafe TestBit(kNotDeleted) //
// hack. The spied object must have the kMustCleanup bit set otherwise  //
// you will get an error.                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

ClassImp(TObjectSpy)
ClassImp(TObjectRefSpy)

//______________________________________________________________________________
TObjectSpy::TObjectSpy(TObject *obj, Bool_t fixMustCleanupBit) :
   TObject(), fObj(obj), fResetMustCleanupBit(kFALSE)
{
   // Register the object that must be spied. The object must have the
   // kMustCleanup bit set. If the object has been deleted during a
   // RecusiveRemove() operation, GetObject() will return 0.

   gROOT->GetListOfCleanups()->Add(this);
   if (fObj && !fObj->TestBit(kMustCleanup)) {
      if (fixMustCleanupBit) {
         fResetMustCleanupBit = kTRUE;
         fObj->SetBit(kMustCleanup, kTRUE);
      } else {
         Error("TObjectSpy", "spied object must have the kMustCleanup bit set");
      }
   }
}

//______________________________________________________________________________
TObjectSpy::~TObjectSpy()
{
   // Cleanup.

   if (fObj && fResetMustCleanupBit)
      fObj->SetBit(kMustCleanup, kFALSE);
   gROOT->GetListOfCleanups()->Remove(this);
}

//______________________________________________________________________________
void TObjectSpy::RecursiveRemove(TObject *obj)
{
   // Sets the object pointer to zero if the object is deleted in the
   // RecursiveRemove() operation.

   if (obj == fObj) {
      fObj = 0;
      fResetMustCleanupBit = kFALSE;
   }
}

//______________________________________________________________________________
void TObjectSpy::SetObject(TObject *obj, Bool_t fixMustCleanupBit)
{
   // Set obj as the spy target.

   if (fObj && fResetMustCleanupBit)
      fObj->SetBit(kMustCleanup, kFALSE);
   fResetMustCleanupBit = kFALSE;

   fObj = obj;

   if (fObj && !fObj->TestBit(kMustCleanup)) {
      if (fixMustCleanupBit) {
         fResetMustCleanupBit = kTRUE;
         fObj->SetBit(kMustCleanup, kTRUE);
      } else {
         Error("TObjectSpy", "spied object must have the kMustCleanup bit set");
      }
   }
}


//______________________________________________________________________________
TObjectRefSpy::TObjectRefSpy(TObject *&obj, Bool_t fixMustCleanupBit) :
   fObj(obj), fResetMustCleanupBit(kFALSE)
{
   // Register the object that must be spied. The object must have the
   // kMustCleanup bit set. If the object has been deleted during a
   // RecusiveRemove() operation, GetObject() will return 0.

   gROOT->GetListOfCleanups()->Add(this);
   if (fObj && !fObj->TestBit(kMustCleanup)) {
      if (fixMustCleanupBit) {
         fResetMustCleanupBit = kTRUE;
         fObj->SetBit(kMustCleanup, kTRUE);
      } else {
         Error("TObjectSpy", "spied object must have the kMustCleanup bit set");
      }
   }
}

//______________________________________________________________________________
TObjectRefSpy::~TObjectRefSpy()
{
   // Cleanup.

   if (fObj && fResetMustCleanupBit)
      fObj->SetBit(kMustCleanup, kFALSE);
   gROOT->GetListOfCleanups()->Remove(this);
}

//______________________________________________________________________________
void TObjectRefSpy::RecursiveRemove(TObject *obj)
{
   // Sets the object pointer to zero if the object is deleted in the
   // RecursiveRemove() operation.

   if (obj == fObj) {
      fObj = 0;
      fResetMustCleanupBit = kFALSE;
   }
}

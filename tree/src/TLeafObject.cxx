// @(#)root/tree:$Name$:$Id$
// Author: Rene Brun   27/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// A TLeaf for a general object derived from TObject.                   //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TLeafObject.h"
#include "TBranch.h"
#include "TClass.h"
#include "TMethodCall.h"
#include "TDataType.h"


ClassImp(TLeafObject)

//______________________________________________________________________________
TLeafObject::TLeafObject(): TLeaf()
{
//*-*-*-*-*-*Default constructor for LeafObject*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*        =================================
   fClass      = 0;
   fObjAddress = 0;
}

//______________________________________________________________________________
TLeafObject::TLeafObject(const char *name, const char *type)
       :TLeaf(name,type)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create a LeafObject*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==================
//*-*

  SetTitle(type);
  fClass      = gROOT->GetClass(type);
  fObjAddress = 0;
}

//______________________________________________________________________________
TLeafObject::~TLeafObject()
{
//*-*-*-*-*-*Default destructor for a LeafObject*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ==================================

}


//______________________________________________________________________________
void TLeafObject::FillBasket(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*-*-*Pack leaf elements in Basket output buffer*-*-*-*-*-*-*
//*-*                  =========================================

   TObject *object  = GetObject();
   if (object) {
      object->Streamer(b);
   } else {
     if (fClass) {
        object = (TObject *)fClass->New();
        object->SetBit(kInvalidObject);
        object->SetUniqueID(123456789);
        object->Streamer(b);
        delete object;
     } else {
        Error("FillBasket","Attempt to write a NULL object in leaf:%s",GetName());
     }
   }
}

//______________________________________________________________________________
TMethodCall *TLeafObject::GetMethodCall(char *name)
{
//*-*-*-*-*-*-*-*Returns pointer to method corresponding to name*-*-*-*-*-*-*
//*-*            ============================================
//*-*
//*-*    name is a string with the general form  "method(list of params)"
//*-*   If list of params is omitted, () is assumed;
//*-*

   char *params = strchr(name,'(');
   if (params) { *params = 0; params++; }
   else params = ")";

   if (!fClass) fClass      = gROOT->GetClass(GetTitle());
   TMethodCall *m = new TMethodCall(fClass, name, params);
   if (m->GetMethod()) return m;
   Error("GetMethodCall","Unknown method:%s",name);
   delete m;
   return 0;
}

//______________________________________________________________________________
const char *TLeafObject::GetTypeName() const
{
//*-*-*-*-*-*-*-*Returns name of leaf type*-*-*-*-*-*-*-*-*-*-*-*
//*-*            =========================

   return fTitle.Data();
}

//______________________________________________________________________________
void TLeafObject::ReadBasket(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*-*-*Read leaf elements from Basket input buffer*-*-*-*-*-*
//*-*                  ===========================================

   if (fClass) {
      TObject *object;
      if (!fObjAddress) {
         Long_t *voidobj = new Long_t[1];
         fObjAddress  = (void **)voidobj;
         *fObjAddress = (TObject *)fClass->New();
      }
      object = (TObject*)(*fObjAddress);
      if (fBranch->IsAutoDelete()) {
         delete object;
         object = (TObject *)fClass->New();
      }
      if (!object) return;
      object->Streamer(b);
      // in case we had written a null pointer a Zombie object was created
      // we must delete it
      if (object->TestBit(kInvalidObject)) {
         if (object->GetUniqueID() == 123456789) {
            delete object;
            object = 0;
         }
      }
      *fObjAddress = object;
   } else GetBranch()->SetAddress(0);
}

//______________________________________________________________________________
void TLeafObject::SetAddress(void *add)
{
//*-*-*-*-*-*-*-*-*-*-*Set leaf buffer data address*-*-*-*-*-*
//*-*                  ============================

   fObjAddress = (void **)add;
}

//______________________________________________________________________________
void TLeafObject::Streamer(TBuffer &b)
{
   // Stream an object of class TLeafObject.

   if (b.IsReading()) {
      b.ReadVersion();  //Version_t v = b.ReadVersion();
      TLeaf::Streamer(b);
      fObjAddress = 0;
      fClass  = gROOT->GetClass(fTitle.Data());
      if (!fClass) Warning("Streamer","Cannot find class:%s",fTitle.Data());
      } else {
      b.WriteVersion(TLeafObject::IsA());
      TLeaf::Streamer(b);
   }
}

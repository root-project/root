// @(#)root/tree:$Name:  $:$Id: TLeafElement.cxx,v 1.5 2001/01/12 14:30:36 brun Exp $
// Author: Rene Brun   14/01/2001

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
#include "TLeafElement.h"
#include "TStreamerInfo.h"
#include "TStreamerElement.h"
#include "TBranchElement.h"
#include "TClass.h"
#include "TMethodCall.h"
#include "TDataType.h"


ClassImp(TLeafElement)

//______________________________________________________________________________
TLeafElement::TLeafElement(): TLeaf()
{
//*-*-*-*-*-*Default constructor for LeafObject*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*        =================================
   fClass      = 0;
   fObjAddress = 0;
   fVirtual    = kTRUE;
   fInfo       = 0;
   fElement    = 0;
}

//______________________________________________________________________________
TLeafElement::TLeafElement(TStreamerInfo *sinfo, TStreamerElement *element, Int_t id, const char *type)
       :TLeaf(type,type)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create a LeafObject*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==================
//*-*

  SetTitle(type);
  fClass      = gROOT->GetClass(type);
  fObjAddress = 0;
  fVirtual    = kTRUE;
  fInfo       = sinfo;
  fElement    = element;
  fID         = id;
}

//______________________________________________________________________________
TLeafElement::~TLeafElement()
{
//*-*-*-*-*-*Default destructor for a LeafObject*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ==================================

}


//______________________________________________________________________________
void TLeafElement::FillBasket(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*-*-*Pack leaf elements in Basket output buffer*-*-*-*-*-*-*
//*-*                  =========================================

   char **apointer = (char**)fBranch->GetAddress();
   char *pointer = (char*)(*apointer);
   if (fID >= 0) fInfo->WriteBuffer(b,pointer,fID);
   printf("TLeafElement::FillBasket, id=%d, leaf= %s\n",fID,GetName());
}

//______________________________________________________________________________
TMethodCall *TLeafElement::GetMethodCall(const char *name)
{
//*-*-*-*-*-*-*-*Returns pointer to method corresponding to name*-*-*-*-*-*-*
//*-*            ============================================
//*-*
//*-*    name is a string with the general form  "method(list of params)"
//*-*   If list of params is omitted, () is assumed;
//*-*

   char *namecpy = new char[strlen(name)+1];
   strcpy(namecpy,name);
   char *params = strchr(namecpy,'(');
   if (params) { *params = 0; params++; }
   else params = ")";

   if (!fClass) fClass      = gROOT->GetClass(GetTitle());
   TMethodCall *m = new TMethodCall(fClass, namecpy, params);
   delete [] namecpy;
   if (m->GetMethod()) return m;
   Error("GetMethodCall","Unknown method:%s",name);
   delete m;
   return 0;
}

//______________________________________________________________________________
const char *TLeafElement::GetTypeName() const
{
//*-*-*-*-*-*-*-*Returns name of leaf type*-*-*-*-*-*-*-*-*-*-*-*
//*-*            =========================

   return fTitle.Data();
}

//______________________________________________________________________________
void TLeafElement::ReadBasket(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*-*-*Read leaf elements from Basket input buffer*-*-*-*-*-*
//*-*                  ===========================================

/*
   char classname[128];
   UChar_t n;
   if (fVirtual) {
      b >> n;
      b.ReadFastArray(classname,n+1);
      fClass      = gROOT->GetClass(GetTitle());
   }
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
*/
}

//______________________________________________________________________________
void TLeafElement::SetAddress(void *add)
{
//*-*-*-*-*-*-*-*-*-*-*Set leaf buffer data address*-*-*-*-*-*
//*-*                  ============================

   fObjAddress = add;
}

//______________________________________________________________________________
void TLeafElement::Streamer(TBuffer &b)
{
   // Stream an object of class TLeafElement.

   if (b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TLeafElement::Class()->ReadBuffer(b, this, R__v, R__s, R__c);
         fObjAddress = 0;
         fClass  = gROOT->GetClass(fTitle.Data());
         if (!fClass) Warning("Streamer","Cannot find class:%s",fTitle.Data());
         return;
      }
      //====process old versions before automatic schema evolution
      TLeaf::Streamer(b);
      fObjAddress = 0;
      fClass  = gROOT->GetClass(fTitle.Data());
      if (!fClass) Warning("Streamer","Cannot find class:%s",fTitle.Data());
      if (R__v < 1) fVirtual = kFALSE;
      //====end of old versions
      
   } else {
      TLeafElement::Class()->WriteBuffer(b,this);
   }
}

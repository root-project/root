// @(#)root/tree:$Name:  $:$Id: TLeafD.cxx,v 1.7 2001/01/24 16:32:24 brun Exp $
// Author: Rene Brun   12/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// A TLeaf for a 64 bit floating point data type.                       //
//////////////////////////////////////////////////////////////////////////

#include "TLeafD.h"
#include "TBranch.h"

ClassImp(TLeafD)

//______________________________________________________________________________
TLeafD::TLeafD(): TLeaf()
{
//*-*-*-*-*-*Default constructor for LeafD*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ============================

   fValue = 0;
   fPointer = 0;
}

//______________________________________________________________________________
TLeafD::TLeafD(const char *name, const char *type)
       :TLeaf(name,type)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create a LeafD*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==============
//*-*

   fLenType = 8;
   fMinimum = 0;
   fMaximum = 0;
   fValue   = 0;
   fPointer = 0;
}

//______________________________________________________________________________
TLeafD::~TLeafD()
{
//*-*-*-*-*-*Default destructor for a LeafD*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ===============================

   if (ResetAddress(0,kTRUE)) delete [] fValue;
}


//______________________________________________________________________________
void TLeafD::Export(TClonesArray *list, Int_t n)
{
//*-*-*-*-*-*Export element from local leaf buffer to ClonesArray*-*-*-*-*
//*-*        ====================================================

   Int_t j = 0;
   for (Int_t i=0;i<n;i++) {
      memcpy((char*)list->UncheckedAt(i) + fOffset,&fValue[j], 8*fLen);
      j += fLen;
   }
}


//______________________________________________________________________________
void TLeafD::FillBasket(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*-*-*Pack leaf elements in Basket output buffer*-*-*-*-*-*-*
//*-*                  ==========================================

   Int_t len = GetLen();
   if (fPointer) fValue = *fPointer;
   b.WriteFastArray(fValue,len);
}


//______________________________________________________________________________
void TLeafD::Import(TClonesArray *list, Int_t n)
{
//*-*-*-*-*-*Import element from ClonesArray into local leaf buffer*-*-*-*-*
//*-*        ======================================================

   const Double_t kDoubleUndefined = -9999.;
   Int_t j = 0;
   char *clone;
   for (Int_t i=0;i<n;i++) {
      clone = (char*)list->UncheckedAt(i);
      if (clone) memcpy(&fValue[j],clone + fOffset, 8*fLen);
      else       memcpy(&fValue[j],&kDoubleUndefined,  8*fLen);
      j += fLen;
   }
}

//______________________________________________________________________________
void TLeafD::PrintValue(Int_t l) const
{
// Prints leaf value

   Double_t *value = (Double_t *)GetValuePointer();
   printf("%g",value[l]);
}

//______________________________________________________________________________
void TLeafD::ReadBasket(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*-*-*Read leaf elements from Basket input buffer*-*-*-*-*-*
//*-*                  ===========================================

   if (fNdata == 1) {
      b >> fValue[0];
   }else {
      if (fLeafCount) {
         Int_t len = Int_t(fLeafCount->GetValue());
         if (len > fLeafCount->GetMaximum()) {
            printf("ERROR leaf:%s, len=%d and max=%d\n",GetName(),len,fLeafCount->GetMaximum());
            len = fLeafCount->GetMaximum();
         }
         b.ReadFastArray(fValue,len*fLen);
      } else {
         b.ReadFastArray(fValue,fLen);
      }
   }
}

//______________________________________________________________________________
void TLeafD::ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n)
{
//*-*-*-*-*-*-*-*-*-*-*Read leaf elements from Basket input buffer*-*-*-*-*-*
//  and export buffer to TClonesArray objects

   b.ReadFastArray(fValue,n*fLen);

   Int_t j = 0;
   for (Int_t i=0;i<n;i++) {
      memcpy((char*)list->UncheckedAt(i) + fOffset,&fValue[j], 8*fLen);
      j += fLen;
   }
}

//______________________________________________________________________________
void TLeafD::SetAddress(void *add)
{
//*-*-*-*-*-*-*-*-*-*-*Set leaf buffer data address*-*-*-*-*-*
//*-*                  ============================

   if (ResetAddress(add)) {
      delete [] fValue;
   }
   if (add) {
      if (fLeafCount) {
         fPointer = (Double_t**) add;
         fNdata = fLen;
         Int_t ncountmax = Int_t(fLeafCount->GetMaximum()+1);
         if (ncountmax > fNdata || *fPointer == 0) {
            delete [] *fPointer;
            if (ncountmax > fNdata) fNdata = ncountmax;
            *fPointer = new Double_t[fNdata];
         }
         fValue = *fPointer;
      } else {
         fValue = (Double_t*)add;
      }
   } else {
      fValue = new Double_t[fNdata];
   }
}

// @(#)root/tree:$Name:  $:$Id: TLeafI.cxx,v 1.8 2001/01/24 16:32:24 brun Exp $
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
// A TLeaf for an Integer data type.                                    //
//////////////////////////////////////////////////////////////////////////

#include "TLeafI.h"
#include "TBranch.h"

ClassImp(TLeafI)

//______________________________________________________________________________
TLeafI::TLeafI(): TLeaf()
{
//*-*-*-*-*-*Default constructor for LeafI*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ============================

   fValue = 0;
   fPointer = 0;
}

//______________________________________________________________________________
TLeafI::TLeafI(const char *name, const char *type)
       :TLeaf(name,type)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create a LeafI*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==============
//*-*

   fLenType = 4;
   fMinimum = 0;
   fMaximum = 0;
   fValue   = 0;
   fPointer = 0;
}

//______________________________________________________________________________
TLeafI::~TLeafI()
{
//*-*-*-*-*-*Default destructor for a LeafI*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ===============================

   if (ResetAddress(0,kTRUE)) delete [] fValue;
}


//______________________________________________________________________________
void TLeafI::Export(TClonesArray *list, Int_t n)
{
//*-*-*-*-*-*Export element from local leaf buffer to ClonesArray*-*-*-*-*
//*-*        ======================================================

   Int_t *value = fValue;
   for (Int_t i=0;i<n;i++) {
      char *first = (char*)list->UncheckedAt(i);
      Int_t *ii = (Int_t*)&first[fOffset];
      for (Int_t j=0;j<fLen;j++) {
         ii[j] = value[j];
      }
      value += fLen;
   }
}

//______________________________________________________________________________
void TLeafI::FillBasket(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*-*-*Pack leaf elements in Basket output buffer*-*-*-*-*-*-*
//*-*                  =========================================

   Int_t i;
   Int_t len = GetLen();
   if (fPointer) fValue = *fPointer;
   if (IsRange()) {
      if (fValue[0] > fMaximum) fMaximum = fValue[0];
   }
   if (IsUnsigned()) {
      for (i=0;i<len;i++) b << (UInt_t)fValue[i];
   } else {
      b.WriteFastArray(fValue,len);
   }
}

//______________________________________________________________________________
const char *TLeafI::GetTypeName() const
{
//*-*-*-*-*-*-*-*Returns name of leaf type*-*-*-*-*-*-*-*-*-*-*-*
//*-*            =========================

   if (fIsUnsigned) return "UInt_t";
   return "Int_t";
}


//______________________________________________________________________________
Double_t TLeafI::GetValue(Int_t i) const
{
//*-*-*-*-*-*-*-*Returns current value of leaf*-*-*-*-*-*-*-*-*-*-*-*
//*-*            =============================

   if (fIsUnsigned) return (UInt_t)fValue[i];
   return fValue[i];
}



//______________________________________________________________________________
void TLeafI::Import(TClonesArray *list, Int_t n)
{
//*-*-*-*-*-*Import element from ClonesArray into local leaf buffer*-*-*-*-*
//*-*        ======================================================

   const Int_t kIntUndefined = -9999;
   Int_t j = 0;
   char *clone;
   for (Int_t i=0;i<n;i++) {
      clone = (char*)list->UncheckedAt(i);
      if (clone) memcpy(&fValue[j],clone + fOffset, 4*fLen);
      else       memcpy(&fValue[j],&kIntUndefined,  4*fLen);
      j += fLen;
   }
}

//______________________________________________________________________________
void TLeafI::PrintValue(Int_t l) const
{
// Prints leaf value

   Int_t *value = (Int_t*)GetValuePointer();
   printf("%d",value[l]);
}

//______________________________________________________________________________
void TLeafI::ReadBasket(TBuffer &b)
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
void TLeafI::ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n)
{
//*-*-*-*-*-*-*-*-*-*-*Read leaf elements from Basket input buffer*-*-*-*-*-*
//  and export buffer to TClonesArray objects

   if (n*fLen == 1) {
      b >> fValue[0];
   } else {
      b.ReadFastArray(fValue,n*fLen);
   }
   Int_t *value = fValue;
   for (Int_t i=0;i<n;i++) {
      char *first = (char*)list->UncheckedAt(i);
      Int_t *ii = (Int_t*)&first[fOffset];
      for (Int_t j=0;j<fLen;j++) {
         ii[j] = value[j];
      }
      value += fLen;
   }
}

//______________________________________________________________________________
void TLeafI::SetAddress(void *add)
{
//*-*-*-*-*-*-*-*-*-*-*Set leaf buffer data address*-*-*-*-*-*
//*-*                  ============================

   if (ResetAddress(add)) {
      delete [] fValue;
   }
   if (add) {
      if (TestBit(kIndirectAddress)) {
         fPointer = (Int_t**) add;
         fNdata = fLen;
         Int_t ncountmax = fLen*(fLeafCount->GetMaximum() + 1);
         if (ncountmax > fNdata || *fPointer == 0) {
            delete [] *fPointer;
            if (ncountmax > fNdata) fNdata = ncountmax;
            *fPointer = new Int_t[fNdata];
         }
         fValue = *fPointer;
      } else {
         fValue = (Int_t*)add;
      }
   } else {
     fValue = new Int_t[fNdata];
   }
}


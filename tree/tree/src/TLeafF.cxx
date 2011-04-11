// @(#)root/tree:$Id$
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
// A TLeaf for a 32 bit floating point data type.                       //
//////////////////////////////////////////////////////////////////////////

#include "TLeafF.h"
#include "TBranch.h"
#include "TClonesArray.h"
#include "Riostream.h"

ClassImp(TLeafF)

//______________________________________________________________________________
TLeafF::TLeafF(): TLeaf()
{
//*-*-*-*-*-*Default constructor for LeafF*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ============================

   fLenType = 4;
   fMinimum = 0;
   fMaximum = 0;
   fValue   = 0;
   fPointer = 0;
}

//______________________________________________________________________________
TLeafF::TLeafF(TBranch *parent, const char *name, const char *type)
   :TLeaf(parent, name,type)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create a LeafF*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==============
//*-*

   fLenType = 4;
   fMinimum = 0;
   fMaximum = 0;
   fValue   = 0;
   fPointer = 0;
}

//______________________________________________________________________________
TLeafF::~TLeafF()
{
//*-*-*-*-*-*Default destructor for a LeafF*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ===============================

   if (ResetAddress(0,kTRUE)) delete [] fValue;
}



//______________________________________________________________________________
void TLeafF::Export(TClonesArray *list, Int_t n)
{
//*-*-*-*-*-*Export element from local leaf buffer to ClonesArray*-*-*-*-*
//*-*        ====================================================

   Float_t *value = fValue;
   for (Int_t i=0;i<n;i++) {
      char *first = (char*)list->UncheckedAt(i);
      Float_t *ff = (Float_t*)&first[fOffset];
      for (Int_t j=0;j<fLen;j++) {
         ff[j] = value[j];
      }
      value += fLen;
   }
}

//______________________________________________________________________________
void TLeafF::FillBasket(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*-*-*Pack leaf elements in Basket output buffer*-*-*-*-*-*-*
//*-*                  ==========================================

   Int_t len = GetLen();
   if (fPointer) fValue = *fPointer;
   b.WriteFastArray(fValue,len);
}


//______________________________________________________________________________
void TLeafF::Import(TClonesArray *list, Int_t n)
{
//*-*-*-*-*-*Import element from ClonesArray into local leaf buffer*-*-*-*-*
//*-*        ======================================================

   const Float_t kFloatUndefined = -9999.;
   Int_t j = 0;
   char *clone;
   for (Int_t i=0;i<n;i++) {
      clone = (char*)list->UncheckedAt(i);
      if (clone) memcpy(&fValue[j],clone + fOffset,  4*fLen);
      else       memcpy(&fValue[j],&kFloatUndefined, 4*fLen);
      j += fLen;
   }
}

//______________________________________________________________________________
void TLeafF::PrintValue(Int_t l) const
{
// Prints leaf value

   Float_t *value = (Float_t *)GetValuePointer();
   printf("%g",value[l]);
}

//______________________________________________________________________________
void TLeafF::ReadBasket(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*-*-*Read leaf elements from Basket input buffer*-*-*-*-*-*
//*-*                  ===========================================

   if (!fLeafCount && fNdata == 1) {
      b.ReadFloat(fValue[0]);
   }else {
      if (fLeafCount) {
         Long64_t entry = fBranch->GetReadEntry();
         if (fLeafCount->GetBranch()->GetReadEntry() != entry) {
            fLeafCount->GetBranch()->GetEntry(entry);
         }
         Int_t len = Int_t(fLeafCount->GetValue());
         if (len > fLeafCount->GetMaximum()) {
            printf("ERROR leaf:%s, len=%d and max=%d\n",GetName(),len,fLeafCount->GetMaximum());
            len = fLeafCount->GetMaximum();
         }
         fNdata = len*fLen;
         b.ReadFastArray(fValue,len*fLen);
      } else {
         b.ReadFastArray(fValue,fLen);
      }
   }
}

//______________________________________________________________________________
void TLeafF::ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n)
{
//*-*-*-*-*-*-*-*-*-*-*Read leaf elements from Basket input buffer*-*-*-*-*-*
//  and export buffer to TClonesArray objects

   if (n*fLen == 1) {
      b >> fValue[0];
   } else {
      b.ReadFastArray(fValue,n*fLen);
   }

   Float_t *value = fValue;
   for (Int_t i=0;i<n;i++) {
      char *first = (char*)list->UncheckedAt(i);
      Float_t *ff = (Float_t*)&first[fOffset];
      for (Int_t j=0;j<fLen;j++) {
         ff[j] = value[j];
      }
      value += fLen;
   }
}

//______________________________________________________________________________
void TLeafF::ReadValue(istream &s)
{
// read a float from istream s and store it into the branch buffer
   Float_t *value = (Float_t*)GetValuePointer();
   for (Int_t i=0;i<fLen;i++) s >> value[i];
}

//______________________________________________________________________________
void TLeafF::SetAddress(void *add)
{
//*-*-*-*-*-*-*-*-*-*-*Set leaf buffer data address*-*-*-*-*-*
//*-*                  ============================

   if (ResetAddress(add) && (add!= fValue)) {
      delete [] fValue;
   }

   if (add) {
      if (TestBit(kIndirectAddress)) {
         fPointer = (Float_t**) add;
         Int_t ncountmax = fLen;
         if (fLeafCount) ncountmax = fLen*(fLeafCount->GetMaximum() + 1);
         if ((fLeafCount && ncountmax > Int_t(fLeafCount->GetValue())) ||
             ncountmax > fNdata || *fPointer == 0) {
            if (*fPointer) delete [] *fPointer;
            if (ncountmax > fNdata) fNdata = ncountmax;
            *fPointer = new Float_t[fNdata];
         }
         fValue = *fPointer;
      } else {
         fValue = (Float_t*)add;
      }
   } else {
      fValue = new Float_t[fNdata];
      fValue[0] = 0;
   }
}

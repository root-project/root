// @(#)root/tree:$Id$
// Author: Philippe Canal  20/1/05

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// A TLeaf for a bool data type.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TLeafO.h"
#include "TBranch.h"
#include "TClonesArray.h"
#include "Riostream.h"

ClassImp(TLeafO)

//______________________________________________________________________________
TLeafO::TLeafO(): TLeaf()
{
//*-*-*-*-*-*Default constructor for LeafB*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ============================

   fValue   = 0;
   fPointer = 0;
   fMinimum = 0;
   fMaximum = 0;
   fLenType = sizeof(bool);
}

//______________________________________________________________________________
TLeafO::TLeafO(TBranch *parent, const char *name, const char *type)
   : TLeaf(parent,name,type)
{
//*-*-*-*-*-*-*-*-*-*-*-*-*Create a LeafB*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                      ==============
//*-*

   fLenType = sizeof(bool);
   fMinimum = 0;
   fMaximum = 0;
   fValue   = 0;
   fPointer = 0;
}

//______________________________________________________________________________
TLeafO::~TLeafO()
{
//*-*-*-*-*-*Default destructor for a LeafB*-*-*-*-*-*-*-*-*-*-*-*
//*-*        ===============================

   if (ResetAddress(0,kTRUE)) delete [] fValue;
}


//______________________________________________________________________________
void TLeafO::Export(TClonesArray *list, Int_t n)
{
//*-*-*-*-*-*Export element from local leaf buffer to ClonesArray*-*-*-*-*
//*-*        ====================================================

   Int_t j = 0;
   for (Int_t i=0;i<n;i++) {
      memcpy((char*)list->UncheckedAt(i) + fOffset,&fValue[j], fLen);
      j += fLen;
   }
}


//______________________________________________________________________________
void TLeafO::FillBasket(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*-*-*Pack leaf elements in Basket output buffer*-*-*-*-*-*-*
//*-*                  ==========================================

   Int_t len = GetLen();
   if (fPointer) fValue = *fPointer;
   b.WriteFastArray(fValue,len);
}

//______________________________________________________________________________
const char *TLeafO::GetTypeName() const
{
//*-*-*-*-*-*-*-*Returns name of leaf type*-*-*-*-*-*-*-*-*-*-*-*
//*-*            =========================

   return "Bool_t";
}


//______________________________________________________________________________
void TLeafO::Import(TClonesArray *list, Int_t n)
{
//*-*-*-*-*-*Import element from ClonesArray into local leaf buffer*-*-*-*-*
//*-*        ======================================================

   Int_t j = 0;
   for (Int_t i=0;i<n;i++) {
      memcpy(&fValue[j],(char*)list->UncheckedAt(i) + fOffset, fLen);
      j += fLen;
   }
}

//______________________________________________________________________________
void TLeafO::PrintValue(Int_t l) const
{
// Prints leaf value

   char *value = (char*)GetValuePointer();
   printf("%d",(Int_t)value[l]);
}


//______________________________________________________________________________
void TLeafO::ReadBasket(TBuffer &b)
{
//*-*-*-*-*-*-*-*-*-*-*Read leaf elements from Basket input buffer*-*-*-*-*-*
//*-*                  ===========================================

   if (!fLeafCount && fNdata == 1) {
      b >> fValue[0];
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
void TLeafO::ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n)
{
//*-*-*-*-*-*-*-*-*-*-*Read leaf elements from Basket input buffer*-*-*-*-*-*
//  and export buffer to TClonesArray objects

   b.ReadFastArray(fValue,n*fLen);

   Int_t j = 0;
   for (Int_t i=0;i<n;i++) {
      memcpy((char*)list->UncheckedAt(i) + fOffset,&fValue[j], fLen);
      j += fLen;
   }
}

//______________________________________________________________________________
void TLeafO::ReadValue(istream &s)
{
// read a string from istream s and store it into the branch buffer
   char *value = (char*)GetValuePointer();
   s >> value;
}

//______________________________________________________________________________
void TLeafO::SetAddress(void *add)
{
//*-*-*-*-*-*-*-*-*-*-*Set leaf buffer data address*-*-*-*-*-*
//*-*                  ============================

   if (ResetAddress(add)) {
      delete [] fValue;
   }
   if (add) {
      if (TestBit(kIndirectAddress)) {
         fPointer = (Bool_t**) add;
         Int_t ncountmax = fLen;
         if (fLeafCount) ncountmax = fLen*(fLeafCount->GetMaximum() + 1);
         if ((fLeafCount && ncountmax > Int_t(fLeafCount->GetValue())) ||
             ncountmax > fNdata || *fPointer == 0) {
            if (*fPointer) delete [] *fPointer;
            if (ncountmax > fNdata) fNdata = ncountmax;
            *fPointer = new Bool_t[fNdata];
         }
         fValue = *fPointer;
      } else {
         fValue = (bool*)add;
      }
   } else {
      fValue = new bool[fNdata];
      fValue[0] = 0;
   }
}

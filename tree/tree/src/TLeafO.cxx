// @(#)root/tree:$Id$
// Author: Philippe Canal  20/1/05

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TLeafO
\ingroup tree

A TLeaf for a bool data type.
*/

#include "TLeafO.h"
#include "TBranch.h"
#include "TBuffer.h"
#include "TClonesArray.h"
#include <iostream>


////////////////////////////////////////////////////////////////////////////////
/// Default constructor for LeafO.

TLeafO::TLeafO(): TLeaf()
{
   fValue   = nullptr;
   fPointer = nullptr;
   fMinimum = false;
   fMaximum = false;
   fLenType = sizeof(bool);
}

////////////////////////////////////////////////////////////////////////////////
/// Create a LeafO.

TLeafO::TLeafO(TBranch *parent, const char *name, const char *type)
   : TLeaf(parent,name,type)
{
   fLenType = sizeof(bool);
   fMinimum = false;
   fMaximum = false;
   fValue   = nullptr;
   fPointer = nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Default destructor for a LeafO.

TLeafO::~TLeafO()
{
   if (ResetAddress(nullptr,true)) delete [] fValue;
}

////////////////////////////////////////////////////////////////////////////////
/// Export element from local leaf buffer to ClonesArray.

void TLeafO::Export(TClonesArray *list, Int_t n)
{
   Int_t j = 0;
   for (Int_t i=0;i<n;i++) {
      memcpy((char*)list->UncheckedAt(i) + fOffset,&fValue[j], fLen);
      j += fLen;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Pack leaf elements in Basket output buffer.

void TLeafO::FillBasket(TBuffer &b)
{
   Int_t len = GetLen();
   if (fPointer) fValue = *fPointer;
   if (IsRange()) {
      if (fValue[0] > fMaximum) fMaximum = fValue[0];
   }
   b.WriteFastArray(fValue,len);
}

////////////////////////////////////////////////////////////////////////////////
/// Returns name of leaf type.

const char *TLeafO::GetTypeName() const
{
   return "Bool_t";
}

////////////////////////////////////////////////////////////////////////////////
/// Copy/set fMinimum and fMaximum to include/be wide than those of the parameter

bool TLeafO::IncludeRange(TLeaf *input)
{
    if (input) {
        if (input->GetMaximum() > this->GetMaximum())
            this->SetMaximum( input->GetMaximum() );
        if (input->GetMinimum() < this->GetMinimum())
            this->SetMinimum( input->GetMinimum() );
        return true;
    } else {
        return false;
    }
}

////////////////////////////////////////////////////////////////////////////////
/// Import element from ClonesArray into local leaf buffer.

void TLeafO::Import(TClonesArray *list, Int_t n)
{
   Int_t j = 0;
   for (Int_t i=0;i<n;i++) {
      memcpy(&fValue[j],(char*)list->UncheckedAt(i) + fOffset, fLen);
      j += fLen;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Prints leaf value.

void TLeafO::PrintValue(Int_t l) const
{
   char *value = (char*)GetValuePointer();
   printf("%d",(Int_t)value[l]);
}

////////////////////////////////////////////////////////////////////////////////
/// Read leaf elements from Basket input buffer.

void TLeafO::ReadBasket(TBuffer &b)
{
   if (!fLeafCount && fNdata == 1) {
      b.ReadBool(fValue[0]);
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

////////////////////////////////////////////////////////////////////////////////
/// Read leaf elements from Basket input buffer and export buffer to
/// TClonesArray objects.

void TLeafO::ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n)
{
   b.ReadFastArray(fValue,n*fLen);

   Int_t j = 0;
   for (Int_t i=0;i<n;i++) {
      memcpy((char*)list->UncheckedAt(i) + fOffset,&fValue[j], fLen);
      j += fLen;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Read a string from std::istream s and store it into the branch buffer.

void TLeafO::ReadValue(std::istream &s, Char_t /*delim = ' '*/)
{
   char *value = (char*)GetValuePointer();
   for (Int_t i=0;i<fLen;i++) s >> value[i];
}

////////////////////////////////////////////////////////////////////////////////
/// Set leaf buffer data address.

void TLeafO::SetAddress(void *add)
{
   if (ResetAddress(add)) {
      delete [] fValue;
   }
   if (add) {
      if (TestBit(kIndirectAddress)) {
         fPointer = (bool**) add;
         Int_t ncountmax = fLen;
         if (fLeafCount) ncountmax = fLen*(fLeafCount->GetMaximum() + 1);
         if ((fLeafCount && ncountmax > Int_t(fLeafCount->GetValue())) ||
             ncountmax > fNdata || *fPointer == nullptr) {
            if (*fPointer) delete [] *fPointer;
            if (ncountmax > fNdata) fNdata = ncountmax;
            *fPointer = new bool[fNdata];
         }
         fValue = *fPointer;
      } else {
         fValue = (bool*)add;
      }
   } else {
      fValue = new bool[fNdata];
      fValue[0] = false;
   }
}

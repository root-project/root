// @(#)root/tree:$Id$
// Author: Rene Brun   12/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLeafB
#define ROOT_TLeafB


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLeafB                                                               //
//                                                                      //
// A TLeaf for an 8 bit Integer data type.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TLeaf.h"

class TLeafB : public TLeaf {

protected:
   Char_t       fMinimum;         ///<  Minimum value if leaf range is specified
   Char_t       fMaximum;         ///<  Maximum value if leaf range is specified
   Char_t       *fValue;          ///<! Pointer to data buffer
   Char_t       **fPointer;       ///<! Address of a pointer to data buffer!

public:
   TLeafB();
   TLeafB(TBranch *parent, const char* name, const char* type);
   virtual ~TLeafB();

   void            Export(TClonesArray* list, Int_t n) override;
   void            FillBasket(TBuffer& b) override;
   DeserializeType GetDeserializeType() const override { return DeserializeType::kZeroCopy; }
   Int_t           GetMaximum() const override { return fMaximum; }
   Int_t           GetMinimum() const override { return fMinimum; }
   const char     *GetTypeName() const override;
   Double_t        GetValue(Int_t i = 0) const override { return IsUnsigned() ? (Double_t)((UChar_t) fValue[i]) : (Double_t)fValue[i]; }
   void           *GetValuePointer() const override { return fValue; }
   Bool_t          IncludeRange(TLeaf *) override;
   void            Import(TClonesArray* list, Int_t n) override;
   void            PrintValue(Int_t i = 0) const override;
   void            ReadBasket(TBuffer&) override;
   void            ReadBasketExport(TBuffer&, TClonesArray* list, Int_t n) override;
   void            ReadValue(std::istream &s, Char_t delim = ' ') override;
   void            SetAddress(void* addr = nullptr) override;
   virtual void    SetMaximum(Char_t max) { fMaximum = max; }
   virtual void    SetMinimum(Char_t min) { fMinimum = min; }

   // Deserialize N events from an input buffer.  Since chars are stored unchanged, there
   // is nothing to do here but return true if we don't have variable-length arrays.
   bool            ReadBasketFast(TBuffer&, Long64_t) override { return true; }

   ClassDefOverride(TLeafB,1);  //A TLeaf for an 8 bit Integer data type.
};

#endif

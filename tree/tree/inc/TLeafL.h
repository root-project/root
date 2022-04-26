// @(#)root/tree:$Id$
// Author: Rene Brun   19/12/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLeafL
#define ROOT_TLeafL


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLeafL                                                               //
//                                                                      //
// A TLeaf for a 64 bit integer data type.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TLeaf.h"

class TLeafL : public TLeaf {

protected:
   Long64_t     fMinimum;         ///<  Minimum value if leaf range is specified
   Long64_t     fMaximum;         ///<  Maximum value if leaf range is specified
   Long64_t    *fValue;           ///<! Pointer to data buffer
   Long64_t   **fPointer;         ///<! Address of pointer to data buffer

public:
   TLeafL();
   TLeafL(TBranch *parent, const char *name, const char *type);
   virtual ~TLeafL();

   void            Export(TClonesArray *list, Int_t n) override;
   void            FillBasket(TBuffer &b) override;
   DeserializeType GetDeserializeType() const override { return DeserializeType::kInPlace; }
   const char     *GetTypeName() const override;
   Int_t           GetMaximum() const override { return (Int_t)fMaximum; }
   Int_t           GetMinimum() const override { return (Int_t)fMinimum; }
   Double_t        GetValue(Int_t i=0) const override;
   Long64_t        GetValueLong64(Int_t i = 0) const override;
   LongDouble_t    GetValueLongDouble(Int_t i = 0) const override;
   void           *GetValuePointer() const override { return fValue; }
   Bool_t          IncludeRange(TLeaf *) override;
   void            Import(TClonesArray *list, Int_t n) override;
   void            PrintValue(Int_t i=0) const override;
   void            ReadBasket(TBuffer &b) override;
   void            ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n) override;
   bool            ReadBasketFast(TBuffer&, Long64_t) override;
   void            ReadValue(std::istream& s, Char_t delim = ' ') override;
   void            SetAddress(void *add=nullptr) override;
   virtual void    SetMaximum(Long64_t max) {fMaximum = max;}
   virtual void    SetMinimum(Long64_t min) {fMinimum = min;}

   ClassDefOverride(TLeafL,1);  //A TLeaf for a 64 bit Integer data type.
};

// if leaf is a simple type, i must be set to 0
// if leaf is an array, i is the array element number to be returned
inline Long64_t TLeafL::GetValueLong64(Int_t i) const { return fValue[i]; }

#endif

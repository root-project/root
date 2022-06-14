// @(#)root/tree:$Id$
// Author: Enrico Guiraud

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLeafG
#define ROOT_TLeafG


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLeafG                                                               //
//                                                                      //
// A TLeaf for a long integer data type.                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TLeaf.h"

class TLeafG : public TLeaf {

protected:
   Long_t     fMinimum;       ///<  Minimum value if leaf range is specified
   Long_t     fMaximum;       ///<  Maximum value if leaf range is specified
   Long_t    *fValue;         ///<! Pointer to data buffer
   Long_t   **fPointer;       ///<! Address of pointer to data buffer

public:
   TLeafG();
   TLeafG(TBranch *parent, const char *name, const char *type);
   virtual ~TLeafG();

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
   virtual void    SetMaximum(Long_t max) {fMaximum = max;}
   virtual void    SetMinimum(Long_t min) {fMinimum = min;}

   ClassDefOverride(TLeafG,1);  //A TLeaf for a long integer data type.
};

// if leaf is a simple type, i must be set to 0
// if leaf is an array, i is the array element number to be returned
inline Long64_t TLeafG::GetValueLong64(Int_t i) const { return fValue[i]; }

#endif

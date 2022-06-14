// @(#)root/tree:$Id$
// Author: Rene Brun   12/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLeafS
#define ROOT_TLeafS


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLeafS                                                               //
//                                                                      //
// A TLeaf for a 16 bit Integer data type.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TLeaf.h"

class TLeafS : public TLeaf {

protected:
   Short_t       fMinimum;         ///<  Minimum value if leaf range is specified
   Short_t       fMaximum;         ///<  Maximum value if leaf range is specified
   Short_t       *fValue;          ///<! Pointer to data buffer
   Short_t       **fPointer;       ///<! Address of pointer to data buffer

public:
   TLeafS();
   TLeafS(TBranch *parent, const char *name, const char *type);
   virtual ~TLeafS();

   void            Export(TClonesArray *list, Int_t n) override;
   void            FillBasket(TBuffer &b) override;
   DeserializeType GetDeserializeType() const override { return DeserializeType::kInPlace; }
   Int_t           GetMaximum() const override { return fMaximum; }
   Int_t           GetMinimum() const override { return fMinimum; }
   const char     *GetTypeName() const override;
   Double_t        GetValue(Int_t i=0) const override;
   void           *GetValuePointer() const override { return fValue; }
   Bool_t          IncludeRange(TLeaf *) override;
   void            Import(TClonesArray *list, Int_t n) override;
   void            PrintValue(Int_t i=0) const override;
   void            ReadBasket(TBuffer &b) override;
   void            ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n) override;
   bool            ReadBasketFast(TBuffer&, Long64_t) override;
   void            ReadValue(std::istream& s, Char_t delim = ' ') override;
   void            SetAddress(void *add=nullptr) override;
   virtual void    SetMaximum(Short_t max) { fMaximum = max; }
   virtual void    SetMinimum(Short_t min) { fMinimum = min; }

   ClassDefOverride(TLeafS,1);  //A TLeaf for a 16 bit Integer data type.
};

#endif

// @(#)root/tree:$Id$
// Author: Rene Brun   12/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLeafD
#define ROOT_TLeafD


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLeafD                                                               //
//                                                                      //
// A TLeaf for a 64 bit floating point data type.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TLeaf.h"

class TLeafD : public TLeaf {

protected:
   Double_t       fMinimum;         ///<  Minimum value if leaf range is specified
   Double_t       fMaximum;         ///<  Maximum value if leaf range is specified
   Double_t       *fValue;          ///<! Pointer to data buffer
   Double_t       **fPointer;       ///<! Address of pointer to data buffer

public:
   TLeafD();
   TLeafD(TBranch *parent, const char *name, const char *type);
   virtual ~TLeafD();

   void            Export(TClonesArray *list, Int_t n) override;
   void            FillBasket(TBuffer &b) override;
   DeserializeType GetDeserializeType() const override { return DeserializeType::kInPlace; }
   const char     *GetTypeName() const override { return "Double_t"; }
   Double_t        GetValue(Int_t i=0) const override;
   void           *GetValuePointer() const override { return fValue; }
   void            Import(TClonesArray *list, Int_t n) override;
   void            PrintValue(Int_t i=0) const override;
   void            ReadBasket(TBuffer &b) override;
   void            ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n) override;
   void            ReadValue(std::istream& s, Char_t delim = ' ') override;
   void            SetAddress(void *add=nullptr) override;

   bool            ReadBasketFast(TBuffer&, Long64_t) override;

   ClassDefOverride(TLeafD,1);  //A TLeaf for a 64 bit floating point data type.
};

// if leaf is a simple type, i must be set to 0
// if leaf is an array, i is the array element number to be returned
inline Double_t TLeafD::GetValue(Int_t i) const { return fValue[i]; }

#endif

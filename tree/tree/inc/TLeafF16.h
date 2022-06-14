// @(#)root/tree:$Id$
// Author: Simon Spies 23/02/19

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLeafF16
#define ROOT_TLeafF16

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLeafF16                                                             //
//                                                                      //
// A TLeaf for a 24 bit truncated floating point data type.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TLeaf.h"

class TStreamerElement;

class TLeafF16 : public TLeaf {

protected:
   Float16_t fMinimum;         ///<  Minimum value if leaf range is specified
   Float16_t fMaximum;         ///<  Maximum value if leaf range is specified
   Float16_t *fValue;          ///<! Pointer to data buffer
   Float16_t **fPointer;       ///<! Address of pointer to data buffer!
   TStreamerElement *fElement; ///<! StreamerElement used for TBuffer read / write

public:
   TLeafF16();
   TLeafF16(TBranch *parent, const char *name, const char *type);
   virtual ~TLeafF16();

   DeserializeType GetDeserializeType() const override { return DeserializeType::kExternal; }
   void            Export(TClonesArray *list, Int_t n) override;
   void            FillBasket(TBuffer &b) override;
   const char     *GetTypeName() const override { return "Float16_t"; }
   Double_t        GetValue(Int_t i = 0) const override;
   void           *GetValuePointer() const override { return fValue; }
   void            Import(TClonesArray *list, Int_t n) override;
   void            PrintValue(Int_t i = 0) const override;
   void            ReadBasket(TBuffer &b) override;
   void            ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n) override;
   void            ReadValue(std::istream &s, Char_t delim = ' ') override;
   void            SetAddress(void *add = nullptr) override;

   ClassDefOverride(TLeafF16, 1); // A TLeaf for a 24 bit truncated floating point data type.
};

// if leaf is a simple type, i must be set to 0
// if leaf is an array, i is the array element number to be returned
inline Double_t TLeafF16::GetValue(Int_t i) const
{
   return fValue[i];
}

#endif

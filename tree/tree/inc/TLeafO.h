// @(#)root/tree:$Id$
// Author: Philippe Canal  20/1/05

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLeafO
#define ROOT_TLeafO


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLeafO                                                               //
//                                                                      //
// A TLeaf for a bool data type.                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TLeaf.h"

class TLeafO : public TLeaf {

protected:
   Bool_t       fMinimum;         ///<  Minimum value if leaf range is specified
   Bool_t       fMaximum;         ///<  Maximum value if leaf range is specified
   Bool_t       *fValue;          ///<! Pointer to data buffer
   Bool_t       **fPointer;       ///<! Address of a pointer to data buffer!

public:
   TLeafO();
   TLeafO(TBranch *parent, const char *name, const char *type);
   virtual ~TLeafO();

   void            Export(TClonesArray *list, Int_t n) override;
   void            FillBasket(TBuffer &b) override;
   DeserializeType GetDeserializeType() const override { return DeserializeType::kZeroCopy; }
   Int_t           GetMaximum() const override {return fMaximum;}
   Int_t           GetMinimum() const override {return fMinimum;}
   const char     *GetTypeName() const override;
   Double_t        GetValue(Int_t i=0) const override;
   void           *GetValuePointer() const override { return fValue; }
   Bool_t          IncludeRange(TLeaf *) override;
   void            Import(TClonesArray *list, Int_t n) override;
   void            PrintValue(Int_t i=0) const override;
   void            ReadBasket(TBuffer &b) override;
   void            ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n) override;
   void            ReadValue(std::istream& s, Char_t delim = ' ') override;
   void            SetAddress(void *add=nullptr) override;
   virtual void    SetMaximum(Bool_t max) { fMaximum = max; }
   virtual void    SetMinimum(Bool_t min) { fMinimum = min; }

   // Deserialize N events from an input buffer.  Since chars are stored unchanged, there
   // is nothing to do here but return true if we don't have variable-length arrays.
   bool            ReadBasketFast(TBuffer&, Long64_t) override { return true; }

   ClassDefOverride(TLeafO,1);  //A TLeaf for an 8 bit Integer data type.
};

inline Double_t TLeafO::GetValue(Int_t i) const { return fValue[i]; }

#endif

// @(#)root/tree:$Id$
// Author: Rene Brun   12/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLeafF
#define ROOT_TLeafF


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLeafF                                                               //
//                                                                      //
// A TLeaf for a 32 bit floating point data type.                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TLeaf.h"

class TLeafF : public TLeaf {

protected:
   Float_t       fMinimum;         ///<  Minimum value if leaf range is specified
   Float_t       fMaximum;         ///<  Maximum value if leaf range is specified
   Float_t       *fValue;          ///<! Pointer to data buffer
   Float_t       **fPointer;       ///<! Address of pointer to data buffer!

public:
   TLeafF();
   TLeafF(TBranch *parent, const char *name, const char *type);
   virtual ~TLeafF();

   virtual void    Export(TClonesArray *list, Int_t n);
   virtual void    FillBasket(TBuffer &b);
   virtual DeserializeType GetDeserializeType() const { return kInPlace; }
   const char     *GetTypeName() const { return "Float_t"; }
   Double_t        GetValue(Int_t i=0) const;
   virtual void   *GetValuePointer() const { return fValue; }
   virtual void    Import(TClonesArray *list, Int_t n);
   virtual void    PrintValue(Int_t i=0) const;
   virtual void    ReadBasket(TBuffer &b);
   virtual void    ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n);
   virtual void    ReadValue(std::istream& s, Char_t delim = ' ');
   virtual void    SetAddress(void *add=0);

   virtual bool    ReadBasketFast(TBuffer&, Long64_t);
   virtual bool    ReadBasketSerialized(TBuffer&, Long64_t) { return true; }

   ClassDef(TLeafF,1);  //A TLeaf for a 32 bit floating point data type.
};

// if leaf is a simple type, i must be set to 0
// if leaf is an array, i is the array element number to be returned
inline Double_t TLeafF::GetValue(Int_t i) const { return fValue[i]; }

#endif

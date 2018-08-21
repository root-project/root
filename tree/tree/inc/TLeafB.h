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

   virtual void    Export(TClonesArray* list, Int_t n);
   virtual void    FillBasket(TBuffer& b);
   virtual DeserializeType GetDeserializeType() const { return fLeafCount ? kDestructive : kZeroCopy; }
   virtual Int_t   GetMaximum() const { return fMaximum; }
   virtual Int_t   GetMinimum() const { return fMinimum; }
   const char     *GetTypeName() const;
   Double_t        GetValue(Int_t i = 0) const { return IsUnsigned() ? (Double_t)((UChar_t) fValue[i]) : (Double_t)fValue[i]; }
   virtual void   *GetValuePointer() const { return fValue; }
   virtual Bool_t  IncludeRange(TLeaf *);
   virtual void    Import(TClonesArray* list, Int_t n);
   virtual void    PrintValue(Int_t i = 0) const;
   virtual void    ReadBasket(TBuffer&);
   virtual void    ReadBasketExport(TBuffer&, TClonesArray* list, Int_t n);
   virtual void    ReadValue(std::istream &s, Char_t delim = ' ');
   virtual void    SetAddress(void* addr = 0);
   virtual void    SetMaximum(Char_t max) { fMaximum = max; }
   virtual void    SetMinimum(Char_t min) { fMinimum = min; }

   // Deserialize N events from an input buffer.  Since chars are stored unchanged, there
   // is nothing to do here but return true if we don't have variable-length arrays.
   virtual bool    ReadBasketFast(TBuffer&, Long64_t) { return !fLeafCount; }

   ClassDef(TLeafB,1);  //A TLeaf for an 8 bit Integer data type.
};

#endif

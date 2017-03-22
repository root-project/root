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

   virtual void    Export(TClonesArray *list, Int_t n);
   virtual void    FillBasket(TBuffer &b);
   const char     *GetTypeName() const;
   virtual Int_t   GetMaximum() const {return (Int_t)fMaximum;}
   virtual Int_t   GetMinimum() const {return (Int_t)fMinimum;}
   virtual Double_t     GetValue(Int_t i=0) const;
   virtual Long64_t     GetValueLong64(Int_t i = 0) const ;
   virtual LongDouble_t GetValueLongDouble(Int_t i = 0) const;
   virtual void   *GetValuePointer() const {return fValue;}
   virtual Bool_t  IncludeRange(TLeaf *);
   virtual void    Import(TClonesArray *list, Int_t n);
   virtual void    PrintValue(Int_t i=0) const;
   virtual void    ReadBasket(TBuffer &b);
   virtual void    ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n);
   virtual bool    ReadBasketFast(TBuffer&, Long64_t);
   virtual bool    ReadBasketSerialized(TBuffer&, Long64_t) {return GetDeserializeType() == kInPlace; }
   virtual void    ReadValue(std::istream& s, Char_t delim = ' ');
   virtual void    SetAddress(void *add=0);
   virtual void    SetMaximum(Long64_t max) {fMaximum = max;}
   virtual void    SetMinimum(Long64_t min) {fMinimum = min;}

   ClassDef(TLeafL,1);  //A TLeaf for a 64 bit Integer data type.
};

// if leaf is a simple type, i must be set to 0
// if leaf is an array, i is the array element number to be returned
inline Long64_t TLeafL::GetValueLong64(Int_t i) const { return fValue[i]; }

#endif

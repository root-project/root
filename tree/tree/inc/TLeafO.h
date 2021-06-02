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

   virtual void    Export(TClonesArray *list, Int_t n);
   virtual void    FillBasket(TBuffer &b);
   virtual DeserializeType GetDeserializeType() const { return DeserializeType::kZeroCopy; }
   virtual Int_t   GetMaximum() const {return fMaximum;}
   virtual Int_t   GetMinimum() const {return fMinimum;}
   const char     *GetTypeName() const;
   Double_t        GetValue(Int_t i=0) const;
   virtual void   *GetValuePointer() const {return fValue;}
   virtual Bool_t  IncludeRange(TLeaf *);
   virtual void    Import(TClonesArray *list, Int_t n);
   virtual void    PrintValue(Int_t i=0) const;
   virtual void    ReadBasket(TBuffer &b);
   virtual void    ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n);
   virtual void    ReadValue(std::istream& s, Char_t delim = ' ');
   virtual void    SetAddress(void *add=0);
   virtual void    SetMaximum(Bool_t max) { fMaximum = max; }
   virtual void    SetMinimum(Bool_t min) { fMinimum = min; }

   // Deserialize N events from an input buffer.  Since chars are stored unchanged, there
   // is nothing to do here but return true if we don't have variable-length arrays.
   virtual bool    ReadBasketFast(TBuffer&, Long64_t) { return true; }
   virtual bool    ReadBasketSerialized(TBuffer&, Long64_t) { return true; }

   ClassDef(TLeafO,1);  //A TLeaf for an 8 bit Integer data type.
};

inline Double_t TLeafO::GetValue(Int_t i) const { return fValue[i]; }

#endif

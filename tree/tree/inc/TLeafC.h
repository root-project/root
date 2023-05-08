// @(#)root/tree:$Id$
// Author: Rene Brun   17/03/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLeafC
#define ROOT_TLeafC


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLeafC                                                               //
//                                                                      //
// A TLeaf for a variable length string.                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TLeaf.h"

class TLeafC : public TLeaf {

protected:
   Int_t        fMinimum;         ///<  Minimum value if leaf range is specified
   Int_t        fMaximum;         ///<  Maximum value if leaf range is specified
   Char_t       *fValue;          ///<! Pointer to data buffer
   Char_t       **fPointer;       ///<! Address of pointer to data buffer

public:
   TLeafC();
   TLeafC(TBranch *parent, const char *name, const char *type);
   virtual ~TLeafC();

   void            Export(TClonesArray *list, Int_t n) override;
   void            FillBasket(TBuffer &b) override;
   Int_t           GetMaximum() const override {return fMaximum;}
   Int_t           GetMinimum() const override {return fMinimum;}
   const char     *GetTypeName() const override;
   Double_t        GetValue(Int_t i=0) const override;
   void           *GetValuePointer() const override {return fValue;}
   virtual char   *GetValueString() const {return fValue;}
   Bool_t          IncludeRange(TLeaf *) override;
   void            Import(TClonesArray *list, Int_t n) override;
   void            PrintValue(Int_t i=0) const override;
   void            ReadBasket(TBuffer &b) override;
   void            ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n) override;
   void            ReadValue(std::istream& s, Char_t delim = ' ') override;
   void            SetAddress(void *add = nullptr) override;
   virtual void    SetMaximum(Int_t max) {fMaximum = max;}
   virtual void    SetMinimum(Int_t min) {fMinimum = min;}

   ClassDefOverride(TLeafC,1);  //A TLeaf for a variable length string.
};

inline Double_t TLeafC::GetValue(Int_t i) const { return fValue[i]; }

#endif

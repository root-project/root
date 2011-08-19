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

#ifndef ROOT_TLeaf
#include "TLeaf.h"
#endif

class TLeafC : public TLeaf {

protected:
   Int_t        fMinimum;         //Minimum value if leaf range is specified
   Int_t        fMaximum;         //Maximum value if leaf range is specified
   Char_t       *fValue;          //!Pointer to data buffer
   Char_t       **fPointer;       //!Address of pointer to data buffer

public:
   TLeafC();
   TLeafC(TBranch *parent, const char *name, const char *type);
   virtual ~TLeafC();

   virtual void    Export(TClonesArray *list, Int_t n);
   virtual void    FillBasket(TBuffer &b);
   virtual Int_t   GetMaximum() const {return fMaximum;}
   virtual Int_t   GetMinimum() const {return fMinimum;}
   const char     *GetTypeName() const;
   Double_t        GetValue(Int_t i=0) const;
   virtual void   *GetValuePointer() const {return fValue;}
   char           *GetValueString()  const {return fValue;}
   virtual void    Import(TClonesArray *list, Int_t n);
   virtual void    PrintValue(Int_t i=0) const;
   virtual void    ReadBasket(TBuffer &b);
   virtual void    ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n);
   virtual void    ReadValue(istream & s);
   virtual void    SetAddress(void *add=0);
   virtual void    SetMaximum(Int_t max) {fMaximum = max;}
   virtual void    SetMinimum(Int_t min) {fMinimum = min;}
   
   ClassDef(TLeafC,1);  //A TLeaf for a variable length string.
};

inline Double_t TLeafC::GetValue(Int_t i) const { return fValue[i]; }

#endif

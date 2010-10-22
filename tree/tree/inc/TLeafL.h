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


#ifndef ROOT_TLeaf
#include "TLeaf.h"
#endif

class TLeafL : public TLeaf {

protected:
   Long64_t     fMinimum;         //Minimum value if leaf range is specified
   Long64_t     fMaximum;         //Maximum value if leaf range is specified
   Long64_t    *fValue;           //!Pointer to data buffer
   Long64_t   **fPointer;         //!Address of pointer to data buffer

public:
   TLeafL();
   TLeafL(TBranch *parent, const char *name, const char *type);
   virtual ~TLeafL();

   virtual void    Export(TClonesArray *list, Int_t n);
   virtual void    FillBasket(TBuffer &b);
   const char     *GetTypeName() const;
   virtual Int_t   GetMaximum() const {return (Int_t)fMaximum;}
   virtual Int_t   GetMinimum() const {return (Int_t)fMinimum;}
   Double_t        GetValue(Int_t i=0) const;
   virtual void   *GetValuePointer() const {return fValue;}
   virtual void    Import(TClonesArray *list, Int_t n);
   virtual void    PrintValue(Int_t i=0) const;
   virtual void    ReadBasket(TBuffer &b);
   virtual void    ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n);
   virtual void    ReadValue(istream & s);
   virtual void    SetAddress(void *add=0);
   virtual void    SetMaximum(Long64_t max) {fMaximum = max;}
   virtual void    SetMinimum(Long64_t min) {fMinimum = min;}
   
   ClassDef(TLeafL,1);  //A TLeaf for a 64 bit Integer data type.
};

#endif

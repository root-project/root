// @(#)root/tree:$Name$:$Id$
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

#ifndef ROOT_TLeaf
#include "TLeaf.h"
#endif

class TLeafD : public TLeaf {

protected:
    Double_t       fMinimum;         //Minimum value if leaf range is specified
    Double_t       fMaximum;         //Maximum value if leaf range is specified
    Double_t       *fValue;          //!Pointer to data buffer

public:
    TLeafD();
    TLeafD(const char *name, const char *type);
    virtual ~TLeafD();

    virtual void    Export(TClonesArray *list, Int_t n);
    virtual void    FillBasket(TBuffer &b);
    const char     *GetTypeName() const {return "Double_t";}
    Float_t         GetValue(Int_t i=0);
    virtual void   *GetValuePointer() {return fValue;}
    virtual void    Import(TClonesArray *list, Int_t n);
    virtual void    Print(Option_t *option="");
    virtual void    ReadBasket(TBuffer &b);
    virtual void    ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n);
    virtual void    SetAddress(void *add=0);

    ClassDef(TLeafD,1)  //A TLeaf for a 64 bit floating point data type.
};

inline Float_t TLeafD::GetValue(Int_t i) { return fValue[i]; }

#endif

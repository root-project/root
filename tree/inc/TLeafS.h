// @(#)root/tree:$Name$:$Id$
// Author: Rene Brun   12/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLeafS
#define ROOT_TLeafS


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLeafS                                                               //
//                                                                      //
// A TLeaf for a 16 bit Integer data type.                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TLeaf
#include "TLeaf.h"
#endif

class TLeafS : public TLeaf {

protected:
    Short_t       fMinimum;         //Minimum value if leaf range is specified
    Short_t       fMaximum;         //Maximum value if leaf range is specified
    Short_t       *fValue;          //!Pointer to data buffer

public:
    TLeafS();
    TLeafS(const char *name, const char *type);
    virtual ~TLeafS();

    virtual void    Export(TClonesArray *list, Int_t n);
    virtual void    FillBasket(TBuffer &b);
    const char     *GetTypeName() const;
    Float_t         GetValue(Int_t i=0);
    virtual void   *GetValuePointer() {return fValue;}
    virtual void    Import(TClonesArray *list, Int_t n);
    virtual void    Print(Option_t *option="");
    virtual void    ReadBasket(TBuffer &b);
    virtual void    ReadBasketExport(TBuffer &b, TClonesArray *list, Int_t n);
    virtual void    SetAddress(void *add=0);

    ClassDef(TLeafS,1)  //A TLeaf for a 16 bit Integer data type.
};

#endif

// @(#)root/tree:$Name:  $:$Id: TLeafElement.h,v 1.3 2001/01/18 09:44:12 brun Exp $
// Author: Rene Brun   14/01/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TLeafElement
#define ROOT_TLeafElement


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TLeafElement                                                          //
//                                                                      //
// A TLeaf for a general object derived from TObject.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TLeaf
#include "TLeaf.h"
#endif

class TMethodCall;

class TLeafElement : public TLeaf {

protected:
    char               *fAbsAddress;   //! Absolute leaf Address
    Int_t               fID;           //element serial number in fInfo
    Int_t               fType;         //leaf type
        
public:
    TLeafElement();
    TLeafElement(const char *name, Int_t id, Int_t type);
    virtual ~TLeafElement();

    virtual void     FillBasket(TBuffer &b);
    TMethodCall     *GetMethodCall(const char *name);
    virtual Double_t GetValue(Int_t i=0) const;
    virtual void    *GetValuePointer() const { return fAbsAddress; }
    virtual void     PrintValue(Int_t i=0) const;
    virtual void     ReadBasket(TBuffer &b);
    virtual void     SetAddress(void *add=0);
    
    ClassDef(TLeafElement,1)  //A TLeaf for a general object derived from TObject.
};

#endif

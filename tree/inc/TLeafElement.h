// @(#)root/tree:$Name:  $:$Id: TLeafElement.h,v 1.6 2001/04/09 08:11:43 brun Exp $
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
#ifndef ROOT_TBranchElement
#include "TBranchElement.h"
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

    virtual Int_t    GetLen() const {return ((TBranchElement*)fBranch)->GetNdata()*fLen;}
    TMethodCall     *GetMethodCall(const char *name);
    virtual Int_t    GetNdata() const {return ((TBranchElement*)fBranch)->GetNdata()*fLen;}
    virtual Double_t GetValue(Int_t i=0) const {return ((TBranchElement*)fBranch)->GetValue(i, fLen);}
    virtual void    *GetValuePointer() const { return fAbsAddress; }
    virtual Bool_t   IsOnTerminalBranch() const;
    virtual void     PrintValue(Int_t i=0) const {((TBranchElement*)fBranch)->PrintValue(i);}
    virtual void     SetLeafCount(TLeaf *leaf) {fLeafCount = leaf;}
    
    ClassDef(TLeafElement,1)  //A TLeaf for a general object derived from TObject.
};

#endif

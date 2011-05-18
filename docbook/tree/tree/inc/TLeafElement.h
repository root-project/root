// @(#)root/tree:$Id$
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
   TLeafElement(TBranch *parent, const char *name, Int_t id, Int_t type);
   virtual ~TLeafElement();
   
   virtual Int_t    GetLen() const {return ((TBranchElement*)fBranch)->GetNdata()*fLen;}
   TMethodCall     *GetMethodCall(const char *name);
   virtual Int_t    GetMaximum() const {return ((TBranchElement*)fBranch)->GetMaximum();}
   virtual Int_t    GetNdata() const {return ((TBranchElement*)fBranch)->GetNdata()*fLen;}
   virtual const char *GetTypeName() const {return ((TBranchElement*)fBranch)->GetTypeName();}
   virtual Double_t GetValue(Int_t i=0) const {return ((TBranchElement*)fBranch)->GetValue(i, fLen, kFALSE);}
   virtual Double_t GetValueSubArray(Int_t i=0, Int_t j=0) const {return ((TBranchElement*)fBranch)->GetValue(i, j, kTRUE);}
   virtual void    *GetValuePointer() const { return ((TBranchElement*)fBranch)->GetValuePointer(); }
   virtual Bool_t   IsOnTerminalBranch() const;
   virtual void     PrintValue(Int_t i=0) const {((TBranchElement*)fBranch)->PrintValue(i);}
   virtual void     SetLeafCount(TLeaf *leaf) {fLeafCount = leaf;}
   
   ClassDef(TLeafElement,1);  //A TLeaf for a general object derived from TObject.
};

#endif

// @(#)root/tree:$Name:  $:$Id: TBranchObject.h,v 1.2 2000/09/05 09:21:24 brun Exp $
// Author: Rene Brun   11/02/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBranchObject
#define ROOT_TBranchObject


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBranchObject                                                        //
//                                                                      //
// A Branch for the case of an object.                                  //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TBranch
#include "TBranch.h"
#endif

class TBranchObject : public TBranch {

protected:
    enum { kWarn = BIT(12) };

    TString     fClassName;        //Class name of referenced object
    TObject     *fOldObject;       //!Pointer to old object

public:
    TBranchObject();
    TBranchObject(const char *name, const char *classname, void *addobj, Int_t basketsize=32000, Int_t splitlevel = 0, Int_t compress=-1);
    virtual ~TBranchObject();

    virtual void    Browse(TBrowser *b);
    virtual Int_t   Fill();
    virtual Int_t   GetEntry(Int_t entry=0, Int_t getall = 0);
    Bool_t          IsFolder() const;
    virtual void    Print(Option_t *option="") const;
    virtual void    Reset(Option_t *option="");
    virtual void    SetAddress(void *addobj);
    virtual void    SetAutoDelete(Bool_t autodel=kTRUE);
    virtual void    SetBasketSize(Int_t buffsize);
    virtual void    UpdateAddress();

    ClassDef(TBranchObject,1)  //Branch in case of an object
};

#endif

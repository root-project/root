// @(#)root/tree:$Name:  $:$Id: TBranchElement.h,v 1.4 2001/01/25 15:02:29 brun Exp $
// Author: Rene Brun   14/01/2001

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBranchElement
#define ROOT_TBranchElement


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBranchElement                                                       //
//                                                                      //
// A Branch for the case of an object.                                  //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TBranch
#include "TBranch.h"
#endif

class TStreamerInfo;

class TBranchElement : public TBranch {

protected:
    enum { kWarn = BIT(12) };

    TString             fClassName;     //Class name of referenced object
    Int_t               fClassVersion;  //Version number of class
    Int_t               fID;            //element serial number in fInfo
    Int_t               fType;          //branch type
    Int_t               fStreamerType;  //branch streamer type
    TStreamerInfo      *fInfo;          //!Pointer to StreamerInfo
    
public:
    TBranchElement();
    TBranchElement(const char *name, TStreamerInfo *sinfo, Int_t id, char *pointer, Int_t basketsize=32000, Int_t splitlevel = 0, Int_t compress=-1);
    virtual ~TBranchElement();

    virtual void    Browse(TBrowser *b);
    virtual Int_t   Fill();
    virtual void    FillLeaves(TBuffer &b);
    virtual const char  *GetClassName() const {return fClassName.Data();}
    virtual Int_t   GetEntry(Int_t entry=0, Int_t getall = 0);
    virtual Int_t   GetID() {return fID;}
    TStreamerInfo  *GetInfo() const {return fInfo;}
    virtual Int_t   GetType() {return fType;}
    virtual Int_t   GetStreamerType() {return fStreamerType;}
    Bool_t          IsFolder() const;
    virtual void    Print(Option_t *option="") const;
    virtual void    ReadLeaves(TBuffer &b);
    virtual void    Reset(Option_t *option="");
    virtual void    SetAddress(void *addobj);
    virtual void    SetAutoDelete(Bool_t autodel=kTRUE);
    virtual void    SetBasketSize(Int_t buffsize);
    virtual void    SetClassName(const char *name) {fClassName=name;}
    virtual void    SetType(Int_t btype) {fType=btype;}
    virtual Int_t   Unroll(const char *name, TClass *cltop, TClass *cl,Int_t basketsize, Int_t splitlevel, Int_t btype);

    ClassDef(TBranchElement,1)  //Branch in case of an object
};

#endif

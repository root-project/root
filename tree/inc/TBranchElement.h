// @(#)root/tree:$Name:  $:$Id: TBranchElement.h,v 1.13 2001/04/30 15:08:50 brun Exp $
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

class TFolder;
class TStreamerInfo;

class TBranchElement : public TBranch {

protected:
    enum { kWarn = BIT(13), kBranchFolder = BIT(14) };

    TString             fClassName;     //Class name of referenced object
    TString             fParentName;    //Name of parent class
    TString             fClonesName;    //Name of class in TClonesArray (if any)
    Int_t               fClassVersion;  //Version number of class
    Int_t               fID;            //element serial number in fInfo
    Int_t               fType;          //branch type
    Int_t               fStreamerType;  //branch streamer type
    Int_t               fMaximum;       //Maximum entries for a TClonesArray or variable array
    Int_t               fNdata;         //!Number of data in this branch
    TBranchElement     *fBranchCount;   //pointer to branchcount branch
    TStreamerInfo      *fInfo;          //!Pointer to StreamerInfo
    char               *fObject;        //!Pointer to object at *fAddress
    
public:
    TBranchElement();
    TBranchElement(const char *name, TStreamerInfo *sinfo, Int_t id, char *pointer, Int_t basketsize=32000, Int_t splitlevel = 0, Int_t compress=-1);
    TBranchElement(const char *name, TClonesArray *clones, Int_t basketsize=32000, Int_t splitlevel = 0, Int_t compress=-1);
    virtual ~TBranchElement();

    virtual void     Browse(TBrowser *b);
            Int_t    Fill();
            void     FillLeaves(TBuffer &b);
    TBranchElement  *GetBranchCount() const {return fBranchCount;}
    virtual const char  *GetClassName() const {return fClassName.Data();}
            Int_t    GetEntry(Int_t entry=0, Int_t getall = 0);
            Int_t    GetID() const {return fID;}
    TStreamerInfo   *GetInfo();
    char    *GetObject() const { return fObject; };
    virtual const char  *GetParentName() const {return fParentName.Data();}
    virtual Int_t    GetMaximum() const;
    TBranchElement  *GetMother() const;
            Int_t    GetNdata() const { return fNdata; };
            Int_t    GetType()  const {return fType;}
            Int_t    GetStreamerType() const {return fStreamerType;}
    TBranchElement  *GetSubBranch(const TBranchElement *br) const;
    virtual const char *GetTypeName() const;
            Double_t GetValue(Int_t i, Int_t len) const;
            Bool_t   IsBranchFolder() const {return TestBit(kBranchFolder);}
            Bool_t   IsFolder() const;
    virtual void     Print(Option_t *option="") const;
            void     PrintValue(Int_t i) const;
    virtual void     ReadLeaves(TBuffer &b);
    virtual void     Reset(Option_t *option="");
    virtual void     SetAddress(void *addobj);
    virtual void     SetAutoDelete(Bool_t autodel=kTRUE);
    virtual void     SetBasketSize(Int_t buffsize);
    virtual void     SetBranchCount(TBranchElement *bre);
    virtual void     SetBranchFolder() {SetBit(kBranchFolder);}
    virtual void     SetClassName(const char *name) {fClassName=name;}
    virtual void     SetParentName(const char *name) {fParentName=name;}
    virtual void     SetType(Int_t btype) {fType=btype;}
    virtual Int_t    Unroll(const char *name, TClass *cltop, TClass *cl,Int_t basketsize, Int_t splitlevel, Int_t btype);

    ClassDef(TBranchElement,5)  //Branch in case of an object
};

#endif

// @(#)root/tree:$Name:  $:$Id: TBranchClones.h,v 1.3 2000/11/21 20:46:58 brun Exp $
// Author: Rene Brun   11/02/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBranchClones
#define ROOT_TBranchClones


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TBranchClones                                                        //
//                                                                      //
// A Branch for the case of an array of clone objects.                  //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TBranch
#include "TBranch.h"
#endif

#ifndef ROOT_TClonesArray
#include "TClonesArray.h"
#endif

class TBranchClones : public TBranch {

protected:
    TClonesArray     *fList;           //!Pointer to the clonesarray
    Int_t            fRead;            //!flag = 1 if clonesarray has been read
    Int_t            fN;               //!Number of elements in ClonesArray
    Int_t            fNdataMax;        //!Maximum value of fN
    TString          fClassName;       //name of the class of the objets in the ClonesArray
    TBranch          *fBranchCount;    //Branch with clones count

public:
    TBranchClones();
    TBranchClones(const char *name, void *clonesaddress, Int_t basketsize=32000,Int_t compress=-1, Int_t splitlevel=1);
    virtual ~TBranchClones();

    virtual void    Browse(TBrowser *b);
    virtual Int_t   Fill();
    virtual Int_t   GetEntry(Int_t entry=0, Int_t getall = 0);
    virtual Int_t   GetN() const {return fN;}
    TClonesArray    *GetList() const {return fList;}
    Bool_t          IsFolder() const {return kTRUE;}
    virtual void    Print(Option_t *option="") const;
    virtual void    Reset(Option_t *option="");
    virtual void    SetAddress(void *add);
    virtual void    SetBasketSize(Int_t buffsize);

    ClassDef(TBranchClones,2)  //Branch in case of an array of clone objects
};

#endif

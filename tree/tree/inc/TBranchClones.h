// @(#)root/tree
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


#include "TBranch.h"

class TClonesArray;
class TTreeCloner;

class TBranchClones : public TBranch {

protected:
   TClonesArray     *fList;           ///<! Pointer to the clonesarray
   Int_t            fRead;            ///<! flag = 1 if clonesarray has been read
   Int_t            fN;               ///<! Number of elements in ClonesArray
   Int_t            fNdataMax;        ///<! Maximum value of fN
   TString          fClassName;       ///<  Name of the class of the objets in the ClonesArray
   TBranch          *fBranchCount;    ///<  Branch with clones count

   friend class TTreeCloner;

   void Init(TTree *tree, TBranch *parent, const char *name, void *clonesaddress, Int_t basketsize=32000,Int_t compress=-1, Int_t splitlevel=1);
   Int_t   FillImpl(ROOT::Internal::TBranchIMTHelper *) override;

public:
   TBranchClones();
   TBranchClones(TTree *tree, const char *name, void *clonesaddress, Int_t basketsize=32000,Int_t compress=-1, Int_t splitlevel=1);
   TBranchClones(TBranch *parent, const char *name, void *clonesaddress, Int_t basketsize=32000,Int_t compress=-1, Int_t splitlevel=1);
   ~TBranchClones() override;

   void    Browse(TBrowser *b) override;
   const char* GetClassName() const override { return fClassName; }
   Int_t   GetEntry(Long64_t entry=0, Int_t getall = 0) override;
   virtual Int_t   GetN() const {return fN;}
   TClonesArray    *GetList() const {return fList;}
   bool            IsFolder() const override {return true;}
   void    Print(Option_t *option="") const override;
   void    Reset(Option_t *option="") override;
   void    ResetAfterMerge(TFileMergeInfo *) override;
   void    SetAddress(void *add) override;
   void    SetBasketSize(Int_t buffsize) override;
   void    SetTree(TTree *tree) override { fTree = tree; fBranchCount->SetTree(tree); }
   void    UpdateFile() override;

private:

   ClassDefOverride(TBranchClones,2);  //Branch in case of an array of clone objects
};

#endif

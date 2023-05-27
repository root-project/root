// @(#)root/treeplayer:$Id$
// Author: Philippe Canal 09/06/2006

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT__TSelectorEntries
#define ROOT__TSelectorEntries

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSelectorEntries                                                     //
//                                                                      //
// A specialized TSelector for TTree::GetEntries(selection)             //
// The selection is passed either via the constructor or via            //
// SetSelection.  The number of entries passing the selection (or       //
// at least one element of the arrays or collections used in the        //
// selection is passing the selection) is stored in fSeletedRwos        //
// which can be retrieved via GetSelectedRows.                          //
// See a usage example in TTreePlayer::GetEntries.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <TSelector.h>

class TTree;
class TTreeFormula;

class TSelectorEntries : public TSelector {
   Bool_t          fOwnInput;       ///<  True if we created the input list.
public :
   TTree          *fChain;          ///<! Pointer to the analyzed TTree or TChain
   TTreeFormula   *fSelect;         ///<  Pointer to selection formula
   Long64_t        fSelectedRows;   ///<  Number of selected entries
   Bool_t          fSelectMultiple; ///<  True if selection has a variable index

   TSelectorEntries(TTree *tree = nullptr, const char *selection = nullptr);
   TSelectorEntries(const char *selection);
   ~TSelectorEntries() override;
   Int_t    Version() const override { return 2; }
   void     Begin(TTree *tree) override;
   void     SlaveBegin(TTree *tree) override;
   void     Init(TTree *tree) override;
   Bool_t   Notify() override;
   Bool_t   Process(Long64_t entry) override;
   Int_t    GetEntry(Long64_t entry, Int_t getall = 0) override;
   virtual Long64_t GetSelectedRows() const { return fSelectedRows; }
   void     SetOption(const char *option) override { fOption = option; }
   void     SetObject(TObject *obj) override { fObject = obj; }
   virtual void     SetSelection(const char *selection);
   TList   *GetOutputList() const override { return fOutput; }
   void     SlaveTerminate() override;
   void     Terminate() override;

   ClassDefOverride(TSelectorEntries,1); //A specialized TSelector for TTree::GetEntries(selection)
};

#endif


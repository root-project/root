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
// selection is passing the slection) is stored in fSeletedRwos         //
// which can be retrieved via GetSelectedRows.                          //
// See a usage example in TTreePlayer::GetEntries.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <TSelector.h>

class TTree;
class TTreeFormula;

class TSelectorEntries : public TSelector {
public :
   TTree          *fChain;          //! pointer to the analyzed TTree or TChain
   TTreeFormula   *fSelect;         //  Pointer to selection formula
   Long64_t        fSelectedRows;   //  Number of selected entries
   Bool_t          fSelectMultiple; //  true if selection has a variable index

   TSelectorEntries(TTree *tree = 0, const char *selection = 0);
   TSelectorEntries(const char *selection);
   virtual ~TSelectorEntries();
   virtual Int_t    Version() const { return 2; }
   virtual void     Begin(TTree *tree);
   virtual void     SlaveBegin(TTree *tree);
   virtual void     Init(TTree *tree);
   virtual Bool_t   Notify();
   virtual Bool_t   Process(Long64_t entry);
   virtual Int_t    GetEntry(Long64_t entry, Int_t getall = 0);
   virtual Long64_t GetSelectedRows() const { return fSelectedRows; }
   virtual void     SetOption(const char *option) { fOption = option; }
   virtual void     SetObject(TObject *obj) { fObject = obj; }
   virtual void     SetSelection(const char *selection);
   virtual TList   *GetOutputList() const { return fOutput; }
   virtual void     SlaveTerminate();
   virtual void     Terminate();

   ClassDef(TSelectorEntries,1); //A specialized TSelector for TTree::GetEntries(selection)
};

#endif


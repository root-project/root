// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//______________________________________________________________________________
// TTreeTools
//
// Collection of classes for TTree interaction.

#include "TEveTreeTools.h"
#include "TTree.h"
#include "TTreeFormula.h"

/******************************************************************************/
/******************************************************************************/

//______________________________________________________________________________
// TEveSelectorToEventList
//
// TSelector that stores entry numbers of matching TTree entries into
// an event-list.


ClassImp(TEveSelectorToEventList)

//______________________________________________________________________________
TEveSelectorToEventList::TEveSelectorToEventList(TEventList* evl, const char* sel) :
   TSelectorDraw(), fEvList(evl)
{
   // Constructor.

   fInput.Add(new TNamed("varexp", ""));
   fInput.Add(new TNamed("selection", sel));
   SetInputList(&fInput);
}

//______________________________________________________________________________
Bool_t TEveSelectorToEventList::Process(Long64_t entry)
{
   // Process entry.

   if(GetSelect()->EvalInstance(0) != 0)
      fEvList->Enter(entry);
   return kTRUE;
}


//______________________________________________________________________________
// TEvePointSelector, TEvePointSelectorConsumer
//
// TEvePointSelector is a sub-class of TSelectorDraw for direct
// extraction of point-like data from a Tree.
//
// TEvePointSelectorConsumer is a virtual base for classes that can be
// filled from TTree data via the TEvePointSelector class.

ClassImp(TEvePointSelector)
ClassImp(TEvePointSelectorConsumer)

//______________________________________________________________________________
TEvePointSelector::TEvePointSelector(TTree* t,
                                     TEvePointSelectorConsumer* c,
                                     const char* vexp, const char* sel) :
   TSelectorDraw(),

   fTree      (t),
   fConsumer  (c),
   fVarexp    (vexp),
   fSelection (sel),
   fSubIdExp  (),
   fSubIdNum  (0)
{
   // Constructor.

   SetInputList(&fInput);
}

//______________________________________________________________________________
Long64_t TEvePointSelector::Select(const char* selection)
{
   // Process the tree, select points matching 'selection'.

   TString var(fVarexp);
   if (fSubIdExp.IsNull()) {
      fSubIdNum = 0;
   } else {
      fSubIdNum = fSubIdExp.CountChar(':') + 1;
      var += ":" + fSubIdExp;
   }

   TString sel;
   if (selection != 0)
      sel = selection;
   else
      sel = fSelection;

   fInput.Delete();
   fInput.Add(new TNamed("varexp",    var.Data()));
   fInput.Add(new TNamed("selection", sel.Data()));

   if (fConsumer)
      fConsumer->InitFill(fSubIdNum);

   // 'para' option -> hack allowing arbitrary dimensions.
   if(fTree)
      fTree->Process(this, "goff para");

   return fSelectedRows;
}

//______________________________________________________________________________
Long64_t TEvePointSelector::Select(TTree* t, const char* selection)
{
   // Process tree 't', select points matching 'selection'.

   fTree = t;
   return Select(selection);
}

//______________________________________________________________________________
void TEvePointSelector::TakeAction()
{
   // Callback from tree-player after a chunk of data has been processed.
   // This is forwarded to the current point-consumer.

   fSelectedRows += fNfill;
   // printf("TEvePointSelector::TakeAction nfill=%d, nall=%lld\n", fNfill, fSelectedRows);
   if (fConsumer) {
      fConsumer->TakeAction(this);
   }
}

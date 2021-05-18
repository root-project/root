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

/** \class TEveSelectorToEventList
\ingroup TEve
TSelector that stores entry numbers of matching TTree entries into
an event-list.
*/

ClassImp(TEveSelectorToEventList);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEveSelectorToEventList::TEveSelectorToEventList(TEventList* evl, const char* sel) :
   TSelectorDraw(), fEvList(evl)
{
   fInputList.SetOwner(kTRUE);
   fInputList.Add(new TNamed("varexp", ""));
   fInputList.Add(new TNamed("selection", sel));
   SetInputList(&fInputList);
}

////////////////////////////////////////////////////////////////////////////////
/// Process entry.

Bool_t TEveSelectorToEventList::Process(Long64_t entry)
{
   if(GetSelect()->EvalInstance(0) != 0)
      fEvList->Enter(entry);
   return kTRUE;
}

/** \class TEvePointSelector
\ingroup TEve
TEvePointSelector is a sub-class of TSelectorDraw for direct
extraction of point-like data from a Tree.
*/

/** \class TEvePointSelectorConsumer
\ingroup TEve
TEvePointSelectorConsumer is a virtual base for classes that can be
filled from TTree data via the TEvePointSelector class.
*/

ClassImp(TEvePointSelector);
ClassImp(TEvePointSelectorConsumer);

////////////////////////////////////////////////////////////////////////////////
/// Constructor.

TEvePointSelector::TEvePointSelector(TTree* t,
                                     TEvePointSelectorConsumer* c,
                                     const char* vexp, const char* sel) :
   TSelectorDraw(),

   fSelectTree(t),
   fConsumer  (c),
   fVarexp    (vexp),
   fSelection (sel),
   fSubIdExp  (),
   fSubIdNum  (0)
{
   fInputList.SetOwner(kTRUE);
   SetInputList(&fInputList);
}

////////////////////////////////////////////////////////////////////////////////
/// Process the tree, select points matching 'selection'.

Long64_t TEvePointSelector::Select(const char* selection)
{
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

   fInputList.Delete();
   fInputList.Add(new TNamed("varexp",    var.Data()));
   fInputList.Add(new TNamed("selection", sel.Data()));

   if (fConsumer)
      fConsumer->InitFill(fSubIdNum);

   if (fSelectTree)
      fSelectTree->Process(this, "goff");

   return fSelectedRows;
}

////////////////////////////////////////////////////////////////////////////////
/// Process tree 't', select points matching 'selection'.

Long64_t TEvePointSelector::Select(TTree* t, const char* selection)
{
   fSelectTree = t;
   return Select(selection);
}

////////////////////////////////////////////////////////////////////////////////
/// Callback from tree-player after a chunk of data has been processed.
/// This is forwarded to the current point-consumer.

void TEvePointSelector::TakeAction()
{
   fSelectedRows += fNfill;
   // printf("TEvePointSelector::TakeAction nfill=%d, nall=%lld\n", fNfill, fSelectedRows);
   if (fConsumer) {
      fConsumer->TakeAction(this);
   }
}

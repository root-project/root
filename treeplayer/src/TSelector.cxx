// @(#)root/treeplayer:$Name$:$Id$
// Author: Rene Brun   05/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// A TSelector object is used by the TTree::Draw, TTree::Scan,          //
//  TTree::Loop to navigate in a TTree and make selections.             //
//                                                                      //
//  The following members functions are called by the TTree functions.  //
//    Init:      to initialize the selector                             //
//    Start:     called everytime a loop on the tree starts.            //
//               a convenient place to create your histograms.          //
//    Finish:    called at the end of a loop on a TTree.                //
//               a convenient place to draw/fit your histograms.        //
//    BeginFile: called when a TChain is processed at the begining of   //
//               of a new file.                                         //
//    EndFile:   called when a TChain is processed at the end of a file.//                                         //
//    Execute:   called for each selected entry.                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSelector.h"
#include "TTree.h"
#include "TFile.h"

ClassImp(TSelector)

//______________________________________________________________________________
TSelector::TSelector(): TNamed()
{
   // Default constructor for a Selector.

}

//______________________________________________________________________________
TSelector::TSelector(const char *name, const char *title)
    :TNamed(name,title)
{
   // Create a Selector.

}

//______________________________________________________________________________
TSelector::~TSelector()
{
   // Destructor for a Selector.

}

//______________________________________________________________________________
void TSelector::BeginFile()
{
   // Called by TTree::Loop when a new file begins in a TChain.

   printf("%s::BeginFile called for file: %s\n",GetName(),fTree->GetCurrentFile()->GetName());
}

//______________________________________________________________________________
void TSelector::EndFile()
{
   // Called by TTree::Loop when a file ends in a TChain.

   printf("%s::EndFile called for file: %s\n",GetName(),fTree->GetCurrentFile()->GetName());
}

//______________________________________________________________________________
void TSelector::Execute(TTree *tree, Int_t entry)
{
   // Process entry number entry.

   tree->GetEntry(entry);
}

//______________________________________________________________________________
void TSelector::Finish(Option_t *)
{
   // Called at the end of TTree::Loop.

   printf("%s::Finish called\n",GetName());
}


//______________________________________________________________________________
void TSelector::Init(TTree *tree, Option_t *)
{
   // Called to initialize/reinitialize the selector.

   fTree = tree;
}

//______________________________________________________________________________
void TSelector::Start(Option_t *)
{
   // Called at the begining of TTree::Loop.

   printf("%s::Start called\n",GetName());
}

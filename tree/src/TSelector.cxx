// @(#)root/treeplayer:$Name:  $:$Id: TSelector.cxx,v 1.1.1.1 2000/05/16 17:00:44 rdm Exp $
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
//  TTree::Loop, TTree::Process to navigate in a TTree and make         //
//  selections.                                                         //
//                                                                      //
//  The following members functions are called by the TTree functions.  //
//    Begin:     called everytime a loop on the tree starts.            //
//               a convenient place to create your histograms.          //
//    Finish:    called at the end of a loop on a TTree.                //
//               a convenient place to draw/fit your histograms.        //
//    Select:    called at the beginning of each entry to return a flag //
//               true if the entry must be analyzed.                    //
//               a convenient place to draw/fit your histograms.        //
//    Analyze:   called in the entry loop for all entries accepted      //
//               by Select.                                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TSelector.h"

ClassImp(TSelector)

//______________________________________________________________________________
TSelector::TSelector(): TObject()
{
   // Default constructor for a Selector.

}

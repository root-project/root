// @(#)root/treeplayer:$Id$
// Author: Akos Hajdu 22/06/2015

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeSelectorReaderGenerator
#define ROOT_TTreeSelectorReaderGenerator

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TTreeSelectorReaderGenerator                                         //
//                                                                      //
// Generate a Selector using TTreeReaderValues and TTreeReaderArrays to //
// access the data in the tree.                                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_Tlist
#include "TList.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

class TTree;

namespace ROOT {

   class TTreeSelectorReaderGenerator
   {
      TTree   *fTree;          // Pointer to the tree
      TString  fClassname;     // Class name of the selector
      UInt_t   fMaxUnrolling;  // Depth of unrolling for non-split classes
      TList    fListOfHeaders; // List of included headers
      
      void   AddHeader(TClass *cl);
      
      void   AnalyzeTree(TTree *tree);
      void   WriteSelector();
   
   public:
      TTreeSelectorReaderGenerator(TTree* tree, const char *classname, UInt_t maxUnrolling);
      
   };

}

using ROOT::TTreeSelectorReaderGenerator;

#endif

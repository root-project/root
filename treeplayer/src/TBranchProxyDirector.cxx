// @(#)root/base:$Name:  $:$Id: TBranchProxyDirector.cxx,v 1.1 2004/02/20 19:23:30 cvsuser Exp $
// Author: Philippe Canal  13/05/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TBranchProxyDirector                                                   //
//                                                                        //
// This class is used to 'drive' and hold a serie of TBranchProxy objects //
// which represent and give access to the content of TTree object.        //
// This is intended to be used as part of a generate Selector class       //
// which will hold the directory and its associate                        //
//                                                                        //
////////////////////////////////////////////////////////////////////////////

#include "TBranchProxyDirector.h"
#include "TBranchProxy.h"
#include "TTree.h"

#include <algorithm>

namespace ROOT {

   void Reset(TBranchProxy*x) {x->Reset();} 

   TBranchProxyDirector::TBranchProxyDirector(TTree* tree, Long64_t i) : 
      fTree(tree),
      fEntry(i) 
   {
      // Simple constructor
   }

   TBranchProxyDirector::TBranchProxyDirector(TTree* tree, Int_t i) :  
      // cint has a problem casting int to long long
      fTree(tree),
      fEntry(i) 
   {
      // Simple constructor
   }
      
   void TBranchProxyDirector::Attach(TBranchProxy* p) {

      // Attach a TBranchProxy object to this director.  The director just
      // 'remembers' this BranchProxy and does not own it.  It will be use
      // to apply Tree wide operation (like reseting).
      fDirected.push_back(p);
   }

   Long64_t TBranchProxyDirector::GetReadEntry() const {
      
      // return the entry currently being read
      return fEntry;
   }

   TTree* TBranchProxyDirector::GetTree() const {

      // Returns the tree object currently looks at.
      return fTree;
   }

   void TBranchProxyDirector::SetReadEntry(Long64_t entry) {

      // move to a new entry to read
      fEntry = entry;
   }

   TTree* TBranchProxyDirector::SetTree(TTree *newtree) {
      
      // Set the BranchProxy to be looking at a new tree.
      // Reset all.
      // Return the old tree.
      
      TTree* oldtree = fTree;
      fTree = newtree;
      fEntry = -1;
      //if (fInitialized) fInitialized = setup();
      //fprintf(stderr,"calling SetTree for %p\n",this);
      std::for_each(fDirected.begin(),fDirected.end(),Reset);
      return oldtree;
   }

} // namespace ROOT

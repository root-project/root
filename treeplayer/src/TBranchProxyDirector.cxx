// @(#)root/base:$Name:  $:$Id: TBranchProxyDirector.cxx,v 1.2 2004/06/28 17:00:36 brun Exp $
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
#include "TFriendProxy.h"
#include "TTree.h"

#include <algorithm>

namespace std {} using namespace std;

namespace ROOT {

   // Helper function to call Reset on each TBranchProxy
   void Reset(TBranchProxy *x) { x->Reset(); } 

   // Helper function to call SetReadEntry on all TFriendProxy
   void ResetReadEntry(TFriendProxy *x) { x->ResetReadEntry(); }

   // Helper class to call Update on all TFriendProxy
   struct Update {
      Update(TTree *newtree) : fNewTree(newtree) {}
      TTree *fNewTree;
      void operator()(TFriendProxy *x) { x->Update(fNewTree); }
   };


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

   void TBranchProxyDirector::Attach(TFriendProxy* p) {

      // Attach a TFriendProxy object to this director.  The director just
      // 'remembers' this BranchProxy and does not own it.  It will be use
      // to apply Tree wide operation (like reseting).
      fFriends.push_back(p);
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
      for_each(fFriends.begin(),fFriends.end(),ResetReadEntry);
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
      for_each(fDirected.begin(),fDirected.end(),Reset);
      Update update(fTree);
      for_each(fFriends.begin(),fFriends.end(),update);
      return oldtree;
   }

} // namespace ROOT

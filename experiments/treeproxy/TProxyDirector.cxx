// @(#)root/base:$Name:  $:$Id: TObject.cxx,v 1.51 2003/05/01 07:42:36 brun Exp $
// Author: Philippe Canal  13/05/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProxyDirector                                                       //
//                                                                      //
// This class is used to 'drive' and hold a serie of TProxy objects     //
// which represent and give access to the content of TTree object.      //
// This is intended to be used as part of a generate Selector class     //
// which will hold the directory and its associate                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TProxyDirector.h"
#include "TProxy.h"
#include "TTree.h"

#include <algorithm>

namespace ROOT {

   void reset(TProxy*x) {x->reset();} 

   TProxyDirector::TProxyDirector(TTree* tree, Long64_t i) : 
      fTree(tree),
      fEntry(i) 
   {
      // Simple constructor
   }

   TProxyDirector::TProxyDirector(TTree* tree, Int_t i) :  
      // cint has a problem casting int to long long
      fTree(tree),
      fEntry(i) 
   {
      // Simple constructor
   }
      
   void TProxyDirector::Attach(TProxy* p) {

      // Attach a TProxy object to this director.  The director just
      // 'remembers' this proxy and does not own it.  It will be use
      // to apply Tree wide operation (like reseting).
      fDirected.push_back(p);
   }

   Long64_t TProxyDirector::GetReadEntry() const {
      
      // return the entry currently being read
      return fEntry;
   }

   TTree* TProxyDirector::GetTree() const {

      // Returns the tree object currently looks at.
      return fTree;
   }

   void TProxyDirector::SetReadEntry(Long64_t entry) {

      // move to a new entry to read
      fEntry = entry;
   }

   TTree* TProxyDirector::SetTree(TTree *newtree) {
      
      // Set the proxy to be looking at a new tree.
      // Reset all.
      // Return the old tree.
      
      TTree* oldtree = fTree;
      fTree = newtree;
      fEntry = -1;
      //if (fInitialized) fInitialized = setup();
      //fprintf(stderr,"calling SetTree for %p\n",this);
      std::for_each(fDirected.begin(),fDirected.end(),reset);
      return oldtree;
   }

} // namespace ROOT

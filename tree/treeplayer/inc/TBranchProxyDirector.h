// @(#)root/base:$Id$
// Author: Philippe Canal 13/05/2003

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TBranchProxyDirector
#define ROOT_TBranchProxyDirector

#include "Rtypes.h"
#include <vector>
#include <list>
#include <algorithm>

class TH1F;
class TTree;

namespace ROOT {
namespace Detail {
   class TBranchProxy;
   class TFriendProxy;
}

namespace Internal{
   class TFriendProxy;

   // Helper function to call SetReadEntry on all TFriendProxy
   void ResetReadEntry(TFriendProxy *fp);

   class TBranchProxyDirector {

      //This class could actually be the selector itself.
      TTree   *fTree;  // TTree we are currently looking at.
      Long64_t fEntry; // Entry currently being read.

      std::list<Detail::TBranchProxy*> fDirected;
      std::vector<TFriendProxy*> fFriends;

      TBranchProxyDirector(const TBranchProxyDirector &) : fTree(0), fEntry(-1) {;}
      TBranchProxyDirector& operator=(const TBranchProxyDirector&) {return *this;}

   public:

      TBranchProxyDirector(TTree* tree, Long64_t i);
      TBranchProxyDirector(TTree* tree, Int_t i);     // cint has (had?) a problem casting int to long long

      void     Attach(Detail::TBranchProxy* p);
      void     Attach(TFriendProxy* f);
      TH1F*    CreateHistogram(const char *options);
      Long64_t GetReadEntry() const { return fEntry; }
      TTree*   GetTree() const { return fTree; };
      // void   Print();
      void     SetReadEntry(Long64_t entry) {
         // move to a new entry to read
         fEntry = entry;
         if (!fFriends.empty()) {
            std::for_each(fFriends.begin(), fFriends.end(), ResetReadEntry);
         }
      }
      TTree*   SetTree(TTree *newtree);

   };

} // namespace Internal
} // namespace ROOT

#endif

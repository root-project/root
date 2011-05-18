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

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include <list>

#ifdef R__OLDHPACC
namespace std {
   using ::list;
}
#endif

class TH1F;
class TTree;

namespace ROOT {
   class TBranchProxy;
   class TFriendProxy;

   class TBranchProxyDirector {

      //This class could actually be the selector itself.
      TTree   *fTree;  // TTree we are currently looking at.
      Long64_t fEntry; // Entry currently being read.

      std::list<TBranchProxy*> fDirected;
      std::list<TFriendProxy*> fFriends;

      TBranchProxyDirector(const TBranchProxyDirector &) : fTree(0), fEntry(-1) {;}
      TBranchProxyDirector& operator=(const TBranchProxyDirector&) {return *this;}

   public:

      TBranchProxyDirector(TTree* tree, Long64_t i);
      TBranchProxyDirector(TTree* tree, Int_t i);     // cint has (had?) a problem casting int to long long

      void     Attach(TBranchProxy* p);
      void     Attach(TFriendProxy* f);
      TH1F*    CreateHistogram(const char *options);
      Long64_t GetReadEntry() const { return fEntry; }
      TTree*   GetTree() const { return fTree; };
      // void   Print();
      void     SetReadEntry(Long64_t entry);
      TTree*   SetTree(TTree *newtree);

   };

} /* namespace ROOT */

#endif

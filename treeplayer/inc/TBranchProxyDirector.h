// @(#)root/base:$Name:  $:$Id: TBranchProxyDirector.h,v 1.1 2004/02/20 19:23:30 cvsuser Exp $
// Author: Philippe Canal 13/05/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TBRANCHPROXYDIRECTOR_H
#define TBRANCHPROXYDIRECTOR_H

#include "TTree.h"
#include <list>

namespace ROOT {
   class TBranchProxy;

   class TBranchProxyDirector {

      //This class could actually be the selector itself.
      TTree *fTree;
      Long64_t fEntry;
      std::list<TBranchProxy*> fDirected;

   public:
      
      TBranchProxyDirector(TTree* tree, Long64_t i);
      TBranchProxyDirector(TTree* tree, Int_t i);     // cint has (had?) a problem casting int to long long
      
      void     Attach(TBranchProxy* p);
      Long64_t GetReadEntry() const;
      TTree*   GetTree() const;
      // void   Print();
      void     SetReadEntry(Long64_t entry);
      TTree*   SetTree(TTree *newtree);

   };

} /* namespace ROOT */

#endif /* TBRANCHPROXYDIRECTOR_H */

// @(#)root/base:$Name:  $:$Id: $
// Author: Philippe Canal 13/05/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun, Fons Rademakers and al.           *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef TPROXYDIRECTOR_H
#define TPROXYDIRECTOR_H

#include <list>
class TTree;

namespace ROOT {
   class TProxy;

   class TProxyDirector {

      //This class could actually be the selector itself.
      TTree *fTree;
      Long64_t fEntry;
      std::list<TProxy*> fDirected;

   public:
      
      TProxyDirector(TTree* tree, Long64_t i);
      TProxyDirector(TTree* tree, Int_t i);     // cint has (had?) a problem casting int to long long
      
      void     Attach(TProxy* p);
      Long64_t GetReadEntry() const;
      TTree*   GetTree() const;
      // void   Print();
      void     SetReadEntry(Long64_t entry);
      TTree*   SetTree(TTree *newtree);

   };

} /* namespace ROOT */

#endif /* TPROXYDIRECTOR_H */

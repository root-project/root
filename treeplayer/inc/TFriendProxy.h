// @(#)root/treeplayer:$Id$
// Author: Philippe Canal 01/06/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFriendProxy
#define ROOT_TFriendProxy

#ifndef ROOT_TBranchProxyDirector
#include "TBranchProxyDirector.h"
#endif

class TTree;

namespace ROOT {

   class TFriendProxy {
   protected:
      TBranchProxyDirector fDirector; // contain pointer to TTree and entry to be read
      Int_t  fIndex; // Index of this tree in the list of friends

   public:
      TFriendProxy();
      TFriendProxy(TBranchProxyDirector *director, TTree *main, Int_t index);

      Long64_t GetReadEntry() const;
      void     ResetReadEntry();
      void     Update(TTree *newmain);
   };

}

#endif

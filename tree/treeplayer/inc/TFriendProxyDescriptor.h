// @(#)rooeeplayer:$Id$
// Author: Philippe Canal 06/06/2004

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers and al.        *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFriendProxyDescriptor
#define ROOT_TFriendProxyDescriptor

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TList
#include "TList.h"
#endif


namespace ROOT {

   class TFriendProxyDescriptor : public TNamed {

      Bool_t fDuplicate;
      Int_t  fIndex;
      TList  fListOfTopProxies;

   private:
      TFriendProxyDescriptor(const TFriendProxyDescriptor &b);
      TFriendProxyDescriptor& operator=(const TFriendProxyDescriptor &b);

   public:
      TFriendProxyDescriptor(const char *treename, const char *aliasname, Int_t index);

      Int_t  GetIndex() const { return fIndex; }
      TList *GetListOfTopProxies() { return &fListOfTopProxies; }

      Bool_t IsEquivalent(const TFriendProxyDescriptor *other);

      void OutputClassDecl(FILE *hf, int offset, UInt_t maxVarname);
      void OutputDecl(FILE *hf, int offset, UInt_t maxVarname);

      Bool_t IsDuplicate() { return fDuplicate; }
      void   SetDuplicate() { fDuplicate = kTRUE; }

      ClassDef(TFriendProxyDescriptor,0); // Describe a branch from a TTreeFriend.
   };
}

#endif

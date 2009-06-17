// @(#)root/base:$Id$
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
// TFriendProxy                                                         //
//                                                                      //
// Concrete implementation of the proxy around a Friend Tree.           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TFriendProxy.h"
#include "TTree.h"
#include "TList.h"
#include "TFriendElement.h"

ClassImp(ROOT::TFriendProxy);

namespace ROOT {

   //------------------------------------------------------------------------------------
   TFriendProxy::TFriendProxy() : fDirector(0,-1), fIndex(-1)
   {
   }

   //------------------------------------------------------------------------------------
   TFriendProxy::TFriendProxy(TBranchProxyDirector *director, TTree *main, Int_t index) :
      fDirector(0,-1), fIndex(index)
   {
      // Constructor.

      if (main) {
         TObject *obj = main->GetListOfFriends()->At(fIndex);
         TFriendElement *element = dynamic_cast<TFriendElement*>( obj );
         if (element) fDirector.SetTree(element->GetTree());
      }
      director->Attach(this);
   }

   //------------------------------------------------------------------------------------
   Long64_t TFriendProxy::GetReadEntry() const
   {
      // Return the entry number currently being looked at.

      return fDirector.GetReadEntry();
   }

   //------------------------------------------------------------------------------------
   void TFriendProxy::ResetReadEntry()
   {
      // Refresh the cached read entry number from the original tree.

      // The 2nd call to GetTree is to insure we get the 'local' tree's entry in the case of a
      // chain.
      if (fDirector.GetTree()) fDirector.SetReadEntry(fDirector.GetTree()->GetTree()->GetReadEntry());
   }

  //------------------------------------------------------------------------------------
   void TFriendProxy::Update(TTree *newmain)
   {
      // Update the address of the underlying tree.

      if (newmain) {
         TObject *obj = newmain->GetListOfFriends()->At(fIndex);
         TFriendElement *element = dynamic_cast<TFriendElement*>( obj );
         if (element) fDirector.SetTree(element->GetTree());
         else fDirector.SetTree(0);
      } else {
         fDirector.SetTree(0);
      }
   }
}

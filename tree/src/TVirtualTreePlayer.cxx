// @(#)root/tree:$Name:  $:$Id: TVirtualTreePlayer.cxx,v 1.1.1.1 2000/05/16 17:00:45 rdm Exp $
// Author: Rene Brun   30/08/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TVirtualTreePlayer.h"
#include "TPluginManager.h"

TClass  *TVirtualTreePlayer::fgPlayer = 0;

ClassImp(TVirtualTreePlayer)

//______________________________________________________________________________
TVirtualTreePlayer *TVirtualTreePlayer::TreePlayer(TTree *obj)
{
   // Static function returning a pointer to a Tree player.
   // The player will process the specified obj. If the Tree player
   // does not exist a default player is created.

   // if no player set yet,  create a default painter via the PluginManager
   if (!fgPlayer) {
      if (gROOT->LoadClass("TProof","Proof")) return 0;
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualTreePlayer"))) {
         if (h->LoadPlugin() == -1)
            return 0;
         TVirtualTreePlayer::SetPlayer(h->GetClass());
         if (!fgPlayer) return 0;
      }
   }
   
   //create an instance of the Tree player
   TVirtualTreePlayer *p = (TVirtualTreePlayer*)fgPlayer->New();
   if (p) p->SetTree(obj);
   return p;
}

//______________________________________________________________________________
void TVirtualTreePlayer::SetPlayer(const char *player)
{
   // Static function to set an alternative Tree player.

   fgPlayer = gROOT->GetClass(player);
}

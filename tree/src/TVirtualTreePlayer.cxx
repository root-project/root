// @(#)root/tree:$Name$:$Id$
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

TClass  *TVirtualTreePlayer::fgPlayer = 0;

ClassImp(TVirtualTreePlayer)

//______________________________________________________________________________
TVirtualTreePlayer *TVirtualTreePlayer::TreePlayer(TTree *obj)
{
   // Static function returning a pointer to a Tree player.
   // The player will process the specified obj. If the Tree player
   // does not exist a default player is created.

   // if no player set yet, set TTreePlayer by default
   if (!fgPlayer) {
      if (gROOT->LoadClass("TProof","Proof")) return 0;
      if (gROOT->LoadClass("TTreePlayer","TreePlayer")) return 0;
      TVirtualTreePlayer::SetPlayer("TTreePlayer");
      if (!fgPlayer) return 0;
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

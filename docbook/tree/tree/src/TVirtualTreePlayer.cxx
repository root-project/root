// @(#)root/tree:$Id$
// Author: Rene Brun   30/08/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualTreePlayer
// 
// Abstract base class defining the interface for the plugins that
// implement Draw, Scan, Process, MakeProxy, etc. for a TTree object.
// See the individual documentations in TTree. 
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TROOT.h"
#include "TVirtualTreePlayer.h"
#include "TPluginManager.h"
#include "TClass.h"

TClass              *TVirtualTreePlayer::fgPlayer  = 0;
TVirtualTreePlayer  *TVirtualTreePlayer::fgCurrent = 0;

ClassImp(TVirtualTreePlayer)

//______________________________________________________________________________
TVirtualTreePlayer *TVirtualTreePlayer::TreePlayer(TTree *obj)
{
   // Static function returning a pointer to a Tree player.
   // The player will process the specified obj. If the Tree player
   // does not exist a default player is created.

   // if no player set yet,  create a default painter via the PluginManager
   if (!fgPlayer) {
      TPluginHandler *h;
      if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualTreePlayer"))) {
         if (h->LoadPlugin() == -1)
            return 0;
         TVirtualTreePlayer::SetPlayer(h->GetClass());
      }
      if (!fgPlayer) return 0;
   }

   //create an instance of the Tree player
   TVirtualTreePlayer *p = (TVirtualTreePlayer*)fgPlayer->New();
   if (p) p->SetTree(obj);
   fgCurrent = p;
   return p;
}

//______________________________________________________________________________
TVirtualTreePlayer::~TVirtualTreePlayer()
{
   // Common destructor.

   if (fgCurrent==this) {
      // Make sure fgCurrent does not point to a deleted player.
      fgCurrent=0;
   }
}

//______________________________________________________________________________
TVirtualTreePlayer *TVirtualTreePlayer::GetCurrentPlayer()
{
   // Static function: return the current player (if any)

   return fgCurrent;
}

//______________________________________________________________________________
void TVirtualTreePlayer::SetPlayer(const char *player)
{
   // Static function to set an alternative Tree player.

   fgPlayer = TClass::GetClass(player);
}


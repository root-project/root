// @(#)root/tree:$Id$
// Author: Rene Brun   30/08/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TVirtualTreePlayer
\ingroup tree

Abstract base class defining the interface for the plugins that
implement Draw, Scan, Process, MakeProxy, etc. for a TTree object.
See the individual documentations in TTree.
*/

#include "TROOT.h"
#include "TVirtualTreePlayer.h"
#include "TPluginManager.h"
#include "TClass.h"

TClass              *TVirtualTreePlayer::fgPlayer  = 0;
TVirtualTreePlayer  *TVirtualTreePlayer::fgCurrent = 0;

ClassImp(TVirtualTreePlayer);

////////////////////////////////////////////////////////////////////////////////
/// Static function returning a pointer to a Tree player.
/// The player will process the specified obj. If the Tree player
/// does not exist a default player is created.

TVirtualTreePlayer *TVirtualTreePlayer::TreePlayer(TTree *obj)
{
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

////////////////////////////////////////////////////////////////////////////////
/// Common destructor.

TVirtualTreePlayer::~TVirtualTreePlayer()
{
   if (fgCurrent==this) {
      // Make sure fgCurrent does not point to a deleted player.
      fgCurrent=0;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Static function: return the current player (if any)

TVirtualTreePlayer *TVirtualTreePlayer::GetCurrentPlayer()
{
   return fgCurrent;
}

////////////////////////////////////////////////////////////////////////////////
/// Static function to set an alternative Tree player.

void TVirtualTreePlayer::SetPlayer(const char *player)
{
   fgPlayer = TClass::GetClass(player);
}


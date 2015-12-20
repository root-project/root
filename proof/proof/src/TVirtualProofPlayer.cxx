// @(#)root/proof:$Id$
// Author: Fons Rademakers   15/03/07

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class TVirtualProofPlayer
\ingroup proofkernel

Abstract interface for the PROOF player.
See the concrete implementations under 'proofplayer' for details.

*/

#include "TVirtualProofPlayer.h"
#include "TPluginManager.h"
#include "TROOT.h"
#include "TError.h"

////////////////////////////////////////////////////////////////////////////////
/// Create a PROOF player.

TVirtualProofPlayer *TVirtualProofPlayer::Create(const char *player,
                                                 TProof *pr, TSocket *s)
{
   TPluginHandler *h;
   TVirtualProofPlayer *p = 0;

   if (!player || !*player) {
      ::Error("TVirtualProofPlayer::Create", "player name missing");
      return 0;
   }

   if ((h = gROOT->GetPluginManager()->FindHandler("TVirtualProofPlayer", player))) {
      if (h->LoadPlugin() == -1)
         return 0;
      if (!strcmp(player, "slave"))
         p = (TVirtualProofPlayer *) h->ExecPlugin(1, s);
      else
         p = (TVirtualProofPlayer *) h->ExecPlugin(1, pr);
   }

   return p;
}

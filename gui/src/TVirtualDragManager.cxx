// $Id: TVirtualDragManager.cxx,v 1.2 2004/09/08 17:34:19 rdm Exp $
// Author: Valeriy Onuchin   02/08/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualDragManager                                                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualDragManager.h"
#include "TROOT.h"
#include "TPluginManager.h"



ClassImp(TVirtualDragManager)


TVirtualDragManager *gDragManager = 0;

//______________________________________________________________________________
TVirtualDragManager::TVirtualDragManager()
{
   // ctor

   Init();
}

//______________________________________________________________________________
TVirtualDragManager *TVirtualDragManager::Instance()
{
   // Load plugin and create drag manager object.

   if (gDragManager) return gDragManager;

   static Bool_t loaded = kFALSE;
   static TPluginHandler *h = 0;

   // load plugin
   if (!loaded) {
      h = gROOT->GetPluginManager()->FindHandler("TVirtualDragManager");
      if (h) {
         if (h->LoadPlugin() == -1) return 0;
         loaded = kTRUE;
      }
   }
   if (loaded) gDragManager = (TVirtualDragManager*)h->ExecPlugin(0);

   return gDragManager;
}

//______________________________________________________________________________
void TVirtualDragManager::Init()
{
   //

   fDragging = kFALSE;
   fMoveWaiting = kFALSE;
   fDropping = kFALSE;
   fPasting = kFALSE;
   fTarget = 0;
   fSource = 0;
   fFrameUnder = 0;
   fPasteFrame = 0;
   fDragType = kDragNone;
}

// $Id$
// Author: Valeriy Onuchin   02/08/04

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This source is based on Xclass95, a Win95-looking GUI toolkit.
    Copyright (C) 1996, 1997 David Barth, Ricky Ralston, Hector Peraza.

    Xclass95 is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

**************************************************************************/

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
   // Load plugin and create drag manager object

   if (gDragManager) return gDragManager;

   static Bool_t loaded = kFALSE;
   static TPluginHandler *h = 0;

   // load plugin
   if (!loaded) {
      h = gROOT->GetPluginManager()->FindHandler("TVirtualDragManager", "GuiBld");

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

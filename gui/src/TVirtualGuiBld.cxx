// $Id: TVirtualGuiBld.cxx,v 1.4 2004/09/12 10:55:26 brun Exp $
// Author: Valeriy Onuchin   12/08/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TVirtualGuiBld                                                       //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TVirtualGuiBld.h"
#include "TVirtualDragManager.h"
#include "TPluginManager.h"

ClassImp(TVirtualGuiBld)
ClassImp(TGuiBldAction)

TVirtualGuiBld *gGuiBuilder = 0;


//______________________________________________________________________________
TGuiBldAction::TGuiBldAction(const char *name, const char *title, Int_t type) :
   TNamed(name, title), fType(type)
{
   // dtor
}

//______________________________________________________________________________
TGuiBldAction::~TGuiBldAction()
{
   // dtor
}

//______________________________________________________________________________
TVirtualGuiBld::TVirtualGuiBld()
{
   // ctor

   gDragManager = TVirtualDragManager::Instance();
   gGuiBuilder  = this;
   fAction      = 0;
}

//______________________________________________________________________________
TVirtualGuiBld::~TVirtualGuiBld()
{
   // dtor

   gGuiBuilder = 0;
}

//______________________________________________________________________________
TVirtualGuiBld *TVirtualGuiBld::Instance()
{
   // Load plugin and create gGuiBuilder object

   if (gGuiBuilder) return gGuiBuilder;

   static Bool_t loaded = kFALSE;
   static TPluginHandler *h = 0;

   // load plugin
   if (!loaded) {
      h = gROOT->GetPluginManager()->FindHandler("TVirtualGuiBld", "GuiBld");

      if (h) {
         if (h->LoadPlugin() == -1) return 0;
         loaded = kTRUE;
      }
   }
   if (loaded) gGuiBuilder = (TVirtualGuiBld*)h->ExecPlugin(0);

   return gGuiBuilder;
}

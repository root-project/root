// $Id: TVirtualGuiBld.cxx,v 1.2 2004/09/08 17:34:19 rdm Exp $
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

// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TEveEventManager.h"

#include "TObjString.h"
#include "TInterpreter.h"

//______________________________________________________________________________
// TEveEventManager
//
// Base class for event management and navigation.

ClassImp(TEveEventManager)

//______________________________________________________________________________
TEveEventManager::TEveEventManager(const Text_t* n, const Text_t* t) :
   TEveElementList(n, t),
   fNewEventCommands()
{
   // Constructor.
}

/******************************************************************************/

//______________________________________________________________________________
void TEveEventManager::AfterNewEventLoaded()
{
   // Virtual function to be called after a new event is loaded.
   // It iterates over the list of registered commands
   // (fNewEventCommands) and executes them in given order.

   TIter next(&fNewEventCommands);
   TObject* o;
   while ((o = next())) {
      TObjString* s = dynamic_cast<TObjString*>(o);
      if (s)
         gInterpreter->ProcessLine(s->String());
   }
}

//______________________________________________________________________________
void TEveEventManager::AddNewEventCommand(const Text_t* cmd)
{
   // Register a command to be executed on each new event.

   fNewEventCommands.Add(new TObjString(cmd));
}

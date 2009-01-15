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
TEveEventManager::TEveEventManager(const char* n, const char* t) :
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

   for (std::vector<TString>::iterator i = fNewEventCommands.begin(); i != fNewEventCommands.end(); ++i)
   {
      gInterpreter->ProcessLine(*i);
   }
}

//______________________________________________________________________________
void TEveEventManager::AddNewEventCommand(const TString& cmd)
{
   // Register a command to be executed on each new event.

   fNewEventCommands.push_back(cmd);
}

//______________________________________________________________________________
void TEveEventManager::RemoveNewEventCommand(const TString& cmd)
{
   // Remove the first command equal to cmd.

   for (std::vector<TString>::iterator i = fNewEventCommands.begin(); i != fNewEventCommands.end(); ++i)
   {
      if (cmd == *i) {
         fNewEventCommands.erase(i);
         break;
      }
   }
}

//______________________________________________________________________________
void TEveEventManager::ClearNewEventCommands()
{
   // Clear the list of commands to be executed on each new event.

   fNewEventCommands.clear();
}

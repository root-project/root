// @(#)root/eve:$Id$
// Authors: Matevz Tadel & Alja Mrak-Tadel: 2006, 2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TEveEventManager
#define ROOT_TEveEventManager

#include "TEveElement.h"
#include "TList.h"

class TEveEventManager : public TEveElementList
{
protected:
   TList        fNewEventCommands;

public:
   TEveEventManager(const Text_t* n="TEveEventManager", const Text_t* t="");
   virtual ~TEveEventManager() {}

   TList& GetNewEventCommands() { return fNewEventCommands; }

   virtual void Open() {}
   virtual void GotoEvent(Int_t /*event*/) {}
   virtual void NextEvent() {}
   virtual void PrevEvent() {}
   virtual void Close() {}

   virtual void AfterNewEventLoaded();
   virtual void AddNewEventCommand(const Text_t* cmd);

   ClassDef(TEveEventManager, 1); // Base class for event management and navigation.
};

#endif

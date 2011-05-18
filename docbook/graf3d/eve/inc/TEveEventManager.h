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

#include <vector>

class TEveEventManager : public TEveElementList
{
protected:
   std::vector<TString>  fNewEventCommands;

public:
   TEveEventManager(const char* n="TEveEventManager", const char* t="");
   virtual ~TEveEventManager() {}

   std::vector<TString>& GetNewEventCommands() { return fNewEventCommands; }

   virtual void Open() {}
   virtual void GotoEvent(Int_t /*event*/) {}
   virtual void NextEvent() {}
   virtual void PrevEvent() {}
   virtual void Close() {}

   virtual void AfterNewEventLoaded();

   virtual void AddNewEventCommand(const TString& cmd);
   virtual void RemoveNewEventCommand(const TString& cmd);
   virtual void ClearNewEventCommands();

   ClassDef(TEveEventManager, 1); // Base class for event management and navigation.
};

#endif

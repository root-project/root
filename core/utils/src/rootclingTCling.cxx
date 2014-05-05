// @(#)root/utils:$Id$
// Author: Axel Naumann, 2014-04-07

/*************************************************************************
 * Copyright (C) 1995-2014, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

// Provides bindings to TCling (compiled with rtti) from rootcling (compiled
// without rtti).

#include "TCling.h"
#include "TROOT.h"
#include "TFile.h"
#include "TClass.h"
#include "TStreamerInfo.h"
#include <iostream>

std::string gPCMFilename;
std::vector<std::string> gClassesToStore;

extern "C"
cling::Interpreter* TCling__GetInterpreter()
{
   static bool sInitialized = false;
   gROOT; // trigger initialization
   if (!sInitialized) {
      gCling->SetClassAutoloading(false);
      sInitialized = true;
   }
   return ((TCling*)gCling)->GetInterpreter();
}

extern "C"
void InitializeStreamerInfoROOTFile(const char* filename)
{
   gPCMFilename = filename;
}

extern "C"
void AddStreamerInfoToROOTFile(const char* normName)
{
   gClassesToStore.push_back(normName);
}

extern "C"
bool CloseStreamerInfoROOTFile()
{
   // Write all persistent TClasses.

   // Avoid plugins.
   TVirtualStreamerInfo::SetFactory(new TStreamerInfo());

   TObjArray classData;
   for (const auto& normName: gClassesToStore) {
      TClass* cl = TClass::GetClass(normName.c_str(), kTRUE /*load*/);
      if (!cl) {
         std::cerr << "ERROR in CloseStreamerInfoROOTFile(): cannot find class "
                   << normName << '\n';
         return false;
      }
      // If the class is not persistent we return success.
      if (cl->GetClassVersion() == 0)
         continue;
      // If this is a proxied collection then offsets are not needed.
      if (cl->GetCollectionProxy())
         continue;
      // streamerInfos.AddLast(...)
   }

   // Don't use TFile::Open(); we don't need plugins.
   TFile dictFile(gPCMFilename.c_str(), "RECREATE");
   if (dictFile.IsZombie())
      return false;
   // Instead of plugins:
   classData.Write("__StreamerInfoOffsets", TObject::kSingleKey);
   return true;
}

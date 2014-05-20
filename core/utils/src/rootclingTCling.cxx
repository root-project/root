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
#include "TProtoClass.h"

std::string gPCMFilename;
std::vector<std::string> gClassesToStore;
std::vector<std::string> gTypedefsToStore;
std::vector<std::string> gAncestorPCMsNames;

extern "C"
const char*** TROOT__GetExtraInterpreterArgs() {
   return &TROOT::GetExtraInterpreterArgs();
}

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
   gClassesToStore.emplace_back(normName);
}

extern "C"
void AddTypedefToROOTFile(const char* tdname)
{
   gTypedefsToStore.push_back(tdname);
}

extern "C"
void AddAncestorPCMROOTFile(const char* pcmName)
{
   gAncestorPCMsNames.emplace_back(pcmName);
}

extern "C"
bool CloseStreamerInfoROOTFile()
{
   // Write all persistent TClasses.

   // Avoid plugins.
   TVirtualStreamerInfo::SetFactory(new TStreamerInfo());

   TObjArray protoClasses;
   for (const auto normName: gClassesToStore) {
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
      cl->Property(); // Force initialization of the bits and property fields.
      protoClasses.AddLast(new TProtoClass(cl));
   }

   TObjArray typedefs;
   for (const auto dtname: gTypedefsToStore) {
      TDataType* dt = (TDataType*)gROOT->GetListOfTypes()->FindObject(dtname.c_str());
      if (!dt) {
         std::cerr << "ERROR in CloseStreamerInfoROOTFile(): cannot find class "
                   << dtname << '\n';
         return false;
      }
      if (dt->GetType() == -1) {
         dt->Property(); // Force initialization of the bits and property fields.
         dt->GetTypeName(); // Force caching of type name.
         typedefs.AddLast(dt);
      }
   }

   // Don't use TFile::Open(); we don't need plugins.
   TFile dictFile(gPCMFilename.c_str(), "RECREATE");
   if (dictFile.IsZombie())
      return false;
   // Instead of plugins:
   protoClasses.Write("__ProtoClasses", TObject::kSingleKey);
   protoClasses.Delete();
   typedefs.Write("__Typedefs", TObject::kSingleKey);

   dictFile.WriteObjectAny(&gAncestorPCMsNames, "std::vector<std::string>", "__AncestorPCMsNames");


   return true;
}

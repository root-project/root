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

TFile* gDictFile = 0;

/*
  Done by ~TROOT() before.
struct DictFileCloser {
   ~DictFileCloser() {
      delete gDictFile;
   }
} dictFileCloser;
*/

extern "C"
cling::Interpreter* TCling__GetInterpreter()
{
   gROOT; // trigger initialization
   return ((TCling*)gCling)->GetInterpreter();
}

extern "C"
void InitializeStreamerInfoROOTFile(const char* filename)
{
   // Don't use TFile::Open(); we don't need plugins.
   gDictFile = new TFile(filename, "RECREATE");
   // Instead of plugins:
   TVirtualStreamerInfo::SetFactory(new TStreamerInfo());
}

extern "C"
bool AddStreamerInfoToROOTFile(const char* normName)
{
   TClass* cl = TClass::GetClass(normName, kTRUE /*load*/);
   if (!cl)
      return false;
   TVirtualStreamerInfo* SI = cl->GetStreamerInfo();
   //FIXME: merge with TStreamerOffsets branch, then:
   // SI->BuildOffsets();
   if (!SI)
      return false;
   SI->ForceWriteInfo(gDictFile, true);
   return true;
}

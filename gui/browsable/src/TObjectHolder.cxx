/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/Browsable/TObjectHolder.hxx>

#include "TROOT.h"
#include <sstream>

using namespace ROOT::Experimental::Browsable;

///////////////////////////////////////////////////////////////////////////
/// Return TObject instance with ownership
/// If object is not owned by the holder, it will be cloned (with few exceptions)

void *TObjectHolder::TakeObject()
{
   auto res = fObj;

   if (fOwner) {
      fObj = nullptr;
      fOwner = false;
   } else if (fObj && !fObj->IsA()->InheritsFrom("TDirectory") && !fObj->IsA()->InheritsFrom("TTree")) {
      res = fObj->Clone();
      if (res && res->InheritsFrom("TH1")) {
         std::stringstream cmd;
         cmd << "((TH1 *) " << std::hex << std::showbase << (size_t)res << ")->SetDirectory(nullptr);";
         gROOT->ProcessLine(cmd.str().c_str());
      }
   } else {
      res = nullptr;
   }

   return res;
}

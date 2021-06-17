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
/// Check if object is not registered in some global lists
/// Prevent double deletion

void TObjectHolder::ClearROOTOwnership(TObject *obj)
{
   if (obj && obj->InheritsFrom("TH1")) {
      std::stringstream cmd;
      cmd << "((TH1 *) " << std::hex << std::showbase << (size_t)obj << ")->SetDirectory(nullptr);";
      gROOT->ProcessLine(cmd.str().c_str());
   } else if (obj && obj->InheritsFrom("TF1")) {
      std::stringstream cmd;
      cmd << "((TF1 *) " << std::hex << std::showbase << (size_t)obj << ")->AddToGlobalList(kFALSE);";
      gROOT->ProcessLine(cmd.str().c_str());
   }
}

///////////////////////////////////////////////////////////////////////////
/// Return TObject instance with ownership
/// If object is not owned by the holder, it will be cloned (except TDirectory or TTree classes)

void *TObjectHolder::TakeObject()
{
   auto res = fObj;

   if (fOwner) {
      fObj = nullptr;
      fOwner = false;
   } else if (fObj && !fObj->IsA()->InheritsFrom("TDirectory") && !fObj->IsA()->InheritsFrom("TTree")) {
      res = fObj->Clone();
      ClearROOTOwnership(res);
   } else {
      res = nullptr;
   }

   return res;
}

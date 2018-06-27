/// \file RObjectDrawable.cxx
/// \ingroup CanvasPainter ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-05-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RObjectDrawable.hxx>

#include <ROOT/RDisplayItem.hxx>
#include <ROOT/TLogger.hxx>
#include <ROOT/RMenuItem.hxx>
#include <ROOT/RPadPainter.hxx>

#include "TClass.h"
#include "TROOT.h"

#include <exception>

void ROOT::Experimental::RObjectDrawable::Paint(Internal::RPadPainter &pad)
{
   pad.AddDisplayItem(std::make_unique<RObjectDisplayItem>(fObj.get(), fOpts.GetOptionString()));
}

void ROOT::Experimental::RObjectDrawable::PopulateMenu(RMenuItems &items)
{
   TObject *obj = fObj.get();

   // fill context menu items for the ROOT class
   items.PopulateObjectMenu(obj, obj->IsA());
}

void ROOT::Experimental::RObjectDrawable::Execute(const std::string &exec)
{
   TObject *obj = fObj.get();

   TString cmd;
   cmd.Form("((%s*) %p)->%s;", obj->ClassName(), obj, exec.c_str());
   printf("RObjectDrawable::Execute Obj %s Cmd %s\n", obj->GetName(), cmd.Data());
   gROOT->ProcessLine(cmd);
}

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RObjectDrawable.hxx>

#include <ROOT/RDisplayItem.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RMenuItems.hxx>

#include "TROOT.h"

#include <exception>
#include <sstream>
#include <iostream>


using namespace ROOT::Experimental;

std::unique_ptr<RDisplayItem> RObjectDrawable::Display(const RPadBase &, Version_t vers) const
{
   if (GetVersion() > vers)
      return std::make_unique<RObjectDisplayItem>(fObj.get(), fOpts);
   return nullptr;
}

void RObjectDrawable::PopulateMenu(RMenuItems &items)
{
   // fill context menu items for the ROOT class
   items.PopulateObjectMenu(fObj.get(), fObj.get()->IsA());
}

void RObjectDrawable::Execute(const std::string &exec)
{
   TObject *obj = fObj.get();

   std::stringstream cmd;
   cmd << "((" << obj->ClassName() << "* ) " << std::hex << std::showbase << (size_t)obj << ")->" << exec << ";";
   std::cout << "RObjectDrawable::Execute Obj " << obj->GetName() << "Cmd " << cmd.str() << std::endl;
   gROOT->ProcessLine(cmd.str().c_str());
}

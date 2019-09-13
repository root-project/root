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
#include <ROOT/RLogger.hxx>
#include <ROOT/RMenuItem.hxx>

#include "TROOT.h"


#include <exception>
#include <sstream>
#include <iostream>

void ROOT::Experimental::RObjectDrawable::PopulateMenu(RMenuItems &items)
{
   // fill context menu items for the ROOT class
   items.PopulateObjectMenu(fObj.get(), fObj.get()->IsA());
}

void ROOT::Experimental::RObjectDrawable::Execute(const std::string &exec)
{
   TObject *obj = fObj.get();

   std::stringstream cmd;
   cmd << "((" << obj->ClassName() << "* ) " << std::hex << std::showbase << (size_t)obj << ")->" << exec << ";";
   std::cout << "RObjectDrawable::Execute Obj " << obj->GetName() << "Cmd " << cmd.str() << std::endl;
   gROOT->ProcessLine(cmd.str().c_str());
}

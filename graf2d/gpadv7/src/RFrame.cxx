/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RFrame.hxx"

#include "ROOT/RLogger.hxx"
#include "ROOT/RPadUserAxis.hxx"
#include "ROOT/RMenuItem.hxx"

#include "TROOT.h"

#include <cassert>
#include <sstream>

ROOT::Experimental::RFrame::RFrame(std::vector<std::unique_ptr<RPadUserAxisBase>> &&coords) : RFrame()
{
   fUserCoord=  std::move(coords);
   fPalette = RPalette::GetPalette("default");
}

void ROOT::Experimental::RFrame::Execute(const std::string &arg)
{
   std::stringstream cmd;
   cmd << "((RFrame *) " << std::hex << std::showbase << (size_t)this << ")->" << arg << ";";
   gROOT->ProcessLine(cmd.str().c_str());
}

void ROOT::Experimental::RFrame::GrowToDimensions(size_t nDimensions)
{
   std::size_t oldSize = fUserCoord.size();
   if (oldSize >= nDimensions)
      return;
   fUserCoord.resize(nDimensions);
   for (std::size_t idx = oldSize; idx < nDimensions; ++idx)
      if (!fUserCoord[idx])
         fUserCoord[idx].reset(new RPadCartesianUserAxis);
}

void ROOT::Experimental::RFrame::PopulateMenu(RMenuItems &items)
{
   auto is_x = items.GetSpecifier() == "x";
   auto is_y = items.GetSpecifier() == "y";

   if (is_x || is_y) {
      RAttrAxis &attr = is_x ? AttrX() : AttrY();
      std::string name = is_x ? "AttrX()" : "AttrY()";
      items.AddChkMenuItem("Log scale", "Change log scale", attr.GetLog(), name + ".SetLog" + (attr.GetLog() ? "(false)" : "(true)"));
   }
}


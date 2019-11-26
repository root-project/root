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

#include <cassert>

ROOT::Experimental::RFrame::RFrame(std::vector<std::unique_ptr<RPadUserAxisBase>> &&coords) : RFrame()
{
   fUserCoord=  std::move(coords);
   fPalette = RPalette::GetPalette("default");
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

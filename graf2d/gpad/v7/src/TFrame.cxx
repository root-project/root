/// \file TFrame.cxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2017-09-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/TFrame.hxx"

#include "ROOT/TLogger.hxx"
#include "ROOT/TPadUserAxis.hxx"

#include <cassert>

ROOT::Experimental::TFrame::TFrame(std::vector<std::unique_ptr<Detail::TPadUserAxisBase>> &&coords, const DrawingOpts &opts)
   : fUserCoord(std::move(coords)), fPalette(TPalette::GetPalette("default")), fPos(opts.fPos.Get()), fSize(opts.fSize.Get())
{
   if (fUserCoord.empty()) {
      fUserCoord.emplace_back(new TPadCartesianUserAxis);
      fUserCoord.emplace_back(new TPadCartesianUserAxis);
   }
}

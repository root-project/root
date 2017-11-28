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
#include "ROOT/TPadUserCoordBase.hxx"

#include <cassert>

namespace {
using namespace ROOT::Experimental;
// FIXME: Replace by array of TFrameAxis!
class TPadUserCoordDefault: public Detail::TPadUserCoordBase {

public:
   std::array<TPadCoord::Normal, 2> ToNormal(const std::array<TPadCoord::User, 2> &user) const override
   {
      R__ERROR_HERE("Gpad") << "Not yet implemented!";
      return {{user[0].fVal, user[1].fVal}};
   }
};
} // namespace

ROOT::Experimental::TFrame::TFrame(std::unique_ptr<Detail::TPadUserCoordBase> &&coords, const TPadPos &pos,
                                   const TPadExtent &size)
   : fUserCoord(std::move(coords)), fPalette(TPalette::GetPalette("default")), fPos(pos), fSize(size)
{
   if (!fUserCoord)
      fUserCoord.reset(new TPadUserCoordDefault);
}

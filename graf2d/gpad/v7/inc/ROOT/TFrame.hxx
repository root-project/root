/// \file ROOT/TFrame.hxx
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

#ifndef ROOT7_TFrame
#define ROOT7_TFrame

#include "ROOT/TDrawingOptsBase.hxx"
#include "ROOT/TPadExtent.hxx"
#include "ROOT/TPadPos.hxx"
#include "ROOT/TPadUserCoordBase.hxx"
#include "ROOT/TPalette.hxx"

#include <memory>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TFrame
  Holds a user coordinate system with a palette.
  */

class TFrame {
public:
   class DrawingOpts: public TDrawingOptsBase<DrawingOpts> {
   public:
      /// Position of the frame in parent TPad coordinates.
      TPadPos fPos = {0.1_normal, 0.1_normal};
      /// Size of the frame in parent TPad coordinates.
      TPadExtent fSize = {0.8_normal, 0.8_normal};

      DrawingOpts(TPadBase &pad): TDrawingOptsBase<DrawingOpts>(pad, "Frame") {}
   };

private:
   /// Mapping of user coordinates to normal coordinates.
   std::unique_ptr<Detail::TPadUserCoordBase> fUserCoord;

   /// Palette used to visualize user coordinates.
   TPalette fPalette;

   /// Offset with respect to parent TPad.
   TPadPos fPos;

   /// Size of the frame, in parent TPad coordinates.
   TPadExtent fSize;

public:
   /// Constructor taking user coordinate system, position and extent.
   explicit TFrame(std::unique_ptr<Detail::TPadUserCoordBase> &&coords, const TPadPos &pos = DrawingOpts::Default().fPos,
          const TPadExtent &size = DrawingOpts::Default().fSize);

   // Constructor taking position and extent.
   explicit TFrame(const TPadPos &pos = DrawingOpts::Default().fPos, const TPadExtent &size = DrawingOpts::Default().fSize)
      : TFrame(nullptr, pos, size)
   {}

   /// Get the current user coordinate system.
   Detail::TPadUserCoordBase &GetUserCoord() const;

   /// Get the current user coordinate system.
   std::unique_ptr<Detail::TPadUserCoordBase> SwapUserCoordSystem(std::unique_ptr<Detail::TPadUserCoordBase> &&newCoord)
   {
      std::unique_ptr<Detail::TPadUserCoordBase> ret(std::move(newCoord));
      std::swap(ret, fUserCoord);
      return ret;
   }

   /// Convert user coordinates to normal coordinates.
   std::array<TPadCoord::Normal, 2> UserToNormal(const std::array<TPadCoord::User, 2> &pos) const
   {
      return fUserCoord->ToNormal(pos);
   }
};

} // namespace Experimental
} // namespace ROOT

#endif

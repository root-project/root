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

#include <memory>

namespace ROOT {
namespace Experimental {
   class TPalette;

   namespace Internal {
      class TPadUserCoordBase;
   }

/** \class ROOT::Experimental::TFrame
  Holds a user coordinate system with a palette.
  */

class TFrame {
public:
   struct DrawingOpts: public TDrawingOptsBase<DrawingOpts> {
      /// Position of the frame in parent TPad coordinates.
      TPadPos    fPos = {0.1_normal, 0.1_normal};
      /// Size of the frame in parent TPad coordinates.
      TPadExtent fSize = {0.8_normal, 0.8_normal};
   };
private:
   /// Mapping of user coordinates to normal coordinates.
   std::unique_ptr<Internal::TPadUserCoordBase> fUserCoord;

   /// Palette used to visualize user coordinates.
   std::unique_ptr<TPalette> fPalette;

   /// Offset with respect to parent TPad.
   TPadPos fPos;

   /// Size of the frame, in parent TPad coordinates.
   TPadExtent fSize;

public:
   /// Default constructor, initializing to cover 

   // Constructor taking position and extent. Defaults to leaving 10% of the
   // pad size empty around the frame.
   TFrame(const TPadPos& pos = DrawingOpts::Default().fPos),
          const TPadExtent& size = DrawingOpts::Default().fSize));

   TFrame(std::unique_ptr<Internal::TPadUserCoordBase>&& coords,
          const TPadPos& pos = DrawingOpts::Default().fPos,
          const TPadExtent& size = DrawingOpts::Default().fSize):
   fUserCoord(std::move(coords)), TFrame(pos, size);

   ~TFrame();

   /// Get the current user coordinate system.
   Internal::TPadUserCoordBase &GetUserCoord() const;

   /// Get the current user coordinate system.
   std::unique_ptr<Internal::TPadUserCoordBase>
   SwapUserCoordSystem(std::unique_ptr<Internal::TPadUserCoordBase&& newCoord) {
      std::unique_ptr<Internal::TPadUserCoordBase> ret(std::move(newCoord));
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

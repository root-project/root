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

#include "ROOT/TDrawingAttr.hxx"
#include "ROOT/TDrawingOptsBase.hxx"
#include "ROOT/TPadExtent.hxx"
#include "ROOT/TPadPos.hxx"
#include "ROOT/TPadUserAxis.hxx"
#include "ROOT/TPalette.hxx"

#include <memory>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::TFrame
  Holds a user coordinate system with a palette.
  */

class TFrame {
public:
   class DrawingOpts: public TDrawingOptsBase {
   public:
      /// Position of the frame in parent TPad coordinates.
      TDrawingAttr<TPadPos> fPos{*this, "frame.pos", 0.1_normal, 0.1_normal};
      /// Size of the frame in parent TPad coordinates.
      TDrawingAttr<TPadExtent> fSize{*this, "frame.size", 0.8_normal, 0.8_normal};
   };

private:
   /// Mapping of user coordinates to normal coordinates, one entry per dimension.
   std::vector<std::unique_ptr<Detail::TPadUserAxisBase>> fUserCoord;

   /// Palette used to visualize user coordinates.
   TPalette fPalette;

   /// Offset with respect to parent TPad.
   TPadPos fPos;

   /// Size of the frame, in parent TPad coordinates.
   TPadExtent fSize;

public:
   /// Constructor taking user coordinate system, position and extent.
   explicit TFrame(std::vector<std::unique_ptr<Detail::TPadUserAxisBase>> &&coords, const DrawingOpts &opts);

   // Constructor taking position and extent.
   explicit TFrame(const DrawingOpts &opts)
      : TFrame({}, opts)
   {}

   /// Get the current user coordinate system for a given dimension.
   Detail::TPadUserAxisBase &GetUserAxis(size_t dimension) const { return *fUserCoord[dimension]; }

   /// Set the user coordinate system.
   void SetUserAxis(std::vector<std::unique_ptr<Detail::TPadUserAxisBase>> &&axes) { fUserCoord = std::move(axes); }

   /// Convert user coordinates to normal coordinates.
   std::array<TPadLength::Normal, 2> UserToNormal(const std::array<TPadLength::User, 2> &pos) const
   {
      return {{fUserCoord[0]->ToNormal(pos[0]), fUserCoord[1]->ToNormal(pos[1])}};
   }
};

} // namespace Experimental
} // namespace ROOT

#endif

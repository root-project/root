/// \file ROOT/RFrame.hxx
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

#ifndef ROOT7_RFrame
#define ROOT7_RFrame

#include "ROOT/RAttrBox.hxx"
#include "ROOT/RDrawingOptsBase.hxx"
#include "ROOT/RPadUserAxis.hxx"
#include "ROOT/RPalette.hxx"

#include <memory>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RFrame
  Holds a user coordinate system with a palette.
  */

class RFrame {
public:
   class DrawingOpts: public RDrawingOptsBase, public RAttrBox {
   public:
      DrawingOpts():
         RAttrBox("frame", this)
      {}
   };

private:
   /// Mapping of user coordinates to normal coordinates, one entry per dimension.
   std::vector<std::unique_ptr<RPadUserAxisBase>> fUserCoord;

   /// Palette used to visualize user coordinates.
   RPalette fPalette;

public:
   // Default constructor
   RFrame()
   {
      GrowToDimensions(2);
   }

   /// Constructor taking user coordinate system, position and extent.
   explicit RFrame(std::vector<std::unique_ptr<RPadUserAxisBase>> &&coords, const DrawingOpts &opts);

   // Constructor taking position and extent.
   explicit RFrame(const DrawingOpts &opts)
      : RFrame({}, opts)
   {}

   /// Create `nDimensions` default axes for the user coordinate system.
   void GrowToDimensions(size_t nDimensions);

   /// Get the number of axes.
   size_t GetNDimensions() const { return fUserCoord.size(); }

   /// Get the current user coordinate system for a given dimension.
   RPadUserAxisBase &GetUserAxis(size_t dimension) const { return *fUserCoord[dimension]; }

   /// Set the user coordinate system.
   void SetUserAxis(std::vector<std::unique_ptr<RPadUserAxisBase>> &&axes) { fUserCoord = std::move(axes); }

   /// Convert user coordinates to normal coordinates.
   std::array<RPadLength::Normal, 2> UserToNormal(const std::array<RPadLength::User, 2> &pos) const
   {
      return {{fUserCoord[0]->ToNormal(pos[0]), fUserCoord[1]->ToNormal(pos[1])}};
   }
};

} // namespace Experimental
} // namespace ROOT

#endif

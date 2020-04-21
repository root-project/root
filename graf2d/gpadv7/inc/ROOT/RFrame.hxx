/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RFrame
#define ROOT7_RFrame

#include "ROOT/RDrawable.hxx"

#include "ROOT/RAttrLine.hxx"
#include "ROOT/RAttrFill.hxx"
#include "ROOT/RAttrMargins.hxx"
#include "ROOT/RAttrAxis.hxx"

#include "ROOT/RPadUserAxis.hxx"

#include <memory>

class TRootIOCtor;

namespace ROOT {
namespace Experimental {

/** \class RFrame
\ingroup GpadROOT7
\brief Holds an area where drawing on user coordinate-system can be performed.
\author Axel Naumann <axel@cern.ch>
\author Sergey Linev <s.linev@gsi.de>
\date 2017-09-26
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RFrame : public RDrawable  {

   friend class RPadBase;

   class RFrameAttrs : public RAttrBase {
      friend class RFrame;
      R__ATTR_CLASS(RFrameAttrs, "", AddBool("gridx", false).AddBool("gridy",false));
   };

private:
   RAttrMargins fMargins{this, "margin_"};     ///<!
   RAttrLine fAttrBorder{this, "border_"};     ///<!
   RAttrFill fAttrFill{this, "fill_"};         ///<!
   RAttrAxis fAttrX{this, "x_"};               ///<!
   RAttrAxis fAttrY{this, "y_"};               ///<!
   RAttrAxis fAttrZ{this, "z_"};               ///<!
   RFrameAttrs fAttr{this,""};                 ///<! own frame attributes


   /// Mapping of user coordinates to normal coordinates, one entry per dimension.
   std::vector<std::unique_ptr<RPadUserAxisBase>> fUserCoord;

   RFrame(const RFrame &) = delete;
   RFrame &operator=(const RFrame &) = delete;

   // Default constructor
   RFrame() : RDrawable("frame")
   {
      GrowToDimensions(2);
   }

   /// Constructor taking user coordinate system, position and extent.
   explicit RFrame(std::vector<std::unique_ptr<RPadUserAxisBase>> &&coords);

protected:

   void PopulateMenu(RMenuItems &) override;

   void GetAxisRanges(unsigned ndim, const RAttrAxis &axis, RUserRanges &ranges) const;

public:

   RFrame(TRootIOCtor*) : RFrame() {}

   const RAttrMargins &GetMargins() const { return fMargins; }
   RFrame &SetMargins(const RAttrMargins &margins) { fMargins = margins; return *this; }
   RAttrMargins &Margins() { return fMargins; }

   const RAttrLine &GetAttrBorder() const { return fAttrBorder; }
   RFrame &SetAttrBorder(const RAttrLine &border) { fAttrBorder = border; return *this; }
   RAttrLine &AttrBorder() { return fAttrBorder; }

   const RAttrFill &GetAttrFill() const { return fAttrFill; }
   RFrame &SetAttrFill(const RAttrFill &fill) { fAttrFill = fill; return *this; }
   RAttrFill &AttrFill() { return fAttrFill; }

   const RAttrAxis &GetAttrX() const { return fAttrX; }
   RFrame &SetAttrX(const RAttrAxis &axis) { fAttrX = axis; return *this; }
   RAttrAxis &AttrX() { return fAttrX; }

   const RAttrAxis &GetAttrY() const { return fAttrY; }
   RFrame &SetAttrY(const RAttrAxis &axis) { fAttrY = axis; return *this; }
   RAttrAxis &AttrY() { return fAttrY; }

   const RAttrAxis &GetAttrZ() const { return fAttrZ; }
   RFrame &SetAttrZ(const RAttrAxis &axis) { fAttrZ = axis; return *this; }
   RAttrAxis &AttrZ() { return fAttrZ; }

   RFrame &SetGridX(bool on = true) { fAttr.SetValue("gridx", on); return *this; }
   bool GetGridX() const { return fAttr.GetValue<bool>("gridx"); }

   RFrame &SetGridY(bool on = true) { fAttr.SetValue("gridy", on); return *this; }
   bool GetGridY() const { return fAttr.GetValue<bool>("gridy"); }

   void GetVisibleRanges(RUserRanges &) const override;

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

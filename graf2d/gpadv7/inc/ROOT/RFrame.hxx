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

#include "ROOT/RDrawableRequest.hxx"
#include "ROOT/RAttrLine.hxx"
#include "ROOT/RAttrFill.hxx"
#include "ROOT/RAttrMargins.hxx"
#include "ROOT/RAttrAxis.hxx"

#include "ROOT/RPadUserAxis.hxx"

#include <memory>
#include <map>

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

public:

   class RUserRanges {
      std::vector<double> values;  ///< min/max values for all dimensions
      std::vector<bool> flags;     ///< flag if values available
   public:
      // Default constructor - for I/O
      RUserRanges() = default;

      // Constructor for 1-d ranges
      RUserRanges(double xmin, double xmax)
      {
         AssignMin(0, xmin);
         AssignMax(0, xmax);
      }

      // Constructor for 2-d ranges
      RUserRanges(double xmin, double xmax, double ymin, double ymax)
      {
         AssignMin(0, xmin);
         AssignMax(0, xmax);
         AssignMin(1, ymin);
         AssignMax(1, ymax);
      }

      // Extend number of dimensions which can be stored in the object
      void Extend(unsigned ndim = 3)
      {
         if (ndim*2 > values.size()) {
            values.resize(ndim*2, 0.);
            flags.resize(ndim*2, false);
         }
      }

      bool HasMin(unsigned ndim) const { return (ndim*2 < flags.size()) && flags[ndim*2]; }
      double GetMin(unsigned ndim) const { return (ndim*2 < values.size()) ? values[ndim*2] : 0.; }

      // Assign minimum for specified dimension
      void AssignMin(unsigned ndim, double value, bool force = false)
      {
         if (!HasMin(ndim) || force) {
            Extend(ndim+1);
            values[ndim*2] = value;
            flags[ndim*2] = true;
         }
      }

      bool HasMax(unsigned ndim) const { return (ndim*2+1 < flags.size()) && flags[ndim*2+1]; }
      double GetMax(unsigned ndim) const { return (ndim*2+1 < values.size()) ? values[ndim*2+1] : 0.; }

      // Assign maximum for specified dimension
      void AssignMax(unsigned ndim, double value, bool force = false)
      {
         if (!HasMax(ndim) || force) {
            Extend(ndim+1);
            values[ndim*2+1] = value;
            flags[ndim*2+1] = true;
         }
      }

      // Returns true if any value is specified
      bool IsAny() const
      {
         for (auto fl : flags)
            if (fl) return true;
         return false;
      }
   };

   class RZoomRequest : public RDrawableRequest {
      RUserRanges ranges; // specified ranges
   public:
      RZoomRequest() = default;
      std::unique_ptr<RDrawableReply> Process() override
      {
         auto frame = dynamic_cast<RFrame *>(GetContext().GetDrawable());
         if (frame) frame->SetClientRanges(0, ranges);
         return nullptr;
      }
   };

private:

   class RFrameAttrs : public RAttrBase {
      friend class RFrame;
      R__ATTR_CLASS(RFrameAttrs, "", AddBool("gridx", false).AddBool("gridy",false));
   };

   RAttrMargins fMargins{this, "margin_"};     ///<!
   RAttrLine fAttrBorder{this, "border_"};     ///<!
   RAttrFill fAttrFill{this, "fill_"};         ///<!
   RAttrAxis fAttrX{this, "x_"};               ///<!
   RAttrAxis fAttrY{this, "y_"};               ///<!
   RAttrAxis fAttrZ{this, "z_"};               ///<!
   RFrameAttrs fAttr{this,""};                 ///<! own frame attributes
   std::map<unsigned, RUserRanges> fClientRanges; ///<! individual client ranges

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

   void SetClientRanges(unsigned connid, const RUserRanges &ranges);

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

   void GetClientRanges(unsigned connid, RUserRanges &ranges);

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

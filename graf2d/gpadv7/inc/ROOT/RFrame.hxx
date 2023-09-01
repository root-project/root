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
#include "ROOT/RAttrBorder.hxx"
#include "ROOT/RAttrFill.hxx"
#include "ROOT/RAttrMargins.hxx"
#include "ROOT/RAttrAxis.hxx"
#include "ROOT/RAttrValue.hxx"

#include <memory>
#include <map>

class TRootIOCtor;

namespace ROOT {
namespace Experimental {


/** \class RFrame
\ingroup GpadROOT7
\brief Holds an area where drawing on user coordinate-system can be performed.
\authors Axel Naumann <axel@cern.ch> Sergey Linev <s.linev@gsi.de>
\date 2017-09-26
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RFrame : public RDrawable  {

   friend class RPadBase;

public:

   class RUserRanges {
      std::vector<double> values;  ///< min/max values for all dimensions
      std::vector<bool> flags;     ///< flag if values available

      void UpdateDim(unsigned ndim, const RUserRanges &src)
      {
         if (src.IsUnzoom(ndim)) {
            ClearMinMax(ndim);
         } else {
            if (src.HasMin(ndim))
               AssignMin(ndim, src.GetMin(ndim));
            if (src.HasMax(ndim))
               AssignMax(ndim, src.GetMax(ndim));
         }
      }

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
      void AssignMin(unsigned ndim, double value)
      {
          Extend(ndim+1);
          values[ndim*2] = value;
          flags[ndim*2] = true;
      }

      bool HasMax(unsigned ndim) const { return (ndim*2+1 < flags.size()) && flags[ndim*2+1]; }
      double GetMax(unsigned ndim) const { return (ndim*2+1 < values.size()) ? values[ndim*2+1] : 0.; }

      // Assign maximum for specified dimension
      void AssignMax(unsigned ndim, double value)
      {
          Extend(ndim+1);
          values[ndim*2+1] = value;
          flags[ndim*2+1] = true;
      }

      void ClearMinMax(unsigned ndim)
      {
         if (ndim*2+1 < flags.size())
            flags[ndim*2] = flags[ndim*2+1] = false;

         if (ndim*2+1 < values.size())
            values[ndim*2] = values[ndim*2+1] = 0.;
      }

      /** Returns true if axis configured as unzoomed, can be specified from client */
      bool IsUnzoom(unsigned ndim) const
      {
         return (ndim*2+1 < flags.size()) && (ndim*2+1 < values.size()) &&
               !flags[ndim*2] && !flags[ndim*2+1] &&
               (values[ndim*2] < -0.5) && (values[ndim*2+1] < -0.5);
      }

      // Returns true if any value is specified
      bool IsAny() const
      {
         for (auto fl : flags)
            if (fl) return true;
         return false;
      }

      void Update(const RUserRanges &src)
      {
         UpdateDim(0, src);
         UpdateDim(1, src);
         UpdateDim(2, src);
         UpdateDim(3, src);
         UpdateDim(4, src);
      }
   };

private:
   std::map<unsigned, RUserRanges> fClientRanges; ///<! individual client ranges

   RFrame(const RFrame &) = delete;
   RFrame &operator=(const RFrame &) = delete;

   // Default constructor
   RFrame() : RDrawable("frame")
   {
   }

   void SetClientRanges(unsigned connid, const RUserRanges &ranges, bool ismainconn);

protected:

   void PopulateMenu(RMenuItems &) override;

   void GetAxisRanges(unsigned ndim, const RAttrAxis &axis, RUserRanges &ranges) const;
   void AssignZoomRange(unsigned ndim, RAttrAxis &axis, const RUserRanges &ranges);

public:

   class RZoomRequest : public RDrawableRequest {
      RUserRanges ranges; // specified ranges
   public:
      RZoomRequest() = default;
      std::unique_ptr<RDrawableReply> Process() override
      {
         auto frame = dynamic_cast<RFrame *>(GetContext().GetDrawable());
         if (frame) frame->SetClientRanges(GetContext().GetConnId(), ranges, GetContext().IsMainConn());
         return nullptr;
      }
   };

   RAttrMargins margins{this, "margins"};              ///<! frame margins relative to pad
   RAttrBorder border{this, "border"};                 ///<! frame border attributes
   RAttrFill fill{this, "fill"};                       ///<! frame fill attributes
   RAttrAxis x{this, "x"};                             ///<! drawing attributes for X axis
   RAttrAxis y{this, "y"};                             ///<! drawing attributes for Y axis
   RAttrAxis z{this, "z"};                             ///<! drawing attributes for Z axis
   RAttrAxis x2{this, "x2"};                           ///<! drawing attributes for X2 axis
   RAttrAxis y2{this, "y2"};                           ///<! drawing attributes for Y2 axis
   RAttrValue<bool> drawAxes{this, "drawAxes", false}; ///<! draw axes by frame
   RAttrValue<bool> gridX{this, "gridX", false};       ///<! show grid for X axis
   RAttrValue<bool> gridY{this, "gridY", false};       ///<! show grid for Y axis
   RAttrValue<bool> swapX{this, "swapX", false};       ///<! swap position of X axis
   RAttrValue<bool> swapY{this, "swapY", false};       ///<! swap position of Y axis
   RAttrValue<int> ticksX{this, "ticksX", 1};          ///<! X ticks drawing: 0 - off, 1 - normal, 2 - both sides, 3 - both sides with labels
   RAttrValue<int> ticksY{this, "ticksY", 1};          ///<! Y ticks drawing: 0 - off, 1 - normal, 2 - both sides, 3 - both sides with labels

   RFrame(TRootIOCtor*) : RFrame() {}

   void GetClientRanges(unsigned connid, RUserRanges &ranges);
};


} // namespace Experimental
} // namespace ROOT

#endif

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "ROOT/RFrame.hxx"

#include "ROOT/RLogger.hxx"
#include "ROOT/RMenuItems.hxx"

#include "TROOT.h"

#include <cassert>
#include <sstream>

using namespace ROOT::Experimental;

////////////////////////////////////////////////////////////////////////////
/// Internal - extract range for specified axis

void RFrame::GetAxisRanges(unsigned ndim, const RAttrAxis &axis, RUserRanges &ranges) const
{
   if (axis.zoommin.Has())
      ranges.AssignMin(ndim, axis.zoommin);

   if (axis.zoommax.Has())
      ranges.AssignMax(ndim, axis.zoommax);
}

////////////////////////////////////////////////////////////////////////////
/// Internal - assign client zoomed range to specified axis

void RFrame::AssignZoomRange(unsigned ndim, RAttrAxis &axis, const RUserRanges &ranges)
{
   if (ranges.IsUnzoom(ndim)) {
      axis.zoommin.Clear();
      axis.zoommax.Clear();
   } else {
      if (ranges.HasMin(ndim))
         axis.zoommin = ranges.GetMin(ndim);
      if (ranges.HasMax(ndim))
         axis.zoommax = ranges.GetMax(ndim);
   }
}

////////////////////////////////////////////////////////////////////////////
/// Provide context menu items

void RFrame::PopulateMenu(RMenuItems & /* items */)
{
   // do not use online context menu for frame - make it fully client side
/*   auto is_x = items.GetSpecifier() == "x";
   auto is_y = items.GetSpecifier() == "y";

   if (is_x || is_y) {
      RAttrAxis &attr = is_x ? x : y;
      std::string name = is_x ? "x" : "y";
      auto cl = TClass::GetClass<RAttrAxis>();
      auto log = attr.GetLog();
      items.AddChkMenuItem("Linear scale", "Set linear scale", !log, name + ".log = 0", cl);
      items.AddChkMenuItem("Log scale", "Logarithmic scale", !log, name + ".log = 10", cl);
   }
*/
}

////////////////////////////////////////////////////////////////////////////
/// Remember client range, can be used for drawing or stats box calculations

void RFrame::SetClientRanges(unsigned connid, const RUserRanges &ranges, bool ismainconn)
{
   if (ismainconn) {
      AssignZoomRange(0, x, ranges);
      AssignZoomRange(1, y, ranges);
      AssignZoomRange(2, z, ranges);
      AssignZoomRange(3, x2, ranges);
      AssignZoomRange(4, y2, ranges);
   }

   if (fClientRanges.find(connid) == fClientRanges.end()) {
      RUserRanges ranges0;
      GetClientRanges(connid, ranges0);
      fClientRanges[connid] = ranges0;
   }

   fClientRanges[connid].Update(ranges);
}

////////////////////////////////////////////////////////////////////////////
/// Return ranges configured for the client

void RFrame::GetClientRanges(unsigned connid, RUserRanges &ranges)
{
   auto iter = fClientRanges.find(connid);

   if (iter != fClientRanges.end()) {
      ranges = iter->second;
   } else {
      GetAxisRanges(0, x, ranges);
      GetAxisRanges(1, y, ranges);
      GetAxisRanges(2, z, ranges);
   }
}


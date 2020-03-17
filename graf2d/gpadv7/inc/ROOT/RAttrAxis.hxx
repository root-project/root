/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrAxis
#define ROOT7_RAttrAxis

#include <ROOT/RAttrBase.hxx>
#include <ROOT/RAttrLine.hxx>
#include <ROOT/RAttrText.hxx>

namespace ROOT {
namespace Experimental {

/** \class RAttrAxis
\ingroup GpadROOT7
\author Sergey Linev <s.linev@gsi.de>
\date 2020-02-20
\brief All kind of drawing a axis: line, text, ticks, min/max, log, invert, ...
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrAxis : public RAttrBase {

   RAttrLine fAttrLine{this, "line_"};   ///<!  axis line attributes
   RAttrText fAttrText{this, "text_"};   ///<!  axis text attributes

   R__ATTR_CLASS(RAttrAxis, "axis_", AddDefaults(fAttrLine).AddDefaults(fAttrText)
                                   .AddDouble("min", 0.).AddDouble("max", 1.)
                                   .AddDouble("zoommin", 0.).AddDouble("zoommax", 0.)
                                   .AddBool("log", false).AddBool("invert", false));

   const RAttrLine &GetAttrLine() const { return fAttrLine; }
   RAttrAxis &SetAttrLine(const RAttrLine &line) { fAttrLine = line; return *this; }
   RAttrLine &AttrLine() { return fAttrLine; }

   const RAttrText &GetAttrText() const { return fAttrText; }
   RAttrAxis &SetAttrText(const RAttrText &text) { fAttrText = text; return *this; }
   RAttrText &AttrText() { return fAttrText; }

   // min range, graphics will not show value less then this
   RAttrAxis &SetMin(double min) { SetValue("min", min); return *this; }
   RAttrAxis &SetMax(double max) { SetValue("max", max); return *this; }
   double GetMin() const { return GetValue<double>("min"); }
   double GetMax() const { return GetValue<double>("max"); }

   RAttrAxis &SetMinMax(double min, double max) { SetMin(min); SetMax(max); return *this; }
   void ClearMinMax() { ClearValue("min"); ClearValue("max"); }

   RAttrAxis &SetZoomMin(double min) { SetValue("zoommin", min); return *this; }
   RAttrAxis &SetZoomMax(double max) { SetValue("zoommax", max); return *this; }
   double GetZoomMin() const { return GetValue<double>("zoommin"); }
   double GetZoomMax() const { return GetValue<double>("zoommax"); }

   RAttrAxis &SetZoomMinMax(double min, double max) { SetZoomMin(min); SetZoomMax(max); return *this; }
   void ClearZoomMinMax() { ClearValue("zoommin"); ClearValue("zoommax"); }

   RAttrAxis &SetLog(bool on = true) { SetValue("log", on); return *this; }
   bool GetLog() const { return GetValue<bool>("log"); }

   RAttrAxis &SetInvert(bool on = true) { SetValue("invert", on); return *this; }
   bool GetInvert() const { return GetValue<bool>("invert"); }

};

} // namespace Experimental
} // namespace ROOT

#endif

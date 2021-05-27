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
#include <ROOT/RAttrValue.hxx>
#include <ROOT/RPadLength.hxx>
#include <cmath>

namespace ROOT {
namespace Experimental {

/** \class RAttrAxis
\ingroup GpadROOT7
\author Sergey Linev <s.linev@gsi.de>
\date 2020-02-20
\brief All supported axes attributes for: line, ticks, labels, title, min/max, log, reverse, ...
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrAxis : public RAttrBase {

   RAttrValue<double> fMin{"min", this, 0.};                             ///<! axis min
   RAttrValue<double> fMax{"max", this, 0.};                             ///<! axis max
   RAttrValue<double> fZoomMin{"zoommin", this, 0.};                     ///<! axis zoom min
   RAttrValue<double> fZoomMax{"zoommax", this, 0.};                     ///<! axis zoom max
   RAttrValue<double> fLog{"log", this, 0};                              ///<! log scale, <1 off, 1 - base10, 2 - base 2, 2.71 - exp, 3, 4, ...
   RAttrValue<double> fSymlog{"symlog", this, 0};                        ///<! symlog scale constant, 0 - off
   RAttrValue<bool> fReverse{"reverse", this, false};                    ///<! reverse scale
   RAttrValue<bool> fTimeDisplay{"time", this, false};                   ///<! time display
   RAttrValue<double> fTimeOffset{"time_offset", this, 0};               ///<! time offset to display
   RAttrValue<std::string> fTimeFormat{"time_format", this, ""};         ///<! time format
   RAttrValue<RColor> fLineColor{"line_color", this, RColor::kBlack};    ///<! line color
   RAttrValue<double> fLineWidth{"line_width", this, 1.};                ///<! line width
   RAttrValue<int> fLineStyle{"line_style", this, 1};                    ///<! line style
   RAttrValue<std::string> fEndingStyle{"ending_style", this, ""};       ///<! axis ending style - none, arrow, circle
   RAttrValue<RPadLength> fEndingSize{"ending_size", this, 0.02_normal}; ///<! axis ending size
   RAttrValue<std::string> fTicksSide{"ticks_side", this, "normal"};     ///<! ticks position - normal, invert, both
   RAttrValue<RPadLength> fTicksSize{"ticks_size", this, 0.02_normal};   ///<! ticks size
   RAttrValue<RColor> fTicksColor{"ticks_color", this, RColor::kBlack};  ///<! ticks color
   RAttrValue<int> fTicksWidth{"ticks_width", this, 1};                  ///<! ticks width
   RAttrText fLabelsAttr{"labels", this};                                ///<! text attributes for labels
   RAttrValue<RPadLength> fLabelsOffset{"labels_offset", this, {}};      ///<! axis labels offset - relative
   RAttrValue<bool> fLabelsCenter{"labels_center", this, false};         ///<! center labels
   RAttrText fTitleAttr{"title", this};                                  ///<! axis title text attributes
   RAttrValue<std::string> fTitle{"title", this, ""};                    ///<! axis title
   RAttrValue<std::string> fTitlePos{"title_position", this, "right"};   ///<! axis title position - left, right, center
   RAttrValue<RPadLength> fTitleOffset{"title_offset", this, {}};        ///<! axis title offset - relative

   R__ATTR_CLASS(RAttrAxis, "axis");

   RAttrAxis &SetMin(double min) { fMin = min; return *this; }
   RAttrAxis &SetMax(double max) { fMax = max; return *this; }
   double GetMin() const { return fMin; }
   double GetMax() const { return fMax; }
   bool HasMin() const { return fMin.Has(); }
   bool HasMax() const { return fMax.Has(); }

   RAttrAxis &SetMinMax(double min, double max) { SetMin(min); SetMax(max); return *this; }
   void ClearMinMax() { fMin.Clear(); fMax.Clear(); }

   RAttrAxis &SetZoomMin(double min) { fZoomMin = min; return *this; }
   RAttrAxis &SetZoomMax(double max) { fZoomMax = max; return *this; }
   double GetZoomMin() const { return fZoomMin; }
   double GetZoomMax() const { return fZoomMax; }
   bool HasZoomMin() const { return fZoomMin.Has(); }
   bool HasZoomMax() const { return fZoomMax.Has(); }

   RAttrAxis &SetZoom(double min, double max) { SetZoomMin(min); SetZoomMax(max); return *this; }
   void ClearZoom() { fZoomMin.Clear(); fZoomMax.Clear(); }

   RAttrAxis &SetLog(double base = 10) { fLog = (base < 1) ? 0 : base; return *this; }
   double GetLog() const { return fLog; }
   bool IsLogScale() const { return GetLog() > 0.999999; }
   bool IsLog10() const { auto l = GetLog(); return (std::fabs(l-1.) < 1e-6) || (std::fabs(l-10.) < 1e-6); }
   bool IsLog2() const { return std::fabs(GetLog() - 2.) < 1e-6; }
   bool IsLn() const { return std::fabs(GetLog() - 2.71828) < 0.1; }

   RAttrAxis &SetSymlog(double const_value = 0.01)
   {
      if (const_value <= 0.)
         fSymlog.Clear();
      else
         fSymlog = const_value;
      return *this;
   }
   double GetSymlog() const { return fSymlog; }
   bool HasSymlog() const { return fSymlog.Has(); }
   void ClearSymlog() { fSymlog.Clear(); }

   RAttrAxis &SetReverse(bool on = true) { fReverse = on; return *this; }
   bool GetReverse() const { return fReverse; }

   RAttrAxis &SetTimeDisplay(const std::string &fmt = "", double offset = -1)
   {
      fTimeDisplay = true;
      if (!fmt.empty()) fTimeFormat = fmt;
      if (offset >= 0) fTimeOffset = offset;
      return *this;
   }
   bool GetTimeDisplay() const { return fTimeDisplay; }
   void ClearTimeDisplay()
   {
      fTimeDisplay.Clear();
      fTimeOffset.Clear();
      fTimeFormat.Clear();
   }

   ///The width of the line.
   RAttrAxis &SetLineWidth(double width) { fLineWidth = width; return *this; }
   double GetLineWidth() const { return fLineWidth; }

   ///The style of the line.
   RAttrAxis &SetLineStyle(int style) { fLineStyle = style; return *this; }
   int GetLineStyle() const { return fLineStyle; }

   ///The color of the line.
   RAttrAxis &SetLineColor(const RColor &color) { fLineColor = color; return *this; }
   RColor GetLineColor() const { return fLineColor; }

   RAttrAxis &SetEndingSize(const RPadLength &sz) { fEndingSize = sz; return *this; }
   RPadLength GetEndingSize() const { return fEndingSize; }

   RAttrAxis &SetEndingStyle(const std::string &st) { fEndingStyle = st; return *this; }
   RAttrAxis &SetEndingArrow() { return SetEndingStyle("arrow"); }
   RAttrAxis &SetEndingCircle() { return SetEndingStyle("cicrle"); }
   std::string GetEndingStyle() const { return fEndingStyle; }

   RAttrAxis &SetTicksSize(const RPadLength &sz) { fTicksSize = sz; return *this; }
   RPadLength GetTicksSize() const { return fTicksSize; }

   RAttrAxis &SetTicksSide(const std::string &side) { fTicksSide = side; return *this; }
   RAttrAxis &SetTicksNormal() { return SetTicksSide("normal"); }
   RAttrAxis &SetTicksInvert() { return SetTicksSide("invert"); }
   RAttrAxis &SetTicksBoth() { return SetTicksSide("both"); }
   std::string GetTicksSide() const { return fTicksSide; }

   RAttrAxis &SetTicksColor(const RColor &color) { fTicksColor = color; return *this; }
   RColor GetTicksColor() const { return fTicksColor; }

   RAttrAxis &SetTicksWidth(int width) { fTicksWidth = width; return *this; }
   int GetTicksWidth() const { return fTicksWidth; }

   RAttrAxis &SetLabelsOffset(const RPadLength &len) { fLabelsOffset = len; return *this; }
   RPadLength GetLabelsOffset() const { return fLabelsOffset; }

   const RAttrText &GetLabelsAttr() const { return fLabelsAttr; }
   RAttrAxis &SetLabelsAttr(const RAttrText &attr) { fLabelsAttr = attr; return *this; }
   RAttrText &LabelsAttr() { return fLabelsAttr; }

   RAttrAxis &SetLabelsCenter(bool on = true) { fLabelsCenter = on; return *this; }
   bool GetLabelsCenter() const { return fLabelsCenter; }

   const RAttrText &GetTitleAttr() const { return fTitleAttr; }
   RAttrAxis &SetTitleAttr(const RAttrText &attr) { fTitleAttr = attr; return *this; }
   RAttrText &TitleAttr() { return fTitleAttr; }

   RAttrAxis &SetTitle(const std::string &title) { fTitle = title; return *this; }
   std::string GetTitle() const { return fTitle; }

   RAttrAxis &SetTitlePos(const std::string &pos) { fTitlePos = pos; return *this; }
   RAttrAxis &SetTitleLeft() { return SetTitlePos("left"); }
   RAttrAxis &SetTitleCenter() { return SetTitlePos("center"); }
   RAttrAxis &SetTitleRight() { return SetTitlePos("right"); }
   std::string GetTitlePos() const { return fTitlePos; }

   RAttrAxis &SetTitleOffset(const RPadLength &len) { fTitleOffset = len; return *this; }
   RPadLength GetTitleOffset() const { return fTitleOffset; }

   const RAttrAxis &AttrAxis() const { return *this; }
   RAttrAxis &AttrAxis() { return *this; }

};

} // namespace Experimental
} // namespace ROOT

#endif

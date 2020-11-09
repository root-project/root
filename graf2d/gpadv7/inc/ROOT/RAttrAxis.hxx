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

   RAttrLine            fAttrLine{this, "line_"};        ///<! line attributes
   RAttrText            fAttrText{this, "text_"};        ///<! text attributes
   RAttrValue<double>   fMin{this, "min", 0.};           ///<! axis min
   RAttrValue<double>   fMax{this, "max", 1.};           ///<! axis max
   RAttrValue<double>   fZoomMin{this, "zoommin", 0.};   ///<! axis zoom min
   RAttrValue<double>   fZoomMax{this, "zoommax", 0.};   ///<! axis zoom max
   RAttrValue<int>      fLog{this, "log", 0};            ///<! log scale
   RAttrValue<bool>     fReverse{this, "reverse", false};  ///<! reverse scale, chagge min/max values
   RAttrValue<std::string> fTitle{this, "title", ""};                     ///<! axis title
   RAttrValue<std::string> fTitlePos{this, "title_position", "right"};    ///<! axis title position - left, right, center
   RAttrValue<RPadLength>  fTitleOffset{this, "title_offset", 0.2_normal};  ///<! axis title offset

   RAttrValue<std::string> fTicksSide{this, "ticks_side", "normal"};     ///<! ticks position - normal, invert, both
   RAttrValue<RPadLength>  fTicksSize{this, "ticks_size", 0.03_normal};  ///<! ticks size

   RAttrValue<std::string> fEndingStyle{this, "ending_style", ""};         ///<! axis ending style - none, arrow, circle
   RAttrValue<RPadLength>  fEndingSize{this, "ending_size", 0.02_normal};  ///<! axis ending size


   R__ATTR_CLASS(RAttrAxis, "axis_", AddDefaults(fAttrLine).AddDefaults(fAttrText)
                                    .AddDefaults(fMin).AddDefaults(fMax)
                                    .AddDefaults(fZoomMin).AddDefaults(fZoomMax)
                                    .AddDefaults(fLog).AddDefaults(fReverse)
                                    .AddDefaults(fTitle).AddDefaults(fTitlePos).AddDefaults(fTitleOffset)
                                    .AddDefaults(fTicksSide).AddDefaults(fTicksSize)
                                    .AddDefaults(fEndingStyle).AddDefaults(fEndingSize));

   const RAttrLine &GetAttrLine() const { return fAttrLine; }
   RAttrAxis &SetAttrLine(const RAttrLine &line) { fAttrLine = line; return *this; }
   RAttrLine &AttrLine() { return fAttrLine; }

   const RAttrText &GetAttrText() const { return fAttrText; }
   RAttrAxis &SetAttrText(const RAttrText &text) { fAttrText = text; return *this; }
   RAttrText &AttrText() { return fAttrText; }

   // min range, graphics will not show value less then this
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

   RAttrAxis &SetZoomMinMax(double min, double max) { SetZoomMin(min); SetZoomMax(max); return *this; }
   void ClearZoomMinMax() { fZoomMin.Clear(); fZoomMax.Clear(); }

   RAttrAxis &SetLog(int base = 10) { fLog = base; return *this; }
   int GetLog() const { return fLog; }

   RAttrAxis &SetReverse(bool on = true) { fReverse = on; return *this; }
   bool GetReverse() const { return fReverse; }

   RAttrAxis &SetTitle(const std::string &title) { fTitle = title; return *this; }
   std::string GetTitle() const { return fTitle; }

   RAttrAxis &SetTitlePos(const std::string &pos) { fTitlePos = pos; return *this; }
   RAttrAxis &SetTitleLeft() { return SetTitlePos("left"); }
   RAttrAxis &SetTitleCenter() { return SetTitlePos("center"); }
   RAttrAxis &SetTitleRight() { return SetTitlePos("right"); }
   std::string GetTitlePos() const { return fTitlePos; }

   RAttrAxis &SetTitleOffset(const RPadLength &len) { fTitleOffset = len; return *this; }
   RPadLength GetTitleOffset() const { return fTitleOffset; }

   RAttrAxis &SetTicksSize(const RPadLength &sz) { fTicksSize = sz; return *this; }
   RPadLength GetTicksSize() const { return fTicksSize; }

   RAttrAxis &SetTicksSide(const std::string &side) { fTicksSide = side; return *this; }
   RAttrAxis &SetTicksNormal() { return SetTicksSide("normal"); }
   RAttrAxis &SetTicksInvert() { return SetTicksSide("invert"); }
   RAttrAxis &SetTicksBoth() { return SetTicksSide("both"); }
   std::string GetTicksSide() const { return fTicksSide; }

   RAttrAxis &SetEndingSize(const RPadLength &sz) { fEndingSize = sz; return *this; }
   RPadLength GetEndingSize() const { return fEndingSize; }

   RAttrAxis &SetEndingStyle(const std::string &st) { fEndingStyle = st; return *this; }
   RAttrAxis &SetEndingArrow() { return SetEndingStyle("arrow"); }
   RAttrAxis &SetEndingCircle() { return SetEndingStyle("cicrle"); }
   std::string GetEndingStyle() const { return fEndingStyle; }

};

} // namespace Experimental
} // namespace ROOT

#endif

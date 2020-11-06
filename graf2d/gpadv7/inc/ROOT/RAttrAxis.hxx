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
   RAttrValue<bool>     fInvert{this, "invert", false};  ///<! invert scale
   RAttrValue<std::string> fTitle{this, "title", ""};                     ///<! axis title
   RAttrValue<std::string> fTitlePos{this, "title_position", "right"};    ///<! axis title position
   RAttrValue<RPadLength>  fTitleOffset{this, "title_offset", 0.2_normal};  ///<! axis title offset

   R__ATTR_CLASS(RAttrAxis, "axis_", AddDefaults(fAttrLine).AddDefaults(fAttrText)
                                    .AddDefaults(fMin).AddDefaults(fMax)
                                    .AddDefaults(fZoomMin).AddDefaults(fZoomMax)
                                    .AddDefaults(fLog).AddDefaults(fInvert)
                                    .AddDefaults(fTitle).AddDefaults(fTitlePos).AddDefaults(fTitleOffset));

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

   RAttrAxis &SetInvert(bool on = true) { fInvert = on; return *this; }
   bool GetInvert() const { return fInvert; }

   RAttrAxis &SetTitle(const std::string &title) { fTitle = title; return *this; }
   std::string GetTitle() const { return fTitle; }

   RAttrAxis &SetTitlePos(const std::string &pos) { fTitlePos = pos; return *this; }
   RAttrAxis &SetTitleLeft() { return SetTitlePos("left"); }
   RAttrAxis &SetTitleCenter() { return SetTitlePos("center"); }
   RAttrAxis &SetTitleRight() { return SetTitlePos("right"); }
   std::string GetTitlePos() const { return fTitlePos; }

   RAttrAxis &SetTitleOffset(const RPadLength &len) { fTitleOffset = len; return *this; }
   RPadLength GetTitleOffset() const { return fTitleOffset; }

};

} // namespace Experimental
} // namespace ROOT

#endif

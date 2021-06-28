/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrAxis
#define ROOT7_RAttrAxis

#include <ROOT/RAttrAggregation.hxx>
#include <ROOT/RAttrLine.hxx>
#include <ROOT/RAttrText.hxx>
#include <ROOT/RAttrValue.hxx>
#include <ROOT/RPadLength.hxx>
#include <cmath>

namespace ROOT {
namespace Experimental {

/** \class RAttrAxisLabels
\ingroup GpadROOT7
\author Sergey Linev <s.linev@gsi.de>
\date 2021-06-28
\brief Axis labels drawing attributes
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrAxisLabels : public RAttrText {
   R__ATTR_CLASS_DERIVED(RAttrAxisLabels, "labels", RAttrText)

   RAttrValue<RPadLength> offset{this, "offset", {}};      ///<! labels offset - relative to "default" position
   RAttrValue<bool> center{this, "center", false};         ///<! center labels
   RAttrValue<bool> hide{this, "hide", false};             ///<! hide labels
};

/** \class RAttrAxisTitle
\ingroup GpadROOT7
\author Sergey Linev <s.linev@gsi.de>
\date 2021-06-28
\brief Axis title and its drawing attributes
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrAxisTitle : public RAttrText {
   R__ATTR_CLASS_DERIVED(RAttrAxisTitle, "title", RAttrText)

   RAttrValue<std::string> value{this, "value", ""};            ///<! axis title value
   RAttrValue<std::string> position{this, "position", "right"};   ///<! axis title position - left, right, center
   RAttrValue<RPadLength> offset{this, "offset", {}};        ///<! axis title offset - relative to "default" position

   RAttrAxisTitle& operator=(const std::string &_title) { value = _title; return *this; }

   void SetLeft() { position = "left"; }
   void SetCenter() { position = "center"; }
   void SetRight() { position = "right"; }
};

/** \class RAttrAxisTicks
\ingroup GpadROOT7
\author Sergey Linev <s.linev@gsi.de>
\date 2021-06-28
\brief Axis ticks attributes
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrAxisTicks : public RAttrAggregation {
   R__ATTR_CLASS(RAttrAxisTicks, "ticks");

   RAttrValue<std::string> side{this, "side", "normal"};     ///<! ticks position - normal, invert, both
   RAttrValue<RPadLength> size{this, "size", 0.02_normal};   ///<! ticks size
   RAttrValue<RColor> color{this, "color", RColor::kBlack};  ///<! ticks color
   RAttrValue<int> width{this, "width", 1};                  ///<! ticks width

   void SetNormal() { side = "normal"; }
   void SetInvert() { side = "invert"; }
   void SetBoth() { side = "both"; }

};

/** \class RAttrAxis
\ingroup GpadROOT7
\author Sergey Linev <s.linev@gsi.de>
\date 2020-02-20
\brief All supported axes attributes for: line, ticks, labels, title, min/max, log, reverse, ...
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAttrAxis : public RAttrAggregation {
   R__ATTR_CLASS(RAttrAxis, "axis");

   RAttrLine line{this, "line"};                                    ///<! line attributes
   RAttrLineEnding ending{this, "ending"};                          ///<! ending attributes
   RAttrAxisLabels labels{this, "labels"};                          ///<! labels attributes
   RAttrAxisTitle title{this, "title"};                             ///<! title attributes
   RAttrAxisTicks ticks{this, "ticks"};                             ///<! ticks attributes
   RAttrValue<double> min{this, "min", 0.};                         ///<! axis min
   RAttrValue<double> max{this, "max", 0.};                         ///<! axis max
   RAttrValue<double> zoommin{this, "zoommin", 0.};                 ///<! axis zoom min
   RAttrValue<double> zoommax{this, "zoommax", 0.};                 ///<! axis zoom max
   RAttrValue<double> log{this, "log", 0};                          ///<! log scale, <1 off, 1 - base10, 2 - base 2, 2.71 - exp, 3, 4, ...
   RAttrValue<double> symlog{this, "symlog", 0};                    ///<! symlog scale constant, 0 - off
   RAttrValue<bool> reverse{this, "reverse", false};                ///<! reverse scale
   RAttrValue<bool> time{this, "time", false};                      ///<! time scale
   RAttrValue<double> time_offset{this, "time_offset", 0};          ///<! time offset to display
   RAttrValue<std::string> time_format{this, "time_format", ""};    ///<! time format

   RAttrAxis &SetMinMax(double _min, double _max) { min = _min; max = _max; return *this; }
   RAttrAxis &ClearMinMax() { min.Clear(); max.Clear(); return *this; }

   RAttrAxis &SetZoom(double _zoommin, double _zoommax) { zoommin = _zoommin; zoommax = _zoommax; return *this; }
   RAttrAxis &ClearZoom() { zoommin.Clear(); zoommax.Clear(); return *this; }

   bool IsLogScale() const { return this->log > 0.999999; }
   bool IsLog10() const { auto l = this->log; return (std::fabs(l-1.) < 1e-6) || (std::fabs(l-10.) < 1e-6); }
   bool IsLog2() const { return std::fabs(this->log - 2.) < 1e-6; }
   bool IsLn() const { return std::fabs(this->log - 2.71828) < 0.1; }

   RAttrAxis &SetTimeDisplay(const std::string &fmt = "", double offset = -1)
   {
      this->time = true;
      if (!fmt.empty()) time_format = fmt;
      if (offset >= 0) time_offset = offset;
      return *this;
   }
   void ClearTimeDisplay()
   {
      this->time.Clear();
      time_offset.Clear();
      time_format.Clear();
   }

   RAttrAxis &SetTitle(const std::string &_title) { title.value = _title; return *this; }
   std::string GetTitle() const { return title.value; }
};

} // namespace Experimental
} // namespace ROOT

#endif

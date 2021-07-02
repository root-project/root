/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_ROnFrameDrawable
#define ROOT7_ROnFrameDrawable

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrValue.hxx>

namespace ROOT {
namespace Experimental {

/** \class ROnFrameDrawable
\ingroup GpadROOT7
\brief Base class for drawable which can be drawn on frame or on pad. Introduces "onFrame" and "clipping" attributes. If onFrame = true, one can enable clipping of such drawables.
Dedicated for classes like RLine, RText, RBox and similar
\author Sergey Linev <s.linev@gsi.de>
\date 2021-06-29
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class ROnFrameDrawable : public RDrawable {
protected:
   ROnFrameDrawable(const ROnFrameDrawable &) = delete;
   ROnFrameDrawable &operator=(const ROnFrameDrawable &) = delete;

   explicit ROnFrameDrawable(const std::string &type) : RDrawable(type) {}

public:
   virtual ~ROnFrameDrawable() = default;

   RAttrValue<bool> onFrame{this, "onFrame", false};  ///<! is drawn on the frame or not
   RAttrValue<bool> clipping{this, "clipping", false}; ///<! is clipping on when drawn on the frame
};

} // namespace Experimental
} // namespace ROOT

#endif

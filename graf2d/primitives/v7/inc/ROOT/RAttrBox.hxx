/// \file ROOT/RAttrBox.hxx
/// \ingroup Graf ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2018-10-17
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAttrBox
#define ROOT7_RAttrBox

#include <ROOT/RAttrLine.hxx>
#include <ROOT/RColor.hxx>
#include <ROOT/RDrawingAttr.hxx>
#include <ROOT/RPadExtent.hxx>
#include <ROOT/RPadPos.hxx>

namespace ROOT {
namespace Experimental {

/** class ROOT::Experimental::RAttrBox
 Drawing attributes for a box: rectangular lines with size and position.
 */
class RAttrBox: public RDrawingAttrBase {
public:
   RAttrBox() = default;

   RAttrBox(const char* name, RDrawingAttrHolderBase* holder, RDrawingAttrBase *parent = nullptr):
      RDrawingAttrBase(name, holder, parent, {"pos", "size"})
   {}

   RAttrLine border{"border", GetHolder(), this};

   /// The position of the box.
   void SetPos(const RPadPos& pos) { Set(0, PosToString(pos)); }
   std::pair<RPadPos, bool> GetPos() const
   {
      auto ret = Get(0);
      return {PosFromString("box position", ret.first), ret.second};
   }

   ///The size of the box.
   void SetSize(const RPadExtent& size) { Set(1, ExtentToString(size)); }
   std::pair<RPadExtent, bool> GetSize() const
   {
      auto ret = Get(1);
      return {ExtentFromString("box size", ret.first), ret.second};
   }
};

} // namespace Experimental
} // namespace ROOT

#endif

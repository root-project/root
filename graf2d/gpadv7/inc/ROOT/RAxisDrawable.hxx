/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RAxisDrawable
#define ROOT7_RAxisDrawable

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrAxis.hxx>
#include <ROOT/RPadPos.hxx>
#include <ROOT/RPadLength.hxx>
#include <ROOT/RLogger.hxx>

#include <vector>
#include <string>

namespace ROOT {
namespace Experimental {

/** \class RAxisDrawable
\ingroup GrafROOT7
\brief Axis drawing
\author Sergey Linev <S.Linev@gsi.de>
\date 2020-11-03
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RAxisDrawable : public RDrawable {

   RPadPos fPos;                          ///< axis start point
   bool fVertical{false};                 ///< is vertical axis
   RPadLength fLength;                    ///< axis length
   std::vector<std::string> fLabels;      ///< axis labels

public:

   RAttrAxis axis{this, "axis"};     ///<! axis attributes

   RAxisDrawable() : RDrawable("axis") {}

   RAxisDrawable(const RPadPos &pos, bool vertical, const RPadLength &len) : RAxisDrawable()
   {
      SetPos(pos);
      SetVertical(vertical);
      SetLength(len);
   }

   RAxisDrawable &SetPos(const RPadPos &pos) { fPos = pos; return *this; }
   RAxisDrawable &SetVertical(bool vertical = true) { fVertical = vertical; return *this; }
   RAxisDrawable &SetLength(const RPadLength &len) { fLength = len; return *this; }
   RAxisDrawable &SetLabels(const std::vector<std::string> &lbls) { fLabels = lbls; return *this; }

   bool IsVertical() const { return fVertical; }
   const RPadPos& GetPos() const { return fPos; }
   const RPadLength& GetLength() const { return fLength; }
   const std::vector<std::string> &GetLabels() const { return fLabels; }
};



} // namespace Experimental
} // namespace ROOT

#endif

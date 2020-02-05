/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RLegend
#define ROOT7_RLegend

#include <ROOT/RBox.hxx>

#include <ROOT/RAttrText.hxx>

#include <initializer_list>
#include <memory>

namespace ROOT {
namespace Experimental {

class RLegend;

namespace Internal {

/** \class RLegendEntry
\ingroup GrafROOT7
\brief An entry in RLegend, references RDrawable and its attributes
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-09
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


class RLegendEntry {

   friend class ROOT::Experimental::RLegend;

   std::string fLabel;    ///< label shown for the entry

   std::string fLine, fFill, fMarker;  ///< prefixes for line, fill, marker attributes

   RIOShared<RDrawable> fDrawable;  ///< reference to RDrawable

public:

   RLegendEntry() = default;

   RLegendEntry(std::shared_ptr<RDrawable> drawable, const std::string &lbl = "")
   {
      fDrawable = drawable;
      fLabel = lbl;
   }

   RLegendEntry &SetLabel(const std::string &lbl) { fLabel = lbl; return *this; }
   const std::string &GetLabel() const { return fLabel; }

   RLegendEntry &SetLine(const std::string &lbl) { fLine = lbl; return *this; }
   const std::string &GetLine() const { return fLine; }

   RLegendEntry &SetFill(const std::string &lbl) { fFill = lbl; return *this; }
   const std::string &GetFill() const { return fFill; }

   RLegendEntry &SetMarker(const std::string &lbl) { fMarker = lbl; return *this; }
   const std::string &GetMarker() const { return fMarker; }

};

}

/** \class RLegend
\ingroup GrafROOT7
\brief A legend for several drawables
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-09
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RLegend : public RBox {

   std::string fTitle;                  ///< legend title

   std::vector<Internal::RLegendEntry> fEntries; ///< list of entries which should be displayed

   RAttrText  fAttrTitle{this, "title_"};    ///<! title attributes

protected:

   void CollectShared(Internal::RIOSharedVector_t &vect) override
   {
      for (auto &entry : fEntries) {
         vect.emplace_back(&entry.fDrawable);
         if (entry.fDrawable)
            entry.fDrawable->CollectShared(vect);
      }
   }

public:

   RLegend() : RBox("legend") {}

   RLegend(const RPadPos &p1, const RPadPos &p2, const std::string &title = "") : RLegend()
   {
      SetP1(p1);
      SetP2(p2);
      SetTitle(title);
   }

   RLegend &SetTitle(const std::string &title) { fTitle = title; return *this; }
   const std::string &GetTitle() const { return fTitle; }

   const RAttrText &GetAttrTitle() const { return fAttrTitle; }
   RLegend &SetAttrTitle(const RAttrText &attr) { fAttrTitle = attr; return *this; }
   RAttrText &AttrTitle() { return fAttrTitle; }

   Internal::RLegendEntry &AddEntry(std::shared_ptr<RDrawable> drawable, const std::string &lbl = "")
   {
      fEntries.emplace_back(drawable, lbl);
      return fEntries.back();
   }

   auto NumEntries() const { return fEntries.size(); }


};

} // namespace Experimental
} // namespace ROOT

#endif

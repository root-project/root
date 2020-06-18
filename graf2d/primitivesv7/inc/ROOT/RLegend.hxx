/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RLegend
#define ROOT7_RLegend

#include <ROOT/RPave.hxx>
#include <ROOT/RDisplayItem.hxx>

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

   std::string fDrawableId;        ///< drawable id, used only when display item

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

class RLegend : public RPave {

   friend class RDisplayLegend;

   std::string fTitle;                  ///< legend title

   std::vector<Internal::RLegendEntry> fEntries; ///< list of entries which should be displayed

protected:

   void CollectShared(Internal::RIOSharedVector_t &vect) override
   {
      for (auto &entry : fEntries) {
         vect.emplace_back(&entry.fDrawable);
         if (entry.fDrawable)
            entry.fDrawable->CollectShared(vect);
      }
   }

   /** hide I/O pointers when creating display item */
   std::unique_ptr<RDisplayItem> Display(const RDisplayContext &) override
   {
      for (auto &entry : fEntries) {
         entry.fDrawableId = RDisplayItem::ObjectIDFromPtr(entry.fDrawable.get());
         entry.fDrawable.reset_io();
      }

      return std::make_unique<RDrawableDisplayItem>(*this);
   }

   /** when display item destroyed - restore I/O pointers */
   void OnDisplayItemDestroyed(RDisplayItem *) const override
   {
      for (auto &centry : fEntries) {
         auto entry = const_cast<Internal::RLegendEntry *>(&centry);
         entry->fDrawable.restore_io();
         entry->fDrawableId.clear();
      }
   }

public:

   RLegend() : RPave("legend") {}

   RLegend(const std::string &title) : RLegend()
   {
      SetTitle(title);
   }

   RLegend &SetTitle(const std::string &title) { fTitle = title; return *this; }
   const std::string &GetTitle() const { return fTitle; }

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

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RLegend
#define ROOT7_RLegend

#include <ROOT/RPave.hxx>
#include <ROOT/RAttrLine.hxx>
#include <ROOT/RAttrFill.hxx>
#include <ROOT/RAttrMarker.hxx>
#include <ROOT/RDisplayItem.hxx>

#include <initializer_list>
#include <memory>
#include <vector>

namespace ROOT {
namespace Experimental {

/** \class RLegend
\ingroup GrafROOT7
\brief A legend for several drawables
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-09
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RLegend : public RPave {

public:

   /** \class RCustomDrawable
   \ingroup GrafROOT7
   \brief Special drawable to let provide line, fill or marker attributes for legend
   \author Sergey Linev <S.Linev@gsi.de>
   \date 2021-07-06
   \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
   is welcome!
   */

   class RCustomDrawable final : public RDrawable {
   public:
      RCustomDrawable() : RDrawable("lentry") {}

      RAttrLine line{this, "line"};          ///<! line attributes
      RAttrFill fill{this, "fill"};          ///<! fill attributes
      RAttrMarker marker{this, "marker"};    ///<! marker attributes
   };

   /** \class REntry
   \ingroup GrafROOT7
   \brief An entry in RLegend, references RDrawable and its attributes
   \author Sergey Linev <S.Linev@gsi.de>
   \date 2019-10-09
   \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
   is welcome!
   */

   class REntry {

      friend class RLegend;

      std::string fLabel; ///< label shown for the entry

      bool fLine{true}, fFill{false}, fMarker{false}, fError{false}; ///< enable line, fill, marker, error showing

      Internal::RIOShared<RDrawable> fDrawable; ///< reference to RDrawable

      std::string fDrawableId; ///< drawable id, used only when display item

      bool IsCustomDrawable() const { return dynamic_cast<const RCustomDrawable *>(fDrawable.get()) != nullptr; }

      void DecodeOptions(const std::string &opt)
      {
         if (opt.find('l') != std::string::npos) SetLine(true);
         if (opt.find('f') != std::string::npos) SetFill(true);
         if (opt.find('m') != std::string::npos) SetMarker(true);
         if (opt.find('e') != std::string::npos) SetError(true);
      }

   public:
      REntry() = default;

      /** Create entry without reference to existing drawable object, can assign attributes */
      REntry(const std::string &lbl, const std::string &opt)
      {
         fDrawableId = "custom";
         fLabel = lbl;
         DecodeOptions(opt);
      }

      /** Create entry with reference to existing drawable object */
      REntry(std::shared_ptr<RDrawable> drawable, const std::string &lbl, const std::string &opt)
      {
         fDrawable = drawable;
         fLabel = lbl;
         DecodeOptions(opt);
      }

      REntry &SetLabel(const std::string &lbl) { fLabel = lbl; return *this; }
      const std::string &GetLabel() const { return fLabel; }

      REntry &SetLine(bool on = true) { fLine = on; return *this; }
      bool GetLine() const { return fLine; }

      REntry &SetFill(bool on = true) { fFill = on; return *this; }
      bool GetFill() const { return fFill; }

      REntry &SetMarker(bool on = true) { fMarker = on; return *this; }
      bool GetMarker() const { return fMarker; }

      REntry &SetError(bool on = true) { fError = on; return *this; }
      bool GetError() const { return fError; }

      std::shared_ptr<RDrawable> GetDrawable() const { return fDrawable.get_shared(); }
   };

private:
   std::string fTitle; ///< legend title

   std::vector<REntry> fEntries; ///< list of entries which should be displayed

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
         if (!entry.IsCustomDrawable()) {
            entry.fDrawableId = RDisplayItem::ObjectIDFromPtr(entry.fDrawable.get());
            entry.fDrawable.reset_io();
         }
      }

      return std::make_unique<RDrawableDisplayItem>(*this);
   }

   /** when display item destroyed - restore I/O pointers */
   void OnDisplayItemDestroyed(RDisplayItem *) const override
   {
      for (auto &centry : fEntries) {
         if (!centry.IsCustomDrawable()) {
            auto entry = const_cast<REntry *>(&centry);
            entry->fDrawable.restore_io();
            entry->fDrawableId.clear();
         }
      }
   }

public:
   RLegend() : RPave("legend") {}

   RLegend(const std::string &title) : RLegend() { SetTitle(title); }

   RLegend(const RPadPos &offset, const RPadExtent &size) : RLegend()
   {
      offsetX = offset.Horiz();
      offsetY = offset.Vert();
      width = size.Horiz();
      height = size.Vert();
   }

   RLegend &SetTitle(const std::string &title) { fTitle = title; return *this; }
   const std::string &GetTitle() const { return fTitle; }

   std::shared_ptr<RCustomDrawable> AddEntry(const std::string &lbl, const std::string &opt = "")
   {
      fEntries.emplace_back(lbl, opt);
      auto drawable = std::make_shared<RCustomDrawable>();
      fEntries.back().fDrawable = drawable;
      return drawable;
   }

   void AddEntry(const std::shared_ptr<RDrawable> &drawable, const std::string &lbl, const std::string &opt = "")
   {
      fEntries.emplace_back(drawable, lbl, opt);
   }

   auto NumEntries() const { return fEntries.size(); }

   auto &GetEntry(int n) { return fEntries[n]; }
};

} // namespace Experimental
} // namespace ROOT

#endif

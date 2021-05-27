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

      bool fLine{true}, fFill{false}, fMarker{false}; ///< enable line, fill, marker showing


   protected:
      virtual void CollectShared(Internal::RIOSharedVector_t &) {}

      virtual void BeforeDisplay() {}

      virtual void AfterDisplay() const {}

   public:
      REntry() = default;
      virtual ~REntry() = default;

      REntry &SetLabel(const std::string &lbl) { fLabel = lbl; return *this; }
      const std::string &GetLabel() const { return fLabel; }

      REntry &SetLine(bool on = true) { fLine = on; return *this; }
      bool GetLine() const { return fLine; }

      REntry &SetFill(bool on = true) { fFill = on; return *this; }
      bool GetFill() const { return fFill; }

      REntry &SetMarker(bool on = true) { fMarker = on; return *this; }
      bool GetMarker() const { return fMarker; }
   };

   /** \class RDrawableEntry
   \ingroup GrafROOT7
   \brief An entry in RLegend, which references other RDrawable with its attributes
   \author Sergey Linev <S.Linev@gsi.de>
   \date 2021-05-27
   \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
   is welcome!
   */

   class RDrawableEntry : public REntry {
      Internal::RIOShared<RDrawable> fDrawable; ///< reference to RDrawable
      std::string fDrawableId; ///< drawable id, used only when display item
   protected:

      void CollectShared(Internal::RIOSharedVector_t &vect) override
      {
         vect.emplace_back(&fDrawable);
         if (fDrawable)
            fDrawable->CollectShared(vect);
      }

      void BeforeDisplay() override
      {
         fDrawableId = RDisplayItem::ObjectIDFromPtr(fDrawable.get());
         fDrawable.reset_io();
      }

      void AfterDisplay() const override
      {
         auto entry = const_cast<RDrawableEntry *>(this);
         entry->fDrawable.restore_io();
         entry->fDrawableId.clear();
      }

   public:

      RDrawableEntry() = default;

      /** Create entry with reference to existing drawable object */
      RDrawableEntry(std::shared_ptr<RDrawable> drawable, const std::string &lbl)
      {
         fDrawable = drawable;
         fLabel = lbl;
      }

   };

   /** \class RDrawableEntry
   \ingroup GrafROOT7
   \brief An entry in RLegend, which references other RDrawable with its attributes
   \author Sergey Linev <S.Linev@gsi.de>
   \date 2021-05-27
   \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
   is welcome!
   */

   class RCustomEntry : public REntry, protected RDrawable, public RAttrLine, public RAttrFill, public RAttrMarker {
   public:

      RCustomEntry(const std::string &lbl = "") : REntry(), RDrawable("lentry"), RAttrLine(this), RAttrFill(this), RAttrMarker(this)
      {
         SetLabel(lbl);
      }

   };

private:
   std::string fTitle; ///< legend title

   std::vector<std::unique_ptr<REntry>> fEntries; ///< list of entries which should be displayed

protected:
   void CollectShared(Internal::RIOSharedVector_t &vect) override
   {
      for (auto &entry : fEntries)
         entry->CollectShared(vect);
   }

   /** hide I/O pointers when creating display item */
   std::unique_ptr<RDisplayItem> Display(const RDisplayContext &) override
   {
      for (auto &entry : fEntries)
         entry->BeforeDisplay();

      return std::make_unique<RDrawableDisplayItem>(*this);
   }

   /** when display item destroyed - restore I/O pointers */
   void OnDisplayItemDestroyed(RDisplayItem *) const override
   {
      for (auto &centry : fEntries)
         centry->AfterDisplay();
   }

public:
   RLegend() : RPave("legend") {}

   RLegend(const std::string &title) : RLegend() { SetTitle(title); }

   void SetTitle(const std::string &title) { fTitle = title; }
   const std::string &GetTitle() const { return fTitle; }

   RCustomEntry *AddEntry(const std::string &lbl)
   {
      fEntries.emplace_back(std::make_unique<RCustomEntry>(lbl));
      return static_cast<RCustomEntry *>(fEntries.back().get());
   }

   RDrawableEntry *AddEntry(std::shared_ptr<RDrawable> drawable, const std::string &lbl)
   {
      fEntries.emplace_back(std::make_unique<RDrawableEntry>(drawable, lbl));
      return static_cast<RDrawableEntry *>(fEntries.back().get());
   }

   auto NumEntries() const { return fEntries.size(); }

   REntry *GetEntry(int n) { return fEntries[n].get(); }
};

} // namespace Experimental
} // namespace ROOT

#endif

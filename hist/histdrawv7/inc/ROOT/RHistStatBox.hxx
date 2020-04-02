/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RHistStatBox
#define ROOT7_RHistStatBox

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrText.hxx>
#include <ROOT/RAttrLine.hxx>
#include <ROOT/RAttrFill.hxx>
#include <ROOT/RPadPos.hxx>
#include <ROOT/RDisplayItem.hxx>
#include <ROOT/RHist.hxx>
#include <ROOT/RHistImpl.hxx>

#include <memory>
#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {

class RDisplayHistStat : public RIndirectDisplayItem {
   std::vector<std::string> fLines;   ///< text lines displayed in the stat box
public:
   RDisplayHistStat() = default;

   RDisplayHistStat(const RDrawable &box) : RIndirectDisplayItem(box) {}

   void AddLine(const std::string &line) { fLines.emplace_back(line); }
};


/** \class ROOT::Experimental::RHistStatBox
\ingroup GrafROOT7
\brief Statistic box for RHist class
\author Sergey Linev <s.linev@gsi.de>
\date 2020-04-01
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


template <int DIMENSIONS>
class RHistStatBox : public RDrawable {
public:
   using HistImpl_t = Detail::RHistImplPrecisionAgnosticBase<DIMENSIONS>;

private:
   Internal::RIOShared<HistImpl_t> fHistImpl;  ///< I/O capable reference on histogram

   class RHistStatBoxAttrs : public RAttrBase {
      // friend class RHistStatBox<DIMENSIONS>;
      R__ATTR_CLASS(RHistStatBoxAttrs, "", AddString("cornerx","0.02").AddString("cornery","0.02").AddString("width","0.5").AddString("height","0.2"));
   };

   std::string fTitle;                       ///< stat box title

   RAttrText fAttrText{this, "text_"};       ///<! text attributes
   RAttrLine fAttrBorder{this, "border_"};   ///<! border attributes
   RAttrFill fAttrFill{this, "fill_"};       ///<! line attributes
   RHistStatBoxAttrs fAttr{this,""};         ///<! title direct attributes

protected:

   bool IsFrameRequired() const final { return true; }

   void CollectShared(Internal::RIOSharedVector_t &vect) override { vect.emplace_back(&fHistImpl); }

   virtual void FillStatistic(RDisplayHistStat &, const RPadBase &) const {}

   std::unique_ptr<RDisplayItem> Display(const RPadBase &pad) const override
   {
      auto res = std::make_unique<RDisplayHistStat>(*this);

      if (!GetTitle().empty())
         res->AddLine(GetTitle());

      FillStatistic(*res.get(), pad);

      return res;
   }

public:

   RHistStatBox() : RDrawable("stats") {}

   template <class HIST>
   RHistStatBox(const std::shared_ptr<HIST> &hist, const std::string &title = "") : RHistStatBox()
   {
      fHistImpl = std::shared_ptr<HistImpl_t>(hist, hist->GetImpl());
      SetTitle(title);
   }

   std::shared_ptr<HistImpl_t> GetHist() const { return fHistImpl.get_shared(); }


   void SetTitle(const std::string &title) { fTitle = title; }
   const std::string &GetTitle() const { return fTitle; }

   RHistStatBox &SetCornerX(const RPadLength &pos)
   {
      if (pos.Empty())
         fAttr.ClearValue("cornerx");
      else
         fAttr.SetValue("cornerx", pos.AsString());

      return *this;
   }

   RPadLength GetCornerX() const
   {
      RPadLength res;
      auto value = fAttr.template GetValue<std::string>("cornerx");
      if (!value.empty())
         res.ParseString(value);
      return res;
   }

   RHistStatBox &SetCornerY(const RPadLength &pos)
   {
      if (pos.Empty())
         fAttr.ClearValue("cornery");
      else
         fAttr.SetValue("cornery", pos.AsString());

      return *this;
   }

   RPadLength GetCornerY() const
   {
      RPadLength res;
      auto value = fAttr.template GetValue<std::string>("cornery");
      if (!value.empty())
         res.ParseString(value);
      return res;
   }

   RHistStatBox &SetWidth(const RPadLength &pos)
   {
      if (pos.Empty())
         fAttr.ClearValue("width");
      else
         fAttr.SetValue("width", pos.AsString());

      return *this;
   }

   RPadLength GetWidth() const
   {
      RPadLength res;
      auto value = fAttr.template GetValue<std::string>("width");
      if (!value.empty())
         res.ParseString(value);
      return res;
   }

   RHistStatBox &SetHeight(const RPadLength &pos)
   {
      if (pos.Empty())
         fAttr.ClearValue("height");
      else
         fAttr.SetValue("height", pos.AsString());

      return *this;
   }

   RPadLength GetHeight() const
   {
      RPadLength res;
      auto value = fAttr.template GetValue<std::string>("height");
      if (!value.empty())
         res.ParseString(value);
      return res;
   }

   const RAttrText &GetAttrText() const { return fAttrText; }
   RHistStatBox &SetAttrText(const RAttrText &attr) { fAttrText = attr; return *this; }
   RAttrText &AttrText() { return fAttrText; }

   const RAttrLine &GetAttrBorder() const { return fAttrBorder; }
   RHistStatBox &SetAttrBorder(const RAttrLine &border) { fAttrBorder = border; return *this; }
   RAttrLine &AttrBorder() { return fAttrBorder; }

   const RAttrFill &GetAttrFill() const { return fAttrFill; }
   RHistStatBox &SetAttrFill(const RAttrFill &fill) { fAttrFill = fill; return *this; }
   RAttrFill &AttrFill() { return fAttrFill; }
};


class RHist1StatBox final : public RHistStatBox<1> {
protected:
   void FillStatistic(RDisplayHistStat &, const RPadBase &) const override;
public:
   RHist1StatBox() = default;
   template <class HIST>
   RHist1StatBox(const std::shared_ptr<HIST> &hist, const std::string &title = "") : RHistStatBox<1>(hist, title) {}
};


class RHist2StatBox final : public RHistStatBox<2> {
protected:

   void FillStatistic(RDisplayHistStat &, const RPadBase &) const override;

public:
   RHist2StatBox() = default;

   template <class HIST>
   RHist2StatBox(const std::shared_ptr<HIST> &hist, const std::string &title = "") : RHistStatBox<2>(hist, title) {}
};

class RHist3StatBox final : public RHistStatBox<3> {
protected:

   void FillStatistic(RDisplayHistStat &, const RPadBase &) const override;

public:
   RHist3StatBox() = default;

   template <class HIST>
   RHist3StatBox(const std::shared_ptr<HIST> &hist, const std::string &title = "") : RHistStatBox<3>(hist, title) {}
};



} // namespace Experimental
} // namespace ROOT

#endif

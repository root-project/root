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


class RHistStatRequest : public RDrawableRequest {
   std::vector<double> xmin; // vector of axis min values
   std::vector<double> xmax; // vector of axis max values

public:

   std::unique_ptr<RDrawableReply> Process() override;

   double GetMin(unsigned indx) const { return indx < xmin.size() ? xmin[indx] : 0; }
   double GetMax(unsigned indx) const { return indx < xmax.size() ? xmax[indx] : 0; }
};

class RHistStatReply : public RDrawableReply {
   std::vector<std::string> lines;   ///< text lines displayed in the stat box
public:

   void AddLine(const std::string &line) { lines.emplace_back(line); }

   // pin vtable
   virtual ~RHistStatReply() = default;
};

class RDisplayHistStat : public RIndirectDisplayItem {
public:
   RDisplayHistStat() = default;
   RDisplayHistStat(const RDrawable &dr) : RIndirectDisplayItem(dr) {}
   virtual ~RDisplayHistStat() = default;
};


/** \class ROOT::Experimental::RHistStatBoxBase
\ingroup GrafROOT7
\brief Base class for histogram statistic box, provides graphics attributes and virtual method for fill statistic
\author Sergey Linev <s.linev@gsi.de>
\date 2020-04-01
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RHistStatBoxBase : public RDrawable {

friend class RHistStatRequest; // access fill statistic method

private:

   class RHistStatBoxAttrs : public RAttrBase {
      friend class RHistStatBoxBase;
      R__ATTR_CLASS(RHistStatBoxAttrs, "", AddString("cornerx","0.02").AddString("cornery","0.02").AddString("width","0.5").AddString("height","0.2"));
   };

   std::string fTitle;                       ///< stat box title

   RAttrText fAttrText{this, "text_"};       ///<! text attributes
   RAttrLine fAttrBorder{this, "border_"};   ///<! border attributes
   RAttrFill fAttrFill{this, "fill_"};       ///<! line attributes
   RHistStatBoxAttrs fAttr{this,""};         ///<! stat box direct attributes

protected:

   bool IsFrameRequired() const final { return true; }

   virtual void FillStatistic(const RHistStatRequest &, RHistStatReply &) const {}

   std::unique_ptr<RDisplayItem> Display(const RPadBase &, Version_t) const override
   {
      // do not send stat box itself while it includes histogram which is not required on client side
      return std::make_unique<RDisplayHistStat>(*this);
   }

public:

   RHistStatBoxBase() : RDrawable("stats") {}

   void SetTitle(const std::string &title) { fTitle = title; }
   const std::string &GetTitle() const { return fTitle; }

   RHistStatBoxBase &SetCornerX(const RPadLength &pos)
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

   RHistStatBoxBase &SetCornerY(const RPadLength &pos)
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

   RHistStatBoxBase &SetWidth(const RPadLength &pos)
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

   RHistStatBoxBase &SetHeight(const RPadLength &pos)
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
   RHistStatBoxBase &SetAttrText(const RAttrText &attr) { fAttrText = attr; return *this; }
   RAttrText &AttrText() { return fAttrText; }

   const RAttrLine &GetAttrBorder() const { return fAttrBorder; }
   RHistStatBoxBase &SetAttrBorder(const RAttrLine &border) { fAttrBorder = border; return *this; }
   RAttrLine &AttrBorder() { return fAttrBorder; }

   const RAttrFill &GetAttrFill() const { return fAttrFill; }
   RHistStatBoxBase &SetAttrFill(const RAttrFill &fill) { fAttrFill = fill; return *this; }
   RAttrFill &AttrFill() { return fAttrFill; }
};



/** \class ROOT::Experimental::RHistStatBox
\ingroup GrafROOT7
\brief Template class for statistic box for RHist class
\author Sergey Linev <s.linev@gsi.de>
\date 2020-04-01
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/


template <int DIMENSIONS>
class RHistStatBox : public RHistStatBoxBase {
public:
   using HistImpl_t = Detail::RHistImplPrecisionAgnosticBase<DIMENSIONS>;

private:
   Internal::RIOShared<HistImpl_t> fHistImpl;  ///< I/O capable reference on histogram

protected:

   void CollectShared(Internal::RIOSharedVector_t &vect) override { vect.emplace_back(&fHistImpl); }

public:

   template <class HIST>
   RHistStatBox(const std::shared_ptr<HIST> &hist, const std::string &title = "")
   {
      fHistImpl = std::shared_ptr<HistImpl_t>(hist, hist->GetImpl());
      SetTitle(title);
   }

   std::shared_ptr<HistImpl_t> GetHist() const { return fHistImpl.get_shared(); }
};


class RHist1StatBox final : public RHistStatBox<1> {
protected:
   void FillStatistic(const RHistStatRequest &, RHistStatReply &) const override;
public:
   RHist1StatBox() = default;
   template <class HIST>
   RHist1StatBox(const std::shared_ptr<HIST> &hist, const std::string &title = "") : RHistStatBox<1>(hist, title) {}
};


class RHist2StatBox final : public RHistStatBox<2> {
protected:

   void FillStatistic(const RHistStatRequest &, RHistStatReply &) const override;

public:
   RHist2StatBox() = default;

   template <class HIST>
   RHist2StatBox(const std::shared_ptr<HIST> &hist, const std::string &title = "") : RHistStatBox<2>(hist, title) {}
};

class RHist3StatBox final : public RHistStatBox<3> {
protected:

   void FillStatistic(const RHistStatRequest &, RHistStatReply &) const override;

public:
   RHist3StatBox() = default;

   template <class HIST>
   RHist3StatBox(const std::shared_ptr<HIST> &hist, const std::string &title = "") : RHistStatBox<3>(hist, title) {}
};



} // namespace Experimental
} // namespace ROOT

#endif

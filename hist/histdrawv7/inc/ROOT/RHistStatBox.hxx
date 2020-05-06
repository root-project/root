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
#include <ROOT/RDrawableRequest.hxx>
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

/** \class ROOT::Experimental::RHistStatRequest
\ingroup GrafROOT7
\brief Data request from stat box, send when client starts drawing
\author Sergey Linev <s.linev@gsi.de>
\date 2020-04-17
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RHistStatRequest : public RDrawableRequest {
   unsigned mask{0xff};      // mask of items to show
public:
   RHistStatRequest() = default;
   unsigned GetMask() const { return mask; }
   std::unique_ptr<RDrawableReply> Process() override;
};

/** \class ROOT::Experimental::RHistStatReply
\ingroup GrafROOT7
\brief Reply of stat box on RHistStatRequest, contains text lines to display
\author Sergey Linev <s.linev@gsi.de>
\date 2020-04-17
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RHistStatReply : public RDrawableReply {
   unsigned mask{0};                 ///< mask used to create lines
   std::vector<std::string> lines;   ///< text lines displayed in the stat box
public:

   std::vector<std::string> &GetLines() { return lines; }

   void SetMask(unsigned _mask) { mask = _mask; }

   // virtual destructor - required to pin vtable
   virtual ~RHistStatReply() = default;
};

/** \class ROOT::Experimental::RDisplayHistStat
\ingroup GrafROOT7
\brief Object send to client for display of RHistStat, required to avoid sending histogram to the client
\author Sergey Linev <s.linev@gsi.de>
\date 2020-04-17
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RDisplayHistStat : public RIndirectDisplayItem {
   unsigned fShowMask{0};   ///< initial show mask
   std::vector<std::string> fEntries; ///< names of entries for context menu
   std::vector<std::string> fLines;   ///< filled lines to show in stat box
public:
   RDisplayHistStat() = default;
   RDisplayHistStat(const RDrawable &dr, unsigned mask, const std::vector<std::string> &entries, const std::vector<std::string> &lines) :
         RIndirectDisplayItem(dr), fShowMask(mask), fEntries(entries), fLines(lines) {}
   virtual ~RDisplayHistStat() = default;

   unsigned GetShowMask() const { return fShowMask; }
   const std::vector<std::string> &GetEntries() const { return fEntries; }
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
      R__ATTR_CLASS(RHistStatBoxAttrs, "", AddPadLength("cornerx",0.02).AddPadLength("cornery",0.02).AddPadLength("width",0.5).AddPadLength("height",0.2));
   };

   std::string fTitle;                       ///< stat box title
   unsigned fShowMask{0xff};                 ///< show stat box lines

   RAttrText fAttrText{this, "text_"};       ///<! text attributes
   RAttrLine fAttrBorder{this, "border_"};   ///<! border attributes
   RAttrFill fAttrFill{this, "fill_"};       ///<! line attributes
   RHistStatBoxAttrs fAttr{this, ""};        ///<! stat box direct attributes

protected:

   enum EShowBits { kShowTitle = 0x1, kShowEntries = 0x2, kShowMean = 0x4, kShowDev = 0x8, kShowRange = 0x10 };

   bool IsFrameRequired() const final { return true; }

   virtual void FillStatistic(unsigned, const RDrawable::RUserRanges &, std::vector<std::string> &) const {}

   virtual const std::vector<std::string> &GetEntriesNames() const;

   std::unique_ptr<RDisplayItem> Display(const RDisplayContext &) override;

public:

   RHistStatBoxBase() : RDrawable("stats") {}

   void SetTitle(const std::string &title) { fTitle = title; }
   const std::string &GetTitle() const { return fTitle; }

   RHistStatBoxBase &SetCornerX(const RPadLength &pos)
   {
      fAttr.SetValue("cornerx", pos);
      return *this;
   }

   RPadLength GetCornerX() const
   {
      return fAttr.template GetValue<RPadLength>("cornerx");
   }

   RHistStatBoxBase &SetCornerY(const RPadLength &pos)
   {
      fAttr.SetValue("cornery", pos);
      return *this;
   }

   RPadLength GetCornerY() const
   {
      return fAttr.template GetValue<RPadLength>("cornery");
   }

   RHistStatBoxBase &SetWidth(const RPadLength &width)
   {
      fAttr.SetValue("width", width);
      return *this;
   }

   RPadLength GetWidth() const
   {
      return fAttr.template GetValue<RPadLength>("width");
   }

   RHistStatBoxBase &SetHeight(const RPadLength &height)
   {
      fAttr.SetValue("height", height);
      return *this;
   }

   RPadLength GetHeight() const
   {
      return fAttr.template GetValue<RPadLength>("height");
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
   void FillStatistic(unsigned, const RDrawable::RUserRanges &, std::vector<std::string> &) const override;
public:
   template <class HIST>
   RHist1StatBox(const std::shared_ptr<HIST> &hist, const std::string &title = "") : RHistStatBox<1>(hist, title) {}
};


class RHist2StatBox final : public RHistStatBox<2> {
protected:
   void FillStatistic(unsigned, const RDrawable::RUserRanges &, std::vector<std::string> &) const override;
public:
   template <class HIST>
   RHist2StatBox(const std::shared_ptr<HIST> &hist, const std::string &title = "") : RHistStatBox<2>(hist, title) {}
};

class RHist3StatBox final : public RHistStatBox<3> {
protected:
   void FillStatistic(unsigned, const RDrawable::RUserRanges &, std::vector<std::string> &) const override;
public:
   template <class HIST>
   RHist3StatBox(const std::shared_ptr<HIST> &hist, const std::string &title = "") : RHistStatBox<3>(hist, title) {}
};



} // namespace Experimental
} // namespace ROOT

#endif

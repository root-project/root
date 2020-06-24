/// \file ROOT/RHistDrawable.h
/// \ingroup HistDraw ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RHistDrawable
#define ROOT7_RHistDrawable

#include <ROOT/RDrawable.hxx>
#include <ROOT/RAttrLine.hxx>
#include <ROOT/RAttrText.hxx>
#include <ROOT/RAttrMarker.hxx>
#include <ROOT/RAttrFill.hxx>
#include <ROOT/RHist.hxx>
#include <ROOT/RHistImpl.hxx>

#include <memory>

namespace ROOT {
namespace Experimental {

template <int DIMENSIONS>
class RHistDrawable : public RDrawable {
public:
   using HistImpl_t = Detail::RHistImplPrecisionAgnosticBase<DIMENSIONS>;

private:
   Internal::RIOShared<HistImpl_t> fHistImpl;  ///< I/O capable reference on histogram

   class RHistAttrs final : public RAttrBase {
      friend class RHistDrawable;
      R__ATTR_CLASS(RHistAttrs, "", AddString("kind","").AddInt("sub",0).AddBool("text", false));
   };

   RHistAttrs  fAttr{this, ""};           ///<! hist direct attributes
   RAttrLine   fAttrLine{this, "line_"};  ///<! hist line attributes
   RAttrFill   fAttrFill{this, "fill_"};  ///<! hist fill attributes
   RAttrText   fAttrText{this, "text_"};  ///<! hist text attributes
   RAttrMarker fMarkerAttr{this, "marker_"}; ///<! hist marker attributes

protected:

   void CollectShared(Internal::RIOSharedVector_t &vect) override { vect.emplace_back(&fHistImpl); }

   bool IsFrameRequired() const final { return true; }

   void PopulateMenu(RMenuItems &) override
   {
      // populate menu
   }

   void SetDrawKind(const std::string &kind, int sub = -1)
   {
      fAttr.SetValue("kind", kind);
      if (sub>=0)
         fAttr.SetValue("sub", sub);
      else
         fAttr.ClearValue("sub");
   }

   void SetDrawText(bool on) { fAttr.SetValue("text", on); }

public:
   RHistDrawable() : RDrawable("hist") {}
   virtual ~RHistDrawable() = default;

   template <class HIST>
   RHistDrawable(const std::shared_ptr<HIST> &hist) : RHistDrawable()
   {
      fHistImpl = std::shared_ptr<HistImpl_t>(hist, hist->GetImpl());
   }

   std::shared_ptr<HistImpl_t> GetHist() const { return fHistImpl.get_shared(); }

   const RAttrLine &GetAttrLine() const { return fAttrLine; }
   RHistDrawable &SetAttrLine(const RAttrLine &attr) { fAttrLine = attr; return *this; }
   RAttrLine &AttrLine() { return fAttrLine; }

   const RAttrFill &GetAttrFill() const { return fAttrFill; }
   RHistDrawable &SetAttrFill(const RAttrFill &fill) { fAttrFill = fill; return *this; }
   RAttrFill &AttrFill() { return fAttrFill; }

   const RAttrText &GetAttrText() const { return fAttrText; }
   RHistDrawable &SetAttrText(const RAttrText &attr) { fAttrText = attr; return *this; }
   RAttrText &AttrText() { return fAttrText; }

   const RAttrMarker &GetAttrMarker() const { return fMarkerAttr; }
   RHistDrawable &SetAttrMarker(const RAttrMarker &attr) { fMarkerAttr = attr; return *this; }
   RAttrMarker &AttrMarker() { return fMarkerAttr; }
};

// template <int DIMENSIONS> inline RHistDrawable<DIMENSIONS>::RHistDrawable() : RDrawable("hist") {}


class RHist1Drawable final : public RHistDrawable<1> {
public:
   RHist1Drawable() = default;

   template <class HIST>
   RHist1Drawable(const std::shared_ptr<HIST> &hist) : RHistDrawable<1>(hist) {}

   RHist1Drawable &Bar(int kind = 0) { SetDrawKind("bar", kind); return *this; }
   RHist1Drawable &Error(int kind = 0) { SetDrawKind("err", kind); return *this; }
   RHist1Drawable &Marker() { SetDrawKind("p"); return *this; }
   RHist1Drawable &Star() { AttrMarker().SetStyle(3); return Marker(); }
   RHist1Drawable &Hist() { SetDrawKind("hist"); return *this; }
   RHist1Drawable &Lego(int kind = 0) { SetDrawKind("lego", kind); return *this; }
   RHist1Drawable &Text(bool on = true) { SetDrawText(on); return *this; }
};


class RHist2Drawable final : public RHistDrawable<2> {

public:
   RHist2Drawable() = default;

   template <class HIST>
   RHist2Drawable(const std::shared_ptr<HIST> &hist) : RHistDrawable<2>(hist) {}

   RHist2Drawable &Color() { SetDrawKind("col"); return *this; }
   RHist2Drawable &Lego(int kind = 0) { SetDrawKind("lego", kind); return *this; }
   RHist2Drawable &Surf(int kind = 0) { SetDrawKind("surf", kind); return *this; }
   RHist2Drawable &Error() { SetDrawKind("err"); return *this; }
   RHist2Drawable &Contour(int kind = 0) { SetDrawKind("cont", kind); return *this; }
   RHist2Drawable &Scatter() { SetDrawKind("scat"); return *this; }
   RHist2Drawable &Arrow() { SetDrawKind("arr"); return *this; }
   RHist2Drawable &Text(bool on = true) { SetDrawText(on); return *this; }
};


class RHist3Drawable final : public RHistDrawable<3> {
   RAttrColor fColor{this, "color_"};     ///<! bin color, used which box option

public:
   RHist3Drawable() = default;

   template <class HIST>
   RHist3Drawable(const std::shared_ptr<HIST> &hist) : RHistDrawable<3>(hist) {}

   RHist3Drawable &Color() { SetDrawKind("col"); return *this; }
   RHist3Drawable &Box(int kind = 0) { SetDrawKind("box", kind); return *this; }
   RHist3Drawable &Sphere(int kind = 0) { SetDrawKind("sphere", kind); return *this; }
   RHist3Drawable &Scatter() { SetDrawKind("scat"); return *this; }

   /// The color when box option is used
   RHist3Drawable &SetColor(const RColor &color) { fColor = color; return *this; }
   RColor GetColor() const { return fColor.GetColor(); }
   RAttrColor &AttrColor() { return fColor; }
};


inline auto GetDrawable(const std::shared_ptr<RH1D> &histimpl)
{
   return std::make_shared<RHist1Drawable>(histimpl);
}

inline auto GetDrawable(const std::shared_ptr<RH1I> &histimpl)
{
   return std::make_shared<RHist1Drawable>(histimpl);
}

inline auto GetDrawable(const std::shared_ptr<RH1C> &histimpl)
{
   return std::make_shared<RHist1Drawable>(histimpl);
}

inline auto GetDrawable(const std::shared_ptr<RH1F> &histimpl)
{
   return std::make_shared<RHist1Drawable>(histimpl);
}

inline auto GetDrawable(const std::shared_ptr<RH2D> &histimpl)
{
   return std::make_shared<RHist2Drawable>(histimpl);
}

inline auto GetDrawable(const std::shared_ptr<RH2I> &histimpl)
{
   return std::make_shared<RHist2Drawable>(histimpl);
}

inline auto GetDrawable(const std::shared_ptr<RH2C> &histimpl)
{
   return std::make_shared<RHist2Drawable>(histimpl);
}

inline auto GetDrawable(const std::shared_ptr<RH2F> &histimpl)
{
   return std::make_shared<RHist2Drawable>(histimpl);
}

inline auto GetDrawable(const std::shared_ptr<RH3D> &histimpl)
{
   return std::make_shared<RHist3Drawable>(histimpl);
}

inline auto GetDrawable(const std::shared_ptr<RH3I> &histimpl)
{
   return std::make_shared<RHist3Drawable>(histimpl);
}

inline auto GetDrawable(const std::shared_ptr<RH3C> &histimpl)
{
   return std::make_shared<RHist3Drawable>(histimpl);
}

inline auto GetDrawable(const std::shared_ptr<RH3F> &histimpl)
{
   return std::make_shared<RHist3Drawable>(histimpl);
}

} // namespace Experimental
} // namespace ROOT

#endif

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
#include <ROOT/RHist.hxx>
#include <ROOT/RHistImpl.hxx>
#include <ROOT/RMenuItems.hxx>

#include <memory>

namespace ROOT {
namespace Experimental {

template <int DIMENSIONS, class PRECISION, template <int D_, class P_> class... STAT>
class RHist;

namespace Detail {
template <int DIMENSIONS>
class RHistImplPrecisionAgnosticBase;
}

template <int DIMENSIONS>
class RHistDrawable : public RDrawable {
public:
   using HistImpl_t = Detail::RHistImplPrecisionAgnosticBase<DIMENSIONS>;

private:
   Internal::RIOShared<HistImpl_t> fHistImpl;  ///< I/O capable reference on histogram

protected:

   void CollectShared(Internal::RIOSharedVector_t &vect) override { vect.emplace_back(&fHistImpl); }

   bool IsFrameRequired() const final { return true; }

   void PopulateMenu(RMenuItems &) override
   {
      // populate menu
   }

public:
   RHistDrawable();
   virtual ~RHistDrawable() = default;

   template <class HIST>
   RHistDrawable(const std::shared_ptr<HIST> &hist) : RHistDrawable()
   {
      fHistImpl = std::shared_ptr<HistImpl_t>(hist, hist->GetImpl());
   }

   std::shared_ptr<HistImpl_t> GetHist() const { return fHistImpl.get_shared(); }

//   template <class HIST>
//   RHistDrawable(std::unique_ptr<HIST> &&hist)
//      : fHistImpl(std::unique_ptr<HistImpl_t>(std::move(*hist).TakeImpl())), fOpts(opts)
//   {}

};

template <int DIMENSIONS> inline RHistDrawable<DIMENSIONS>::RHistDrawable() : RDrawable("hist") {}


class RHist1Drawable final : public RHistDrawable<1> {
private:

   RAttrLine  fAttrLine{this, "line_"};        ///<! line attributes

public:
   RHist1Drawable() = default;

   template <class HIST>
   RHist1Drawable(const std::shared_ptr<HIST> &hist) : RHistDrawable<1>(hist) {}

   const RAttrLine &GetAttrLine() const { return fAttrLine; }
   RHist1Drawable &SetAttrLine(const RAttrLine &attr) { fAttrLine = attr; return *this; }
   RAttrLine &AttrLine() { return fAttrLine; }
};


class RHist2Drawable final : public RHistDrawable<2> {
   class RHist2Attrs final : public RAttrBase {
      friend class RHist2Drawable;
      R__ATTR_CLASS(RHist2Attrs, "", AddString("kind","").AddInt("sub",0).AddBool("text", false));
   };

   RHist2Attrs fAttr{this, ""};           ///<! hist2 direct attributes
   RAttrLine   fAttrLine{this, "line_"};  ///<! line attributes, used for error or some lego plots
   RAttrText   fAttrText{this, "text_"};  ///<! text attributes, used with Text drawing

public:
   RHist2Drawable() = default;

   template <class HIST>
   RHist2Drawable(const std::shared_ptr<HIST> &hist) : RHistDrawable<2>(hist) {}

   RHist2Drawable &Color() { fAttr.SetValue("kind", std::string("col")); fAttr.ClearValue("sub"); return *this; }
   RHist2Drawable &Lego(int kind = 0) { fAttr.SetValue("kind", std::string("lego")); fAttr.SetValue("sub", kind); return *this; }
   RHist2Drawable &Surf(int kind = 0) { fAttr.SetValue("kind", std::string("surf")); fAttr.SetValue("sub", kind); return *this; }
   RHist2Drawable &Error() { fAttr.SetValue("kind", std::string("err")); fAttr.ClearValue("sub"); return *this; }
   RHist2Drawable &Contour(int kind = 0) { fAttr.SetValue("kind", std::string("cont")); fAttr.SetValue("sub", kind); return *this; }
   RHist2Drawable &Scatter() { fAttr.SetValue("kind", std::string("scat")); fAttr.ClearValue("sub"); return *this; }
   RHist2Drawable &Text(bool on = true) { fAttr.SetValue("text", on); return *this; }

   const RAttrLine &GetAttrLine() const { return fAttrLine; }
   RHist2Drawable &SetAttrLine(const RAttrLine &attr) { fAttrLine = attr; return *this; }
   RAttrLine &AttrLine() { return fAttrLine; }

   const RAttrText &GetAttrText() const { return fAttrText; }
   RHist2Drawable &SetAttrText(const RAttrText &attr) { fAttrText = attr; return *this; }
   RAttrText &AttrText() { return fAttrText; }

};


class RHist3Drawable final : public RHistDrawable<3> {
   class RHist3Attrs final : public RAttrBase {
      friend class RHist3Drawable;
      R__ATTR_CLASS(RHist3Attrs, "", AddString("kind","").AddInt("sub",0));
   };

   RHist3Attrs fAttr{this, ""};           ///<! hist2 direct attributes
   RAttrColor fColor{this, "color_"};     ///<! bin color, used which box option
   RAttrColor fLineColor{this, "line_color_"};     ///<! bin color, used which box option

public:
   RHist3Drawable() = default;

   template <class HIST>
   RHist3Drawable(const std::shared_ptr<HIST> &hist) : RHistDrawable<3>(hist) {}

   RHist3Drawable &Color() { fAttr.SetValue("kind", std::string("col")); fAttr.ClearValue("sub"); return *this; }
   RHist3Drawable &Box(int kind = 0) { fAttr.SetValue("kind", std::string("box")); fAttr.SetValue("sub", kind); return *this; }
   RHist3Drawable &Sphere(int kind = 0) { fAttr.SetValue("kind", std::string("sphere")); fAttr.SetValue("sub", kind); return *this; }
   RHist3Drawable &Scatter() { fAttr.SetValue("kind", std::string("scat")); fAttr.ClearValue("sub"); return *this; }

   /// The color when box option is used
   RHist3Drawable &SetColor(const RColor &color) { fColor = color; return *this; }
   RColor GetColor() const { return fColor.GetColor(); }
   RAttrColor &AttrColor() { return fColor; }

   /// The line color when box option is used
   RHist3Drawable &SetLineColor(const RColor &color) { fLineColor = color; return *this; }
   RColor GetLineColor() const { return fLineColor.GetColor(); }
   RAttrColor &AttrLineColor() { return fLineColor; }
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

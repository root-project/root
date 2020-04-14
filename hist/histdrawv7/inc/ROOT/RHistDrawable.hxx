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
#include <ROOT/RHist.hxx>
#include <ROOT/RHistImpl.hxx>
#include <ROOT/RMenuItem.hxx>

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

   void Execute(const std::string &) override
   {
      // should execute menu item
   }

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
   RHistDrawable &SetAttrLine(const RAttrLine &attr) { fAttrLine = attr; return *this; }
   RAttrLine &AttrLine() { return fAttrLine; }
};


class RHist2Drawable final : public RHistDrawable<2> {
public:
   RHist2Drawable() = default;

   template <class HIST>
   RHist2Drawable(const std::shared_ptr<HIST> &hist) : RHistDrawable<2>(hist) {}
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
   return std::make_shared<RHistDrawable<2>>(histimpl);
}

inline auto GetDrawable(const std::shared_ptr<RH2C> &histimpl)
{
   return std::make_shared<RHistDrawable<2>>(histimpl);
}

inline auto GetDrawable(const std::shared_ptr<RH2F> &histimpl)
{
   return std::make_shared<RHistDrawable<2>>(histimpl);
}

inline auto GetDrawable(const std::shared_ptr<RH3D> &histimpl)
{
   return std::make_shared<RHistDrawable<3>>(histimpl);
}

inline auto GetDrawable(const std::shared_ptr<RH3I> &histimpl)
{
   return std::make_shared<RHistDrawable<3>>(histimpl);
}

inline auto GetDrawable(const std::shared_ptr<RH3C> &histimpl)
{
   return std::make_shared<RHistDrawable<3>>(histimpl);
}

inline auto GetDrawable(const std::shared_ptr<RH3F> &histimpl)
{
   return std::make_shared<RHistDrawable<3>>(histimpl);
}

} // namespace Experimental
} // namespace ROOT

#endif

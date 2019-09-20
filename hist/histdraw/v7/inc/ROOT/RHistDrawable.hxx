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
#include <ROOT/RPad.hxx>

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
class RHistDrawable final: public RDrawable {
public:
   using HistImpl_t = Detail::RHistImplPrecisionAgnosticBase<DIMENSIONS>;

private:
   Internal::RIOShared<HistImpl_t> fHistImpl;  ///< I/O capable reference on histogram

   RAttrLine  fLineAttr{this, "line_"};        ///<! line attributes

protected:

   void CollectShared(Internal::RIOSharedVector_t &vect) final { vect.emplace_back(&fHistImpl); }

public:
   RHistDrawable();

   template <class HIST>
   RHistDrawable(const std::shared_ptr<HIST> &hist) : RHistDrawable()
   {
      fHistImpl = std::shared_ptr<HistImpl_t>(hist, hist->GetImpl());
   }

   RAttrLine &AttrLine() { return fLineAttr; }
   const RAttrLine &AttrLine() const { return fLineAttr; }

   std::shared_ptr<HistImpl_t> GetHist() const { return fHistImpl.get_shared(); }

   void PopulateMenu(RMenuItems &) final
   {
      // populate menu
   }

   void Execute(const std::string &) final
   {
      // should execute menu item
   }

//   template <class HIST>
//   RHistDrawable(std::unique_ptr<HIST> &&hist)
//      : fHistImpl(std::unique_ptr<HistImpl_t>(std::move(*hist).TakeImpl())), fOpts(opts)
//   {}

   /// Paint the histogram
   void Paint(Internal::RPadPainter &pad) final;

};

extern template class RHistDrawable<1>;
extern template class RHistDrawable<2>;
extern template class RHistDrawable<3>;



template <class... ARGS>
inline auto GetDrawable(const std::shared_ptr<RH1D> &histimpl, ARGS...)
{
   return std::make_shared<RHistDrawable<1>>(histimpl);
}

template <class... ARGS>
inline auto GetDrawable(const std::shared_ptr<RH1I> &histimpl, ARGS...)
{
   return std::make_shared<RHistDrawable<1>>(histimpl);
}

template <class... ARGS>
inline auto GetDrawable(const std::shared_ptr<RH1C> &histimpl, ARGS...)
{
   return std::make_shared<RHistDrawable<1>>(histimpl);
}

template <class... ARGS>
inline auto GetDrawable(const std::shared_ptr<RH1F> &histimpl, ARGS...)
{
   return std::make_shared<RHistDrawable<1>>(histimpl);
}

template <class... ARGS>
inline auto GetDrawable(const std::shared_ptr<RH2D> &histimpl, ARGS...)
{
   return std::make_shared<RHistDrawable<2>>(histimpl);
}

template <class... ARGS>
inline auto GetDrawable(const std::shared_ptr<RH2I> &histimpl, ARGS...)
{
   return std::make_shared<RHistDrawable<2>>(histimpl);
}

template <class... ARGS>
inline auto GetDrawable(const std::shared_ptr<RH2C> &histimpl, ARGS...)
{
   return std::make_shared<RHistDrawable<2>>(histimpl);
}

template <class... ARGS>
inline auto GetDrawable(const std::shared_ptr<RH2F> &histimpl, ARGS...)
{
   return std::make_shared<RHistDrawable<2>>(histimpl);
}


template <class... ARGS>
inline auto GetDrawable(const std::shared_ptr<RH3D> &histimpl, ARGS...)
{
   return std::make_shared<RHistDrawable<3>>(histimpl);
}

template <class... ARGS>
inline auto GetDrawable(const std::shared_ptr<RH3I> &histimpl, ARGS...)
{
   return std::make_shared<RHistDrawable<3>>(histimpl);
}

template <class... ARGS>
inline auto GetDrawable(const std::shared_ptr<RH3C> &histimpl, ARGS...)
{
   return std::make_shared<RHistDrawable<3>>(histimpl);
}

template <class... ARGS>
inline auto GetDrawable(const std::shared_ptr<RH3F> &histimpl, ARGS...)
{
   return std::make_shared<RHistDrawable<3>>(histimpl);
}


namespace Internal {

void LoadHistPainterLibrary();

template <int DIMENSION>
class RHistPainterBase {
   static RHistPainterBase<DIMENSION> *&GetPainterPtr();

protected:
   RHistPainterBase();
   virtual ~RHistPainterBase();

public:
   static RHistPainterBase<DIMENSION> *GetPainter();

   /// Paint a RHist. All we need is access to its GetBinContent()
   virtual void Paint(RHistDrawable<DIMENSION> &obj, RPadPainter &pad) = 0;
};

extern template class RHistPainterBase<1>;
extern template class RHistPainterBase<2>;
extern template class RHistPainterBase<3>;

} // namespace Internal


} // namespace Experimental
} // namespace ROOT

#endif

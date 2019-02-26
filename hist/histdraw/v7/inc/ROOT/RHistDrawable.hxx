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

#include "ROOT/RDrawable.hxx"
#include "ROOT/RHistDrawingOpts.hxx"
#include "ROOT/RHist.hxx"
#include "ROOT/RHistImpl.hxx"
#include "ROOT/RMenuItem.hxx"

#include <memory>

namespace ROOT {
namespace Experimental {

template <int DIMENSIONS, class PRECISION, template <int D_, class P_> class... STAT>
class RHist;

namespace Detail {
template <int DIMENSIONS>
class RHistImplPrecisionAgnosticBase;
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
   virtual void Paint(RDrawable &obj, const RHistDrawingOpts<DIMENSION> &opts, RPadPainter &pad) = 0;
};

extern template class RHistPainterBase<1>;
extern template class RHistPainterBase<2>;
extern template class RHistPainterBase<3>;

} // namespace Internal

template <class DERIVED>
class RHistDrawableBase: public RDrawableBase<DERIVED> {
public:
   virtual ~RHistDrawableBase();

   void PopulateMenu(RMenuItems &) final;

   void Execute(const std::string &) final
   {
      // should execute menu item
   }
};

template <int DIMENSIONS>
class RHistDrawable final: public RHistDrawableBase<RHistDrawable<DIMENSIONS>> {
public:
   using HistImpl_t = Detail::RHistImplPrecisionAgnosticBase<DIMENSIONS>;

private:
   Internal::TUniWeakPtr<HistImpl_t> fHistImpl;
   RHistDrawingOpts<DIMENSIONS> fOpts;

public:
   RHistDrawable();

   template <class HIST>
   RHistDrawable(const std::shared_ptr<HIST> &hist, const RHistDrawingOpts<DIMENSIONS> &opts = {})
      : fHistImpl(std::shared_ptr<HistImpl_t>(hist, hist->GetImpl())), fOpts(opts)
   {}

   template <class HIST>
   RHistDrawable(std::unique_ptr<HIST> &&hist, const RHistDrawingOpts<DIMENSIONS> &opts = {})
      : fHistImpl(std::unique_ptr<HistImpl_t>(std::move(*hist).TakeImpl())), fOpts(opts)
   {}

   /// Paint the histogram
   void Paint(Internal::RPadPainter &pad) final;

   RHistDrawingOpts<DIMENSIONS> &GetOptions() { return fOpts; }
   const RHistDrawingOpts<DIMENSIONS> &GetOptions() const { return fOpts; }
};

extern template class RHistDrawableBase<RHistDrawable<1>>;
extern template class RHistDrawableBase<RHistDrawable<2>>;
extern template class RHistDrawableBase<RHistDrawable<3>>;

extern template class RHistDrawable<1>;
extern template class RHistDrawable<2>;
extern template class RHistDrawable<3>;


/// Interface to graphics taking a shared_ptr<RHist>.
template <int DIMENSIONS, class PRECISION, template <int D_, class P_> class... STAT>
std::shared_ptr<RHistDrawable<DIMENSIONS>>
GetDrawable(const std::shared_ptr<RHist<DIMENSIONS, PRECISION, STAT...>> &hist,
            const RHistDrawingOpts<DIMENSIONS> &opts = {})
{
   return std::make_unique<RHistDrawable<DIMENSIONS>>(hist, opts);
}

/// Interface to graphics taking a unique_ptr<RHist>.
template <int DIMENSIONS, class PRECISION, template <int D_, class P_> class... STAT>
std::shared_ptr<RHistDrawable<DIMENSIONS>>
GetDrawable(std::unique_ptr<RHist<DIMENSIONS, PRECISION, STAT...>> &&hist,
            const RHistDrawingOpts<DIMENSIONS> &opts = {})
{
   return std::make_unique<RHistDrawable<DIMENSIONS>>(std::move(hist), opts);
}

} // namespace Experimental
} // namespace ROOT

#endif

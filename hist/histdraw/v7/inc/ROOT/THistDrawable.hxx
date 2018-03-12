/// \file ROOT/THistDrawable.h
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

#ifndef ROOT7_THistDrawable
#define ROOT7_THistDrawable

#include "ROOT/TDrawable.hxx"
#include "ROOT/THistDrawingOpts.hxx"
#include "ROOT/THistImpl.hxx"
#include "ROOT/TMenuItem.hxx"

#include <memory>

namespace ROOT {
namespace Experimental {

template <int DIMENSIONS, class PRECISION,
          template <int D_, class P_, template <class P__> class STORAGE> class... STAT>
class THist;

namespace Detail {
template <int DIMENSIONS>
class THistImplPrecisionAgnosticBase;
}

namespace Internal {

void LoadHistPainterLibrary();

template <int DIMENSION>
class THistPainterBase {
   static THistPainterBase<DIMENSION> *&GetPainterPtr();

protected:
   THistPainterBase();
   virtual ~THistPainterBase();

public:
   static THistPainterBase<DIMENSION> *GetPainter();

   /// Paint a THist. All we need is access to its GetBinContent()
   virtual void Paint(TDrawable &obj, const THistDrawingOpts<DIMENSION> &opts, TPadPainter &pad) = 0;
};

extern template class THistPainterBase<1>;
extern template class THistPainterBase<2>;
extern template class THistPainterBase<3>;

} // namespace Internal

template <class DERIVED>
class THistDrawableBase: public TDrawableBase<DERIVED> {
public:
   virtual ~THistDrawableBase() = default;

   void PopulateMenu(TMenuItems &) final;

   void Execute(const std::string &) final
   {
      // should execute menu item
   }
};

template <int DIMENSIONS>
class THistDrawable final: public THistDrawableBase<THistDrawable<DIMENSIONS>> {
public:
   using HistImpl_t = Detail::THistImplPrecisionAgnosticBase<DIMENSIONS>;

private:
   Internal::TUniWeakPtr<HistImpl_t> fHistImpl;
   THistDrawingOpts<DIMENSIONS> fOpts;

public:
   THistDrawable();

   template <class HIST>
   THistDrawable(const std::shared_ptr<HIST> &hist, const THistDrawingOpts<DIMENSIONS> &opts = {})
      : fHistImpl(std::shared_ptr<HistImpl_t>(hist, hist->GetImpl())), fOpts(opts)
   {}

   template <class HIST>
   THistDrawable(std::unique_ptr<HIST> &&hist, const THistDrawingOpts<DIMENSIONS> &opts = {})
      : fHistImpl(std::unique_ptr<HistImpl_t>(std::move(hist->TakeImpl()))), fOpts(opts)
   {}

   /// Paint the histogram
   void Paint(Internal::TPadPainter &pad) final;

   THistDrawingOpts<DIMENSIONS> &GetOptions() { return fOpts; }
   const THistDrawingOpts<DIMENSIONS> &GetOptions() const { return fOpts; }
};

extern template class THistDrawableBase<THistDrawable<1>>;
extern template class THistDrawableBase<THistDrawable<2>>;
extern template class THistDrawableBase<THistDrawable<3>>;

extern template class THistDrawable<1>;
extern template class THistDrawable<2>;
extern template class THistDrawable<3>;

} // namespace Experimental
} // namespace ROOT

#endif

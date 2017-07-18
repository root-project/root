/// \file ROOT/THistDrawable.h
/// \ingroup Hist ROOT7
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
#include "ROOT/THistDrawOptions.hxx"
#include "ROOT/TLogger.hxx"

// TODO: from Sergey Linev: should libHist depend from libGpad?
//#include "ROOT/TMenuItem.hxx"

#include <memory>

class TH1;

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
   static THistPainterBase<DIMENSION> *fgPainter;

protected:
   THistPainterBase() { fgPainter = this; }
   virtual ~THistPainterBase();

public:
   static THistPainterBase<DIMENSION> *GetPainter()
   {
      if (!fgPainter) LoadHistPainterLibrary();
      return fgPainter;
   }

   /// Paint a THist. All we need is access to its GetBinContent()
   virtual void Paint(TDrawable &obj, THistDrawOptions<DIMENSION> opts, TVirtualCanvasPainter &canv) = 0;
};

extern template class THistPainterBase<1>;
extern template class THistPainterBase<2>;
extern template class THistPainterBase<3>;

class THistDrawableBase : public TDrawable {
protected:
   std::unique_ptr<TH1> fOldHist;

public:
   TH1 *GetOldHist() const { return fOldHist.get(); }

   THistDrawableBase();
   THistDrawableBase(THistDrawableBase &&);
   virtual ~THistDrawableBase();

   THistDrawableBase &operator=(THistDrawableBase &&);

   void PopulateMenu(TMenuItems &) final;

   void Execute(const std::string &) final
   {
      // should execute menu item
   }
};

template <int DIMENSIONS>
class THistDrawable final : public THistDrawableBase {
public:
   using HistImpl_t = Detail::THistImplPrecisionAgnosticBase<DIMENSIONS>;

private:
   TUniWeakPtr<HistImpl_t> fHistImpl;
   THistDrawOptions<DIMENSIONS> fOpts;

   bool UpdateOldHist();

public:
   template <class HIST>
   THistDrawable(const std::shared_ptr<HIST> &hist, THistDrawOptions<DIMENSIONS> opts)
      : fHistImpl(std::shared_ptr<HistImpl_t>(hist, hist->GetImpl())), fOpts(opts)
   {
   }

   template <class HIST>
   THistDrawable(std::unique_ptr<HIST> &&hist, THistDrawOptions<DIMENSIONS> opts)
      : fHistImpl(std::unique_ptr<HistImpl_t>(std::move(hist->TakeImpl()))), fOpts(opts)
   {
   }

   /// Paint the histogram
   void Paint(TVirtualCanvasPainter &canv) final
   {
      if (UpdateOldHist()) THistPainterBase<DIMENSIONS>::GetPainter()->Paint(*this, fOpts, canv);
   }
};

extern template class THistDrawable<1>;
extern template class THistDrawable<2>;
extern template class THistDrawable<3>;

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif

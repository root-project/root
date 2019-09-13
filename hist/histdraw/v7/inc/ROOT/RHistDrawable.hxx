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

template <int DIMENSIONS>
class RHistDrawable final: public RDrawable {
public:
   using HistImpl_t = Detail::RHistImplPrecisionAgnosticBase<DIMENSIONS>;

private:
   Internal::RIOShared<HistImpl_t> fHistImpl;

public:
   RHistDrawable();

   void PopulateMenu(RMenuItems &) final
   {
      // populate menu
   }

   void Execute(const std::string &) final
   {
      // should execute menu item
   }


   template <class HIST>
   RHistDrawable(const std::shared_ptr<HIST> &hist)
      : fHistImpl(std::shared_ptr<HistImpl_t>(hist, hist->GetImpl()))
   {}

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

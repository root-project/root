/// \file ROOT/THistDrawable.h
/// \ingroup Hist ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_THistDrawable
#define ROOT7_THistDrawable

#include "ROOT/TDrawable.h"
#include "ROOT/THistDrawOptions.h"
#include "ROOT/TLogger.h"

#include <memory>

namespace ROOT {
namespace Experimental {

template<int DIMENSIONS, class PRECISION,
  template <int D_, class P_, template <class P__> class STORAGE> class... STAT>
class THist;

namespace Internal {

void LoadHistPainterLibrary();

template <int DIMENSION>
class THistPainterBase {
  static THistPainterBase<DIMENSION>* fgPainter;

protected:
  THistPainterBase() { fgPainter = this; }
  virtual ~THistPainterBase();

public:
  static THistPainterBase<DIMENSION>* GetPainter() {
    if (!fgPainter)
      LoadHistPainterLibrary();
    return fgPainter;
  }

  /// Paint a THist. All we need is access to its GetBinContent()
  virtual void Paint(TDrawable& obj, THistDrawOptions<DIMENSION> opts) = 0;
};

extern template class THistPainterBase<1>;
extern template class THistPainterBase<2>;
extern template class THistPainterBase<3>;

template<int DIMENSIONS, class PRECISION,
  template <int D_, class P_, template <class P__> class STORAGE> class... STAT>
class THistDrawable final: public TDrawable {
public:
  using Hist_t = THist<DIMENSIONS, PRECISION, STAT...>;
private:
  std::weak_ptr<Hist_t> fHist;
  THistDrawOptions<DIMENSIONS> fOpts;

public:
  THistDrawable(std::weak_ptr<Hist_t> hist,
                THistDrawOptions<DIMENSIONS> opts): fHist(hist), fOpts(opts) {}

  ~THistDrawable() = default;

  /// Paint the histogram
  void Paint() final {
    THistPainterBase<DIMENSIONS>::GetPainter()->Paint(*this, fOpts);
  }
};

} // namespace Internal
} // namespace Experimental
} // namespace ROOT

#endif

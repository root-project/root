/// \file ROOT/THistDrawable.h
/// \ingroup Hist
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-09

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_THistDrawable
#define ROOT7_THistDrawable

#include "ROOT/TCoopPtr.h"
#include "ROOT/TDrawable.h"
#include "ROOT/THistDrawOptions.h"
#include "ROOT/TLogger.h"

#include "TSystem.h"

#include <memory>

namespace ROOT {

template<int DIMENSIONS, class PRECISION> class THist;

namespace Internal {

template <int DIMENSION>
class THistPainterBase {
  static THistPainterBase<DIMENSION>* fgPainter;

protected:
  THistPainterBase() { fgPainter = this; }
  ~THistPainterBase() { fgPainter = nullptr; }

public:
  static THistPainterBase<DIMENSION>* GetPainter() {
    if (!fgPainter)
      gSystem->Load("libHistPainter");
    return fgPainter;
  }

  /// Paint a THist. All we need is access to its GetBinContent()
  virtual void Paint(TDrawable& obj, THistDrawOptions<DIMENSION> opts) = 0;
};

template <int DIMENSION>
THistPainterBase<DIMENSION>* THistPainterBase<DIMENSION>::fgPainter = 0;


template <int DIMENSION, class PRECISION>
class THistDrawable final: public TDrawable {
private:
  TCoopPtr<THist<DIMENSION, PRECISION>> fHist;
  THistDrawOptions<DIMENSION> fOpts;

public:
  THistDrawable(TCoopPtr<THist<DIMENSION, PRECISION>> hist,
                THistDrawOptions<DIMENSION> opts): fHist(hist), fOpts(opts) {}

  ~THistDrawable() = default;

  /// Paint the histogram
  void Paint() final {
    THistPainterBase<DIMENSION>::GetPainter()->Paint(*this, fOpts);
  }
};

} // namespace Internal
} // namespace ROOT

#endif

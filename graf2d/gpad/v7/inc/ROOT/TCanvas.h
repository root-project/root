/// \file TCanvas.h
/// \ingroup Gpad
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-08

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TCanvas
#define ROOT7_TCanvas

#include <experimental/string_view>
#include <vector>

#include "ROOT/TCoopPtr.h"
#include "ROOT/TDrawable.h"

namespace ROOT {

/** \class TCanvas
  Graphic container for `TDrawable`-s.
  */

class TCanvas {
  std::vector<std::unique_ptr<Internal::TDrawable>> fPrimitives;

  /// We need to keep track of canvases; please use Create()
  TCanvas() = default;

public:
  static TCoopPtr<TCanvas> Create();
  static TCoopPtr<TCanvas> Create(std::experimental::string_view name);

  /// Add a something to be painted. The pad claims shared ownership.
  template <class T>
  void Draw(TCoopPtr<T> what) {
    // Requires GetDrawable(what, options) to be known!
    fPrimitives.emplace_back(GetDrawable(what));
  }

  /// Add a something to be painted, with options. The pad claims shared ownership.
  template <class T, class OPTIONS>
  void Draw(TCoopPtr<T> what, const OPTIONS& options) {
    // Requires GetDrawable(what, options) to be known!
    fPrimitives.emplace_back(GetDrawable(what, options));
  }

  void Paint();

  static const std::vector<TCoopPtr<TCanvas>>& GetCanvases();
};

}

#endif

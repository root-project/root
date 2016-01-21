/// \file TCanvas.h
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-08
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!

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
#include <memory>
#include <vector>

#include "ROOT/TDrawable.h"

namespace ROOT {
namespace Experimental {

/** \class TCanvas
  Graphic container for `TDrawable`-s.
  */

class TCanvas {
  std::vector<std::unique_ptr<Internal::TDrawable>> fPrimitives;

  /// We need to keep track of canvases; please use Create()
  TCanvas() = default;

public:
  static std::weak_ptr<TCanvas> Create();
  static std::weak_ptr<TCanvas> Create(std::experimental::string_view name);

  /// Add a something to be painted. The pad claims shared ownership.
  template <class T>
  void Draw(std::shared_ptr<T> what) {
    // Requires GetDrawable(what, options) to be known!
    fPrimitives.emplace_back(GetDrawable(what));
  }

  /// Add a something to be painted, with options. The pad claims shared ownership.
  template <class T, class OPTIONS>
  void Draw(std::shared_ptr<T> what, const OPTIONS& options) {
    // Requires GetDrawable(what, options) to be known!
    fPrimitives.emplace_back(GetDrawable(what, options));
  }

  void Paint();

  static const std::vector<std::weak_ptr<TCanvas>>& GetCanvases();
};

} // namespace Experimental
} // namespace ROOT

#endif

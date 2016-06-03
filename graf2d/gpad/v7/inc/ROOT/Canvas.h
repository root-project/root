/// \file ROOT/Canvas.h
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

namespace Internal {
class TCanvasSharedPtrMaker;
}

class TCanvasPtr;

/** \class ROOT::Experimental::TCanvas
  Graphic container for `TDrawable`-s.
  Access is through TCanvasPtr.
  */

class TCanvas {
  std::vector <std::unique_ptr<Internal::TDrawable>> fPrimitives;

  /// We need to keep track of canvases; please use Create()
  TCanvas() = default;

  /// Private class used to construct a shared_ptr.
  friend class Internal::TCanvasSharedPtrMaker;

public:
  static TCanvasPtr Create();
  static TCanvasPtr Create(std::experimental::string_view name);

  /// Add a something to be painted. The pad claims shared ownership.
  template<class T>
  void Draw(std::shared_ptr <T> what) {
    // Requires GetDrawable(what, options) to be known!
    fPrimitives.emplace_back(GetDrawable(what));
  }

  /// Add a something to be painted, with options. The pad claims shared ownership.
  template<class T, class OPTIONS>
  void Draw(std::shared_ptr <T> what, const OPTIONS &options) {
    // Requires GetDrawable(what, options) to be known!
    fPrimitives.emplace_back(GetDrawable(what, options));
  }

  void Paint();

  static const std::vector <std::weak_ptr<TCanvas>> &GetCanvases();
};


/**
 \class TCanvasPtr
 Points to a TCanvas. Canvases are resources managed by ROOT; access is
 restricted to TCanvasPtr.
 */

class TCanvasPtr {
private:
  std::shared_ptr<TCanvas> fCanvas;

  /// Constructed by Create etc.
  TCanvasPtr(std::shared_ptr<TCanvas>&& canvas): fCanvas(std::move(canvas)) {}

  friend class TCanvas;

public:
  /// Dereference the file pointer, giving access to the TCanvas object.
  TCanvas* operator ->() { return fCanvas.get(); }

  /// Dereference the file pointer, giving access to the TCanvas object.
  /// const overload.
  const TCanvas* operator ->() const { return fCanvas.get(); }

  /// Check the validity of the file pointer.
  operator bool() const { return fCanvas.get(); }
};


} // namespace Experimental
} // namespace ROOT

#endif

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
#include <string>
#include <vector>

#include "ROOT/TDrawable.h"

namespace ROOT {
namespace Experimental {

namespace Internal {
class TCanvasSharedPtrMaker;
class TV5CanvasAdaptor;
}

/** \class ROOT::Experimental::TCanvas
  Graphic container for `TDrawable`-s.
  Access is through TCanvasPtr.
  */

class TCanvas {
public:
  using Primitives_t = std::vector<std::unique_ptr<Internal::TDrawable>>;

private:
  /// Content of the pad.
  Primitives_t fPrimitives;

  /// Title of the canvas.
  std::string fTitle;

  /// Adaptor for painting an old canvas.
  std::unique_ptr<Internal::TV5CanvasAdaptor> fAdaptor;

  /// Disable copy construction for now.
  TCanvas(const TCanvas&) = delete;

  /// Disable assignment for now.
  TCanvas& operator=(const TCanvas&) = delete;

public:
  static std::shared_ptr<TCanvas> Create(const std::string& title);

  /// Create a temporary Canvas; for long-lived ones please use Create().
  TCanvas();

  /// Default destructor.
  ///
  /// Outline the implementation in sources.
  ~TCanvas();

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

  /// Remove an object from the list of primitives.
  //TODO: void Wipe();

  /// Paint the canvas elements ("primitives").
  void Paint();

  /// Get the canvas's title.
  const std::string& GetTitle() const { return fTitle; }

  /// Set the canvas's title.
  void SetTitle(const std::string& title) { fTitle = title; }

  /// Get the elements contained in the canvas.
  const Primitives_t& GetPrimitives();


  static const std::vector<std::shared_ptr<TCanvas>> &GetCanvases();
};

} // namespace Experimental
} // namespace ROOT

#endif

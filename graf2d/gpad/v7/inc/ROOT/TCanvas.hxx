/// \file ROOT/TCanvas.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2015-07-08
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

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

#include "ROOT/TDrawable.hxx"
#include "ROOT/TypeTraits.hxx"
#include "ROOT/TVirtualCanvasPainter.hxx"

namespace ROOT {
namespace Experimental {

namespace Internal {
class TCanvasSharedPtrMaker;
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

   /// If canvas modified.
   bool fModified;

   /// The painter of this canvas, bootstrapping the graphics connection.
   /// Unmapped canvases (those that never had `Draw()` invoked) might not have
   /// a painter.
   std::unique_ptr<Internal::TVirtualCanvasPainter> fPainter;

   /// Disable copy construction for now.
   TCanvas(const TCanvas &) = delete;

   /// Disable assignment for now.
   TCanvas &operator=(const TCanvas &) = delete;

public:
   static std::shared_ptr<TCanvas> Create(const std::string &title);

   /// Create a temporary TCanvas; for long-lived ones please use Create().
   TCanvas() = default;

   /// Default destructor.
   ~TCanvas() = default;

   // TODO: Draw() should return the Drawable&.
   /// Add something to be painted.
   /// The pad observes what's lifetime through a weak pointer.
   template <class T>
   void Draw(const std::shared_ptr<T> &what)
   {
      // Requires GetDrawable(what, options) to be known!
      fPrimitives.emplace_back(GetDrawable(what));
   }

   /// Add something to be painted, with options.
   /// The pad observes what's lifetime through a weak pointer.
   template <class T, class OPTIONS>
   void Draw(const std::shared_ptr<T> &what, const OPTIONS &options)
   {
      // Requires GetDrawable(what, options) to be known!
      fPrimitives.emplace_back(GetDrawable(what, options));
   }

   /// Add something to be painted. The pad claims ownership.
   template <class T>
   void Draw(std::unique_ptr<T> &&what)
   {
      // Requires GetDrawable(what, options) to be known!
      fPrimitives.emplace_back(GetDrawable(std::move(what)));
   }

   /// Add something to be painted, with options. The pad claims ownership.
   template <class T, class OPTIONS>
   void Draw(std::unique_ptr<T> &&what, const OPTIONS &options)
   {
      // Requires GetDrawable(what, options) to be known!
      fPrimitives.emplace_back(GetDrawable(std::move(what), options));
   }

   /// Add a copy of something to be painted.
   template <class T, class = typename std::enable_if<!ROOT::TypeTraits::IsSmartOrDumbPtr<T>::value>::type>
   void Draw(const T &what)
   {
      // Requires GetDrawable(what, options) to be known!
      fPrimitives.emplace_back(GetDrawable(std::make_unique<T>(what)));
   }

   /// Add a copy of something to be painted, with options.
   template <class T, class OPTIONS,
             class = typename std::enable_if<!ROOT::TypeTraits::IsSmartOrDumbPtr<T>::value>::type>
   void Draw(const T &what, const OPTIONS &options)
   {
      // Requires GetDrawable(what, options) to be known!
      fPrimitives.emplace_back(GetDrawable(std::make_unique<T>(what), options));
   }

   /// Remove an object from the list of primitives.
   // TODO: void Wipe();

   void Modified() { fModified = true; }

   /// Actually display the canvas.
   void Show();

   /// Save canvas in image file
   void SaveAs(const std::string &filename);

   /// update drawing
   void Update();

   /// Get the canvas's title.
   const std::string &GetTitle() const { return fTitle; }

   /// Set the canvas's title.
   void SetTitle(const std::string &title) { fTitle = title; }

   /// Get the elements contained in the canvas.
   const Primitives_t &GetPrimitives() const { return fPrimitives; }

   static const std::vector<std::shared_ptr<TCanvas>> &GetCanvases();
};

} // namespace Experimental
} // namespace ROOT

#endif

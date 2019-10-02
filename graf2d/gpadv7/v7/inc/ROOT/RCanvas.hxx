/// \file ROOT/RCanvas.hxx
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

#ifndef ROOT7_RCanvas
#define ROOT7_RCanvas

#include "ROOT/RPadBase.hxx"
#include "ROOT/RVirtualCanvasPainter.hxx"

#include <memory>
#include <string>
#include <vector>

namespace ROOT {
namespace Experimental {

/** \class ROOT::Experimental::RCanvas
  A window's topmost `RPad`.
  */

class RCanvas: public RPadBase {
friend class RPadBase;  /// use for ID generation
private:
   /// Title of the canvas.
   std::string fTitle;

   /// Size of the canvas in pixels,
   std::array<RPadLength::Pixel, 2> fSize;

   /// Modify counter, incremented every time canvas is changed
   uint64_t fModified{0}; ///<!

   uint64_t fIdCounter{2};   ///< counter for objects, id==1 is canvas itself

   /// The painter of this canvas, bootstrapping the graphics connection.
   /// Unmapped canvases (those that never had `Draw()` invoked) might not have
   /// a painter.
   std::unique_ptr<Internal::RVirtualCanvasPainter> fPainter; ///<!

   /// Disable copy construction for now.
   RCanvas(const RCanvas &) = delete;

   /// Disable assignment for now.
   RCanvas &operator=(const RCanvas &) = delete;

   std::string GenerateUniqueId();

public:
   static std::shared_ptr<RCanvas> Create(const std::string &title);

   /// Create a temporary RCanvas; for long-lived ones please use Create().
   RCanvas() = default;

   ~RCanvas() = default;

   const RCanvas *GetCanvas() const override { return this; }

   /// Access to the top-most canvas, if any (non-const version).
   RCanvas *GetCanvas() override { return this; }

   /// Return canvas pixel size as array with two elements - width and height
   const std::array<RPadLength::Pixel, 2> &GetSize() const { return fSize; }

   /// Set canvas pixel size as array with two elements - width and height
   RCanvas &SetSize(const std::array<RPadLength::Pixel, 2> &sz)
   {
      fSize = sz;
      return *this;
   }

   /// Set canvas pixel size - width and height
   RCanvas &SetSize(const RPadLength::Pixel &width, const RPadLength::Pixel &height)
   {
      fSize[0] = width;
      fSize[1] = height;
      return *this;
   }

   /// Display the canvas.
   void Show(const std::string &where = "");

   /// Hide all canvas displays
   void Hide();

   /// Remove canvas from global canvas lists, will be destroyed when shared_ptr will be removed
   void Remove();

   /// Insert panel into the canvas, canvas should be shown at this moment
   template <class PANEL>
   bool AddPanel(std::shared_ptr<PANEL> &panel)
   {
      if (!fPainter) return false;
      return fPainter->AddPanel(panel->GetWindow());
   }

   // Indicates that primitives list was changed or any primitive was modified
   void Modified() { fModified++; }

   // Return if canvas was modified and not yet updated
   bool IsModified() const;

   /// update drawing
   void Update(bool async = false, CanvasCallback_t callback = nullptr);

   /// Run canvas functionality for given time (in seconds)
   void Run(double tm = 0.);

   /// Save canvas in image file
   void SaveAs(const std::string &filename, bool async = false, CanvasCallback_t callback = nullptr);

   /// Get the canvas's title.
   const std::string &GetTitle() const { return fTitle; }

   /// Set the canvas's title.
   RCanvas &SetTitle(const std::string &title)
   {
      fTitle = title;
      return *this;
   }

   void ResolveSharedPtrs();

   /// Convert a `Pixel` position to Canvas-normalized positions.
   std::array<RPadLength::Normal, 2> PixelsToNormal(const std::array<RPadLength::Pixel, 2> &pos) const final
   {
      return {{pos[0] / fSize[0], pos[1] / fSize[1]}};
   }

   static const std::vector<std::shared_ptr<RCanvas>> GetCanvases();
};

} // namespace Experimental
} // namespace ROOT

#endif

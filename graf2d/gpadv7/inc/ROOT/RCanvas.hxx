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
#include "ROOT/RDrawableRequest.hxx"

#include <memory>
#include <string>
#include <vector>
#include <list>

namespace ROOT {
namespace Experimental {

class RChangeAttrRequest : public RDrawableRequest {
   std::vector<std::string> ids;                           ///< array of ids
   std::vector<std::string> names;                         ///< array of attribute names
   std::vector<std::unique_ptr<RAttrMap::Value_t>> values; ///< array of values
   bool update{true};                                      ///< update canvas at the end
   bool fNeedUpdate{false};       ///<! is canvas update required
   RChangeAttrRequest(const RChangeAttrRequest &) = delete;
   RChangeAttrRequest& operator=(const RChangeAttrRequest &) = delete;
public:
   RChangeAttrRequest() = default; // for I/O
   virtual ~RChangeAttrRequest() = default;
   std::unique_ptr<RDrawableReply> Process() override;
   bool NeedCanvasUpdate() const override { return fNeedUpdate; }
};

/** \class RCanvas
\ingroup GpadROOT7
\brief A window's topmost `RPad`.
\author Axel Naumann <axel@cern.ch>
\date 2015-07-08
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RCanvas: public RPadBase {
friend class RPadBase;  /// use for ID generation
friend class RCanvasPainter; /// used for primitives drawing
friend class RChangeAttrRequest; /// to apply attributes changes
private:
   /// Title of the canvas.
   std::string fTitle;

   /// Width of the canvas in pixels
   int fWidth{0};

   /// Height of the canvas in pixels
   int fHeight{0};

   /// Modify counter, incremented every time canvas is changed
   Version_t fModified{1}; ///<!

   /// The painter of this canvas, bootstrapping the graphics connection.
   /// Unmapped canvases (those that never had `Draw()` invoked) might not have
   /// a painter.
   std::unique_ptr<Internal::RVirtualCanvasPainter> fPainter; ///<!

   /// indicate if Show() method was called before
   bool fShown{false}; ///<!

   /// Disable copy construction for now.
   RCanvas(const RCanvas &) = delete;

   /// Disable assignment for now.
   RCanvas &operator=(const RCanvas &) = delete;

   // Increment modify counter
   uint64_t IncModified() { return ++fModified; }

public:
   static std::shared_ptr<RCanvas> Create(const std::string &title);

   /// Create a temporary RCanvas; for long-lived ones please use Create().
   RCanvas() : RPadBase("canvas") {}

   ~RCanvas() = default;

   const RCanvas *GetCanvas() const override { return this; }

   /// Access to the top-most canvas, if any (non-const version).
   RCanvas *GetCanvas() override { return this; }

   /// Set canvas pixel size - width and height
   void SetSize(int width, int height)
   {
      fWidth = width;
      fHeight = height;
   }

   /// Set canvas width
   void SetWidth(int width) { fWidth = width; }

   /// Set canvas height
   void SetHeight(int height) { fHeight = height; }

   /// Get canvas width
   int GetWidth() const { return fWidth; }

   /// Get canvas height
   int GetHeight() const { return fHeight; }

   /// Display the canvas.
   void Show(const std::string &where = "");

   bool IsShown() const { return fShown; }
   void ClearShown() { fShown = false; }

   /// Returns window name used to display canvas
   std::string GetWindowAddr() const;

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

   /// Get modify counter
   uint64_t GetModified() const { return fModified; }

   // Set newest version to all primitives
   void Modified() { SetDrawableVersion(IncModified()); }

   /// Set newest version to specified drawable
   void Modified(std::shared_ptr<RDrawable> drawable)
   {
      // TODO: may be check that drawable belong to the canvas
      if (drawable)
         drawable->SetDrawableVersion(IncModified());
   }

   // Return if canvas was modified and not yet updated
   bool IsModified() const;

   /// update drawing
   void Update(bool async = false, CanvasCallback_t callback = nullptr);

   /// Run canvas functionality for given time (in seconds)
   void Run(double tm = 0.);

   /// Save canvas in image file
   bool SaveAs(const std::string &filename);

   /// Provide JSON which can be used for offline display
   std::string CreateJSON();

   /// Get the canvas's title.
   const std::string &GetTitle() const { return fTitle; }

   /// Set the canvas's title.
   RCanvas &SetTitle(const std::string &title)
   {
      fTitle = title;
      return *this;
   }

   void ResolveSharedPtrs();

   static const std::vector<std::shared_ptr<RCanvas>> GetCanvases();

   static void ReleaseHeldCanvases();
};

} // namespace Experimental
} // namespace ROOT

#endif

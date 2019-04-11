/// \file ROOT/RPadDisplayItem.hxx
/// \ingroup Base ROOT7
/// \author Sergey Linev
/// \date 2017-05-31
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2017, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RPadDisplayItem
#define ROOT7_RPadDisplayItem

#include <ROOT/RDisplayItem.hxx>
#include <ROOT/RFrame.hxx>
#include <ROOT/RPad.hxx>

namespace ROOT {
namespace Experimental {

/// Display item for the pad
/// Includes different graphical properties of the pad itself plus
/// list of created items for all primitives

class RPadDisplayItem : public RDisplayItem {
public:
   // list of snapshot for primitives in pad
   using PadPrimitives_t = std::vector<std::unique_ptr<RDisplayItem>>;


protected:
   const RFrame *fFrame{nullptr};             ///< temporary pointer on frame object
   const RPad::DrawingOpts *fDrawOpts{nullptr}; ///< temporary pointer on pad drawing options
   const RPadExtent *fSize{nullptr};          ///< temporary pointer on pad size attributes
   std::string fTitle;                        ///< title of the pad (used for canvas)
   std::array<RPadLength::Pixel, 2> fWinSize; ///< window size (used for canvas)
   PadPrimitives_t fPrimitives;               ///< display items for all primitives in the pad
public:
   RPadDisplayItem() = default;
   virtual ~RPadDisplayItem() {}
   void SetFrame(const RFrame *f) { fFrame = f; }
   void SetDrawOpts(const RPad::DrawingOpts *opts) { fDrawOpts = opts; }
   void SetSize(const RPadExtent *sz) { fSize = sz; }
   void SetTitle(const std::string &title) { fTitle = title; }
   void SetWindowSize(const std::array<RPadLength::Pixel, 2> &win) { fWinSize = win; }
   PadPrimitives_t &GetPrimitives() { return fPrimitives; }
   void Add(std::unique_ptr<RDisplayItem> &&item) { fPrimitives.push_back(std::move(item)); }
   void Clear()
   {
      fPrimitives.clear();
      fFrame = nullptr;
      fDrawOpts = nullptr;
      fSize = nullptr;
      fWinSize[0] = fWinSize[1] = 0;
      fTitle.clear();
   }
};

} // Experimental
} // ROOT

#endif

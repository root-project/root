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
   const RDrawingAttr *fAttr{nullptr}; ///< temporary pointer on attributes
   const RPadPos *fPos{nullptr};              ///< pad position
   const RPadExtent *fSize{nullptr};          ///< pad size
   std::string fTitle;                        ///< title of the pad (used for canvas)
   std::array<int, 2> fWinSize;               ///< window size (used for canvas)
   PadPrimitives_t fPrimitives;               ///< display items for all primitives in the pad
public:
   RPadDisplayItem() = default;
   virtual ~RPadDisplayItem() {}
   void SetFrame(const RFrame *f) { fFrame = f; }
   void SetAttributes(const RDrawingAttr *f) { fAttr = f; }
   void SetPadPosSize(const RPadPos *pos, const RPadExtent *size) { fPos = pos; fSize = size; }
   void SetTitle(const std::string &title) { fTitle = title; }
   void SetWindowSize(const std::array<RPadLength::Pixel, 2> &win) { fWinSize[0] = (int) win[0].fVal; fWinSize[1] = (int) win[1].fVal; }
   PadPrimitives_t &GetPrimitives() { return fPrimitives; }
   void Add(std::unique_ptr<RDisplayItem> &&item) { fPrimitives.push_back(std::move(item)); }
   void Clear()
   {
      fFrame = nullptr;
      fAttr = nullptr;
      fPos = nullptr;
      fSize = nullptr;
      fTitle.clear();
      fWinSize[0] = fWinSize[1] = 0;
      fPrimitives.clear();
   }
};

} // Experimental
} // ROOT

#endif

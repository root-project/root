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


/** class RPadBaseDisplayItem
\ingroup BaseROOT7
\brief Display item for the RPadBase
Includes primitives and frames
\author Sergey Linev
\date 2017-05-31
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RPadBaseDisplayItem : public RDisplayItem {
public:
   // list of snapshot for primitives in pad
   using PadPrimitives_t = std::vector<std::unique_ptr<RDisplayItem>>;

protected:
   const RFrame *fFrame{nullptr};       ///< temporary pointer on frame object
   const RAttrMap *fAttr{nullptr};      ///< temporary pointer on attributes
   PadPrimitives_t fPrimitives;         ///< display items for all primitives in the pad
public:
   RPadBaseDisplayItem() = default;
   virtual ~RPadBaseDisplayItem() = default;
   void SetFrame(const RFrame *f) { fFrame = f; }
   void SetAttributes(const RAttrMap *f) { fAttr = f; }
   PadPrimitives_t &GetPrimitives() { return fPrimitives; }
   void Add(std::unique_ptr<RDisplayItem> &&item) { fPrimitives.push_back(std::move(item)); }
};


/// Display item for the pad
/// Includes pad graphical properties

class RPadDisplayItem : public RPadBaseDisplayItem {

protected:
   const RPadPos *fPos{nullptr};        ///< pad position
   const RPadExtent *fSize{nullptr};    ///< pad size
public:
   RPadDisplayItem() = default;
   virtual ~RPadDisplayItem() {}
   void SetPadPosSize(const RPadPos *pos, const RPadExtent *size) { fPos = pos; fSize = size; }

   void BuildFullId(const std::string &prefix) override
   {
      RDisplayItem::BuildFullId(prefix);
      std::string subprefix = prefix + std::to_string(GetIndex()) + "_";
      for (auto &item : fPrimitives)
         item->BuildFullId(subprefix);
   }
};

/// Display item for the canvas
/// Includes canvas properties

class RCanvasDisplayItem : public RPadBaseDisplayItem {

protected:
   std::string fTitle;                  ///< title of the pad (used for canvas)
   std::array<int, 2> fWinSize;         ///< window size (used for canvas)
public:
   RCanvasDisplayItem() = default;
   virtual ~RCanvasDisplayItem() = default;
   void SetTitle(const std::string &title) { fTitle = title; }
   void SetWindowSize(const std::array<RPadLength::Pixel, 2> &win) { fWinSize[0] = (int) win[0].fVal; fWinSize[1] = (int) win[1].fVal; }

   void BuildFullId(const std::string &prefix) override
   {
      for (auto &item : fPrimitives)
         item->BuildFullId(prefix);
   }
};



} // Experimental
} // ROOT

#endif

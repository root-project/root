/// \file ROOT/TDisplayItem.h
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

#ifndef ROOT7_TPadDisplayItem
#define ROOT7_TPadDisplayItem

#include <ROOT/TDisplayItem.hxx>

#include <ROOT/TFrame.hxx>

#include <ROOT/TPad.hxx>

namespace ROOT {
namespace Experimental {

// list of snapshot for primitives in pad

using TDisplayItemsVector = std::vector<std::unique_ptr<TDisplayItem>>;

/// Display item for the pad
/// Includes different graphical properties of the pad itself plus
/// list of created items for all primitives

class TPadDisplayItem : public TDisplayItem {
protected:
   const TFrame *fFrame{nullptr};   ///< temporary pointer on frame object
   const TPadDrawingOpts *fDrawOpts{nullptr}; ///< temporary pointer on pad drawing options
   const TPadExtent *fSize{nullptr};  ///< temporary pointer on pad size attributes
   TDisplayItemsVector fPrimitives; ///< display items for all primitives in the pad
public:
   TPadDisplayItem() = default;
   virtual ~TPadDisplayItem() {}
   void SetFrame(const TFrame *f) { fFrame = f; }
   void SetDrawOpts(const TPadDrawingOpts *opts) { fDrawOpts = opts; }
   void SetSize(const TPadExtent *sz) { fSize = sz; }
   TDisplayItemsVector &GetPrimitives() { return fPrimitives; }
   void Add(std::unique_ptr<TDisplayItem> &&item) { fPrimitives.push_back(std::move(item)); }
   void Clear()
   {
      fPrimitives.clear();
      fFrame = nullptr;
      fDrawOpts = nullptr;
      fSize = nullptr;
   }
};

} // Experimental
} // ROOT

#endif

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

namespace ROOT {
namespace Experimental {

// list of snapshot for primitives in pad

using TDisplayItemsVector = std::vector<std::unique_ptr<TDisplayItem>>;

class TPadDisplayItem : public TDisplayItem {
protected:
   const TFrame *fFrame{nullptr};   ///< temporary pointer on frame object
   TDisplayItemsVector fPrimitives; ///< display items for all primitives in the pad
public:
   TPadDisplayItem() = default;
   virtual ~TPadDisplayItem() {}
   void SetFrame(const TFrame *f) { fFrame = f; }
   TDisplayItemsVector &GetPrimitives() { return fPrimitives; }
   void Add(std::unique_ptr<TDisplayItem> &&item) { fPrimitives.push_back(std::move(item)); }
   TDisplayItem *Last() const { return fPrimitives.back().get(); }
   void Clear() { fPrimitives.clear(); }
};

} // Experimental
} // ROOT

#endif

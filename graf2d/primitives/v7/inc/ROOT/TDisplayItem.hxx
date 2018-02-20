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

#ifndef ROOT7_TDisplayItem
#define ROOT7_TDisplayItem

#include <string>
#include <memory>
#include <vector>

namespace ROOT {
namespace Experimental {
class TCanvas;
class TFrame;

/** \class TDisplayItem
  Base class for painting data for JS.
  */

class TDisplayItem {
protected:
   std::string fObjectID;   ///< unique object identifier

public:
   TDisplayItem() = default;
   TDisplayItem(const TDisplayItem &rhs);
   virtual ~TDisplayItem();

   void SetObjectIDAsPtr(void *ptr);
   void SetObjectID(const std::string &id) { fObjectID = id; }
   std::string GetObjectID() const { return fObjectID; }

   static std::string MakeIDFromPtr(void *ptr);
};

// list of snapshot for primitives in pad

class TPadDisplayItem : public TDisplayItem {
protected:
   const TFrame *fFrame{nullptr};               ///< temporary pointer on frame object
   std::vector<TDisplayItem *> fPrimitives;
public:
   TPadDisplayItem() = default;
   virtual ~TPadDisplayItem();
   void SetFrame(const TFrame *f) { fFrame = f; }
   void Add(TDisplayItem *snap) { fPrimitives.push_back(snap); }
   TDisplayItem *Last() const { return fPrimitives[fPrimitives.size() - 1]; }
   void Clear();
};

// direct pointer to some object without ownership

template <class T>
class TOrdinaryDisplayItem : public TDisplayItem {
protected:
   const T *fObject;

public:
   TOrdinaryDisplayItem(const T *addr) : TDisplayItem(), fObject(addr) {}
   TOrdinaryDisplayItem(const TOrdinaryDisplayItem<T> &&rhs) : TDisplayItem(rhs), fObject(rhs.fObject) {}
   virtual ~TOrdinaryDisplayItem() {}

   const T *GetObject() const { return fObject; }
};

// unique pointer of specified class with ownership

template <class T>
class TUniqueDisplayItem : public TDisplayItem {
protected:
   std::unique_ptr<T> fObject;

public:
   TUniqueDisplayItem(T *addr) : TDisplayItem(), fObject(addr) {}
   TUniqueDisplayItem(const TUniqueDisplayItem<T> &&rhs) : TDisplayItem(rhs), fObject(std::move(rhs.fObject)) {}
   virtual ~TUniqueDisplayItem() {}

   T *GetObject() const { return fObject.get(); }
};

} // namespace Experimental
} // namespace ROOT

#endif

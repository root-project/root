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

/** \class TDisplayItem
  Base class for painting data for JS.
  */

class TDisplayItem {
protected:
   std::string fObjectID;
   std::string fOption;
   int fKind;

public:
   TDisplayItem();
   TDisplayItem(const TDisplayItem &rhs);
   virtual ~TDisplayItem();

   void SetObjectIDAsPtr(void *ptr);
   void SetObjectID(const char *id) { fObjectID = id; }
   const char *GetObjectID() const { return fObjectID.c_str(); }

   void SetOption(const char *opt) { fOption = opt; }
   const char *GetOption() { return fOption.c_str(); }

   void SetKind(int kind) { fKind = kind; }
   int GetKind() const { return fKind; }

   static std::string MakeIDFromPtr(void *ptr);
};

// list of snapshot for primitives in pad

class TPadDisplayItem : public TDisplayItem {
protected:
   std::vector<TDisplayItem *> fPrimitives;

public:
   TPadDisplayItem() : TDisplayItem(), fPrimitives() { SetKind(3); }
   virtual ~TPadDisplayItem();
   void Add(TDisplayItem *snap) { fPrimitives.push_back(snap); }
   TDisplayItem *Last() const { return fPrimitives[fPrimitives.size() - 1]; }
   void Clear();
};

// direct pointer to some object without ownership

template <class T>
class TOrdinaryDisplayItem : public TDisplayItem {
protected:
   T *fSnapshot;

public:
   TOrdinaryDisplayItem(T *addr) : TDisplayItem(), fSnapshot(addr) { SetKind(1); }
   TOrdinaryDisplayItem(const TOrdinaryDisplayItem<T> &&rhs) : TDisplayItem(rhs), fSnapshot(rhs.fSnapshot) {}
   virtual ~TOrdinaryDisplayItem() { fSnapshot = 0; }

   T *GetSnapshot() const { return fSnapshot; }
};

// unique pointer of specified class with ownership

template <class T>
class TUniqueDisplayItem : public TDisplayItem {
protected:
   std::unique_ptr<T> fSnapshot;

public:
   TUniqueDisplayItem(T *addr) : TDisplayItem(), fSnapshot(addr) { SetKind(1); }
   TUniqueDisplayItem(const TUniqueDisplayItem<T> &&rhs) : TDisplayItem(rhs), fSnapshot(std::move(rhs.fSnapshot)) {}
   virtual ~TUniqueDisplayItem() {}

   T *GetSnapshot() const { return fSnapshot.get(); }
};

} // namespace Experimental
} // namespace ROOT

#endif

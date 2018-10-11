/// \file ROOT/RDisplayItem.h
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

#ifndef ROOT7_RDisplayItem
#define ROOT7_RDisplayItem

#include <string>
#include <memory>
#include <vector>

namespace ROOT {
namespace Experimental {
class RCanvas;
class RFrame;

/** \class RDisplayItem
  Base class for painting data for JS.
  */

class RDisplayItem {
protected:
   std::string fObjectID; ///< unique object identifier

public:
   RDisplayItem() = default;
   virtual ~RDisplayItem() {}

   void SetObjectID(const std::string &id) { fObjectID = id; }
   std::string GetObjectID() const { return fObjectID; }
};

// direct pointer to some object without ownership

template <class T>
class ROrdinaryDisplayItem : public RDisplayItem {
protected:
   const T *fObject{nullptr}; ///<  direct pointer without ownership

public:
   ROrdinaryDisplayItem(const T *addr) : RDisplayItem(), fObject(addr) {}

   const T *GetObject() const { return fObject; }
};

// unique pointer of specified class with ownership

template <class T>
class RUniqueDisplayItem : public RDisplayItem {
protected:
   std::unique_ptr<T> fObject;

public:
   RUniqueDisplayItem(T *addr) : RDisplayItem(), fObject(addr) {}

   T *GetObject() const { return fObject.get(); }
};

} // namespace Experimental
} // namespace ROOT

#endif

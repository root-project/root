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
   std::string fObjectID; ///< unique object identifier

public:
   TDisplayItem() = default;
   virtual ~TDisplayItem() {}

   void SetObjectID(const std::string &id) { fObjectID = id; }
   std::string GetObjectID() const { return fObjectID; }
};

// direct pointer to some object without ownership

template <class T>
class TOrdinaryDisplayItem : public TDisplayItem {
protected:
   const T *fObject{nullptr}; ///<  direct pointer without ownership

public:
   TOrdinaryDisplayItem(const T *addr) : TDisplayItem(), fObject(addr) {}

   const T *GetObject() const { return fObject; }
};

// unique pointer of specified class with ownership

template <class T>
class TUniqueDisplayItem : public TDisplayItem {
protected:
   std::unique_ptr<T> fObject;

public:
   TUniqueDisplayItem(T *addr) : TDisplayItem(), fObject(addr) {}

   T *GetObject() const { return fObject.get(); }
};

} // namespace Experimental
} // namespace ROOT

#endif

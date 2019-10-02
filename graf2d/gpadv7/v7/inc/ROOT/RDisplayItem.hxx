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

#include <ROOT/RDrawable.hxx>


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

   void SetObjectIDAsPtr(const void *ptr);

   void PrependID(const std::string &prefix) { fObjectID = prefix + fObjectID; }

};

// created from plain drawable without need of extra parameters

class RDrawableDisplayItem : public RDisplayItem {
protected:

   const RDrawable *fDrawable{nullptr};        ///< drawable

public:

   template <class DRAWABLE>
   RDrawableDisplayItem(const DRAWABLE &dr)
   {
      SetObjectID(dr.GetId());
      fDrawable = &dr;
   }

};

} // namespace Experimental
} // namespace ROOT

#endif

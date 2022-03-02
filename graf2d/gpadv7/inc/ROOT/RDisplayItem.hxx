/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RDisplayItem
#define ROOT7_RDisplayItem

#include <string>

namespace ROOT {
namespace Experimental {

class RDrawable;
class RStyle;
class RAttrMap;

/** \class RDisplayItem
\ingroup GpadROOT7
\brief Base class for painting data for JS.
\author Sergey Linev <s.linev@gsi.de>
\date 2017-05-31
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RDisplayItem {
protected:
   std::string fObjectID;   ///< unique object identifier
   RStyle *fStyle{nullptr}; ///< style object
   unsigned fIndex{0};      ///<! index inside current pad, used to produce fully-qualified id, not send to client
   bool fDummy{false};      ///< if true, just placeholder for drawable which does not changed

public:
   RDisplayItem() = default;
   RDisplayItem(bool dummy) : RDisplayItem() { fDummy = dummy; }
   virtual ~RDisplayItem() {}

   void SetObjectID(const std::string &id) { fObjectID = id; }
   std::string GetObjectID() const { return fObjectID; }

   void SetObjectIDAsPtr(const void *ptr);

   void SetStyle(RStyle *style) { fStyle = style; }

   void SetIndex(unsigned indx) { fIndex = indx; }
   unsigned GetIndex() const { return fIndex; }

   virtual void BuildFullId(const std::string &prefix);

   static std::string ObjectIDFromPtr(const void *ptr);
};


/** \class RDrawableDisplayItem
\ingroup GpadROOT7
\brief Generic display item for RDrawable, just reference drawable itself
\author Sergey Linev <s.linev@gsi.de>
\date 2017-05-31
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RDrawableDisplayItem : public RDisplayItem {
protected:

   const RDrawable *fDrawable{nullptr};        ///< drawable

public:

   template <class DRAWABLE>
   RDrawableDisplayItem(const DRAWABLE &dr)
   {
      fDrawable = &dr;
   }

   const RDrawable *GetDrawable() const { return fDrawable; }

   virtual ~RDrawableDisplayItem();

};


/** \class RIndirectDisplayItem
\ingroup GpadROOT7
\brief Extract (reference) only basic attributes from drawable, but not drawable itself
\author Sergey Linev <s.linev@gsi.de>
\date 2020-04-02
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RIndirectDisplayItem : public RDisplayItem {
protected:

   const RAttrMap *fAttr{nullptr};        ///< pointer on drawable attributes
   const std::string *fCssClass{nullptr}; ///< pointer on drawable class
   const std::string *fId{nullptr};       ///< pointer on drawable id

public:

   RIndirectDisplayItem() = default;

   RIndirectDisplayItem(const RDrawable &dr);
};

} // namespace Experimental
} // namespace ROOT

#endif

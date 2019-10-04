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

class TObject;

namespace ROOT {
namespace Experimental {

class RDrawable;
class RStyle;

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

public:
   RDisplayItem() = default;
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

};

/** \class RObjectDisplayItem
\ingroup GpadROOT7
\brief Display item for TObject with drawing options
\author Sergey Linev <s.linev@gsi.de>
\date 2017-05-31
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RObjectDisplayItem : public RDisplayItem {
protected:

   const TObject *fObject{nullptr};        ///< ROOT6 object
   std::string fOption;                    ///< drawing options

public:

   RObjectDisplayItem(const TObject *obj, const std::string &opt)
   {
      fObject = obj;
      fOption = opt;
   }

};

} // namespace Experimental
} // namespace ROOT

#endif

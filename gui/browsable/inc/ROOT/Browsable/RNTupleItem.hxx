/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_Browsable_RNTupleItem
#define ROOT7_Browsable_RNTupleItem

#include <ROOT/Browsable/RItem.hxx>

namespace ROOT {
namespace Browsable {

/** \class RNTupleItem
\ingroup rbrowser
\brief Representation of an RNTuple item in the browser
\author Patryk Tymoteusz Pilichowski
*/

class RNTupleItem : public RItem {
public:
   enum ECategory {
      kField,
      kVisualization
   };

   RNTupleItem() = default;
   RNTupleItem(const std::string &_name, int _nchilds = 0, const std::string &_icon = "", ECategory _category = kField)
      : RItem(_name, _nchilds, _icon), category(_category)
   {
   }
   // must be here, one needs virtual table for correct streaming of sub-classes
   virtual ~RNTupleItem() = default;

   bool IsVisualization() const { return category == kVisualization; }
   bool IsField() const { return category == kField; }

   bool Compare(const RItem *b, const std::string &s) const override
   {
      auto tuple_b = dynamic_cast<const RNTupleItem *>(b);
      if (tuple_b != nullptr && (IsVisualization() || tuple_b->IsVisualization()))
         return IsVisualization();
      return RItem::Compare(b, s);
   }

protected:
   ECategory category{kField};
};

} // namespace Browsable
} // namespace ROOT

#endif

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_Browsable_RLevelIter
#define ROOT7_Browsable_RLevelIter

#include <memory>
#include <string>

namespace ROOT {
namespace Experimental {
namespace Browsable {

class RElement;
class RItem;

/** \class RLevelIter
\ingroup rbrowser
\brief Iterator over single level hierarchy like any array, keys list, ...
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-14
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class RLevelIter {
public:
   virtual ~RLevelIter() = default;

   /** Shift to next element */
   virtual bool Next() = 0;

   /** Is there current element  */
   virtual bool HasItem() const = 0;

   /** Returns current element name  */
   virtual std::string GetName() const = 0;

   /** If element may have childs: 0 - no, >0 - yes, -1 - maybe */
   virtual int CanHaveChilds() const { return 0; }

   /** Returns full information for current element */
   virtual std::shared_ptr<RElement> GetElement() = 0;

   virtual std::unique_ptr<RItem> CreateItem();

   /** Reset iterator to the first element, returns false if not supported */
   virtual bool Reset() { return false; }

   virtual bool Find(const std::string &name);

};


} // namespace Browsable
} // namespace Experimental
} // namespace ROOT

#endif

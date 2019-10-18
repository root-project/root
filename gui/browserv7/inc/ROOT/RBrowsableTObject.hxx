/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_RBrowsableTObject
#define ROOT7_RBrowsableTObject

#include <ROOT/RBrowsable.hxx>

class TDirectory;

namespace ROOT {
namespace Experimental {

/** Representation of single item in the file browser for object from TKey */
class RBrowserTObjectItem : public RBrowserItem {
   std::string className; ///< class name

public:

   RBrowserTObjectItem() = default;

   RBrowserTObjectItem(const std::string &_name, int _nchilds) : RBrowserItem(_name, _nchilds) {}

   // should be here, one needs virtual table for correct streaming of RRootBrowserReply
   virtual ~RBrowserTObjectItem() = default;

   void SetClassName(const std::string &_className) { className = _className; }
};

// ========================================================================================

} // namespace Experimental
} // namespace ROOT


#endif

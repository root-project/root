/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_Browsable_TObjectItem
#define ROOT7_Browsable_TObjectItem

#include <ROOT/Browsable/RItem.hxx>

namespace ROOT {
namespace Experimental {
namespace Browsable {

/** \class TObjectItem
\ingroup rbrowser
\brief Representation of single item in the file browser for generic TObject object
\author Sergey Linev <S.Linev@gsi.de>
\date 2019-10-19
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class TObjectItem : public RItem {
   std::string className; ///< class name
public:

   TObjectItem() = default;

   TObjectItem(const std::string &_name, int _nchilds) : RItem(_name, _nchilds) {}

   // should be here, one needs virtual table for correct streaming of RRootBrowserReply
   virtual ~TObjectItem() = default;

   void SetClassName(const std::string &_className) { className = _className; }
};

} // namespace Browsable
} // namespace Experimental
} // namespace ROOT


#endif

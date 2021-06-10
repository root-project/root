/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TObjectDisplayItem
#define ROOT7_TObjectDisplayItem

#include <ROOT/RDisplayItem.hxx>

#include <string>
#include "TObject.h"

namespace ROOT {
namespace Experimental {


/** \class TObjectDisplayItem
\ingroup GpadROOT7
\brief Display item for TObject with drawing options
\author Sergey Linev <s.linev@gsi.de>
\date 2017-05-31
\warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback is welcome!
*/

class TObjectDisplayItem : public RDisplayItem {
protected:

   int fKind{0};                           ///< object kind
   const TObject *fObject{nullptr};        ///< ROOT6 object
   std::string fOption;                    ///< drawing options
   bool fOwner{false};                     ///<! if object must be deleted

public:

   TObjectDisplayItem(int kind, const TObject *obj, const std::string &opt, bool owner = false)
   {
      fKind = kind;
      fObject = obj;
      fOption = opt;
      fOwner = owner;
   }

   virtual ~TObjectDisplayItem()
   {
      if (fOwner) delete fObject;
   }

};

} // namespace Experimental
} // namespace ROOT

#endif

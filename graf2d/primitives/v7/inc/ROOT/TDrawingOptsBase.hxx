/// \file ROOT/TDrawingOptionsBase.hxx
/// \ingroup Gpad ROOT7
/// \author Axel Naumann <axel@cern.ch>
/// \date 2018-02-12
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2018, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT7_TDrawingOptsBase
#define ROOT7_TDrawingOptsBase

#include <ROOT/TDrawingAttr.hxx>


#include <RStringView.h>
#include <utility>
#include <vector>

namespace ROOT {
namespace Experimental {

class TDrawingOptsBase {
private:
   /// Collection of all attributes (name and member offset) owned by the derived class.
   std::vector<std::pair<const char*, size_t>> fAttrNameOffsets;

public:
   /// Register a TDrawingAttrOrRefBase. NOTE: the name's lifetime must be larger than this.
   void AddAttr(TDrawingAttrBase& attr, const char *name)
   {
      fAttrNameOffsets.push_back(std::pair<const char*, size_t>(name, (char*)&attr - (char*)this));
   }

   /// Synchronize all shared attributes into their local copy.
   void SyncFromShared() {
      for (auto&& attrOffset: fAttrNameOffsets) {
         reinterpret_cast<TDrawingAttrBase*>((char*)this + attrOffset.second)->SyncFromShared();
      }
   }
};

} // namespace Experimental
} // namespace ROOT

#endif // ROOT7_TDrawingOptsBase

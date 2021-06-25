/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_Browsable_RFieldHolder
#define ROOT_Browsable_RFieldHolder

#include <ROOT/Browsable/RHolder.hxx>

#include <ROOT/RNTuple.hxx>
#include <ROOT/RMiniFile.hxx>

#include "TClass.h"

namespace ROOT {
namespace Experimental {
namespace Detail {
class RPageSource;
}
}
}

class RFieldHolder : public ROOT::Experimental::Browsable::RHolder {
   std::shared_ptr<ROOT::Experimental::Detail::RPageSource> fNtplSource;
   std::string fParentName;

   ROOT::Experimental::DescriptorId_t fFieldId;

public:
   RFieldHolder(std::shared_ptr<ROOT::Experimental::Detail::RPageSource> ntplSource,
                const std::string &parent_name,
                ROOT::Experimental::DescriptorId_t id)
                : fNtplSource(ntplSource), fParentName(parent_name), fFieldId(id) {}

   const TClass *GetClass() const override { return TClass::GetClass<ROOT::Experimental::RNTuple>(); }

   /** Returns direct (temporary) object pointer */
   const void *GetObject() const override { return nullptr; }

   auto GetNtplSource() const { return fNtplSource; }
   auto GetParentName() const { return fParentName; }
   auto GetId() const { return fFieldId; }
};

#endif

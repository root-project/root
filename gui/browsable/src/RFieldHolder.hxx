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

#include <ROOT/RNTupleReader.hxx>
#include <ROOT/RMiniFile.hxx>

#include "TClass.h"

namespace ROOT {
namespace Experimental {
namespace Detail {
class RPageSource;
}
}
}

class RFieldHolder : public ROOT::Browsable::RHolder {
   std::shared_ptr<ROOT::Experimental::RNTupleReader> fNtplReader;
   std::string fParentName;

   ROOT::Experimental::DescriptorId_t fFieldId;

public:
   RFieldHolder(std::shared_ptr<ROOT::Experimental::RNTupleReader> ntplReader, const std::string &parent_name,
                ROOT::Experimental::DescriptorId_t id)
      : fNtplReader(ntplReader), fParentName(parent_name), fFieldId(id)
   {
   }

   const TClass *GetClass() const override { return TClass::GetClass<ROOT::RNTuple>(); }

   /** Returns direct (temporary) object pointer */
   const void *GetObject() const override { return nullptr; }

   auto GetNtplReader() const { return fNtplReader; }
   auto GetParentName() const { return fParentName; }
   auto GetId() const { return fFieldId; }
};

#endif

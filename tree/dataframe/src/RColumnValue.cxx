// Author: Enrico Guiraud, Danilo Piparo CERN  09/2018

/*************************************************************************
 * Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RDF/RColumnValue.hxx>
#include <ROOT/RDF/RCustomColumnBase.hxx>
#include <TClass.h>
#include <ROOT/RDF/Utils.hxx> // TypeID2TypeName

#include <string>
#include <typeinfo>
#include <vector>

namespace ROOT {
namespace Internal {
namespace RDF {

// Some extern instaniations to speed-up compilation/interpretation time
// These are not active if c++17 is enabled because of a bug in our clang
// See ROOT-9499.
#if __cplusplus < 201703L
template class RColumnValue<int>;
template class RColumnValue<unsigned int>;
template class RColumnValue<char>;
template class RColumnValue<unsigned char>;
template class RColumnValue<float>;
template class RColumnValue<double>;
template class RColumnValue<Long64_t>;
template class RColumnValue<ULong64_t>;
template class RColumnValue<std::vector<int>>;
template class RColumnValue<std::vector<unsigned int>>;
template class RColumnValue<std::vector<char>>;
template class RColumnValue<std::vector<unsigned char>>;
template class RColumnValue<std::vector<float>>;
template class RColumnValue<std::vector<double>>;
template class RColumnValue<std::vector<Long64_t>>;
template class RColumnValue<std::vector<ULong64_t>>;
#endif

void CheckCustomColumn(RCustomColumnBase *customColumn, const std::type_info &tid)
{
   // Here we compare names and not typeinfos since they may come from two different contexts: a compiled
   // and a jitted one.
   const auto diffTypes = (0 != std::strcmp(customColumn->GetTypeId().name(), tid.name()));
   auto inheritedType = [&]() {
      auto colTClass = TClass::GetClass(customColumn->GetTypeId());
      return colTClass && colTClass->InheritsFrom(TClass::GetClass(tid));
   };

   if (diffTypes && !inheritedType()) {
      const auto tName = TypeID2TypeName(tid);
      const auto colTypeName = TypeID2TypeName(customColumn->GetTypeId());
      std::string errMsg = "RColumnValue: type specified for column \"" + customColumn->GetName() + "\" is ";
      if (tName.empty()) {
         errMsg += tid.name();
         errMsg += " (extracted from type info)";
      } else {
         errMsg += tName;
      }
      errMsg += " but temporary column has type ";
      if (colTypeName.empty()) {
         auto &id = customColumn->GetTypeId();
         errMsg += id.name();
         errMsg += " (extracted from type info)";
      } else {
         errMsg += colTypeName;
      }
      throw std::runtime_error(errMsg);
   }
}

} // namespace RDF
} // namespace Internal
} // namespace ROOT

/// \file RNTupleBrowseUtils.cxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2025-07-25

/*************************************************************************
 * Copyright (C) 1995-2025, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleBrowseUtils.hxx>
#include <ROOT/RNTupleDescriptor.hxx>

ROOT::DescriptorId_t ROOT::Internal::GetNextBrowsableField(DescriptorId_t fieldId, const RNTupleDescriptor &desc)
{
   const auto &fieldDesc = desc.GetFieldDescriptor(fieldId);
   if (fieldDesc.GetLinkIds().empty())
      return fieldId;

   DescriptorId_t result = fieldId;
   if (fieldDesc.GetStructure() == ENTupleStructure::kCollection) {
      // Variable-length collection: vector, map, set, collection proxy etc.
      result = fieldDesc.GetLinkIds()[0];
   } else if (fieldDesc.GetNRepetitions() > 0) {
      // Fixed-size array
      result = fieldDesc.GetLinkIds()[0];
   } else if (fieldDesc.GetTypeName().rfind("std::atomic<", 0) == 0) {
      result = fieldDesc.GetLinkIds()[0];
   }

   return (result != fieldId) ? GetNextBrowsableField(result, desc) : fieldId;
}

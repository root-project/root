/// \file RNTupleView.cxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2024-10-28

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RError.hxx>
#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleView.hxx>
#include <ROOT/RPageStorage.hxx>

#include <deque>

ROOT::RNTupleGlobalRange
ROOT::Internal::GetFieldRange(const ROOT::RFieldBase &field, const ROOT::Internal::RPageSource &pageSource)
{
   const auto &desc = pageSource.GetSharedDescriptorGuard().GetRef();

   auto fnGetPrincipalColumnId = [&desc](ROOT::DescriptorId_t fieldId) -> ROOT::DescriptorId_t {
      R__ASSERT(fieldId != ROOT::kInvalidDescriptorId);
      auto columnIterable = desc.GetColumnIterable(fieldId);
      return (columnIterable.size() > 0) ? columnIterable.begin()->GetPhysicalId() : ROOT::kInvalidDescriptorId;
   };

   auto columnId = fnGetPrincipalColumnId(field.GetOnDiskId());
   if (columnId == ROOT::kInvalidDescriptorId) {
      // We need to iterate the field descriptor tree, not the sub fields of `field`, because in the presence of
      // read rules, the in-memory sub fields may be artificial and not have valid on-disk IDs.
      const auto &linkIds = desc.GetFieldDescriptor(field.GetOnDiskId()).GetLinkIds();
      std::deque<ROOT::DescriptorId_t> subFields(linkIds.begin(), linkIds.end());
      while (!subFields.empty()) {
         auto subFieldId = subFields.front();
         subFields.pop_front();
         columnId = fnGetPrincipalColumnId(subFieldId);
         if (columnId != ROOT::kInvalidDescriptorId)
            break;

         const auto &subLinkIds = desc.GetFieldDescriptor(subFieldId).GetLinkIds();
         subFields.insert(subFields.end(), subLinkIds.begin(), subLinkIds.end());
      }
   }

   if (columnId == ROOT::kInvalidDescriptorId) {
      return ROOT::RNTupleGlobalRange(ROOT::kInvalidNTupleIndex, ROOT::kInvalidNTupleIndex);
   }

   auto arraySize = std::max(std::uint64_t(1), desc.GetFieldDescriptor(field.GetOnDiskId()).GetNRepetitions());
   return ROOT::RNTupleGlobalRange(0, desc.GetNElements(columnId) / arraySize);
}

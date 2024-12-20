/// \file RNTupleView.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2024-10-28
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

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

ROOT::Experimental::RNTupleGlobalRange
ROOT::Experimental::Internal::GetFieldRange(const RFieldBase &field, const RPageSource &pageSource)
{
   const auto &desc = pageSource.GetSharedDescriptorGuard().GetRef();

   auto fnGetPrincipalColumnId = [&desc](DescriptorId_t fieldId) -> DescriptorId_t {
      auto columnIterable = desc.GetColumnIterable(fieldId);
      return (columnIterable.size() > 0) ? columnIterable.begin()->GetPhysicalId() : kInvalidDescriptorId;
   };

   auto columnId = fnGetPrincipalColumnId(field.GetOnDiskId());
   if (columnId == kInvalidDescriptorId) {
      for (const auto &f : field) {
         columnId = fnGetPrincipalColumnId(f.GetOnDiskId());
         if (columnId != kInvalidDescriptorId)
            break;
      }
   }

   if (columnId == kInvalidDescriptorId) {
      return RNTupleGlobalRange(kInvalidNTupleIndex, kInvalidNTupleIndex);
   }

   auto arraySize = std::max(std::uint64_t(1), desc.GetFieldDescriptor(field.GetOnDiskId()).GetNRepetitions());
   return RNTupleGlobalRange(0, desc.GetNElements(columnId) / arraySize);
}

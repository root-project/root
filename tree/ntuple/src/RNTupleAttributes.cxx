/// \file RNTupleAttributes.cxx
/// \ingroup NTuple ROOT7
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2025-02-25
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleAttributes.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleFillContext.hxx>
#include <ROOT/RPageStorageFile.hxx>

static ROOT::RResult<void> ValidateAttributeModel(const ROOT::RNTupleModel &model)
{
   const auto &projFields = ROOT::Internal::GetProjectedFieldsOfModel(model);
   if (!projFields.IsEmpty())
      return R__FAIL("The Model passed to CreateAttributeSet cannot contain projected fields.");

   for (const auto &field : model.GetConstFieldZero()) {
      if (field.GetStructure() == ROOT::ENTupleStructure::kStreamer)
         return R__FAIL(std::string("The Model passed to CreateAttributeSet cannot contain Streamer field '") +
                        field.GetQualifiedFieldName());
   }
   return ROOT::RResult<void>::Success();
}

//
//  RNTupleAttributeRange
//
ROOT::Experimental::RNTupleAttributeRange::RNTupleAttributeRange(std::unique_ptr<REntry> entry,
                                                                 ROOT::NTupleSize_t start)
   : fEntry(std::move(entry)), fStart(start)
{
}

//
//  RNTupleAttributeSet
//
ROOT::RResult<ROOT::Experimental::RNTupleAttributeSet>
ROOT::Experimental::RNTupleAttributeSet::Create(std::string_view name, std::unique_ptr<RNTupleModel> model,
                                                const RNTupleFillContext *mainFillContext, TDirectory &dir)

{
   // Validate model
   if (auto modelValid = ValidateAttributeModel(*model); !modelValid)
      return R__FORWARD_ERROR(modelValid);

   // Add an internal EntryRange field to the model.
   // TODO: the entry range field name is not guaranteed to not be already taken!
   // TODO: do we need a bespoke field type?
   model->MakeField<REntryRange>(kEntryRangeFieldName);
   model->Freeze();

   // Create a sink that points to the same TDirectory as the main RNTuple
   auto opts = ROOT::RNTupleWriteOptions{};
   auto sink = std::make_unique<ROOT::Internal::RPageSinkFile>(name, dir, opts);
   RNTupleFillContext fillContext{std::move(model), std::move(sink)};
   return RNTupleAttributeSet(mainFillContext, std::move(fillContext));
}

ROOT::Experimental::RNTupleAttributeSet::RNTupleAttributeSet(const RNTupleFillContext *mainFillContext,
                                                             RNTupleFillContext fillContext)
   : fFillContext(std::move(fillContext)), fMainFillContext(mainFillContext)
{
}

ROOT::Experimental::RNTupleAttributeRange ROOT::Experimental::RNTupleAttributeSet::BeginRange()
{
   auto entry = fFillContext.GetModel().CreateEntry();
   const auto start = fMainFillContext->GetNEntries();
   auto range = RNTupleAttributeRange(std::move(entry), start);
   return range;
}

void ROOT::Experimental::RNTupleAttributeSet::EndRange(ROOT::Experimental::RNTupleAttributeRange range)
{
   // Get current entry number from the writer and use it as end of entry range
   const auto end = fMainFillContext->GetNEntries();
   auto pRange = range.fEntry->GetPtr<REntryRange>(kEntryRangeFieldName);
   R__ASSERT(pRange);
   *pRange = {range.fStart, end};
   fFillContext.Fill(*range.fEntry);
}

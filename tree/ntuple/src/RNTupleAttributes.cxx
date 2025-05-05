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
ROOT::Experimental::Internal::RNTupleAttributeRange::RNTupleAttributeRange(std::unique_ptr<REntry> entry,
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

ROOT::Experimental::RNTupleAttributeSet::~RNTupleAttributeSet()
{
   if (fOpenRange)
      EndRangeInternal();
}

const std::string &ROOT::Experimental::RNTupleAttributeSet::GetName() const
{
   const auto &name = fFillContext.fSink->GetNTupleName();
   return name;
}

ROOT::Experimental::RNTupleAttributeRangeHandle ROOT::Experimental::RNTupleAttributeSet::BeginRange()
{
   if (fOpenRange)
      throw ROOT::RException(R__FAIL("Called BeginRange() without having closed the currently open range!"));

   auto entry = fFillContext.GetModel().CreateEntry();
   const auto start = fMainFillContext->GetNEntries();
   fOpenRange = Internal::RNTupleAttributeRange(std::move(entry), start);
   auto handle = RNTupleAttributeRangeHandle{*fOpenRange};
   return handle;
}

void ROOT::Experimental::RNTupleAttributeSet::EndRange(ROOT::Experimental::RNTupleAttributeRangeHandle rangeHandle)
{
   if (R__unlikely(!fOpenRange || &rangeHandle.fRange != &*fOpenRange))
      throw ROOT::RException(
         R__FAIL(std::string("Handle passed to EndRange() of Attribute Set \"") + GetName() +
                 "\" is invalid (it is not the Handle returned by the latest call to BeginRange())"));

   EndRangeInternal();
}

void ROOT::Experimental::RNTupleAttributeSet::EndRangeInternal()
{
   // Get current entry number from the writer and use it as end of entry range
   const auto end = fMainFillContext->GetNEntries();
   auto &range = *fOpenRange;
   auto pRange = range.fEntry->GetPtr<REntryRange>(kEntryRangeFieldName);
   R__ASSERT(pRange);
   *pRange = {range.fStart, end};
   fFillContext.Fill(*range.fEntry);

   fOpenRange = std::nullopt;
}

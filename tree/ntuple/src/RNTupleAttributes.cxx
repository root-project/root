/// \file RNTupleAttributes.cxx
/// \ingroup NTuple ROOT7
/// \author Giacomo Parolini <giacomo.parolini@cern.ch>
/// \date 2026-01-27
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

#include <ROOT/RNTupleAttributes.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleFillContext.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/StringUtils.hxx>

using namespace ROOT::Experimental::Internal::RNTupleAttributes;

static ROOT::RResult<void> ValidateAttributeModel(const ROOT::RNTupleModel &model)
{
   const auto &projFields = ROOT::Internal::GetProjectedFieldsOfModel(model);
   if (!projFields.IsEmpty())
      return R__FAIL("The Model used to create an AttributeSet cannot contain projected fields.");

   for (const auto &field : model.GetConstFieldZero()) {
      if (field.GetStructure() == ROOT::ENTupleStructure::kStreamer)
         return R__FAIL(std::string("The Model used to create an AttributeSet cannot contain Streamer field '") +
                        field.GetQualifiedFieldName() + "'");
   }
   return ROOT::RResult<void>::Success();
}

//
//  RNTupleAttrSetWriter
//
std::unique_ptr<ROOT::Experimental::RNTupleAttrSetWriter>
ROOT::Experimental::RNTupleAttrSetWriter::Create(const RNTupleFillContext &mainFillContext,
                                                 std::unique_ptr<ROOT::Internal::RPageSink> sink,
                                                 std::unique_ptr<RNTupleModel> userModel)

{
   ValidateAttributeModel(*userModel).ThrowOnError();

   // We create a "meta model" that's what we'll use to write data to storage. This meta model has 3 fields:
   // the "meta fields" _rangeStart / _rangeLen and an untyped Record field which contains all the top-level fields
   // from the user model as its children. This is done to "namespace" all user-defined attribute fields so that we
   // are free to use whichever name we want for our meta fields.
   // Note that the user model is preserved as-is to allow the user to create entries from it or use its default
   // entry. When we actually write data to storage, we do some pointer trickery to correctly read the values from
   // the user model and store them under the meta model's fields (see RNTupleAttrEntryPair::Append())
   auto metaModel = RNTupleModel::Create();
   metaModel->SetDescription(userModel->GetDescription());
   auto rangeStartPtr = metaModel->MakeField<ROOT::NTupleSize_t>(kRangeStartName);
   auto rangeLenPtr = metaModel->MakeField<ROOT::NTupleSize_t>(kRangeLenName);
   std::vector<std::unique_ptr<RFieldBase>> fields;
   auto subfields = userModel->GetConstFieldZero().GetConstSubfields();
   fields.reserve(subfields.size());
   for (const auto *field : subfields) {
      fields.push_back(field->Clone(field->GetFieldName()));
   }
   auto userRootField = std::make_unique<ROOT::RRecordField>(kUserModelName, std::move(fields));
   metaModel->AddField(std::move(userRootField));

   metaModel->Freeze();
   userModel->Freeze();

   return std::unique_ptr<RNTupleAttrSetWriter>(
      new RNTupleAttrSetWriter(mainFillContext, std::move(sink), std::move(metaModel), std::move(userModel),
                               std::move(rangeStartPtr), std::move(rangeLenPtr)));
}

ROOT::Experimental::RNTupleAttrSetWriter::RNTupleAttrSetWriter(const RNTupleFillContext &mainFillContext,
                                                               std::unique_ptr<ROOT::Internal::RPageSink> sink,
                                                               std::unique_ptr<RNTupleModel> metaModel,
                                                               std::unique_ptr<RNTupleModel> userModel,
                                                               std::shared_ptr<ROOT::NTupleSize_t> rangeStartPtr,
                                                               std::shared_ptr<ROOT::NTupleSize_t> rangeLenPtr)
   : fFillContext(std::move(metaModel), std::move(sink)),
     fMainFillContext(&mainFillContext),
     fUserModel(std::move(userModel)),
     fRangeStartPtr(std::move(rangeStartPtr)),
     fRangeLenPtr(std::move(rangeLenPtr))
{
}

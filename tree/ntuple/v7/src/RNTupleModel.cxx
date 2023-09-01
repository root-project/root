/// \file RNTupleModel.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-15
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/StringUtils.hxx>

#include <atomic>
#include <cstdlib>
#include <memory>
#include <utility>

ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleModel::RProjectedFields::EnsureValidMapping(const Detail::RFieldBase *target,
                                                                       const FieldMap_t &fieldMap)
{
   auto source = fieldMap.at(target);
   const bool hasCompatibleStructure =
      (source->GetStructure() == target->GetStructure()) ||
      ((source->GetStructure() == ENTupleStructure::kCollection) && dynamic_cast<const RCardinalityField *>(target));
   if (!hasCompatibleStructure)
      return R__FAIL("field mapping structural mismatch: " + source->GetName() + " --> " + target->GetName());
   if (source->GetStructure() == ENTupleStructure::kLeaf) {
      if (target->GetType() != source->GetType())
         return R__FAIL("field mapping type mismatch: " + source->GetName() + " --> " + target->GetName());
   }

   // We support projections only across records and collections. In the following, we check that the projected
   // field is on the same path of collection fields in the field tree than the source field.

   // Finds the first non-record parent field of the input field
   auto fnBreakPoint = [](const Detail::RFieldBase *f) -> const Detail::RFieldBase * {
      auto parent = f->GetParent();
      while (parent) {
         if (parent->GetStructure() != ENTupleStructure::kRecord)
            return parent;
         parent = parent->GetParent();
      }
      // We reached the zero field
      return nullptr;
   };

   // If source or target has a variant or reference as a parent, error out
   auto *sourceBreakPoint = fnBreakPoint(source);
   if (sourceBreakPoint && sourceBreakPoint->GetStructure() != ENTupleStructure::kCollection)
      return R__FAIL("unsupported field mapping (source structure)");
   auto *targetBreakPoint = fnBreakPoint(target);
   if (targetBreakPoint && sourceBreakPoint->GetStructure() != ENTupleStructure::kCollection)
      return R__FAIL("unsupported field mapping (target structure)");

   if (!sourceBreakPoint && !targetBreakPoint) {
      // Source and target have no collections as parent
      return RResult<void>::Success();
   }
   if (sourceBreakPoint && targetBreakPoint) {
      if (sourceBreakPoint == targetBreakPoint) {
         // Source and target are children of the same collection
         return RResult<void>::Success();
      }
      if (auto it = fieldMap.find(targetBreakPoint); it != fieldMap.end() && it->second == sourceBreakPoint) {
         // The parent collection of parent is mapped to the parent collection of the source
         return RResult<void>::Success();
      }
      // Source and target are children of different collections
      return R__FAIL("field mapping structure mismatch: " + source->GetName() + " --> " + target->GetName());
   }

   // Either source or target have no collection as a parent, but the other one has; that doesn't fit
   return R__FAIL("field mapping structure mismatch: " + source->GetName() + " --> " + target->GetName());
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleModel::RProjectedFields::Add(std::unique_ptr<Detail::RFieldBase> field,
                                                        const FieldMap_t &fieldMap)
{
   auto result = EnsureValidMapping(field.get(), fieldMap);
   if (!result)
      return R__FORWARD_ERROR(result);
   for (const auto &f : *field) {
      result = EnsureValidMapping(&f, fieldMap);
      if (!result)
         return R__FORWARD_ERROR(result);
   }

   fFieldMap.insert(fieldMap.begin(), fieldMap.end());
   fFieldZero->Attach(std::move(field));
   return RResult<void>::Success();
}

const ROOT::Experimental::Detail::RFieldBase *
ROOT::Experimental::RNTupleModel::RProjectedFields::GetSourceField(const Detail::RFieldBase *target) const
{
   if (auto it = fFieldMap.find(target); it != fFieldMap.end())
      return it->second;
   return nullptr;
}

std::unique_ptr<ROOT::Experimental::RNTupleModel::RProjectedFields>
ROOT::Experimental::RNTupleModel::RProjectedFields::Clone(const RNTupleModel *newModel) const
{
   auto cloneFieldZero = std::unique_ptr<RFieldZero>(static_cast<RFieldZero *>(fFieldZero->Clone("").release()));
   auto clone = std::unique_ptr<RProjectedFields>(new RProjectedFields(std::move(cloneFieldZero)));
   clone->fModel = newModel;
   // TODO(jblomer): improve quadratic search to re-wire the field mappings given the new model and the cloned
   // projected fields. Not too critical as we generally expect a limited number of projected fields
   for (const auto &[k, v] : fFieldMap) {
      for (const auto &f : *clone->GetFieldZero()) {
         if (f.GetQualifiedFieldName() == k->GetQualifiedFieldName()) {
            clone->fFieldMap[&f] = clone->fModel->GetField(v->GetQualifiedFieldName());
            break;
         }
      }
   }
   return clone;
}

ROOT::Experimental::RNTupleModel::RUpdater::RUpdater(RNTupleWriter &writer)
   : fWriter(writer), fOpenChangeset(*fWriter.fModel)
{
}

void ROOT::Experimental::RNTupleModel::RUpdater::BeginUpdate()
{
   fOpenChangeset.fModel.Unfreeze();
}

void ROOT::Experimental::RNTupleModel::RUpdater::CommitUpdate()
{
   fOpenChangeset.fModel.Freeze();
   if (fOpenChangeset.IsEmpty())
      return;
   Detail::RNTupleModelChangeset toCommit{fOpenChangeset.fModel};
   std::swap(fOpenChangeset.fAddedFields, toCommit.fAddedFields);
   std::swap(fOpenChangeset.fAddedProjectedFields, toCommit.fAddedProjectedFields);
   fWriter.fSink->UpdateSchema(toCommit, fWriter.fNEntries);
}

void ROOT::Experimental::RNTupleModel::RUpdater::AddField(std::unique_ptr<Detail::RFieldBase> field)
{
   auto fieldp = field.get();
   fOpenChangeset.fModel.AddField(std::move(field));
   fOpenChangeset.fAddedFields.emplace_back(fieldp);
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleModel::RUpdater::AddProjectedField(std::unique_ptr<Detail::RFieldBase> field,
                                                              std::function<std::string(const std::string &)> mapping)
{
   auto fieldp = field.get();
   auto result = fOpenChangeset.fModel.AddProjectedField(std::move(field), mapping);
   if (result)
      fOpenChangeset.fAddedProjectedFields.emplace_back(fieldp);
   return R__FORWARD_RESULT(result);
}

void ROOT::Experimental::RNTupleModel::EnsureValidFieldName(std::string_view fieldName)
{
   RResult<void> nameValid = Detail::RFieldBase::EnsureValidFieldName(fieldName);
   if (!nameValid) {
      nameValid.Throw();
   }
   auto fieldNameStr = std::string(fieldName);
   if (fFieldNames.insert(fieldNameStr).second == false) {
      throw RException(R__FAIL("field name '" + fieldNameStr + "' already exists in NTuple model"));
   }
}

void ROOT::Experimental::RNTupleModel::EnsureNotFrozen() const
{
   if (IsFrozen())
      throw RException(R__FAIL("invalid attempt to modify frozen model"));
}

void ROOT::Experimental::RNTupleModel::EnsureNotBare() const
{
   if (!fDefaultEntry)
      throw RException(R__FAIL("invalid attempt to use default entry of bare model"));
}

ROOT::Experimental::RNTupleModel::RNTupleModel()
  : fFieldZero(std::make_unique<RFieldZero>())
{}

std::unique_ptr<ROOT::Experimental::RNTupleModel> ROOT::Experimental::RNTupleModel::CreateBare()
{
   auto model = std::unique_ptr<RNTupleModel>(new RNTupleModel());
   model->fProjectedFields = std::make_unique<RProjectedFields>(model.get());
   return model;
}

std::unique_ptr<ROOT::Experimental::RNTupleModel> ROOT::Experimental::RNTupleModel::Create()
{
   auto model = CreateBare();
   model->fDefaultEntry = std::unique_ptr<REntry>(new REntry());
   return model;
}

std::unique_ptr<ROOT::Experimental::RNTupleModel> ROOT::Experimental::RNTupleModel::Clone() const
{
   auto cloneModel = std::unique_ptr<RNTupleModel>(new RNTupleModel());
   auto cloneFieldZero = fFieldZero->Clone("");
   cloneModel->fModelId = fModelId;
   cloneModel->fFieldZero = std::unique_ptr<RFieldZero>(static_cast<RFieldZero *>(cloneFieldZero.release()));
   cloneModel->fFieldNames = fFieldNames;
   cloneModel->fDescription = fDescription;
   cloneModel->fProjectedFields = fProjectedFields->Clone(cloneModel.get());
   if (fDefaultEntry) {
      cloneModel->fDefaultEntry = std::unique_ptr<REntry>(new REntry(fModelId));
      for (const auto &f : cloneModel->fFieldZero->GetSubFields()) {
         cloneModel->fDefaultEntry->AddValue(f->GenerateValue());
      }
   }
   return cloneModel;
}


void ROOT::Experimental::RNTupleModel::AddField(std::unique_ptr<Detail::RFieldBase> field)
{
   EnsureNotFrozen();
   if (!field)
      throw RException(R__FAIL("null field"));
   EnsureValidFieldName(field->GetName());

   if (fDefaultEntry)
      fDefaultEntry->AddValue(field->GenerateValue());
   fFieldZero->Attach(std::move(field));
}

ROOT::Experimental::RResult<void>
ROOT::Experimental::RNTupleModel::AddProjectedField(std::unique_ptr<Detail::RFieldBase> field,
                                                    std::function<std::string(const std::string &)> mapping)
{
   EnsureNotFrozen();
   if (!field)
      return R__FAIL("null field");
   auto fieldName = field->GetName();

   RProjectedFields::FieldMap_t fieldMap;
   auto sourceField = GetField(mapping(fieldName));
   if (!sourceField)
      return R__FAIL("no such field: " + mapping(fieldName));
   fieldMap[field.get()] = sourceField;
   for (const auto &subField : *field) {
      sourceField = GetField(mapping(subField.GetQualifiedFieldName()));
      if (!sourceField)
         return R__FAIL("no such field: " + mapping(fieldName));
      fieldMap[&subField] = sourceField;
   }

   EnsureValidFieldName(fieldName);
   auto result = fProjectedFields->Add(std::move(field), fieldMap);
   if (!result) {
      fFieldNames.erase(fieldName);
      return R__FORWARD_ERROR(result);
   }
   return RResult<void>::Success();
}

std::shared_ptr<ROOT::Experimental::RCollectionNTupleWriter> ROOT::Experimental::RNTupleModel::MakeCollection(
   std::string_view fieldName, std::unique_ptr<RNTupleModel> collectionModel)
{
   EnsureNotFrozen();
   EnsureValidFieldName(fieldName);
   if (!collectionModel) {
      throw RException(R__FAIL("null collectionModel"));
   }
   auto collectionNTuple = std::make_shared<RCollectionNTupleWriter>(std::move(collectionModel->fDefaultEntry));
   auto field = std::make_unique<RCollectionField>(fieldName, collectionNTuple, std::move(collectionModel));
   if (fDefaultEntry)
      fDefaultEntry->AddValue(field->BindValue(collectionNTuple->GetOffsetPtr()));
   fFieldZero->Attach(std::move(field));
   return collectionNTuple;
}

const ROOT::Experimental::Detail::RFieldBase *
ROOT::Experimental::RNTupleModel::GetField(std::string_view fieldName) const
{
   if (fieldName.empty())
      return nullptr;

   auto *field = static_cast<ROOT::Experimental::Detail::RFieldBase *>(fFieldZero.get());
   for (auto subfieldName : ROOT::Split(fieldName, ".")) {
      const auto subfields = field->GetSubFields();
      auto it =
         std::find_if(subfields.begin(), subfields.end(), [&](const auto *f) { return f->GetName() == subfieldName; });
      if (it != subfields.end()) {
         field = *it;
      } else {
         field = nullptr;
         break;
      }
   }

   return field;
}

ROOT::Experimental::REntry *ROOT::Experimental::RNTupleModel::GetDefaultEntry() const
{
   if (!IsFrozen())
      throw RException(R__FAIL("invalid attempt to get default entry of unfrozen model"));
   EnsureNotBare();
   return fDefaultEntry.get();
}

std::unique_ptr<ROOT::Experimental::REntry> ROOT::Experimental::RNTupleModel::CreateEntry() const
{
   if (!IsFrozen())
      throw RException(R__FAIL("invalid attempt to create entry of unfrozen model"));

   auto entry = std::unique_ptr<REntry>(new REntry(fModelId));
   for (const auto &f : fFieldZero->GetSubFields()) {
      entry->AddValue(f->GenerateValue());
   }
   return entry;
}

std::unique_ptr<ROOT::Experimental::REntry> ROOT::Experimental::RNTupleModel::CreateBareEntry() const
{
   if (!IsFrozen())
      throw RException(R__FAIL("invalid attempt to create entry of unfrozen model"));

   auto entry = std::unique_ptr<REntry>(new REntry(fModelId));
   for (const auto &f : fFieldZero->GetSubFields()) {
      entry->AddValue(f->BindValue(nullptr));
   }
   return entry;
}

void ROOT::Experimental::RNTupleModel::Unfreeze()
{
   if (!IsFrozen())
      throw RException(R__FAIL("invalid attempt to unfreeze an unfrozen model"));
   fModelId = 0;
}

void ROOT::Experimental::RNTupleModel::Freeze()
{
   if (IsFrozen())
      return;

   static std::atomic<std::uint64_t> gLastModelId = 0;
   fModelId = ++gLastModelId;
   if (fDefaultEntry)
      fDefaultEntry->fModelId = fModelId;
}

void ROOT::Experimental::RNTupleModel::SetDescription(std::string_view description)
{
   EnsureNotFrozen();
   fDescription = std::string(description);
}

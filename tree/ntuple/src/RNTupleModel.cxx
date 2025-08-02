/// \file RNTupleModel.cxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-15

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RError.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RFieldToken.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleUtils.hxx>
#include <ROOT/RNTupleWriter.hxx>
#include <ROOT/StringUtils.hxx>

#include <atomic>
#include <cstdlib>
#include <memory>
#include <utility>

namespace {
std::uint64_t GetNewModelId()
{
   static std::atomic<std::uint64_t> gLastModelId = 0;
   return ++gLastModelId;
}
} // anonymous namespace

ROOT::RFieldZero &ROOT::Internal::GetFieldZeroOfModel(ROOT::RNTupleModel &model)
{
   if (model.IsExpired()) {
      throw RException(R__FAIL("invalid use of expired model"));
   }
   return *model.fFieldZero;
}

ROOT::Internal::RProjectedFields &ROOT::Internal::GetProjectedFieldsOfModel(ROOT::RNTupleModel &model)
{
   if (model.IsExpired()) {
      throw RException(R__FAIL("invalid use of expired model"));
   }
   return *model.fProjectedFields;
}

//------------------------------------------------------------------------------

ROOT::RResult<void>
ROOT::Internal::RProjectedFields::EnsureValidMapping(const ROOT::RFieldBase *target, const FieldMap_t &fieldMap)
{
   auto source = fieldMap.at(target);

   if (!target->GetColumnRepresentatives()[0].empty()) {
      const auto representative = target->GetColumnRepresentatives()[0][0];
      // If the user is trying to add a projected field upon which they called SetTruncated() or SetQuantized(),
      // they probably have the wrong expectations about what should happen. Warn them about it.
      if (representative == ENTupleColumnType::kReal32Trunc || representative == ENTupleColumnType::kReal32Quant) {
         R__LOG_WARNING(ROOT::Internal::NTupleLog())
            << "calling SetQuantized() or SetTruncated() on a projected field has no effect, as the on-disk "
               "representation of the value is decided by the projection source field. Reading back the field will "
               "yield the correct values, but the value range and bits of precision you set on the projected field "
               "will be ignored.";
      }
   }

   const bool hasCompatibleStructure = (source->GetStructure() == target->GetStructure()) ||
                                       ((source->GetStructure() == ROOT::ENTupleStructure::kCollection) &&
                                        dynamic_cast<const ROOT::RCardinalityField *>(target));
   if (!hasCompatibleStructure)
      return R__FAIL("field mapping structural mismatch: " + source->GetFieldName() + " --> " + target->GetFieldName());
   if ((source->GetStructure() == ROOT::ENTupleStructure::kLeaf) ||
       (source->GetStructure() == ROOT::ENTupleStructure::kStreamer)) {
      if (target->GetTypeName() != source->GetTypeName())
         return R__FAIL("field mapping type mismatch: " + source->GetFieldName() + " --> " + target->GetFieldName());
   }

   auto fnHasArrayParent = [](const ROOT::RFieldBase &f) -> bool {
      auto parent = f.GetParent();
      while (parent) {
         if (parent->GetNRepetitions() > 0)
            return true;
         parent = parent->GetParent();
      }
      return false;
   };
   if (fnHasArrayParent(*source) || fnHasArrayParent(*target)) {
      return R__FAIL("unsupported field mapping across fixed-size arrays");
   }

   // We support projections only across records and collections. In the following, we check that the projected
   // field is on the same path of collection fields in the field tree than the source field.

   // Finds the first non-record parent field of the input field
   auto fnBreakPoint = [](const ROOT::RFieldBase *f) -> const ROOT::RFieldBase * {
      auto parent = f->GetParent();
      while (parent) {
         if ((parent->GetStructure() != ROOT::ENTupleStructure::kRecord) &&
             (parent->GetStructure() != ROOT::ENTupleStructure::kLeaf)) {
            return parent;
         }
         parent = parent->GetParent();
      }
      // We reached the zero field
      return nullptr;
   };

   // If source or target has a variant or reference as a parent, error out
   auto *sourceBreakPoint = fnBreakPoint(source);
   if (sourceBreakPoint && sourceBreakPoint->GetStructure() != ROOT::ENTupleStructure::kCollection)
      return R__FAIL("unsupported field mapping (source structure)");
   auto *targetBreakPoint = fnBreakPoint(target);
   if (targetBreakPoint && sourceBreakPoint->GetStructure() != ROOT::ENTupleStructure::kCollection)
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
      return R__FAIL("field mapping structure mismatch: " + source->GetFieldName() + " --> " + target->GetFieldName());
   }

   // Either source or target have no collection as a parent, but the other one has; that doesn't fit
   return R__FAIL("field mapping structure mismatch: " + source->GetFieldName() + " --> " + target->GetFieldName());
}

ROOT::RResult<void>
ROOT::Internal::RProjectedFields::Add(std::unique_ptr<ROOT::RFieldBase> field, const FieldMap_t &fieldMap)
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

const ROOT::RFieldBase *ROOT::Internal::RProjectedFields::GetSourceField(const ROOT::RFieldBase *target) const
{
   if (auto it = fFieldMap.find(target); it != fFieldMap.end())
      return it->second;
   return nullptr;
}

std::unique_ptr<ROOT::Internal::RProjectedFields>
ROOT::Internal::RProjectedFields::Clone(const RNTupleModel &newModel) const
{
   auto cloneFieldZero =
      std::unique_ptr<ROOT::RFieldZero>(static_cast<ROOT::RFieldZero *>(fFieldZero->Clone("").release()));
   auto clone = std::unique_ptr<RProjectedFields>(new RProjectedFields(std::move(cloneFieldZero)));
   clone->fModel = &newModel;
   // TODO(jblomer): improve quadratic search to re-wire the field mappings given the new model and the cloned
   // projected fields. Not too critical as we generally expect a limited number of projected fields
   for (const auto &[k, v] : fFieldMap) {
      for (const auto &f : clone->GetFieldZero()) {
         if (f.GetQualifiedFieldName() == k->GetQualifiedFieldName()) {
            clone->fFieldMap[&f] = &newModel.GetConstField(v->GetQualifiedFieldName());
            break;
         }
      }
   }
   return clone;
}

ROOT::RNTupleModel::RUpdater::RUpdater(ROOT::RNTupleWriter &writer)
   : fWriter(writer), fOpenChangeset(fWriter.GetUpdatableModel())
{
}

void ROOT::RNTupleModel::RUpdater::BeginUpdate()
{
   fOpenChangeset.fModel.Unfreeze();
   // We set the model ID to zero until CommitUpdate(). That prevents calls to RNTupleWriter::Fill() in the middle
   // of updates
   std::swap(fOpenChangeset.fModel.fModelId, fNewModelId);
}

void ROOT::RNTupleModel::RUpdater::CommitUpdate()
{
   fOpenChangeset.fModel.Freeze();
   std::swap(fOpenChangeset.fModel.fModelId, fNewModelId);
   if (fOpenChangeset.IsEmpty())
      return;
   Internal::RNTupleModelChangeset toCommit{fOpenChangeset.fModel};
   std::swap(fOpenChangeset.fAddedFields, toCommit.fAddedFields);
   std::swap(fOpenChangeset.fAddedProjectedFields, toCommit.fAddedProjectedFields);
   fWriter.GetSink().UpdateSchema(toCommit, fWriter.GetNEntries());
}

void ROOT::Internal::RNTupleModelChangeset::AddField(std::unique_ptr<ROOT::RFieldBase> field)
{
   auto fieldp = field.get();
   fModel.AddField(std::move(field));
   fAddedFields.emplace_back(fieldp);
}

void ROOT::RNTupleModel::RUpdater::AddField(std::unique_ptr<ROOT::RFieldBase> field)
{
   fOpenChangeset.AddField(std::move(field));
}

ROOT::RResult<void> ROOT::Internal::RNTupleModelChangeset::AddProjectedField(std::unique_ptr<ROOT::RFieldBase> field,
                                                                             RNTupleModel::FieldMappingFunc_t mapping)
{
   auto fieldp = field.get();
   auto result = fModel.AddProjectedField(std::move(field), mapping);
   if (result)
      fAddedProjectedFields.emplace_back(fieldp);
   return R__FORWARD_RESULT(result);
}

ROOT::RResult<void>
ROOT::RNTupleModel::RUpdater::AddProjectedField(std::unique_ptr<ROOT::RFieldBase> field, FieldMappingFunc_t mapping)
{
   return R__FORWARD_RESULT(fOpenChangeset.AddProjectedField(std::move(field), std::move(mapping)));
}

void ROOT::RNTupleModel::EnsureValidFieldName(std::string_view fieldName)
{
   RResult<void> nameValid = ROOT::Internal::EnsureValidNameForRNTuple(fieldName, "Field");
   if (!nameValid) {
      nameValid.Throw();
   }
   if (fieldName.empty()) {
      throw RException(R__FAIL("name cannot be empty string \"\""));
   }
   auto fieldNameStr = std::string(fieldName);
   if (fFieldNames.count(fieldNameStr) > 0)
      throw RException(R__FAIL("field name '" + fieldNameStr + "' already exists in NTuple model"));
}

void ROOT::RNTupleModel::EnsureNotFrozen() const
{
   if (IsFrozen())
      throw RException(R__FAIL("invalid attempt to modify frozen model"));
}

void ROOT::RNTupleModel::EnsureNotBare() const
{
   if (IsBare())
      throw RException(R__FAIL("invalid attempt to use default entry of bare model"));
}

ROOT::RNTupleModel::RNTupleModel(std::unique_ptr<ROOT::RFieldZero> fieldZero)
   : fFieldZero(std::move(fieldZero)), fModelId(GetNewModelId()), fSchemaId(fModelId)
{
}

std::unique_ptr<ROOT::RNTupleModel> ROOT::RNTupleModel::CreateBare()
{
   return CreateBare(std::make_unique<ROOT::RFieldZero>());
}

std::unique_ptr<ROOT::RNTupleModel> ROOT::RNTupleModel::CreateBare(std::unique_ptr<ROOT::RFieldZero> fieldZero)
{
   auto model = std::unique_ptr<RNTupleModel>(new RNTupleModel(std::move(fieldZero)));
   model->fProjectedFields = std::make_unique<Internal::RProjectedFields>(*model);
   return model;
}

std::unique_ptr<ROOT::RNTupleModel> ROOT::RNTupleModel::Create()
{
   return Create(std::make_unique<ROOT::RFieldZero>());
}

std::unique_ptr<ROOT::RNTupleModel> ROOT::RNTupleModel::Create(std::unique_ptr<ROOT::RFieldZero> fieldZero)
{
   auto model = CreateBare(std::move(fieldZero));
   model->fDefaultEntry = std::unique_ptr<ROOT::REntry>(new ROOT::REntry(model->fModelId, model->fSchemaId));
   return model;
}

std::unique_ptr<ROOT::RNTupleModel> ROOT::RNTupleModel::Clone() const
{
   auto cloneModel = std::unique_ptr<RNTupleModel>(new RNTupleModel(
      std::unique_ptr<ROOT::RFieldZero>(static_cast<ROOT::RFieldZero *>(fFieldZero->Clone("").release()))));
   cloneModel->fModelId = GetNewModelId();
   // For a frozen model, we can keep the schema id because adding new fields is forbidden. It is reset in Unfreeze()
   // if called by the user.
   if (IsFrozen()) {
      cloneModel->fSchemaId = fSchemaId;
   } else {
      cloneModel->fSchemaId = cloneModel->fModelId;
   }
   cloneModel->fModelState = (fModelState == EState::kExpired) ? EState::kFrozen : fModelState;
   cloneModel->fFieldNames = fFieldNames;
   cloneModel->fDescription = fDescription;
   cloneModel->fProjectedFields = fProjectedFields->Clone(*cloneModel);
   cloneModel->fRegisteredSubfields = fRegisteredSubfields;
   if (fDefaultEntry) {
      cloneModel->fDefaultEntry =
         std::unique_ptr<ROOT::REntry>(new ROOT::REntry(cloneModel->fModelId, cloneModel->fSchemaId));
      for (const auto &f : cloneModel->fFieldZero->GetMutableSubfields()) {
         cloneModel->fDefaultEntry->AddValue(f->CreateValue());
      }
      for (const auto &f : cloneModel->fRegisteredSubfields) {
         cloneModel->AddSubfield(f, *cloneModel->fDefaultEntry);
      }
   }
   return cloneModel;
}

ROOT::RFieldBase *ROOT::RNTupleModel::FindField(std::string_view fieldName) const
{
   if (fieldName.empty())
      return nullptr;

   auto *field = static_cast<ROOT::RFieldBase *>(fFieldZero.get());
   for (auto subfieldName : ROOT::Split(fieldName, ".")) {
      const auto subfields = field->GetMutableSubfields();
      auto it = std::find_if(subfields.begin(), subfields.end(),
                             [&](const auto *f) { return f->GetFieldName() == subfieldName; });
      if (it != subfields.end()) {
         field = *it;
      } else {
         field = nullptr;
         break;
      }
   }

   return field;
}

void ROOT::RNTupleModel::AddField(std::unique_ptr<ROOT::RFieldBase> field)
{
   EnsureNotFrozen();
   if (!field)
      throw RException(R__FAIL("null field"));
   EnsureValidFieldName(field->GetFieldName());

   if (fDefaultEntry)
      fDefaultEntry->AddValue(field->CreateValue());
   fFieldNames.insert(field->GetFieldName());
   fFieldZero->Attach(std::move(field));
}

void ROOT::RNTupleModel::AddSubfield(std::string_view qualifiedFieldName, ROOT::REntry &entry,
                                     bool initializeValue) const
{
   auto field = FindField(qualifiedFieldName);
   if (initializeValue)
      entry.AddValue(field->CreateValue());
   else
      entry.AddValue(field->BindValue(nullptr));
}

void ROOT::RNTupleModel::RegisterSubfield(std::string_view qualifiedFieldName)
{
   if (qualifiedFieldName.empty())
      throw RException(R__FAIL("no field name provided"));

   if (fFieldNames.find(std::string(qualifiedFieldName)) != fFieldNames.end()) {
      throw RException(
         R__FAIL("cannot register top-level field \"" + std::string(qualifiedFieldName) + "\" as a subfield"));
   }

   if (fRegisteredSubfields.find(std::string(qualifiedFieldName)) != fRegisteredSubfields.end())
      throw RException(R__FAIL("subfield \"" + std::string(qualifiedFieldName) + "\" already registered"));

   EnsureNotFrozen();

   auto *field = FindField(qualifiedFieldName);
   if (!field) {
      throw RException(R__FAIL("could not find subfield \"" + std::string(qualifiedFieldName) + "\" in model"));
   }

   auto parent = field->GetParent();
   while (parent && !parent->GetFieldName().empty()) {
      if (parent->GetStructure() == ROOT::ENTupleStructure::kCollection || parent->GetNRepetitions() > 0 ||
          parent->GetStructure() == ROOT::ENTupleStructure::kVariant) {
         throw RException(R__FAIL(
            "registering a subfield as part of a collection, fixed-sized array or std::variant is not supported"));
      }
      parent = parent->GetParent();
   }

   if (fDefaultEntry)
      AddSubfield(qualifiedFieldName, *fDefaultEntry);
   fRegisteredSubfields.emplace(qualifiedFieldName);
}

ROOT::RResult<void>
ROOT::RNTupleModel::AddProjectedField(std::unique_ptr<ROOT::RFieldBase> field, FieldMappingFunc_t mapping)
{
   EnsureNotFrozen();
   if (!field)
      return R__FAIL("null field");
   auto fieldName = field->GetFieldName();

   Internal::RProjectedFields::FieldMap_t fieldMap;
   auto sourceField = FindField(mapping(fieldName));
   if (!sourceField)
      return R__FAIL("no such field: " + mapping(fieldName));
   fieldMap[field.get()] = sourceField;
   for (const auto &subField : *field) {
      sourceField = FindField(mapping(subField.GetQualifiedFieldName()));
      if (!sourceField)
         return R__FAIL("no such field: " + mapping(subField.GetQualifiedFieldName()));
      fieldMap[&subField] = sourceField;
   }

   EnsureValidFieldName(fieldName);
   auto result = fProjectedFields->Add(std::move(field), fieldMap);
   if (!result) {
      return R__FORWARD_ERROR(result);
   }
   fFieldNames.insert(fieldName);
   return RResult<void>::Success();
}

ROOT::RFieldZero &ROOT::RNTupleModel::GetMutableFieldZero()
{
   if (IsFrozen())
      throw RException(R__FAIL("invalid attempt to get mutable zero field of frozen model"));
   return *fFieldZero;
}

ROOT::RFieldBase &ROOT::RNTupleModel::GetMutableField(std::string_view fieldName)
{
   if (IsFrozen())
      throw RException(R__FAIL("invalid attempt to get mutable field of frozen model"));
   auto f = FindField(fieldName);
   if (!f)
      throw RException(R__FAIL("invalid field: " + std::string(fieldName)));

   return *f;
}

const ROOT::RFieldBase &ROOT::RNTupleModel::GetConstField(std::string_view fieldName) const
{
   auto f = FindField(fieldName);
   if (!f)
      throw RException(R__FAIL("invalid field: " + std::string(fieldName)));

   return *f;
}

ROOT::REntry &ROOT::RNTupleModel::GetDefaultEntry()
{
   EnsureNotBare();
   return *fDefaultEntry;
}

const ROOT::REntry &ROOT::RNTupleModel::GetDefaultEntry() const
{
   if (!IsFrozen())
      throw RException(R__FAIL("invalid attempt to get default entry of unfrozen model"));
   EnsureNotBare();
   return *fDefaultEntry;
}

std::unique_ptr<ROOT::REntry> ROOT::RNTupleModel::CreateEntry() const
{
   switch (fModelState) {
   case EState::kBuilding: throw RException(R__FAIL("invalid attempt to create entry of unfrozen model"));
   case EState::kExpired: throw RException(R__FAIL("invalid attempt to create entry of expired model"));
   case EState::kFrozen: break;
   }

   auto entry = std::unique_ptr<ROOT::REntry>(new ROOT::REntry(fModelId, fSchemaId));
   for (const auto &f : fFieldZero->GetMutableSubfields()) {
      entry->AddValue(f->CreateValue());
   }
   for (const auto &f : fRegisteredSubfields) {
      AddSubfield(f, *entry);
   }
   return entry;
}

std::unique_ptr<ROOT::REntry> ROOT::RNTupleModel::CreateBareEntry() const
{
   switch (fModelState) {
   case EState::kBuilding: throw RException(R__FAIL("invalid attempt to create entry of unfrozen model"));
   case EState::kExpired: throw RException(R__FAIL("invalid attempt to create entry of expired model"));
   case EState::kFrozen: break;
   }

   auto entry = std::unique_ptr<ROOT::REntry>(new ROOT::REntry(fModelId, fSchemaId));
   for (const auto &f : fFieldZero->GetMutableSubfields()) {
      entry->AddValue(f->BindValue(nullptr));
   }
   for (const auto &f : fRegisteredSubfields) {
      AddSubfield(f, *entry, false /* initializeValue */);
   }
   return entry;
}

std::unique_ptr<ROOT::Experimental::Detail::RRawPtrWriteEntry> ROOT::RNTupleModel::CreateRawPtrWriteEntry() const
{
   switch (fModelState) {
   case EState::kBuilding: throw RException(R__FAIL("invalid attempt to create entry of unfrozen model"));
   case EState::kExpired: throw RException(R__FAIL("invalid attempt to create entry of expired model"));
   case EState::kFrozen: break;
   }

   auto entry = std::unique_ptr<Experimental::Detail::RRawPtrWriteEntry>(
      new Experimental::Detail::RRawPtrWriteEntry(fModelId, fSchemaId));
   for (const auto &f : fFieldZero->GetMutableSubfields()) {
      entry->AddField(*f);
   }
   // fRegisteredSubfields are not relevant for writing
   return entry;
}

ROOT::RFieldToken ROOT::RNTupleModel::GetToken(std::string_view fieldName) const
{
   const auto &topLevelFields = fFieldZero->GetConstSubfields();
   auto it = std::find_if(topLevelFields.begin(), topLevelFields.end(),
                          [&fieldName](const ROOT::RFieldBase *f) { return f->GetFieldName() == fieldName; });

   if (it == topLevelFields.end()) {
      throw RException(R__FAIL("invalid field name: " + std::string(fieldName)));
   }
   return ROOT::RFieldToken(std::distance(topLevelFields.begin(), it), fSchemaId);
}

ROOT::RFieldBase::RBulkValues ROOT::RNTupleModel::CreateBulk(std::string_view fieldName) const
{
   switch (fModelState) {
   case EState::kBuilding: throw RException(R__FAIL("invalid attempt to create bulk of unfrozen model"));
   case EState::kExpired: throw RException(R__FAIL("invalid attempt to create bulk of expired model"));
   case EState::kFrozen: break;
   }

   auto f = FindField(fieldName);
   if (!f)
      throw RException(R__FAIL("no such field: " + std::string(fieldName)));
   return f->CreateBulk();
}

void ROOT::RNTupleModel::Expire()
{
   switch (fModelState) {
   case EState::kExpired: return;
   case EState::kBuilding: throw RException(R__FAIL("invalid attempt to expire unfrozen model"));
   case EState::kFrozen: break;
   }

   // Ensure that Fill() does not work anymore
   fModelId = 0;
   fModelState = EState::kExpired;
}

void ROOT::RNTupleModel::Unfreeze()
{
   switch (fModelState) {
   case EState::kBuilding: return;
   case EState::kExpired: throw RException(R__FAIL("invalid attempt to unfreeze expired model"));
   case EState::kFrozen: break;
   }

   fModelId = GetNewModelId();
   fSchemaId = fModelId;
   if (fDefaultEntry) {
      fDefaultEntry->fModelId = fModelId;
      fDefaultEntry->fSchemaId = fSchemaId;
   }
   fModelState = EState::kBuilding;
}

void ROOT::RNTupleModel::Freeze()
{
   if (fModelState == EState::kExpired)
      throw RException(R__FAIL("invalid attempt to freeze expired model"));

   fModelState = EState::kFrozen;
}

void ROOT::RNTupleModel::SetDescription(std::string_view description)
{
   EnsureNotFrozen();
   fDescription = std::string(description);
}

std::size_t ROOT::RNTupleModel::EstimateWriteMemoryUsage(const ROOT::RNTupleWriteOptions &options) const
{
   std::size_t bytes = 0;
   std::size_t minPageBufferSize = 0;

   // Start with the size of the page buffers used to fill a persistent sink
   std::size_t nColumns = 0;
   for (auto &&field : *fFieldZero) {
      for (const auto &r : field.GetColumnRepresentatives()) {
         nColumns += r.size();
         minPageBufferSize += r.size() * options.GetInitialUnzippedPageSize();
      }
   }
   bytes = std::min(options.GetPageBufferBudget(), nColumns * options.GetMaxUnzippedPageSize());

   // If using buffered writing with RPageSinkBuf, we create a clone of the model and keep at least
   // the compressed pages in memory.
   if (options.GetUseBufferedWrite()) {
      bytes += minPageBufferSize;
      // Use the target cluster size as an estimate for all compressed pages combined.
      bytes += options.GetApproxZippedClusterSize();
      int compression = options.GetCompression();
      if (compression != 0 && options.GetUseImplicitMT() == ROOT::RNTupleWriteOptions::EImplicitMT::kDefault) {
         // With IMT, compression happens asynchronously which means that the uncompressed pages also stay around. Use a
         // compression factor of 2x as a very rough estimate.
         bytes += 2 * options.GetApproxZippedClusterSize();
      }
   }

   return bytes;
}

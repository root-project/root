/// \file RNTupleComposer.cxx
/// \author Florine de Geus <florine.de.geus@cern.ch>
/// \date 2024-03-26
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleComposer.hxx>

#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/StringUtils.hxx>

#include <TDirectory.h>

#include <iomanip>

std::unique_ptr<ROOT::Internal::RPageSource> ROOT::Experimental::RNTupleOpenSpec::CreatePageSource() const
{
   if (const std::string *storagePath = std::get_if<std::string>(&fStorage))
      return ROOT::Internal::RPageSource::Create(fNTupleName, *storagePath);

   auto dir = std::get<TDirectory *>(fStorage);
   auto ntuple = std::unique_ptr<ROOT::RNTuple>(dir->Get<ROOT::RNTuple>(fNTupleName.c_str()));
   return ROOT::Internal::RPageSourceFile::CreateFromAnchor(*ntuple);
}

std::unique_ptr<ROOT::Experimental::RNTupleComposer>
ROOT::Experimental::RNTupleComposer::Create(RNTupleOpenSpec ntuple, std::string_view processorName)
{
   return std::unique_ptr<RNTupleSingleComposer>(new RNTupleSingleComposer(std::move(ntuple), processorName));
}

std::unique_ptr<ROOT::Experimental::RNTupleComposer>
ROOT::Experimental::RNTupleComposer::CreateChain(std::vector<RNTupleOpenSpec> ntuples, std::string_view processorName)
{
   if (ntuples.empty())
      throw RException(R__FAIL("at least one RNTuple must be provided"));

   std::vector<std::unique_ptr<RNTupleComposer>> innerProcessors;
   innerProcessors.reserve(ntuples.size());

   for (auto &ntuple : ntuples) {
      innerProcessors.emplace_back(Create(std::move(ntuple)));
   }

   return CreateChain(std::move(innerProcessors), processorName);
}

std::unique_ptr<ROOT::Experimental::RNTupleComposer>
ROOT::Experimental::RNTupleComposer::CreateChain(std::vector<std::unique_ptr<RNTupleComposer>> innerProcessors,
                                                 std::string_view processorName)
{
   if (innerProcessors.empty())
      throw RException(R__FAIL("at least one inner processor must be provided"));

   return std::unique_ptr<RNTupleChainComposer>(new RNTupleChainComposer(std::move(innerProcessors), processorName));
}

std::unique_ptr<ROOT::Experimental::RNTupleComposer>
ROOT::Experimental::RNTupleComposer::CreateJoin(RNTupleOpenSpec primaryNTuple, RNTupleOpenSpec auxNTuple,
                                                const std::vector<std::string> &joinFields,
                                                std::string_view processorName)
{
   if (joinFields.size() > 4) {
      throw RException(R__FAIL("a maximum of four join fields is allowed"));
   }

   if (std::unordered_set(joinFields.begin(), joinFields.end()).size() < joinFields.size()) {
      throw RException(R__FAIL("join fields must be unique"));
   }

   std::unique_ptr<RNTupleComposer> primaryProcessor = Create(primaryNTuple, processorName);

   std::unique_ptr<RNTupleComposer> auxProcessor = Create(auxNTuple);

   return CreateJoin(std::move(primaryProcessor), std::move(auxProcessor), joinFields, processorName);
}

std::unique_ptr<ROOT::Experimental::RNTupleComposer>
ROOT::Experimental::RNTupleComposer::CreateJoin(std::unique_ptr<RNTupleComposer> primaryProcessor,
                                                std::unique_ptr<RNTupleComposer> auxProcessor,
                                                const std::vector<std::string> &joinFields,
                                                std::string_view processorName)
{
   if (joinFields.size() > 4) {
      throw RException(R__FAIL("a maximum of four join fields is allowed"));
   }

   if (std::unordered_set(joinFields.begin(), joinFields.end()).size() < joinFields.size()) {
      throw RException(R__FAIL("join fields must be unique"));
   }

   return std::unique_ptr<RNTupleJoinComposer>(
      new RNTupleJoinComposer(std::move(primaryProcessor), std::move(auxProcessor), joinFields, processorName));
}

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleSingleComposer::RNTupleSingleComposer(RNTupleOpenSpec ntuple, std::string_view processorName)
   : RNTupleComposer(processorName), fNTupleSpec(std::move(ntuple))
{
   if (fProcessorName.empty()) {
      fProcessorName = fNTupleSpec.fNTupleName;
   }
}

void ROOT::Experimental::RNTupleSingleComposer::Initialize(
   std::shared_ptr<ROOT::Experimental::Internal::RNTupleProcessorEntry> entry)
{
   // The processor has already been initialized.
   if (IsInitialized())
      return;

   if (!entry)
      fEntry = std::make_shared<Internal::RNTupleProcessorEntry>();
   else
      fEntry = entry;

   fPageSource = fNTupleSpec.CreatePageSource();
   fPageSource->Attach();

   fNEntries = fPageSource->GetNEntries();
}

bool ROOT::Experimental::RNTupleSingleComposer::CanReadFieldFromDisk(std::string_view fieldName)
{
   Initialize();
   auto desc = fPageSource->GetSharedDescriptorGuard();
   auto fieldZeroId = desc->GetFieldZeroId();

   // TODO handle subfields
   return desc->FindFieldId(fieldName, fieldZeroId) != ROOT::kInvalidDescriptorId;
}

std::unique_ptr<ROOT::RFieldBase>
ROOT::Experimental::RNTupleSingleComposer::CreateAndConnectField(const std::string &qualifiedFieldName,
                                                                 const std::string &typeName)
{
   assert(fPageSource);

   std::string onDiskFieldName = qualifiedFieldName;

   // Strip the "_join" prefix (for join fields) from the field name, if present.
   if (onDiskFieldName.find("_join.") == 0) {
      onDiskFieldName = onDiskFieldName.substr(6);
   }

   const auto &desc = fPageSource->GetSharedDescriptorGuard().GetRef();
   ROOT::RFieldZero fieldZero;
   ROOT::Internal::SetAllowFieldSubstitutions(fieldZero, true);

   const auto onDiskFieldId = desc.FindFieldId(onDiskFieldName);

   if (onDiskFieldId == kInvalidDescriptorId) {
      return nullptr;
   }

   std::unique_ptr<ROOT::RFieldBase> field;
   if (typeName.empty()) {
      const auto &fieldDesc = desc.GetFieldDescriptor(onDiskFieldId);
      field = fieldDesc.CreateField(desc);
   } else {
      // Strip the parent field name prefix(es), if present.
      std::string subfieldName = onDiskFieldName;
      auto posDot = onDiskFieldName.find_last_of('.');
      if (posDot != std::string::npos)
         subfieldName = onDiskFieldName.substr(posDot + 1);

      field = ROOT::RFieldBase::Create(subfieldName, typeName).Unwrap();
   }

   field->SetOnDiskId(onDiskFieldId);
   fieldZero.Attach(std::move(field));
   ROOT::Internal::CallConnectPageSourceOnField(fieldZero, *fPageSource);
   return std::move(fieldZero.ReleaseSubfields()[0]);
}

ROOT::Experimental::Internal::RNTupleProcessorEntry::FieldIndex_t
ROOT::Experimental::RNTupleSingleComposer::AddFieldToEntry(const std::string &fieldName, const std::string &typeName,
                                                           void *valuePtr,
                                                           const Internal::RNTupleProcessorProvenance &provenance)
{
   auto fieldIdx = fEntry->FindFieldIndex(fieldName, typeName);
   if (!fieldIdx) {
      // Strip the processor name prefix(es), if present.
      std::string qualifiedFieldName = fieldName;
      if (provenance.IsPresentInFieldName(qualifiedFieldName)) {
         qualifiedFieldName = qualifiedFieldName.substr(provenance.Get().size() + 1);
      }

      auto field = CreateAndConnectField(qualifiedFieldName, typeName);

      if (!field) {
         throw RException(R__FAIL("cannot register field with name \"" + qualifiedFieldName +
                                  "\" because it is not present in the on-disk information of the RNTuple(s) this "
                                  "processor is created from"));
      }

      fieldIdx = fEntry->AddField(qualifiedFieldName, std::move(field), valuePtr, provenance);
   }

   return *fieldIdx;
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleSingleComposer::LoadEntry(ROOT::NTupleSize_t entryNumber)
{
   if (entryNumber >= fNEntries || !fEntry)
      return kInvalidNTupleIndex;

   for (auto fieldIdx : fFieldIdxs) {
      fEntry->ReadValue(fieldIdx, entryNumber);
   }

   fCurrentEntryNumber = entryNumber;
   return entryNumber;
}

void ROOT::Experimental::RNTupleSingleComposer::Connect(
   const std::unordered_set<ROOT::Experimental::Internal::RNTupleProcessorEntry::FieldIndex_t> &fieldIdxs,
   const Internal::RNTupleProcessorProvenance & /* provenance */, bool updateFields)
{
   Initialize();

   fFieldIdxs = fieldIdxs;

   if (updateFields) {
      for (const auto &fieldIdx : fFieldIdxs) {
         const auto &currField = fEntry->GetValue(fieldIdx).GetField();
         auto newField = CreateAndConnectField(fEntry->GetQualifiedFieldName(fieldIdx), currField.GetTypeName());

         fEntry->UpdateField(fieldIdx, std::move(newField));
      }
   }
}

void ROOT::Experimental::RNTupleSingleComposer::AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable,
                                                                      ROOT::NTupleSize_t entryOffset)
{
   Connect(fFieldIdxs);
   joinTable.Add(*fPageSource, Internal::RNTupleJoinTable::kDefaultPartitionKey, entryOffset);
}

void ROOT::Experimental::RNTupleSingleComposer::PrintStructureImpl(std::ostream &output) const
{
   static constexpr int width = 32;

   std::string ntupleNameTrunc = fNTupleSpec.fNTupleName.substr(0, width - 4);
   if (ntupleNameTrunc.size() < fNTupleSpec.fNTupleName.size())
      ntupleNameTrunc = fNTupleSpec.fNTupleName.substr(0, width - 6) + "..";

   output << "+" << std::setfill('-') << std::setw(width - 1) << "+\n";
   output << std::setfill(' ') << "| " << ntupleNameTrunc << std::setw(width - 2 - ntupleNameTrunc.size()) << " |\n";

   if (const std::string *storage = std::get_if<std::string>(&fNTupleSpec.fStorage)) {
      std::string storageTrunc = storage->substr(0, width - 5);
      if (storageTrunc.size() < storage->size())
         storageTrunc = storage->substr(0, width - 8) + "...";

      output << std::setfill(' ') << "| " << storageTrunc << std::setw(width - 2 - storageTrunc.size()) << " |\n";
   } else {
      output << "| " << std::setw(width - 2) << " |\n";
   }

   output << "+" << std::setfill('-') << std::setw(width - 1) << "+\n";
}

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleChainComposer::RNTupleChainComposer(std::vector<std::unique_ptr<RNTupleComposer>> processors,
                                                               std::string_view processorName)
   : RNTupleComposer(processorName), fInnerProcessors(std::move(processors))
{
   if (fProcessorName.empty()) {
      // `CreateChain` ensures there is at least one inner processor.
      fProcessorName = fInnerProcessors[0]->GetProcessorName();
   }

   fInnerNEntries.assign(fInnerProcessors.size(), kInvalidNTupleIndex);
}

void ROOT::Experimental::RNTupleChainComposer::Initialize(
   std::shared_ptr<ROOT::Experimental::Internal::RNTupleProcessorEntry> entry)
{
   if (IsInitialized())
      return;

   if (!entry)
      fEntry = std::make_shared<Internal::RNTupleProcessorEntry>();
   else
      fEntry = entry;

   fInnerProcessors[0]->Initialize(fEntry);
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleChainComposer::GetNEntries()
{
   if (fNEntries == kInvalidNTupleIndex) {
      fNEntries = 0;

      for (unsigned i = 0; i < fInnerProcessors.size(); ++i) {
         if (fInnerNEntries[i] == kInvalidNTupleIndex) {
            fInnerNEntries[i] = fInnerProcessors[i]->GetNEntries();
         }

         fNEntries += fInnerNEntries[i];
      }
   }

   return fNEntries;
}

void ROOT::Experimental::RNTupleChainComposer::Connect(
   const std::unordered_set<ROOT::Experimental::Internal::RNTupleProcessorEntry::FieldIndex_t> &fieldIdxs,
   const Internal::RNTupleProcessorProvenance &provenance, bool /* updateFields */)
{
   Initialize();
   fFieldIdxs = fieldIdxs;
   fProvenance = provenance;
   ConnectInnerProcessor(fCurrentProcessorNumber);
}

void ROOT::Experimental::RNTupleChainComposer::ConnectInnerProcessor(std::size_t processorNumber)
{
   auto &innerProc = fInnerProcessors[processorNumber];
   innerProc->Initialize(fEntry);
   innerProc->Connect(fFieldIdxs, fProvenance, /*updateFields=*/true);
}

ROOT::Experimental::Internal::RNTupleProcessorEntry::FieldIndex_t
ROOT::Experimental::RNTupleChainComposer::AddFieldToEntry(const std::string &fieldName, const std::string &typeName,
                                                          void *valuePtr,
                                                          const Internal::RNTupleProcessorProvenance &provenance)
{
   return fInnerProcessors[fCurrentProcessorNumber]->AddFieldToEntry(fieldName, typeName, valuePtr, provenance);
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleChainComposer::LoadEntry(ROOT::NTupleSize_t entryNumber)
{
   // If the requested entry number is lower than the current entry number, we have to again localise the correct local
   // entry number starting from the first processor in the chain. Otherwise, we can continue looking from the inner
   // processor that is currently connected, which is much faster when the chain consists of many inner processors.
   if (entryNumber < fCurrentEntryNumber) {
      fCurrentProcessorNumber = 0;
      ConnectInnerProcessor(fCurrentProcessorNumber);
   }

   std::size_t currProcessorNumber = fCurrentProcessorNumber;
   ROOT::NTupleSize_t entriesSeen = 0;
   for (unsigned i = 0; i < currProcessorNumber; ++i) {
      if (fInnerNEntries[i] == kInvalidNTupleIndex) {
         fInnerNEntries[i] = fInnerProcessors[i]->GetNEntries();
      }
      entriesSeen += fInnerNEntries[i];
   }
   ROOT::NTupleSize_t localEntryNumber = entryNumber - entriesSeen;

   // As long as the entry fails to load from the current processor, we decrement the local entry number with the number
   // of entries in this processor and try with the next processor until we find the correct local entry number.
   while (fInnerProcessors[currProcessorNumber]->LoadEntry(localEntryNumber) == kInvalidNTupleIndex) {
      if (fInnerNEntries[currProcessorNumber] == kInvalidNTupleIndex) {
         fInnerNEntries[currProcessorNumber] = fInnerProcessors[currProcessorNumber]->GetNEntries();
      }

      localEntryNumber -= fInnerNEntries[currProcessorNumber];

      // The provided global entry number is larger than the number of available entries.
      if (++currProcessorNumber >= fInnerProcessors.size())
         return kInvalidNTupleIndex;

      ConnectInnerProcessor(currProcessorNumber);
   }

   fCurrentProcessorNumber = currProcessorNumber;
   fCurrentEntryNumber = entryNumber;
   return entryNumber;
}

void ROOT::Experimental::RNTupleChainComposer::AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable,
                                                                     ROOT::NTupleSize_t entryOffset)
{
   for (unsigned i = 0; i < fInnerProcessors.size(); ++i) {
      const auto &innerProc = fInnerProcessors[i];
      // TODO can this be done (more) lazily? I.e. only when a match cannot be found in the current inner proc?
      // At this stage, we don't want to fully initialize (i.e. set the entry of) the inner processor yet
      innerProc->Initialize(nullptr);
      innerProc->AddEntriesToJoinTable(joinTable, entryOffset);
      entryOffset += innerProc->GetNEntries();
   }
}

void ROOT::Experimental::RNTupleChainComposer::PrintStructureImpl(std::ostream &output) const
{
   for (const auto &innerProc : fInnerProcessors) {
      innerProc->PrintStructure(output);
   }
}

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleJoinComposer::RNTupleJoinComposer(std::unique_ptr<RNTupleComposer> primaryProcessor,
                                                               std::unique_ptr<RNTupleComposer> auxProcessor,
                                                               const std::vector<std::string> &joinFields,
                                                               std::string_view processorName)
   : RNTupleComposer(processorName),
     fPrimaryProcessor(std::move(primaryProcessor)),
     fAuxiliaryProcessor(std::move(auxProcessor)),
     fJoinFieldNames(joinFields)
{
   if (fProcessorName.empty()) {
      fProcessorName = fPrimaryProcessor->GetProcessorName();
   }
}

void ROOT::Experimental::RNTupleJoinComposer::Initialize(
   std::shared_ptr<ROOT::Experimental::Internal::RNTupleProcessorEntry> entry)
{
   if (IsInitialized())
      return;

   if (!entry)
      fEntry = std::make_shared<Internal::RNTupleProcessorEntry>();
   else
      fEntry = entry;

   fPrimaryProcessor->Initialize(fEntry);
   fAuxiliaryProcessor->Initialize(fEntry);

   if (!fJoinFieldNames.empty()) {
      for (const auto &joinField : fJoinFieldNames) {
         if (!fPrimaryProcessor->CanReadFieldFromDisk(joinField)) {
            throw RException(R__FAIL("could not find join field \"" + joinField + "\" in primary processor \"" +
                                     fPrimaryProcessor->GetProcessorName() + "\""));
         }
         if (!fAuxiliaryProcessor->CanReadFieldFromDisk(joinField)) {
            throw RException(R__FAIL("could not find join field \"" + joinField + "\" in auxiliary processor \"" +
                                     fAuxiliaryProcessor->GetProcessorName() + "\""));
         }

         // We prepend the name of the primary processor in this case to prevent reading from the wrong join field in
         // composed join operations.
         auto fieldIdx = AddFieldToEntry(fProcessorName + "._join." + joinField, "std::uint64_t", nullptr,
                                         Internal::RNTupleProcessorProvenance(fProcessorName));
         fJoinFieldIdxs.insert(fieldIdx);
      }

      fJoinTable = Internal::RNTupleJoinTable::Create(fJoinFieldNames);
   }
}

void ROOT::Experimental::RNTupleJoinComposer::Connect(
   const std::unordered_set<ROOT::Experimental::Internal::RNTupleProcessorEntry::FieldIndex_t> &fieldIdxs,
   const Internal::RNTupleProcessorProvenance &provenance, bool updateFields)
{
   Initialize();

   auto auxProvenance = provenance.Evolve(fAuxiliaryProcessor->GetProcessorName());
   for (const auto &fieldIdx : fieldIdxs) {
      const auto &fieldProvenance = fEntry->GetFieldProvenance(fieldIdx);
      if (fieldProvenance.Contains(auxProvenance))
         fAuxiliaryFieldIdxs.insert(fieldIdx);
      else
         fFieldIdxs.insert(fieldIdx);
   }

   fPrimaryProcessor->Connect(fFieldIdxs, provenance, updateFields);
   fAuxiliaryProcessor->Connect(fAuxiliaryFieldIdxs, auxProvenance, updateFields);
}

ROOT::Experimental::Internal::RNTupleProcessorEntry::FieldIndex_t
ROOT::Experimental::RNTupleJoinComposer::AddFieldToEntry(const std::string &fieldName, const std::string &typeName,
                                                         void *valuePtr,
                                                         const Internal::RNTupleProcessorProvenance &provenance)
{
   auto auxProvenance = provenance.Evolve(fAuxiliaryProcessor->GetProcessorName());
   if (auxProvenance.IsPresentInFieldName(fieldName)) {
      // If the primaryProcessor has a field with the name of the auxProcessor (either as a "proper" field or because
      // the primary processor itself is a join where its auxProcessor bears the same name as the current auxProcessor),
      // there will be name conflicts, so error out.
      if (fPrimaryProcessor->CanReadFieldFromDisk(fieldName)) {
         throw RException(R__FAIL("ambiguous field name: \"" + fieldName +
                                  "\" is present in the primary RNTupleComposer \"" +
                                  fPrimaryProcessor->GetProcessorName() +
                                  "\", but may also refer to a field in the auxiliary RNTupleComposer named \"" +
                                  fAuxiliaryProcessor->GetProcessorName() +
                                  "\". To avoid this ambiguity, rename the auxiliary RNTupleComposer."));
      }

      auto fieldIdx = fAuxiliaryProcessor->AddFieldToEntry(fieldName, typeName, valuePtr, auxProvenance);
      if (fieldIdx)
         fAuxiliaryFieldIdxs.insert(fieldIdx);
      return fieldIdx;
   } else {
      auto fieldIdx = fPrimaryProcessor->AddFieldToEntry(fieldName, typeName, valuePtr, provenance);
      if (fieldIdx)
         fFieldIdxs.insert(fieldIdx);
      return fieldIdx;
   }
}

void ROOT::Experimental::RNTupleJoinComposer::SetAuxiliaryFieldValidity(bool isValid)
{
   for (const auto &fieldIdx : fAuxiliaryFieldIdxs) {
      fEntry->SetFieldValidity(fieldIdx, isValid);
   }
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleJoinComposer::LoadEntry(ROOT::NTupleSize_t entryNumber)
{
   if (fPrimaryProcessor->LoadEntry(entryNumber) == kInvalidNTupleIndex) {
      for (auto fieldIdx : fFieldIdxs) {
         fEntry->SetFieldValidity(fieldIdx, false);
      }
      SetAuxiliaryFieldValidity(false);
      return kInvalidNTupleIndex;
   }

   fCurrentEntryNumber = entryNumber;

   if (!fJoinTable) {
      // The auxiliary processor's fields are valid if the entry could be loaded.
      fAuxiliaryProcessor->LoadEntry(entryNumber);
      return entryNumber;
   }

   if (!fJoinTableIsBuilt) {
      fAuxiliaryProcessor->AddEntriesToJoinTable(*fJoinTable);
      fJoinTableIsBuilt = true;
   }

   // Collect the values of the join fields for this entry.
   std::vector<ROOT::Experimental::Internal::RNTupleJoinTable::JoinValue_t> values;
   values.reserve(fJoinFieldIdxs.size());
   for (const auto &fieldIdx : fJoinFieldIdxs) {
      auto val = fEntry->GetValue(fieldIdx).GetRef<ROOT::Experimental::Internal::RNTupleJoinTable::JoinValue_t>();
      values.push_back(val);
   }

   // Find the entry index corresponding to the join field values for each auxiliary processor and load the
   // corresponding entry.
   const auto entryIdx = fJoinTable->GetEntryIndex(values);

   if (entryIdx == kInvalidNTupleIndex) {
      SetAuxiliaryFieldValidity(false);
   } else {
      SetAuxiliaryFieldValidity(true);
      for (const auto &fieldIdx : fAuxiliaryFieldIdxs) {
         fEntry->ReadValue(fieldIdx, entryIdx);
      }
   }

   return entryNumber;
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleJoinComposer::GetNEntries()
{
   if (fNEntries == kInvalidNTupleIndex)
      fNEntries = fPrimaryProcessor->GetNEntries();
   return fNEntries;
}

void ROOT::Experimental::RNTupleJoinComposer::AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable,
                                                                    ROOT::NTupleSize_t entryOffset)
{
   fPrimaryProcessor->AddEntriesToJoinTable(joinTable, entryOffset);
}

void ROOT::Experimental::RNTupleJoinComposer::PrintStructureImpl(std::ostream &output) const
{
   std::ostringstream primaryStructureStr;
   fPrimaryProcessor->PrintStructure(primaryStructureStr);
   const auto primaryStructure = ROOT::Split(primaryStructureStr.str(), "\n", /*skipEmpty=*/true);
   const auto primaryStructureWidth = primaryStructure.front().size();

   std::ostringstream auxStructureStr;
   fAuxiliaryProcessor->PrintStructure(auxStructureStr);
   const auto auxStructure = ROOT::Split(auxStructureStr.str(), "\n", /*skipEmpty=*/true);

   const auto maxLength = std::max(primaryStructure.size(), auxStructure.size());
   for (unsigned i = 0; i < maxLength; i++) {
      if (i < primaryStructure.size())
         output << primaryStructure[i];
      else
         output << std::setw(primaryStructureWidth) << "";

      if (i < auxStructure.size())
         output << " " << auxStructure[i];

      output << "\n";
   }
}

/// \file RNTupleProcessor.cxx
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

#include <ROOT/RNTupleProcessor.hxx>

#include <ROOT/RFieldBase.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/StringUtils.hxx>

#include <TDirectory.h>

#include <iomanip>

const ROOT::Internal::RNTupleProcessorEntry *
ROOT::Internal::LoadFullRNTupleProcessorEntry(ROOT::RNTupleProcessor &processor, bool includeSubfields)
{
   processor.AddAllFieldsToEntry(RNTupleProcessorProvenance(), includeSubfields);
   return processor.fEntry.get();
}

//------------------------------------------------------------------------------

std::unique_ptr<ROOT::RNTupleProcessor>
ROOT::RNTupleProcessor::Create(RNTupleInputSpec ntuple, const RNTupleProcessorOptions &options)
{
   return std::unique_ptr<RNTupleSingleProcessor>(new RNTupleSingleProcessor(std::move(ntuple), options));
}

std::unique_ptr<ROOT::RNTupleProcessor>
ROOT::RNTupleProcessor::CreateChain(std::vector<RNTupleInputSpec> ntuples, const RNTupleProcessorOptions &options)
{
   if (ntuples.empty())
      throw RException(R__FAIL("at least one RNTuple must be provided"));

   std::vector<std::unique_ptr<RNTupleProcessor>> innerProcessors;
   innerProcessors.reserve(ntuples.size());

   for (auto &ntuple : ntuples) {
      innerProcessors.emplace_back(Create(std::move(ntuple)));
   }

   return CreateChain(std::move(innerProcessors), options);
}

std::unique_ptr<ROOT::RNTupleProcessor>
ROOT::RNTupleProcessor::CreateChain(std::vector<std::unique_ptr<RNTupleProcessor>> innerProcessors,
                                    const RNTupleProcessorOptions &options)
{
   if (innerProcessors.empty())
      throw RException(R__FAIL("at least one inner processor must be provided"));

   return std::unique_ptr<RNTupleChainProcessor>(new RNTupleChainProcessor(std::move(innerProcessors), options));
}

std::unique_ptr<ROOT::RNTupleProcessor>
ROOT::Experimental::CreateRNTupleJoinProcessor(RNTupleInputSpec primaryNTuple, RNTupleInputSpec auxNTuple,
                                               const std::vector<std::string> &joinFields,
                                               const RNTupleProcessorOptions &options)
{
   if (joinFields.size() > 4) {
      throw RException(R__FAIL("a maximum of four join fields is allowed"));
   }

   if (std::unordered_set(joinFields.begin(), joinFields.end()).size() < joinFields.size()) {
      throw RException(R__FAIL("join fields must be unique"));
   }

   std::unique_ptr<RNTupleProcessor> primaryProcessor = RNTupleProcessor::Create(std::move(primaryNTuple), options);

   std::unique_ptr<RNTupleProcessor> auxProcessor = RNTupleProcessor::Create(std::move(auxNTuple));

   return CreateRNTupleJoinProcessor(std::move(primaryProcessor), std::move(auxProcessor), joinFields, options);
}

std::unique_ptr<ROOT::RNTupleProcessor>
ROOT::Experimental::CreateRNTupleJoinProcessor(std::unique_ptr<RNTupleProcessor> primaryProcessor,
                                               std::unique_ptr<RNTupleProcessor> auxProcessor,
                                               const std::vector<std::string> &joinFields,
                                               const RNTupleProcessorOptions &options)
{
   if (joinFields.size() > 4) {
      throw RException(R__FAIL("a maximum of four join fields is allowed"));
   }

   if (std::unordered_set(joinFields.begin(), joinFields.end()).size() < joinFields.size()) {
      throw RException(R__FAIL("join fields must be unique"));
   }

   return std::unique_ptr<RNTupleJoinProcessor>(
      new RNTupleJoinProcessor(std::move(primaryProcessor), std::move(auxProcessor), joinFields, options));
}

//------------------------------------------------------------------------------

ROOT::RNTupleSingleProcessor::RNTupleSingleProcessor(RNTupleInputSpec ntuple, const RNTupleProcessorOptions &options)
   : RNTupleProcessor(options), fNTupleSpec(std::move(ntuple))
{
   if (fOptions.GetProcessorName().empty()) {
      fOptions.SetProcessorName(fNTupleSpec.fNTupleName);
   }
}

namespace {
std::unique_ptr<ROOT::Internal::RPageSource> CreatePageSource(const ROOT::RNTupleInputSpec &inputSpec)
{
   if (const std::string *storagePath = std::get_if<std::string>(&inputSpec.GetStorage()))
      return ROOT::Internal::RPageSource::Create(inputSpec.GetNTupleName(), *storagePath);

   auto dir = std::get<TDirectory *>(inputSpec.GetStorage());
   auto ntuple = std::unique_ptr<ROOT::RNTuple>(dir->Get<ROOT::RNTuple>(inputSpec.GetNTupleName().c_str()));
   return ROOT::Internal::RPageSourceFile::CreateFromAnchor(*ntuple);
}
} // anonymous namespace

void ROOT::RNTupleSingleProcessor::Initialize(std::shared_ptr<Internal::RNTupleProcessorEntry> entry)
{
   // The processor has already been initialized.
   if (IsInitialized() && fPageSource)
      return;

   if (!entry)
      fEntry = std::make_shared<Internal::RNTupleProcessorEntry>();
   else
      fEntry = std::move(entry);

   fPageSource = CreatePageSource(fNTupleSpec);
   fPageSource->Attach();

   fNEntries = fPageSource->GetNEntries();
}

bool ROOT::RNTupleSingleProcessor::CanReadFieldFromDisk(std::string_view fieldName)
{
   Initialize();
   auto desc = fPageSource->GetSharedDescriptorGuard();
   auto fieldZeroId = desc->GetFieldZeroId();

   // TODO handle subfields
   return desc->FindFieldId(fieldName, fieldZeroId) != ROOT::kInvalidDescriptorId;
}

std::unique_ptr<ROOT::RFieldBase>
ROOT::RNTupleSingleProcessor::CreateAndConnectField(const std::string &qualifiedFieldName, const std::string &typeName)
{
   assert(fPageSource);

   // Strip the "R_rntproc_join_" prefix (for join fields) from the field name, if present.
   const std::string onDiskFieldName =
      qualifiedFieldName.find("R_rntproc_join_") == 0 ? qualifiedFieldName.substr(15) : qualifiedFieldName;

   auto descGuard = fPageSource->GetSharedDescriptorGuard();
   const auto &desc = descGuard.GetConstRef();

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
   return ConnectField(std::move(field));
}

std::unique_ptr<ROOT::RFieldBase> ROOT::RNTupleSingleProcessor::ConnectField(std::unique_ptr<ROOT::RFieldBase> field)
{
   ROOT::RFieldZero fieldZero;
   ROOT::Internal::SetAllowFieldSubstitutions(fieldZero, true);
   fieldZero.Attach(std::move(field));
   ROOT::Internal::CallConnectPageSourceOnField(fieldZero, *fPageSource);
   return std::move(fieldZero.ReleaseSubfields()[0]);
}

ROOT::Internal::RNTupleProcessorEntry::FieldIndex_t
ROOT::RNTupleSingleProcessor::AddFieldToEntry(const std::string &canonicalFieldName, const std::string &typeName,
                                              void *valuePtr, const Internal::RNTupleProcessorProvenance &provenance)
{
   auto fieldIdx = fEntry->FindFieldIndex(canonicalFieldName, typeName);
   if (!fieldIdx) {
      // Strip the processor name prefix(es), if present.
      std::string qualifiedFieldName = canonicalFieldName;
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

ROOT::Internal::RNTupleProcessorEntry::FieldIndex_t
ROOT::RNTupleSingleProcessor::AddFieldToEntry(std::unique_ptr<ROOT::RFieldBase> field,
                                              const std::string &canonicalFieldName, void *valuePtr,
                                              const Internal::RNTupleProcessorProvenance &provenance)
{
   auto fieldIdx = fEntry->FindFieldIndex(canonicalFieldName, field->GetTypeName());
   if (!fieldIdx) {
      // Strip the processor name prefix(es), if present.
      std::string qualifiedFieldName = canonicalFieldName;
      if (provenance.IsPresentInFieldName(qualifiedFieldName)) {
         qualifiedFieldName = qualifiedFieldName.substr(provenance.Get().size() + 1);
      }

      field = ConnectField(std::move(field));

      if (!field) {
         throw RException(R__FAIL("cannot register field with name \"" + qualifiedFieldName +
                                  "\" because it is not present in the on-disk information of the RNTuple(s) this "
                                  "processor is created from"));
      }

      fieldIdx = fEntry->AddField(qualifiedFieldName, std::move(field), valuePtr, provenance);
   }

   return *fieldIdx;
}

void ROOT::RNTupleSingleProcessor::AddAllFieldsToEntry(const Internal::RNTupleProcessorProvenance &provenance,
                                                       bool includeSubfields)
{
   Initialize();
   auto descGuard = fPageSource->GetSharedDescriptorGuard();
   const auto &desc = descGuard.GetConstRef();
   auto fnAddSubfields = [this, &desc, &provenance](const ROOT::RFieldDescriptor &field, auto &fn) -> void {
      std::string fieldName = desc.GetQualifiedFieldName(field.GetId());
      if (auto prefix = provenance.Get(); !prefix.empty())
         fieldName = prefix + "." + fieldName;

      AddFieldToEntry(fieldName, field.GetTypeName(), nullptr, provenance);
      for (const auto &subfield : desc.GetFieldIterable(field.GetId())) {
         fn(subfield, fn);
      }
   };

   for (const auto &field : desc.GetTopLevelFields()) {
      if (includeSubfields) {
         fnAddSubfields(field, fnAddSubfields);
      } else {
         std::string fieldName = desc.GetQualifiedFieldName(field.GetId());
         if (auto prefix = provenance.Get(); !prefix.empty())
            fieldName = prefix + "." + fieldName;

         AddFieldToEntry(fieldName, field.GetTypeName(), nullptr, provenance);
      }
   }
}

void ROOT::RNTupleSingleProcessor::SetEntryMapping(ROOT::NTupleSize_t globalEntryNumber, REntryMapping &mapping)
{
   if (globalEntryNumber >= fNEntries || !fEntry)
      mapping.SetEntryNumber(kInvalidNTupleIndex);
   else
      mapping.SetEntryNumber(globalEntryNumber);
}

void ROOT::RNTupleSingleProcessor::LoadEntryImpl(const REntryMapping &mapping)
{
   assert(mapping.GetEntryNumber() != kInvalidNTupleIndex);

   for (auto fieldIdx : fFieldIdxs) {
      fEntry->ReadValue(fieldIdx, mapping.GetEntryNumber());
   }

   fNEntriesProcessed++;
}

namespace {
void ResetColumnRepresentative(ROOT::RFieldBase &field)
{
   field.SetColumnRepresentatives(ROOT::RFieldBase::RColumnRepresentations::Selection_t{});
   for (auto &subfield : field.GetMutableSubfields()) {
      ResetColumnRepresentative(*subfield);
   }
}

void SetOnDiskId(ROOT::RFieldBase &field, ROOT::DescriptorId_t onDiskId, const ROOT::RNTupleDescriptor &desc)
{
   field.SetOnDiskId(onDiskId);

   for (auto &subfield : field.GetMutableSubfields()) {
      onDiskId = desc.FindFieldId(subfield->GetQualifiedFieldName());
      SetOnDiskId(*subfield, onDiskId, desc);
   }
}
} // anonymous namespace

void ROOT::RNTupleSingleProcessor::Connect(
   const std::unordered_set<Internal::RNTupleProcessorEntry::FieldIndex_t> &fieldIdxs,
   const Internal::RNTupleProcessorProvenance & /* provenance */, bool updateFields)
{
   Initialize(fEntry);

   fFieldIdxs = fieldIdxs;

   if (updateFields) {
      auto descGuard = fPageSource->GetSharedDescriptorGuard();
      const auto &desc = descGuard.GetConstRef();

      for (const auto &fieldIdx : fFieldIdxs) {
         const auto &currField = fEntry->GetValue(fieldIdx).GetField();

         auto onDiskId = desc.FindFieldId(currField.GetQualifiedFieldName());
         bool needsClone = onDiskId != ROOT::kInvalidDescriptorId &&
                           currField.GetTypeName() == desc.GetFieldDescriptor(onDiskId).GetTypeName();
         if (const auto fieldOriginIdx = fEntry->GetFieldOriginIdx(fieldIdx);
             fieldOriginIdx != ROOT::kInvalidDescriptorId) {
            const auto &originField = fEntry->GetField(fieldOriginIdx);
            onDiskId = desc.FindFieldId(originField.GetQualifiedFieldName());
            needsClone = true;
         }

         std::unique_ptr<ROOT::RFieldBase> newField;
         if (needsClone) {
            // We have enough information to clone the existing field.
            newField = currField.Clone(currField.GetFieldName());
            // The clone's column representatives (+ its subfields') need to be reset before it can be connected to the
            // new page source.
            ResetColumnRepresentative(*newField);
            SetOnDiskId(*newField, onDiskId, desc);
            newField = ConnectField(std::move(newField));
         } else {
            // We need to create the new field from scratch (e.g. because it is a subfield without a parent field
            // attached).
            newField = CreateAndConnectField(fEntry->GetQualifiedFieldName(fieldIdx), currField.GetTypeName());
         }

         fEntry->UpdateField(fieldIdx, std::move(newField));
      }
   }
}

void ROOT::RNTupleSingleProcessor::Disconnect()
{
   if (fPageSource) {
      fEntry->ResetFields(fFieldIdxs);
      fPageSource.reset();
   }
}

void ROOT::RNTupleSingleProcessor::AddEntriesToJoinTable(Experimental::Internal::RNTupleJoinTable &joinTable,
                                                         ROOT::NTupleSize_t entryOffset)
{
   Connect(fFieldIdxs);
   joinTable.Add(*fPageSource, Experimental::Internal::RNTupleJoinTable::kDefaultPartitionKey, entryOffset);
}

void ROOT::RNTupleSingleProcessor::PrintStructureImpl(std::ostream &output) const
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

ROOT::RNTupleChainProcessor::RNTupleChainProcessor(std::vector<std::unique_ptr<RNTupleProcessor>> processors,
                                                   const RNTupleProcessorOptions &options)
   : RNTupleProcessor(options), fInnerProcessors(std::move(processors))
{
   if (fOptions.GetProcessorName().empty()) {
      // `CreateChain` ensures there is at least one inner processor.
      fOptions.SetProcessorName(fInnerProcessors[0]->fOptions.GetProcessorName());
   }

   fInnerNEntries.assign(fInnerProcessors.size(), kInvalidNTupleIndex);
}

void ROOT::RNTupleChainProcessor::Initialize(std::shared_ptr<Internal::RNTupleProcessorEntry> entry)
{
   if (IsInitialized())
      return;

   if (!entry)
      fEntry = std::make_shared<Internal::RNTupleProcessorEntry>();
   else
      fEntry = std::move(entry);

   fInnerProcessors[0]->Initialize(fEntry);
}

ROOT::NTupleSize_t ROOT::RNTupleChainProcessor::GetNEntries()
{
   if (fNEntries == kInvalidNTupleIndex) {
      fNEntries = 0;

      for (unsigned i = 0; i < fInnerProcessors.size(); ++i) {
         if (fInnerNEntries[i] == kInvalidNTupleIndex) {
            fInnerProcessors[i]->Initialize(fEntry);
            fInnerNEntries[i] = fInnerProcessors[i]->GetNEntries();
         }

         fNEntries += fInnerNEntries[i];
      }
   }

   return fNEntries;
}

void ROOT::RNTupleChainProcessor::Connect(
   const std::unordered_set<Internal::RNTupleProcessorEntry::FieldIndex_t> &fieldIdxs,
   const Internal::RNTupleProcessorProvenance &provenance, bool /* updateFields */)
{
   Initialize();
   fFieldIdxs = fieldIdxs;
   fProvenance = provenance;
   ConnectInnerProcessor(fCurrentProcessorNumber);
}

void ROOT::RNTupleChainProcessor::Disconnect()
{
   for (const auto &innerProc : fInnerProcessors) {
      innerProc->Disconnect();
   }
}

void ROOT::RNTupleChainProcessor::ConnectInnerProcessor(std::size_t processorNumber)
{
   if (fCurrentProcessorNumber != processorNumber) {
      fInnerProcessors[fCurrentProcessorNumber]->Disconnect();
      fCurrentProcessorNumber = processorNumber;
   }

   auto &innerProc = fInnerProcessors[processorNumber];
   innerProc->Initialize(fEntry);
   innerProc->Connect(fFieldIdxs, fProvenance, /*updateFields=*/true);
}

ROOT::Internal::RNTupleProcessorEntry::FieldIndex_t
ROOT::RNTupleChainProcessor::AddFieldToEntry(const std::string &fieldName, const std::string &typeName, void *valuePtr,
                                             const Internal::RNTupleProcessorProvenance &provenance)
{
   return fInnerProcessors[fCurrentProcessorNumber]->AddFieldToEntry(fieldName, typeName, valuePtr, provenance);
}

ROOT::Internal::RNTupleProcessorEntry::FieldIndex_t
ROOT::RNTupleChainProcessor::AddFieldToEntry(std::unique_ptr<ROOT::RFieldBase> field, const std::string &fieldName,
                                             void *valuePtr, const Internal::RNTupleProcessorProvenance &provenance)
{
   return fInnerProcessors[fCurrentProcessorNumber]->AddFieldToEntry(std::move(field), fieldName, valuePtr, provenance);
}

void ROOT::RNTupleChainProcessor::AddAllFieldsToEntry(const Internal::RNTupleProcessorProvenance &provenance,
                                                      bool includeSubfields)
{
   Initialize();
   fInnerProcessors[0]->AddAllFieldsToEntry(provenance, includeSubfields);
}

void ROOT::RNTupleChainProcessor::SetEntryMapping(ROOT::NTupleSize_t globalEntryNumber, REntryMapping &mapping)
{
   std::size_t currProcessorIdx = mapping.GetInnerProcessorIdx();

   // If the requested entry number is lower than the current entry number, we have to again localise the correct local
   // entry number starting from the first processor in the chain. Otherwise, we can continue looking from the inner
   // processor that is currently connected, which is much faster when the chain consists of many inner processors.
   if (fLastLoadedEntry != ROOT::kInvalidNTupleIndex && globalEntryNumber < fLastLoadedEntry) {
      currProcessorIdx = 0;
      ConnectInnerProcessor(currProcessorIdx);
   }

   ROOT::NTupleSize_t entriesSeen = 0;
   for (unsigned i = 0; i < currProcessorIdx; ++i) {
      if (fInnerNEntries[i] == kInvalidNTupleIndex) {
         fInnerNEntries[i] = fInnerProcessors[i]->GetNEntries();
      }

      if (fInnerNEntries[i] >= globalEntryNumber) {
         currProcessorIdx = 0;
         ConnectInnerProcessor(currProcessorIdx);
         break;
      }
      entriesSeen += fInnerNEntries[i];
   }

   ROOT::NTupleSize_t localEntryNumber = globalEntryNumber - entriesSeen;
   fInnerProcessors[currProcessorIdx]->SetEntryMapping(localEntryNumber, mapping.GetInnerProcessorMapping());

   // As long as the entry fails to load from the current processor, we decrement the local entry number with the number
   // of entries in this processor and try with the next processor until we find the correct local entry number.
   while (mapping.GetInnerProcessorMapping().GetEntryNumber() == kInvalidNTupleIndex) {
      if (fInnerNEntries[currProcessorIdx] == kInvalidNTupleIndex) {
         fInnerNEntries[currProcessorIdx] = fInnerProcessors[currProcessorIdx]->GetNEntries();
      }

      localEntryNumber -= fInnerNEntries[currProcessorIdx];

      // The provided global entry number is larger than the number of available entries.
      if (++currProcessorIdx >= fInnerProcessors.size()) {
         mapping.SetEntryNumber(kInvalidNTupleIndex);
         return;
      }

      ConnectInnerProcessor(currProcessorIdx);
      fInnerProcessors[currProcessorIdx]->SetEntryMapping(localEntryNumber, mapping.GetInnerProcessorMapping());
   }

   mapping.SetEntryNumber(globalEntryNumber);
   mapping.SetInnerProcessorIdx(currProcessorIdx);
}

void ROOT::RNTupleChainProcessor::LoadEntryImpl(const REntryMapping &mapping)
{
   assert(mapping.GetEntryNumber() != kInvalidNTupleIndex);

   if (fCurrentProcessorNumber != mapping.GetInnerProcessorIdx())
      ConnectInnerProcessor(mapping.GetInnerProcessorIdx());
   fInnerProcessors[mapping.GetInnerProcessorIdx()]->LoadEntry(mapping.GetInnerProcessorMapping());
   fLastLoadedEntry = mapping.GetEntryNumber();
   fNEntriesProcessed++;
}

void ROOT::RNTupleChainProcessor::AddEntriesToJoinTable(Experimental::Internal::RNTupleJoinTable &joinTable,
                                                        ROOT::NTupleSize_t entryOffset)
{
   for (unsigned i = 0; i < fInnerProcessors.size(); ++i) {
      const auto &innerProc = fInnerProcessors[i];
      // TODO can this be done (more) lazily? I.e. only when a match cannot be found in the current inner proc?
      innerProc->Initialize(fEntry);
      innerProc->AddEntriesToJoinTable(joinTable, entryOffset);
      entryOffset += innerProc->GetNEntries();
   }
}

void ROOT::RNTupleChainProcessor::PrintStructureImpl(std::ostream &output) const
{
   for (const auto &innerProc : fInnerProcessors) {
      innerProc->PrintStructure(output);
   }
}

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleJoinProcessor::RNTupleJoinProcessor(std::unique_ptr<RNTupleProcessor> primaryProcessor,
                                                               std::unique_ptr<RNTupleProcessor> auxProcessor,
                                                               const std::vector<std::string> &joinFields,
                                                               const RNTupleProcessorOptions &options)
   : RNTupleProcessor(options),
     fPrimaryProcessor(std::move(primaryProcessor)),
     fAuxiliaryProcessor(std::move(auxProcessor)),
     fJoinFieldNames(joinFields)
{
   if (fOptions.GetProcessorName().empty()) {
      fOptions.SetProcessorName(fPrimaryProcessor->fOptions.GetProcessorName());
   }
}

void ROOT::Experimental::RNTupleJoinProcessor::Initialize(std::shared_ptr<ROOT::Internal::RNTupleProcessorEntry> entry)
{
   if (IsInitialized())
      return;

   if (!entry)
      fEntry = std::make_shared<ROOT::Internal::RNTupleProcessorEntry>();
   else
      fEntry = std::move(entry);

   fPrimaryProcessor->Initialize(fEntry);
   fAuxiliaryProcessor->Initialize(fEntry);

   if (!fJoinFieldNames.empty()) {
      for (const auto &joinField : fJoinFieldNames) {
         if (!fPrimaryProcessor->CanReadFieldFromDisk(joinField)) {
            throw RException(R__FAIL("could not find join field \"" + joinField + "\" in primary processor \"" +
                                     fPrimaryProcessor->fOptions.GetProcessorName() + "\""));
         }
         if (!fAuxiliaryProcessor->CanReadFieldFromDisk(joinField)) {
            throw RException(R__FAIL("could not find join field \"" + joinField + "\" in auxiliary processor \"" +
                                     fAuxiliaryProcessor->fOptions.GetProcessorName() + "\""));
         }

         // We prepend the name of the primary processor in this case to prevent reading from the wrong join field in
         // composed join operations.
         auto fieldIdx =
            AddFieldToEntry(fOptions.GetProcessorName() + ".R_rntproc_join_" + joinField, "std::uint64_t", nullptr,
                            ROOT::Internal::RNTupleProcessorProvenance(fOptions.GetProcessorName()));
         fJoinFieldIdxs.insert(fieldIdx);
      }

      fJoinTable = Experimental::Internal::RNTupleJoinTable::Create(fJoinFieldNames);
   }
}

void ROOT::Experimental::RNTupleJoinProcessor::Connect(
   const std::unordered_set<ROOT::Internal::RNTupleProcessorEntry::FieldIndex_t> &fieldIdxs,
   const ROOT::Internal::RNTupleProcessorProvenance &provenance, bool updateFields)
{
   Initialize();

   auto auxProvenance = provenance.Evolve(fAuxiliaryProcessor->fOptions.GetProcessorName());
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

void ROOT::Experimental::RNTupleJoinProcessor::Disconnect()
{
   fPrimaryProcessor->Disconnect();
   fAuxiliaryProcessor->Disconnect();
}

ROOT::Internal::RNTupleProcessorEntry::FieldIndex_t
ROOT::Experimental::RNTupleJoinProcessor::AddFieldToEntry(const std::string &fieldName, const std::string &typeName,
                                                          void *valuePtr,
                                                          const ROOT::Internal::RNTupleProcessorProvenance &provenance)
{
   auto auxProvenance = provenance.Evolve(fAuxiliaryProcessor->fOptions.GetProcessorName());
   if (auxProvenance.IsPresentInFieldName(fieldName)) {
      // If the primaryProcessor has a field with the name of the auxProcessor (either as a "proper" field or because
      // the primary processor itself is a join where its auxProcessor bears the same name as the current auxProcessor),
      // there will be name conflicts, so error out.
      if (fPrimaryProcessor->CanReadFieldFromDisk(fieldName)) {
         throw RException(R__FAIL("ambiguous field name: \"" + fieldName +
                                  "\" is present in the primary RNTupleProcessor \"" +
                                  fPrimaryProcessor->fOptions.GetProcessorName() +
                                  "\", but may also refer to a field in the auxiliary RNTupleProcessor named \"" +
                                  fAuxiliaryProcessor->fOptions.GetProcessorName() +
                                  "\". To avoid this ambiguity, rename the auxiliary RNTupleProcessor."));
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

ROOT::Internal::RNTupleProcessorEntry::FieldIndex_t
ROOT::Experimental::RNTupleJoinProcessor::AddFieldToEntry(std::unique_ptr<ROOT::RFieldBase> field,
                                                          const std::string &fieldName, void *valuePtr,
                                                          const ROOT::Internal::RNTupleProcessorProvenance &provenance)
{
   auto auxProvenance = provenance.Evolve(fAuxiliaryProcessor->GetOptions().GetProcessorName());
   if (auxProvenance.IsPresentInFieldName(fieldName)) {
      // If the primaryProcessor has a field with the name of the auxProcessor (either as a "proper" field or because
      // the primary processor itself is a join where its auxProcessor bears the same name as the current auxProcessor),
      // there will be name conflicts, so error out.
      if (fPrimaryProcessor->CanReadFieldFromDisk(fieldName)) {
         throw RException(R__FAIL("ambiguous field name: \"" + fieldName +
                                  "\" is present in the primary RNTupleProcessor \"" +
                                  fPrimaryProcessor->GetOptions().GetProcessorName() +
                                  "\", but may also refer to a field in the auxiliary RNTupleProcessor named \"" +
                                  fAuxiliaryProcessor->GetOptions().GetProcessorName() +
                                  "\". To avoid this ambiguity, rename the auxiliary RNTupleProcessor."));
      }

      auto fieldIdx = fAuxiliaryProcessor->AddFieldToEntry(std::move(field), fieldName, valuePtr, auxProvenance);
      fAuxiliaryFieldIdxs.insert(fieldIdx);
      return fieldIdx;
   } else {
      auto fieldIdx = fPrimaryProcessor->AddFieldToEntry(std::move(field), fieldName, valuePtr, provenance);
      if (fieldIdx)
         fFieldIdxs.insert(fieldIdx);
      return fieldIdx;
   }
}

void ROOT::Experimental::RNTupleJoinProcessor::AddAllFieldsToEntry(
   const ROOT::Internal::RNTupleProcessorProvenance &provenance, bool includeSubfields)
{
   Initialize();
   fPrimaryProcessor->AddAllFieldsToEntry(provenance, includeSubfields);
   auto auxProvenance = provenance.Evolve(fAuxiliaryProcessor->GetOptions().GetProcessorName());
   fAuxiliaryProcessor->AddAllFieldsToEntry(auxProvenance, includeSubfields);
}

void ROOT::Experimental::RNTupleJoinProcessor::SetAuxiliaryFieldValidity(bool isValid)
{
   for (const auto &fieldIdx : fAuxiliaryFieldIdxs) {
      fEntry->SetFieldValidity(fieldIdx, isValid);
   }
}

void ROOT::Experimental::RNTupleJoinProcessor::SetEntryMapping(ROOT::NTupleSize_t globalEntryNumber,
                                                               REntryMapping &mapping)
{
   fPrimaryProcessor->SetEntryMapping(globalEntryNumber, mapping.GetPrimaryProcessorMapping());
   if (mapping.GetPrimaryProcessorMapping().GetEntryNumber() == kInvalidNTupleIndex) {
      mapping.SetEntryNumber(kInvalidNTupleIndex);
      for (auto fieldIdx : fFieldIdxs) {
         fEntry->SetFieldValidity(fieldIdx, false);
      }
      SetAuxiliaryFieldValidity(false);
      return;
   }
   // We only set the entry mapping of the auxiliary ntuple here if the join is aligned. For unaligne joins, we need to
   // first get the correct entry number from the join table, which requires reading join field data from disk. Because
   // we want to do this only when the entry is actually loaded to avoid unnecessary reads, we postpone this to the
   // actual `LoadEntry` call.
   if (!fJoinTable) {
      fAuxiliaryProcessor->SetEntryMapping(globalEntryNumber, mapping.GetAuxiliaryProcessorMapping());
   }
   mapping.SetEntryNumber(globalEntryNumber);
}

void ROOT::Experimental::RNTupleJoinProcessor::LoadEntryImpl(const REntryMapping &mapping)
{
   assert(mapping.GetEntryNumber() != kInvalidNTupleIndex);

   fPrimaryProcessor->LoadEntry(mapping.GetPrimaryProcessorMapping());
   fNEntriesProcessed++;

   if (!fJoinTable) {
      fAuxiliaryProcessor->LoadEntry(mapping.GetAuxiliaryProcessorMapping());
      return;
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
   REntryMapping auxMapping;
   fAuxiliaryProcessor->SetEntryMapping(entryIdx, auxMapping);

   if (entryIdx == kInvalidNTupleIndex) {
      SetAuxiliaryFieldValidity(false);
   } else {
      SetAuxiliaryFieldValidity(true);
      fAuxiliaryProcessor->LoadEntry(auxMapping);
   }
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleJoinProcessor::GetNEntries()
{
   if (fNEntries == kInvalidNTupleIndex)
      fNEntries = fPrimaryProcessor->GetNEntries();
   return fNEntries;
}

void ROOT::Experimental::RNTupleJoinProcessor::AddEntriesToJoinTable(
   Experimental::Internal::RNTupleJoinTable &joinTable, ROOT::NTupleSize_t entryOffset)
{
   fPrimaryProcessor->AddEntriesToJoinTable(joinTable, entryOffset);
}

void ROOT::Experimental::RNTupleJoinProcessor::PrintStructureImpl(std::ostream &output) const
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

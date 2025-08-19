/// \file RNTupleProcessor.cxx
/// \ingroup NTuple
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

std::unique_ptr<ROOT::Internal::RPageSource> ROOT::Experimental::RNTupleOpenSpec::CreatePageSource() const
{
   if (const std::string *storagePath = std::get_if<std::string>(&fStorage))
      return ROOT::Internal::RPageSource::Create(fNTupleName, *storagePath);

   auto dir = std::get<TDirectory *>(fStorage);
   auto ntuple = std::unique_ptr<ROOT::RNTuple>(dir->Get<ROOT::RNTuple>(fNTupleName.c_str()));
   return ROOT::Internal::RPageSourceFile::CreateFromAnchor(*ntuple);
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::Create(RNTupleOpenSpec ntuple, std::string_view processorName)
{
   return std::unique_ptr<RNTupleSingleProcessor>(new RNTupleSingleProcessor(std::move(ntuple), processorName));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateChain(std::vector<RNTupleOpenSpec> ntuples, std::string_view processorName)
{
   if (ntuples.empty())
      throw RException(R__FAIL("at least one RNTuple must be provided"));

   std::vector<std::unique_ptr<RNTupleProcessor>> innerProcessors;
   innerProcessors.reserve(ntuples.size());

   for (auto &ntuple : ntuples) {
      innerProcessors.emplace_back(Create(std::move(ntuple)));
   }

   return CreateChain(std::move(innerProcessors), processorName);
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateChain(std::vector<std::unique_ptr<RNTupleProcessor>> innerProcessors,
                                                  std::string_view processorName)
{
   if (innerProcessors.empty())
      throw RException(R__FAIL("at least one inner processor must be provided"));

   return std::unique_ptr<RNTupleChainProcessor>(new RNTupleChainProcessor(std::move(innerProcessors), processorName));
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateJoin(RNTupleOpenSpec primaryNTuple, RNTupleOpenSpec auxNTuple,
                                                 const std::vector<std::string> &joinFields,
                                                 std::string_view processorName)
{
   if (joinFields.size() > 4) {
      throw RException(R__FAIL("a maximum of four join fields is allowed"));
   }

   if (std::unordered_set(joinFields.begin(), joinFields.end()).size() < joinFields.size()) {
      throw RException(R__FAIL("join fields must be unique"));
   }

   std::unique_ptr<RNTupleProcessor> primaryProcessor = Create(primaryNTuple, processorName);
   std::unique_ptr<RNTupleProcessor> auxProcessor = Create(auxNTuple);

   return CreateJoin(std::move(primaryProcessor), std::move(auxProcessor), joinFields, processorName);
}

std::unique_ptr<ROOT::Experimental::RNTupleProcessor>
ROOT::Experimental::RNTupleProcessor::CreateJoin(std::unique_ptr<RNTupleProcessor> primaryProcessor,
                                                 std::unique_ptr<RNTupleProcessor> auxProcessor,
                                                 const std::vector<std::string> &joinFields,
                                                 std::string_view processorName)
{
   if (joinFields.size() > 4) {
      throw RException(R__FAIL("a maximum of four join fields is allowed"));
   }

   if (std::unordered_set(joinFields.begin(), joinFields.end()).size() < joinFields.size()) {
      throw RException(R__FAIL("join fields must be unique"));
   }

   return std::unique_ptr<RNTupleJoinProcessor>(
      new RNTupleJoinProcessor(std::move(primaryProcessor), std::move(auxProcessor), joinFields, processorName));
}

//------------------------------------------------------------------------------

ROOT::Experimental::RNTupleSingleProcessor::RNTupleSingleProcessor(RNTupleOpenSpec ntuple,
                                                                   std::string_view processorName)
   : RNTupleProcessor(ENTupleProcessorKind::kSingle, processorName), fNTupleSpec(std::move(ntuple))
{
   if (fProcessorName.empty()) {
      fProcessorName = fNTupleSpec.fNTupleName;
   }
}

void ROOT::Experimental::RNTupleSingleProcessor::Connect(
   std::shared_ptr<ROOT::Experimental::Internal::RNTupleProcessorEntry> entry)
{
   if (entry)
      fEntry = entry;

   if (!fEntry)
      fEntry = std::make_shared<Internal::RNTupleProcessorEntry>();

   if (fPageSource)
      return;

   fPageSource = fNTupleSpec.CreatePageSource();
   fPageSource->Attach();
   fNEntries = fPageSource->GetNEntries();
}

void ROOT::Experimental::RNTupleSingleProcessor::Update(ROOT::NTupleSize_t globalIndex)
{
   if (globalIndex == fCurrentEntryNumber || fNEntries == 0)
      return;

   if (globalIndex >= fNEntries)
      throw RException(R__FAIL("index " + std::to_string(globalIndex) + " out of bounds"));
   fCurrentEntryNumber = globalIndex;
}

bool ROOT::Experimental::RNTupleSingleProcessor::HasField(std::string_view fieldName)
{
   Connect();
   const auto &desc = fPageSource->GetSharedDescriptorGuard().GetRef();
   return desc.FindFieldId(fieldName) != ROOT::kInvalidDescriptorId;
}

std::unique_ptr<ROOT::RFieldBase>
ROOT::Experimental::RNTupleSingleProcessor::CreateField(std::string_view fieldName, std::string_view typeName)
{
   Connect();
   std::unique_ptr<ROOT::RFieldBase> field;
   const auto &desc = fPageSource->GetSharedDescriptorGuard().GetRef();
   const auto fieldId = desc.FindFieldId(fieldName);

   if (fieldId == ROOT::kInvalidDescriptorId) {
      return nullptr;
   }

   const auto &fieldDesc = desc.GetFieldDescriptor(fieldId);

   if (typeName.empty())
      field = fieldDesc.CreateField(desc);
   else
      field = ROOT::RFieldBase::Create(std::string(fieldName), std::string(typeName)).Unwrap();

   field->SetOnDiskId(fieldId);
   ROOT::Internal::CallConnectPageSourceOnField(*field, *fPageSource);

   return field;
}

void ROOT::Experimental::RNTupleSingleProcessor::AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable,
                                                                       ROOT::NTupleSize_t entryOffset)
{
   Connect();
   joinTable.Add(*fPageSource, Internal::RNTupleJoinTable::kDefaultPartitionKey, entryOffset);
}

void ROOT::Experimental::RNTupleSingleProcessor::PrintStructureImpl(std::ostream &output) const
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

ROOT::Experimental::RNTupleChainProcessor::RNTupleChainProcessor(
   std::vector<std::unique_ptr<RNTupleProcessor>> processors, std::string_view processorName)
   : RNTupleProcessor(ENTupleProcessorKind::kChain, processorName), fInnerProcessors(std::move(processors))
{
   if (fProcessorName.empty()) {
      // `CreateChain` ensures there is at least one inner processor.
      fProcessorName = fInnerProcessors[0]->GetProcessorName();
   }

   fInnerNEntries.assign(fInnerProcessors.size(), kInvalidNTupleIndex);
}

void ROOT::Experimental::RNTupleChainProcessor::Connect(
   std::shared_ptr<ROOT::Experimental::Internal::RNTupleProcessorEntry> entry)
{
   if (entry)
      fEntry = entry;

   if (!fEntry)
      fEntry = std::make_shared<Internal::RNTupleProcessorEntry>();

   fInnerProcessors[fCurrentProcessorNumber]->Connect(fEntry);
}

void ROOT::Experimental::RNTupleChainProcessor::Update(ROOT::NTupleSize_t globalIndex)
{
   if (globalIndex == fCurrentEntryNumber)
      return;

   ROOT::NTupleSize_t relativeIndex = globalIndex;
   std::size_t currProcessor = 0;
   ROOT::NTupleSize_t nCurrentEntries = fInnerProcessors[currProcessor]->GetNEntries();

   while (relativeIndex >= nCurrentEntries) {
      relativeIndex -= nCurrentEntries;

      if (++currProcessor >= fInnerProcessors.size())
         throw RException(R__FAIL("index " + std::to_string(globalIndex) + " out of bounds"));

      nCurrentEntries = fInnerProcessors[currProcessor]->GetNEntries();
   }

   if (currProcessor != fCurrentProcessorNumber) {
      fCurrentProcessorNumber = currProcessor;
      auto &innerProc = fInnerProcessors[fCurrentProcessorNumber];

      // Processor has not been connected before
      if (!innerProc->fEntry) {
         innerProc->Connect(fEntry);
      }

      for (auto &[fieldName, value] : *fEntry) {
         auto &currField = value.GetField();
         auto newField = innerProc->GetKind() != ENTupleProcessorKind::kJoin
                            ? innerProc->CreateField(value.GetProcessorFieldName(), currField.GetTypeName())
                            : innerProc->CreateField(fieldName, currField.GetTypeName());
         if (!newField) {
            value.SetIsValid(false);
         } else {
            value.SetIsValid(true);
            fEntry->UpdateField(fieldName, std::move(newField));
         }
      }
   }

   fCurrentEntryNumber = globalIndex;
   fInnerProcessors[fCurrentProcessorNumber]->Update(relativeIndex);
}

bool ROOT::Experimental::RNTupleChainProcessor::HasField(std::string_view fieldName)
{
   return fInnerProcessors[fCurrentProcessorNumber]->HasField(fieldName);
}

std::unique_ptr<ROOT::RFieldBase>
ROOT::Experimental::RNTupleChainProcessor::CreateField(std::string_view fieldName, std::string_view typeName)
{
   return fInnerProcessors[fCurrentProcessorNumber]->CreateField(fieldName, typeName);
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleChainProcessor::GetNEntries()
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

ROOT::NTupleSize_t
ROOT::Experimental::RNTupleChainProcessor::GetLocalCurrentEntryIndex(std::string_view fieldName) const
{
   return fInnerProcessors[fCurrentProcessorNumber]->GetLocalCurrentEntryIndex(fieldName);
}

void ROOT::Experimental::RNTupleChainProcessor::AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable,
                                                                      ROOT::NTupleSize_t entryOffset)
{
   for (unsigned i = 0; i < fInnerProcessors.size(); ++i) {
      const auto &innerProc = fInnerProcessors[i];
      innerProc->AddEntriesToJoinTable(joinTable, entryOffset);
      entryOffset += innerProc->GetNEntries();
   }
}

void ROOT::Experimental::RNTupleChainProcessor::PrintStructureImpl(std::ostream &output) const
{
   for (const auto &innerProc : fInnerProcessors) {
      innerProc->PrintStructure(output);
   }
}

//------------------------------------------------------------------------------

namespace ROOT::Experimental::Internal {
class RAuxiliaryProcessorField final : public ROOT::RRecordField {
private:
   using RFieldBase::GenerateColumns;
   void GenerateColumns() final
   {
      throw RException(R__FAIL("RAuxiliaryProcessorField fields must only be used for reading"));
   }

public:
   RAuxiliaryProcessorField(std::string_view fieldName, std::vector<std::unique_ptr<RFieldBase>> itemFields)
      : ROOT::RRecordField(fieldName, "RAuxiliaryProcessorField")
   {
      fOffsets.reserve(itemFields.size());
      for (auto &item : itemFields) {
         fOffsets.push_back(GetItemPadding(fSize, item->GetAlignment()));
      }
      AttachItemFields(std::move(itemFields));
   }
};
} // namespace ROOT::Experimental::Internal

namespace {
bool IsAuxiliaryField(std::string_view fieldName, std::string_view auxProcessorName)
{
   return fieldName.find(std::string(auxProcessorName) + ".") == 0;
}
} // anonymous namespace

ROOT::Experimental::RNTupleJoinProcessor::RNTupleJoinProcessor(std::unique_ptr<RNTupleProcessor> primaryProcessor,
                                                               std::unique_ptr<RNTupleProcessor> auxProcessor,
                                                               const std::vector<std::string> &joinFields,
                                                               std::string_view processorName)
   : RNTupleProcessor(ENTupleProcessorKind::kJoin, processorName),
     fPrimaryProcessor(std::move(primaryProcessor)),
     fAuxiliaryProcessor(std::move(auxProcessor)),
     fJoinFields(std::move(joinFields))
{
   if (fProcessorName.empty()) {
      fProcessorName = fPrimaryProcessor->GetProcessorName();
   }
}

void ROOT::Experimental::RNTupleJoinProcessor::Connect(
   std::shared_ptr<ROOT::Experimental::Internal::RNTupleProcessorEntry> entry)
{
   if (entry)
      fEntry = entry;

   if (!fEntry)
      fEntry = std::make_shared<Internal::RNTupleProcessorEntry>();

   fAuxiliaryEntry = std::make_shared<Internal::RNTupleProcessorEntry>();

   fPrimaryProcessor->Connect(fEntry);
   fAuxiliaryProcessor->Connect(fAuxiliaryEntry);
   fNEntries = fPrimaryProcessor->GetNEntries();

   if (!fJoinFields.empty()) {
      for (const auto &joinFieldName : fJoinFields) {
         if (!fAuxiliaryProcessor->HasField(joinFieldName)) {
            throw RException(R__FAIL("could not find join field \"" + joinFieldName + "\" in auxiliary processor \"" +
                                     fAuxiliaryProcessor->GetProcessorName() + "\""));
         }
         auto joinField = fPrimaryProcessor->CreateField(joinFieldName);
         if (!joinField) {
            throw RException(R__FAIL("could not find join field \"" + joinFieldName + "\" in primary processor \"" +
                                     fPrimaryProcessor->GetProcessorName() + "\""));
         }
         fEntry->AddOrGetValue(joinFieldName, std::move(joinField));
      }

      fJoinTable = Internal::RNTupleJoinTable::Create(fJoinFields);
   }
}

void ROOT::Experimental::RNTupleJoinProcessor::Update(ROOT::NTupleSize_t globalIndex)
{
   if (globalIndex == fCurrentEntryNumber)
      return;

   if (globalIndex >= fNEntries)
      throw RException(R__FAIL("index " + std::to_string(globalIndex) + " out of bounds"));

   fCurrentEntryNumber = globalIndex;
   fPrimaryProcessor->Update(globalIndex);
   fAuxiliaryProcessor->Update(globalIndex);

   if (fJoinTable) {
      // Only happens the first time `Update` is called.
      if (!fJoinTableIsBuilt) {
         fAuxiliaryProcessor->AddEntriesToJoinTable(*fJoinTable);
         fJoinTableIsBuilt = true;
      }

      // Collect the values of the join fields for this entry.
      std::vector<void *> valPtrs;
      valPtrs.reserve(fJoinFields.size());
      for (auto &joinField : fJoinFields) {
         auto value = fEntry->GetValueOrThrow(joinField);
         auto joinFieldIdx = fPrimaryProcessor->GetLocalCurrentEntryIndex(joinField);
         value.Read(joinFieldIdx);
         valPtrs.emplace_back(value.GetPtr<void>().get());
      }

      // Find the entry index corresponding to the join field values for each auxiliary processor and load the
      // corresponding entry.
      const auto entryIdx = fJoinTable->GetEntryIndex(valPtrs);

      if (entryIdx == kInvalidNTupleIndex) {
         for (auto &[_, value] : *fAuxiliaryEntry) {
            value.SetIsValid(false);
         }
      } else {
         for (auto &[_, value] : *fAuxiliaryEntry) {
            value.SetIsValid(true);
         }

         fAuxiliaryProcessor->Update(entryIdx);
      }
   }
}

bool ROOT::Experimental::RNTupleJoinProcessor::HasField(std::string_view fieldName)
{
   if (IsAuxiliaryField(fieldName, fAuxiliaryProcessor->GetProcessorName()))
      return fAuxiliaryProcessor->HasField(fieldName);
   return fPrimaryProcessor->HasField(fieldName);
}

std::unique_ptr<ROOT::RFieldBase>
ROOT::Experimental::RNTupleJoinProcessor::CreateField(std::string_view fieldName, std::string_view typeName)
{
   if (IsAuxiliaryField(fieldName, fAuxiliaryProcessor->GetProcessorName())) {
      auto baseFieldName = fieldName.substr(fAuxiliaryProcessor->GetProcessorName().size() + 1);
      return fAuxiliaryProcessor->CreateField(baseFieldName, typeName);
   }
   return fPrimaryProcessor->CreateField(fieldName, typeName);
}

ROOT::Experimental::Internal::RNTupleProcessorValue &
ROOT::Experimental::RNTupleJoinProcessor::AddFieldToEntry(std::string_view fieldName,
                                                          std::unique_ptr<ROOT::RFieldBase> field)
{
   assert(fEntry);
   assert(fAuxiliaryEntry);
   if (IsAuxiliaryField(fieldName, fAuxiliaryProcessor->GetProcessorName()))
      return fAuxiliaryEntry->AddOrGetValue(fieldName, std::move(field));
   else
      return fEntry->AddOrGetValue(fieldName, std::move(field));
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleJoinProcessor::GetNEntries()
{
   if (fNEntries == kInvalidNTupleIndex)
      fNEntries = fPrimaryProcessor->GetNEntries();
   return fNEntries;
}

ROOT::NTupleSize_t ROOT::Experimental::RNTupleJoinProcessor::GetLocalCurrentEntryIndex(std::string_view fieldName) const
{
   if (IsAuxiliaryField(fieldName, fAuxiliaryProcessor->GetProcessorName())) {
      auto baseFieldName = fieldName.substr(fAuxiliaryProcessor->GetProcessorName().size() + 1);
      return fAuxiliaryProcessor->GetLocalCurrentEntryIndex(baseFieldName);
   }

   return fPrimaryProcessor->GetLocalCurrentEntryIndex(fieldName);
}

void ROOT::Experimental::RNTupleJoinProcessor::AddEntriesToJoinTable(Internal::RNTupleJoinTable &joinTable,
                                                                     ROOT::NTupleSize_t entryOffset)
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

/// \file RNTupleReader.cxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2024-02-20

/*************************************************************************
 * Copyright (C) 1995-2024, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleReader.hxx>

#include <ROOT/RField.hxx>
#include <ROOT/RFieldVisitor.hxx>
#include <ROOT/RNTupleImtTaskScheduler.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RPageStorageFile.hxx>

#include <TROOT.h>

void ROOT::RNTupleReader::ConnectModel(ROOT::RNTupleModel &model)
{
   auto &fieldZero = ROOT::Internal::GetFieldZeroOfModel(model);
   // We must not use the descriptor guard to prevent recursive locking in field.ConnectPageSource
   ROOT::DescriptorId_t fieldZeroId = fSource->GetSharedDescriptorGuard()->GetFieldZeroId();
   fieldZero.SetOnDiskId(fieldZeroId);
   // Iterate only over fieldZero's direct subfields; their descendants are recursively handled in
   // RFieldBase::ConnectPageSource
   for (auto &field : fieldZero.GetMutableSubfields()) {
      // If the model has been created from the descriptor, the on-disk IDs are already set.
      // User-provided models instead need to find their corresponding IDs in the descriptor.
      if (field->GetOnDiskId() == ROOT::kInvalidDescriptorId) {
         field->SetOnDiskId(fSource->GetSharedDescriptorGuard()->FindFieldId(field->GetFieldName(), fieldZeroId));
      }
      ROOT::Internal::CallConnectPageSourceOnField(*field, *fSource);
   }
}

void ROOT::RNTupleReader::InitPageSource(bool enableMetrics)
{
#ifdef R__USE_IMT
   if (IsImplicitMTEnabled() &&
       fSource->GetReadOptions().GetUseImplicitMT() == ROOT::RNTupleReadOptions::EImplicitMT::kDefault) {
      fUnzipTasks = std::make_unique<Experimental::Internal::RNTupleImtTaskScheduler>();
      fSource->SetTaskScheduler(fUnzipTasks.get());
   }
#endif
   fMetrics.ObserveMetrics(fSource->GetMetrics());
   if (enableMetrics)
      EnableMetrics();
   fSource->Attach();
}

ROOT::RNTupleReader::RNTupleReader(std::unique_ptr<ROOT::RNTupleModel> model,
                                   std::unique_ptr<ROOT::Internal::RPageSource> source,
                                   const ROOT::RNTupleReadOptions &options)
   : fSource(std::move(source)), fModel(std::move(model)), fMetrics("RNTupleReader")
{
   // TODO(jblomer): properly support projected fields
   auto &projectedFields = ROOT::Internal::GetProjectedFieldsOfModel(*fModel);
   if (!projectedFields.IsEmpty()) {
      throw RException(R__FAIL("model has projected fields, which is incompatible with providing a read model"));
   }
   fModel->Freeze();
   InitPageSource(options.GetEnableMetrics());
   ConnectModel(*fModel);
}

ROOT::RNTupleReader::RNTupleReader(std::unique_ptr<ROOT::Internal::RPageSource> source,
                                   const ROOT::RNTupleReadOptions &options)
   : fSource(std::move(source)), fModel(nullptr), fMetrics("RNTupleReader")
{
   InitPageSource(options.GetEnableMetrics());
}

ROOT::RNTupleReader::~RNTupleReader() = default;

std::unique_ptr<ROOT::RNTupleReader> ROOT::RNTupleReader::Open(std::unique_ptr<ROOT::RNTupleModel> model,
                                                               std::string_view ntupleName, std::string_view storage,
                                                               const ROOT::RNTupleReadOptions &options)
{
   return std::unique_ptr<RNTupleReader>(
      new RNTupleReader(std::move(model), Internal::RPageSource::Create(ntupleName, storage, options), options));
}

std::unique_ptr<ROOT::RNTupleReader> ROOT::RNTupleReader::Open(std::string_view ntupleName, std::string_view storage,
                                                               const ROOT::RNTupleReadOptions &options)
{
   return std::unique_ptr<RNTupleReader>(
      new RNTupleReader(Internal::RPageSource::Create(ntupleName, storage, options), options));
}

std::unique_ptr<ROOT::RNTupleReader>
ROOT::RNTupleReader::Open(const ROOT::RNTuple &ntuple, const ROOT::RNTupleReadOptions &options)
{
   return std::unique_ptr<RNTupleReader>(
      new RNTupleReader(Internal::RPageSourceFile::CreateFromAnchor(ntuple, options), options));
}

std::unique_ptr<ROOT::RNTupleReader> ROOT::RNTupleReader::Open(std::unique_ptr<ROOT::RNTupleModel> model,
                                                               const ROOT::RNTuple &ntuple,
                                                               const ROOT::RNTupleReadOptions &options)
{
   return std::unique_ptr<RNTupleReader>(
      new RNTupleReader(std::move(model), Internal::RPageSourceFile::CreateFromAnchor(ntuple, options), options));
}

std::unique_ptr<ROOT::RNTupleReader>
ROOT::RNTupleReader::Open(const ROOT::RNTupleDescriptor::RCreateModelOptions &createModelOpts,
                          std::string_view ntupleName, std::string_view storage,
                          const ROOT::RNTupleReadOptions &options)
{
   auto reader = std::unique_ptr<RNTupleReader>(
      new RNTupleReader(Internal::RPageSource::Create(ntupleName, storage, options), options));
   reader->fCreateModelOptions = createModelOpts;
   return reader;
}

std::unique_ptr<ROOT::RNTupleReader>
ROOT::RNTupleReader::Open(const ROOT::RNTupleDescriptor::RCreateModelOptions &createModelOpts,
                          const ROOT::RNTuple &ntuple, const ROOT::RNTupleReadOptions &options)
{
   auto reader = std::unique_ptr<RNTupleReader>(
      new RNTupleReader(Internal::RPageSourceFile::CreateFromAnchor(ntuple, options), options));
   reader->fCreateModelOptions = createModelOpts;
   return reader;
}

const ROOT::RNTupleModel &ROOT::RNTupleReader::GetModel()
{
   if (!fModel) {
      fModel = fSource->GetSharedDescriptorGuard()->CreateModel(
         fCreateModelOptions.value_or(ROOT::RNTupleDescriptor::RCreateModelOptions{}));
      ConnectModel(*fModel);
   }
   return *fModel;
}

std::unique_ptr<ROOT::REntry> ROOT::RNTupleReader::CreateEntry()
{
   return GetModel().CreateEntry();
}

void ROOT::RNTupleReader::PrintInfo(const ENTupleInfo what, std::ostream &output) const
{
   using namespace ROOT::Internal;

   // TODO(lesimon): In a later version, these variables may be defined by the user or the ideal width may be read out
   // from the terminal.
   char frameSymbol = '*';
   int width = 80;
   /*
   if (width < 30) {
      output << "The width is too small! Should be at least 30." << std::endl;
      return;
   }
   */
   switch (what) {
   case ENTupleInfo::kSummary: {
      std::string name;
      std::unique_ptr<ROOT::RNTupleModel> fullModel;
      {
         auto descriptorGuard = fSource->GetSharedDescriptorGuard();
         name = descriptorGuard->GetName();
         ROOT::RNTupleDescriptor::RCreateModelOptions opts;
         opts.SetCreateBare(true);
         // When printing the schema we always try to reconstruct the whole thing even when we are missing the
         // dictionaries.
         opts.SetEmulateUnknownTypes(true);
         fullModel = descriptorGuard->CreateModel(opts);
      }

      for (int i = 0; i < (width / 2 + width % 2 - 4); ++i)
         output << frameSymbol;
      output << " NTUPLE ";
      for (int i = 0; i < (width / 2 - 4); ++i)
         output << frameSymbol;
      output << "\n";
      // FitString defined in RFieldVisitor.cxx
      output << frameSymbol << " N-Tuple : " << RNTupleFormatter::FitString(name, width - 13) << frameSymbol
             << "\n"; // prints line with name of ntuple
      output << frameSymbol << " Entries : " << RNTupleFormatter::FitString(std::to_string(GetNEntries()), width - 13)
             << frameSymbol << "\n"; // prints line with number of entries

      // Traverses through all fields to gather information needed for printing.
      RPrepareVisitor prepVisitor;
      // Traverses through all fields to do the actual printing.
      RPrintSchemaVisitor printVisitor(output);

      // Note that we do not need to connect the model, we are only looking at its tree of fields
      fullModel->GetConstFieldZero().AcceptVisitor(prepVisitor);

      printVisitor.SetFrameSymbol(frameSymbol);
      printVisitor.SetWidth(width);
      printVisitor.SetDeepestLevel(prepVisitor.GetDeepestLevel());
      printVisitor.SetNumFields(prepVisitor.GetNumFields());

      for (int i = 0; i < width; ++i)
         output << frameSymbol;
      output << "\n";
      fullModel->GetConstFieldZero().AcceptVisitor(printVisitor);
      for (int i = 0; i < width; ++i)
         output << frameSymbol;
      output << std::endl;
      break;
   }
   case ENTupleInfo::kStorageDetails: fSource->GetSharedDescriptorGuard()->PrintInfo(output); break;
   case ENTupleInfo::kMetrics: fMetrics.Print(output); break;
   default:
      // Unhandled case, internal error
      R__ASSERT(false);
   }
}

ROOT::RNTupleReader *ROOT::RNTupleReader::GetDisplayReader()
{
   if (!fDisplayReader) {
      ROOT::RNTupleDescriptor::RCreateModelOptions opts;
      opts.SetEmulateUnknownTypes(true);
      auto fullModel = fSource->GetSharedDescriptorGuard()->CreateModel(opts);
      fDisplayReader = std::unique_ptr<RNTupleReader>(
         new RNTupleReader(std::move(fullModel), fSource->Clone(), ROOT::RNTupleReadOptions{}));
   }
   return fDisplayReader.get();
}

void ROOT::RNTupleReader::Show(ROOT::NTupleSize_t index, std::ostream &output)
{
   auto reader = GetDisplayReader();
   const auto &entry = reader->GetModel().GetDefaultEntry();

   reader->LoadEntry(index);
   output << "{";
   for (auto iValue = entry.begin(); iValue != entry.end();) {
      output << std::endl;
      ROOT::Internal::RPrintValueVisitor visitor(*iValue, output, 1 /* level */);
      iValue->GetField().AcceptVisitor(visitor);

      if (++iValue == entry.end()) {
         output << std::endl;
         break;
      } else {
         output << ",";
      }
   }
   output << "}" << std::endl;
}

const ROOT::RNTupleDescriptor &ROOT::RNTupleReader::GetDescriptor()
{
   auto descriptorGuard = fSource->GetSharedDescriptorGuard();
   if (!fCachedDescriptor || fCachedDescriptor->GetGeneration() != descriptorGuard->GetGeneration())
      fCachedDescriptor = descriptorGuard->Clone();
   return *fCachedDescriptor;
}

ROOT::DescriptorId_t ROOT::RNTupleReader::RetrieveFieldId(std::string_view fieldName) const
{
   auto fieldId = fSource->GetSharedDescriptorGuard()->FindFieldId(fieldName);
   if (fieldId == ROOT::kInvalidDescriptorId) {
      throw RException(R__FAIL("no field named '" + std::string(fieldName) + "' in RNTuple '" +
                               fSource->GetSharedDescriptorGuard()->GetName() + "'"));
   }
   return fieldId;
}

/// \file RNTupleReader.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2024-02-20
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

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
#include <ROOT/RPageSourceFriends.hxx>
#include <ROOT/RPageStorageFile.hxx>

#include <TROOT.h>

void ROOT::Experimental::RNTupleReader::ConnectModel(RNTupleModel &model)
{
   auto &fieldZero = model.GetFieldZero();
   // We must not use the descriptor guard to prevent recursive locking in field.ConnectPageSource
   DescriptorId_t fieldZeroId = fSource->GetSharedDescriptorGuard()->GetFieldZeroId();
   fieldZero.SetOnDiskId(fieldZeroId);
   // Iterate only over fieldZero's direct subfields; their descendants are recursively handled in
   // RFieldBase::ConnectPageSource
   for (auto &field : fieldZero.GetSubFields()) {
      // If the model has been created from the descriptor, the on-disk IDs are already set.
      // User-provided models instead need to find their corresponding IDs in the descriptor.
      if (field->GetOnDiskId() == kInvalidDescriptorId) {
         field->SetOnDiskId(fSource->GetSharedDescriptorGuard()->FindFieldId(field->GetFieldName(), fieldZeroId));
      }
      Internal::CallConnectPageSourceOnField(*field, *fSource);
   }
}

void ROOT::Experimental::RNTupleReader::InitPageSource()
{
#ifdef R__USE_IMT
   if (IsImplicitMTEnabled()) {
      fUnzipTasks = std::make_unique<Internal::RNTupleImtTaskScheduler>();
      fSource->SetTaskScheduler(fUnzipTasks.get());
   }
#endif
   fSource->Attach();
   fMetrics.ObserveMetrics(fSource->GetMetrics());
}

ROOT::Experimental::RNTupleReader::RNTupleReader(std::unique_ptr<ROOT::Experimental::RNTupleModel> model,
                                                 std::unique_ptr<ROOT::Experimental::Internal::RPageSource> source)
   : fSource(std::move(source)), fModel(std::move(model)), fMetrics("RNTupleReader")
{
   // TODO(jblomer): properly support projected fields
   if (!fModel->GetProjectedFields().IsEmpty()) {
      throw RException(R__FAIL("model has projected fields, which is incompatible with providing a read model"));
   }
   fModel->Freeze();
   InitPageSource();
   ConnectModel(*fModel);
}

ROOT::Experimental::RNTupleReader::RNTupleReader(std::unique_ptr<ROOT::Experimental::Internal::RPageSource> source)
   : fSource(std::move(source)), fModel(nullptr), fMetrics("RNTupleReader")
{
   InitPageSource();
}

ROOT::Experimental::RNTupleReader::~RNTupleReader() = default;

std::unique_ptr<ROOT::Experimental::RNTupleReader>
ROOT::Experimental::RNTupleReader::Open(std::unique_ptr<RNTupleModel> model, std::string_view ntupleName,
                                        std::string_view storage, const RNTupleReadOptions &options)
{
   return std::unique_ptr<RNTupleReader>(
      new RNTupleReader(std::move(model), Internal::RPageSource::Create(ntupleName, storage, options)));
}

std::unique_ptr<ROOT::Experimental::RNTupleReader>
ROOT::Experimental::RNTupleReader::Open(std::string_view ntupleName, std::string_view storage,
                                        const RNTupleReadOptions &options)
{
   return std::unique_ptr<RNTupleReader>(
      new RNTupleReader(Internal::RPageSource::Create(ntupleName, storage, options)));
}

std::unique_ptr<ROOT::Experimental::RNTupleReader>
ROOT::Experimental::RNTupleReader::Open(ROOT::Experimental::RNTuple *ntuple, const RNTupleReadOptions &options)
{
   return std::unique_ptr<RNTupleReader>(
      new RNTupleReader(Internal::RPageSourceFile::CreateFromAnchor(*ntuple, options)));
}

std::unique_ptr<ROOT::Experimental::RNTupleReader>
ROOT::Experimental::RNTupleReader::Open(std::unique_ptr<RNTupleModel> model, ROOT::Experimental::RNTuple *ntuple,
                                        const RNTupleReadOptions &options)
{
   return std::unique_ptr<RNTupleReader>(
      new RNTupleReader(std::move(model), Internal::RPageSourceFile::CreateFromAnchor(*ntuple, options)));
}

std::unique_ptr<ROOT::Experimental::RNTupleReader>
ROOT::Experimental::RNTupleReader::OpenFriends(std::span<ROpenSpec> ntuples)
{
   std::vector<std::unique_ptr<Internal::RPageSource>> sources;
   for (const auto &n : ntuples) {
      sources.emplace_back(Internal::RPageSource::Create(n.fNTupleName, n.fStorage, n.fOptions));
   }
   return std::unique_ptr<RNTupleReader>(
      new RNTupleReader(std::make_unique<Internal::RPageSourceFriends>("_friends", sources)));
}

const ROOT::Experimental::RNTupleModel &ROOT::Experimental::RNTupleReader::GetModel()
{
   if (!fModel) {
      fModel = fSource->GetSharedDescriptorGuard()->CreateModel();
      ConnectModel(*fModel);
   }
   return *fModel;
}

void ROOT::Experimental::RNTupleReader::PrintInfo(const ENTupleInfo what, std::ostream &output)
{
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
      std::unique_ptr<RNTupleModel> fullModel;
      {
         auto descriptorGuard = fSource->GetSharedDescriptorGuard();
         name = descriptorGuard->GetName();
         fullModel = descriptorGuard->CreateModel();
      }

      for (int i = 0; i < (width / 2 + width % 2 - 4); ++i)
         output << frameSymbol;
      output << " NTUPLE ";
      for (int i = 0; i < (width / 2 - 4); ++i)
         output << frameSymbol;
      output << std::endl;
      // FitString defined in RFieldVisitor.cxx
      output << frameSymbol << " N-Tuple : " << RNTupleFormatter::FitString(name, width - 13) << frameSymbol
             << std::endl; // prints line with name of ntuple
      output << frameSymbol << " Entries : " << RNTupleFormatter::FitString(std::to_string(GetNEntries()), width - 13)
             << frameSymbol << std::endl; // prints line with number of entries

      // Traverses through all fields to gather information needed for printing.
      RPrepareVisitor prepVisitor;
      // Traverses through all fields to do the actual printing.
      RPrintSchemaVisitor printVisitor(output);

      // Note that we do not need to connect the model, we are only looking at its tree of fields
      fullModel->GetFieldZero().AcceptVisitor(prepVisitor);

      printVisitor.SetFrameSymbol(frameSymbol);
      printVisitor.SetWidth(width);
      printVisitor.SetDeepestLevel(prepVisitor.GetDeepestLevel());
      printVisitor.SetNumFields(prepVisitor.GetNumFields());

      for (int i = 0; i < width; ++i)
         output << frameSymbol;
      output << std::endl;
      fullModel->GetFieldZero().AcceptVisitor(printVisitor);
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

ROOT::Experimental::RNTupleReader *ROOT::Experimental::RNTupleReader::GetDisplayReader()
{
   if (!fDisplayReader)
      fDisplayReader = Clone();
   return fDisplayReader.get();
}

void ROOT::Experimental::RNTupleReader::Show(NTupleSize_t index, std::ostream &output)
{
   auto reader = GetDisplayReader();
   const auto &entry = reader->GetModel().GetDefaultEntry();

   reader->LoadEntry(index);
   output << "{";
   for (auto iValue = entry.begin(); iValue != entry.end();) {
      output << std::endl;
      RPrintValueVisitor visitor(*iValue, output, 1 /* level */);
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

const ROOT::Experimental::RNTupleDescriptor &ROOT::Experimental::RNTupleReader::GetDescriptor()
{
   auto descriptorGuard = fSource->GetSharedDescriptorGuard();
   if (!fCachedDescriptor || fCachedDescriptor->GetGeneration() != descriptorGuard->GetGeneration())
      fCachedDescriptor = descriptorGuard->Clone();
   return *fCachedDescriptor;
}

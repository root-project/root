/// \file RNTupleInspector.cxx
/// \ingroup NTuple ROOT7
/// \author Florine de Geus <florine.willemijn.de.geus@cern.ch>
/// \date 2023-01-09
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2023, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RError.hxx>
#include <ROOT/RNTuple.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleInspector.hxx>
#include <ROOT/RError.hxx>

#include <TFile.h>

#include <cstring>

void ROOT::Experimental::RNTupleInspector::CollectSizeData()
{
   fPageSource->Attach();
   auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();
   int compressionSettings = -1;
   std::uint64_t compressedSize = 0;
   std::uint64_t uncompressedSize = 0;

   for (const auto &clusterDescriptor : descriptorGuard->GetClusterIterable()) {
      compressedSize += clusterDescriptor.GetBytesOnStorage();

      if (compressionSettings == -1) {
         compressionSettings = clusterDescriptor.GetColumnRange(0).fCompressionSettings;
      }
   }

   for (uint64_t colId = 0; colId < descriptorGuard->GetNPhysicalColumns(); ++colId) {
      const ROOT::Experimental::RColumnDescriptor &colDescriptor = descriptorGuard->GetColumnDescriptor(colId);

      uint64_t elemSize =
         ROOT::Experimental::Detail::RColumnElementBase::Generate(colDescriptor.GetModel().GetType())->GetSize();

      uint64_t nElems = descriptorGuard->GetNElements(colId);
      uncompressedSize += nElems * elemSize;
   }

   fCompressionSettings = compressionSettings;
   fCompressedSize = compressedSize;
   fUncompressedSize = uncompressedSize;
}

ROOT::Experimental::RResult<std::unique_ptr<ROOT::Experimental::RNTupleInspector>>
ROOT::Experimental::RNTupleInspector::Create(std::unique_ptr<ROOT::Experimental::Detail::RPageSource> pageSource)
{
   auto inspector = std::unique_ptr<RNTupleInspector>(new RNTupleInspector(std::move(pageSource)));

   inspector->CollectSizeData();

   return inspector;
}

ROOT::Experimental::RResult<std::unique_ptr<ROOT::Experimental::RNTupleInspector>>
ROOT::Experimental::RNTupleInspector::Create(ROOT::Experimental::RNTuple *sourceNTuple)
{
   if (!sourceNTuple) {
      return R__FAIL("provided RNTuple is null");
   }

   std::unique_ptr<ROOT::Experimental::Detail::RPageSource> pageSource = sourceNTuple->MakePageSource();

   return ROOT::Experimental::RNTupleInspector::Create(std::move(pageSource));
}

ROOT::Experimental::RResult<std::unique_ptr<ROOT::Experimental::RNTupleInspector>>
ROOT::Experimental::RNTupleInspector::Create(std::string_view ntupleName, std::string_view sourceFileName)
{
   auto sourceFile = std::unique_ptr<TFile>(TFile::Open(std::string(sourceFileName).c_str()));
   if (!sourceFile || sourceFile->IsZombie()) {
      return R__FAIL("cannot open source file " + std::string(sourceFileName));
   }
   auto ntuple = std::unique_ptr<ROOT::Experimental::RNTuple>(
      sourceFile->Get<ROOT::Experimental::RNTuple>(std::string(ntupleName).c_str()));
   if (!ntuple) {
      return R__FAIL("cannot read RNTuple " + std::string(ntupleName) + " from " + std::string(sourceFileName));
   }

   return ROOT::Experimental::RNTupleInspector::Create(ntuple.get());
}

std::string ROOT::Experimental::RNTupleInspector::GetName()
{
   fPageSource->Attach();
   auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();
   return descriptorGuard->GetName();
}

std::uint64_t ROOT::Experimental::RNTupleInspector::GetNEntries()
{
   fPageSource->Attach();
   auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();
   return descriptorGuard->GetNEntries();
}

int ROOT::Experimental::RNTupleInspector::GetCompressionSettings()
{
   return fCompressionSettings;
}

std::uint64_t ROOT::Experimental::RNTupleInspector::GetCompressedSize()
{
   return fCompressedSize;
}

std::uint64_t ROOT::Experimental::RNTupleInspector::GetUncompressedSize()
{
   return fUncompressedSize;
}

float ROOT::Experimental::RNTupleInspector::GetCompressionFactor()
{
   return (float)fUncompressedSize / (float)fCompressedSize;
}

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

#include <cstring>

void ROOT::Experimental::RNTupleInspector::CollectSizeData()
{
   int compressionSettings = -1;
   std::uint64_t compressedSize = 0;
   std::uint64_t uncompressedSize = 0;

   for (const auto &clusterDescriptor : fSourceNTupleDescriptor->GetClusterIterable()) {
      compressedSize += clusterDescriptor.GetBytesOnStorage();

      if (compressionSettings == -1) {
         compressionSettings = clusterDescriptor.GetColumnRange(0).fCompressionSettings;
      }
   }

   for (uint64_t colId = 0; colId < fSourceNTupleDescriptor->GetNColumns(); ++colId) {
      const ROOT::Experimental::RColumnDescriptor &colDescriptor = fSourceNTupleDescriptor->GetColumnDescriptor(colId);

      uint64_t elemSize =
         ROOT::Experimental::Detail::RColumnElementBase::Generate(colDescriptor.GetModel().GetType())->GetSize();

      uint64_t nElems = fSourceNTupleDescriptor->GetNElements(colId);
      uncompressedSize += nElems * elemSize;
   }

   fCompressionSettings = compressionSettings;
   fCompressedSize = compressedSize;
   fUncompressedSize = uncompressedSize;
}

ROOT::Experimental::RResult<std::unique_ptr<ROOT::Experimental::RNTupleInspector>>
ROOT::Experimental::RNTupleInspector::Create(std::unique_ptr<RNTupleReader> &sourceNTupleReader)
{
   auto inspector = std::unique_ptr<RNTupleInspector>(new RNTupleInspector());

   inspector->fSourceNTupleDescriptor = sourceNTupleReader->GetDescriptor();

   if (!inspector->fSourceNTupleDescriptor) {
      return R__FAIL("cannot get descriptor for provided RNTuple");
   }

   inspector->CollectSizeData();
   return inspector;
}

ROOT::Experimental::RResult<std::unique_ptr<ROOT::Experimental::RNTupleInspector>>
ROOT::Experimental::RNTupleInspector::Create(ROOT::Experimental::RNTuple *sourceNTuple)
{
   auto reader = ROOT::Experimental::RNTupleReader::Open(sourceNTuple);

   if (!reader) {
      return R__FAIL("cannot get reader for provided RNTuple");
   }

   return ROOT::Experimental::RNTupleInspector::Create(reader);
}

std::string ROOT::Experimental::RNTupleInspector::GetName()
{
   return fSourceNTupleDescriptor->GetName();
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

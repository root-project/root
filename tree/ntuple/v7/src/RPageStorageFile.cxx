/// \file RPageStorageFile.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2019-11-25
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RField.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RPageAllocator.hxx>
#include <ROOT/RPagePool.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/RRawFile.hxx>

#include <RVersion.h>
#include <TError.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <utility>


ROOT::Experimental::Detail::RPageSinkFile::RPageSinkFile(std::string_view ntupleName, std::string_view path,
   const RNTupleWriteOptions &options)
   : RPageSink(ntupleName, options)
   , fMetrics("RPageSinkRoot")
   , fPageAllocator(std::make_unique<RPageAllocatorHeap>())
{
   R__WARNING_HERE("NTuple") << "The RNTuple file format will change. " <<
      "Do not store real data with this version of RNTuple!";

   fWriter = std::unique_ptr<Internal::RNTupleFileWriter>(Internal::RNTupleFileWriter::Recreate(
      ntupleName, path, options.GetCompression(), options.GetContainerFormat()));
}


ROOT::Experimental::Detail::RPageSinkFile::RPageSinkFile(std::string_view ntupleName, TFile &file,
   const RNTupleWriteOptions &options)
   : RPageSink(ntupleName, options)
   , fMetrics("RPageSinkRoot")
   , fPageAllocator(std::make_unique<RPageAllocatorHeap>())
{
   R__WARNING_HERE("NTuple") << "The RNTuple file format will change. " <<
      "Do not store real data with this version of RNTuple!";

   fWriter = std::unique_ptr<Internal::RNTupleFileWriter>(Internal::RNTupleFileWriter::Append(ntupleName, file));
}


ROOT::Experimental::Detail::RPageSinkFile::RPageSinkFile(std::string_view ntupleName, std::string_view path,
   const RNTupleWriteOptions &options, std::unique_ptr<TFile> &file)
   : RPageSink(ntupleName, options)
   , fMetrics("RPageSinkRoot")
   , fPageAllocator(std::make_unique<RPageAllocatorHeap>())
{
   R__WARNING_HERE("NTuple") << "The RNTuple file format will change. " <<
      "Do not store real data with this version of RNTuple!";
   fWriter = std::unique_ptr<Internal::RNTupleFileWriter>(
      Internal::RNTupleFileWriter::Recreate(ntupleName, path, file));
}


ROOT::Experimental::Detail::RPageSinkFile::~RPageSinkFile()
{
}


void ROOT::Experimental::Detail::RPageSinkFile::CreateImpl(const RNTupleModel & /* model */)
{
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();
   auto szHeader = descriptor.SerializeHeader(nullptr);
   auto buffer = std::unique_ptr<unsigned char[]>(new unsigned char[szHeader]);
   descriptor.SerializeHeader(buffer.get());

   auto zipBuffer = std::unique_ptr<unsigned char[]>(new unsigned char[szHeader]);
   auto szZipHeader = fCompressor(buffer.get(), szHeader, fOptions.GetCompression(),
      [&zipBuffer](const void *b, size_t n, size_t o){ memcpy(zipBuffer.get() + o, b, n); } );
   fWriter->WriteNTupleHeader(zipBuffer.get(), szZipHeader, szHeader);
}


ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkFile::CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page)
{
   unsigned char *buffer = reinterpret_cast<unsigned char *>(page.GetBuffer());
   bool isAdoptedBuffer = true;
   auto packedBytes = page.GetSize();
   auto element = columnHandle.fColumn->GetElement();
   const auto isMappable = element->IsMappable();

   if (!isMappable) {
      packedBytes = (page.GetNElements() * element->GetBitsOnStorage() + 7) / 8;
      buffer = new unsigned char[packedBytes];
      isAdoptedBuffer = false;
      element->Pack(buffer, page.GetBuffer(), page.GetNElements());
   }
   auto zippedBytes = packedBytes;

   if (fOptions.GetCompression() != 0) {
      zippedBytes = fCompressor(buffer, packedBytes, fOptions.GetCompression());
      if (!isAdoptedBuffer)
         delete[] buffer;
      buffer = const_cast<unsigned char *>(reinterpret_cast<const unsigned char *>(fCompressor.GetZipBuffer()));
      isAdoptedBuffer = true;
   }

   auto offsetData = fWriter->WriteBlob(buffer, zippedBytes, packedBytes);
   fClusterMinOffset = std::min(offsetData, fClusterMinOffset);
   fClusterMaxOffset = std::max(offsetData, fClusterMaxOffset);

   if (!isAdoptedBuffer)
      delete[] buffer;

   RClusterDescriptor::RLocator result;
   result.fPosition = offsetData;
   result.fBytesOnStorage = zippedBytes;
   return result;
}


ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkFile::CommitClusterImpl(ROOT::Experimental::NTupleSize_t /* nEntries */)
{
   RClusterDescriptor::RLocator result;
   result.fPosition = fClusterMinOffset;
   result.fBytesOnStorage = fClusterMaxOffset - fClusterMinOffset;
   fClusterMinOffset = std::uint64_t(-1);
   fClusterMaxOffset = 0;
   return result;
}


void ROOT::Experimental::Detail::RPageSinkFile::CommitDatasetImpl()
{
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();
   auto szFooter = descriptor.SerializeFooter(nullptr);
   auto buffer = std::unique_ptr<unsigned char []>(new unsigned char[szFooter]);
   descriptor.SerializeFooter(buffer.get());

   auto zipBuffer = std::unique_ptr<unsigned char[]>(new unsigned char[szFooter]);
   auto szZipFooter = fCompressor(buffer.get(), szFooter, fOptions.GetCompression(),
      [&zipBuffer](const void *b, size_t n, size_t o){ memcpy(zipBuffer.get() + o, b, n); } );
   fWriter->WriteNTupleFooter(zipBuffer.get(), szZipFooter, szFooter);
   fWriter->Commit();
}


ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSinkFile::ReservePage(ColumnHandle_t columnHandle, std::size_t nElements)
{
   if (nElements == 0)
      nElements = kDefaultElementsPerPage;
   auto elementSize = columnHandle.fColumn->GetElement()->GetSize();
   return fPageAllocator->NewPage(columnHandle.fId, elementSize, nElements);
}

void ROOT::Experimental::Detail::RPageSinkFile::ReleasePage(RPage &page)
{
   fPageAllocator->DeletePage(page);
}


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageAllocatorFile::NewPage(
   ColumnId_t columnId, void *mem, std::size_t elementSize, std::size_t nElements)
{
   RPage newPage(columnId, mem, elementSize * nElements, elementSize);
   newPage.TryGrow(nElements);
   return newPage;
}

void ROOT::Experimental::Detail::RPageAllocatorFile::DeletePage(const RPage& page)
{
   if (page.IsNull())
      return;
   delete[] reinterpret_cast<unsigned char *>(page.GetBuffer());
}


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::RPageSourceFile::RPageSourceFile(std::string_view ntupleName,
   const RNTupleReadOptions &options)
   : RPageSource(ntupleName, options)
   , fMetrics("RPageSourceFile")
   , fPageAllocator(std::make_unique<RPageAllocatorFile>())
   , fPagePool(std::make_shared<RPagePool>())
{
}


ROOT::Experimental::Detail::RPageSourceFile::RPageSourceFile(std::string_view ntupleName, std::string_view path,
   const RNTupleReadOptions &options)
   : RPageSourceFile(ntupleName, options)
{
   fFile = ROOT::Internal::RRawFile::Create(path);
   R__ASSERT(fFile);
   fReader = Internal::RMiniFileReader(fFile.get());
}


ROOT::Experimental::Detail::RPageSourceFile::~RPageSourceFile()
{
}


ROOT::Experimental::RNTupleDescriptor ROOT::Experimental::Detail::RPageSourceFile::AttachImpl()
{
   RNTupleDescriptorBuilder descBuilder;
   const auto fNTuple = fReader.GetNTuple(fNTupleName);

   auto buffer = std::unique_ptr<unsigned char[]>(new unsigned char[fNTuple.fLenHeader]);
   auto zipBuffer = std::unique_ptr<unsigned char[]>(new unsigned char[fNTuple.fNBytesHeader]);
   fReader.ReadBuffer(zipBuffer.get(), fNTuple.fNBytesHeader, fNTuple.fSeekHeader);
   fDecompressor(zipBuffer.get(), fNTuple.fNBytesHeader, fNTuple.fLenHeader, buffer.get());
   descBuilder.SetFromHeader(buffer.get());

   buffer = std::unique_ptr<unsigned char[]>(new unsigned char[fNTuple.fLenFooter]);
   zipBuffer = std::unique_ptr<unsigned char[]>(new unsigned char[fNTuple.fNBytesFooter]);
   fReader.ReadBuffer(zipBuffer.get(), fNTuple.fNBytesFooter, fNTuple.fSeekFooter);
   fDecompressor(zipBuffer.get(), fNTuple.fNBytesFooter, fNTuple.fLenFooter, buffer.get());
   descBuilder.AddClustersFromFooter(buffer.get());

   return descBuilder.MoveDescriptor();
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceFile::PopulatePageFromCluster(
   ColumnHandle_t columnHandle, const RClusterDescriptor &clusterDescriptor, ClusterSize_t::ValueType clusterIndex)
{
   const auto columnId = columnHandle.fId;
   const auto clusterId = clusterDescriptor.GetId();
   const auto &pageRange = clusterDescriptor.GetPageRange(columnId);

   // TODO(jblomer): binary search
   RClusterDescriptor::RPageRange::RPageInfo pageInfo;
   decltype(clusterIndex) firstInPage = 0;
   for (const auto &pi : pageRange.fPageInfos) {
      if (firstInPage + pi.fNElements > clusterIndex) {
         pageInfo = pi;
         break;
      }
      firstInPage += pi.fNElements;
   }
   R__ASSERT(firstInPage <= clusterIndex);
   R__ASSERT((firstInPage + pageInfo.fNElements) > clusterIndex);

   const auto element = columnHandle.fColumn->GetElement();
   const auto elementSize = element->GetSize();

   auto pageSize = pageInfo.fLocator.fBytesOnStorage;
   auto pageBuffer = new unsigned char[
      std::max(pageSize, static_cast<std::uint32_t>(elementSize * pageInfo.fNElements))];
   fReader.ReadBuffer(pageBuffer, pageInfo.fLocator.fBytesOnStorage, pageInfo.fLocator.fPosition);

   const auto bytesOnStorage = (element->GetBitsOnStorage() * pageInfo.fNElements + 7) / 8;
   if (pageSize != bytesOnStorage) {
      fDecompressor(pageBuffer, pageSize, bytesOnStorage);
      pageSize = bytesOnStorage;
   }

   if (!element->IsMappable()) {
      pageSize = elementSize * pageInfo.fNElements;
      auto unpackedBuffer = new unsigned char[pageSize];
      element->Unpack(unpackedBuffer, pageBuffer, pageInfo.fNElements);
      delete[] pageBuffer;
      pageBuffer = unpackedBuffer;
   }

   const auto indexOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
   auto newPage = fPageAllocator->NewPage(columnId, pageBuffer, elementSize, pageInfo.fNElements);
   newPage.SetWindow(indexOffset + firstInPage, RPage::RClusterInfo(clusterId, indexOffset));
   fPagePool->RegisterPage(newPage,
      RPageDeleter([](const RPage &page, void * /*userData*/)
      {
         RPageAllocatorFile::DeletePage(page);
      }, nullptr));
   return newPage;
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceFile::PopulatePage(
   ColumnHandle_t columnHandle, NTupleSize_t globalIndex)
{
   const auto columnId = columnHandle.fId;
   auto cachedPage = fPagePool->GetPage(columnId, globalIndex);
   if (!cachedPage.IsNull())
      return cachedPage;

   const auto clusterId = fDescriptor.FindClusterId(columnId, globalIndex);
   R__ASSERT(clusterId != kInvalidDescriptorId);
   const auto &clusterDescriptor = fDescriptor.GetClusterDescriptor(clusterId);
   const auto selfOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
   R__ASSERT(selfOffset <= globalIndex);
   return PopulatePageFromCluster(columnHandle, clusterDescriptor, globalIndex - selfOffset);
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceFile::PopulatePage(
   ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex)
{
   const auto clusterId = clusterIndex.GetClusterId();
   const auto index = clusterIndex.GetIndex();
   const auto columnId = columnHandle.fId;
   auto cachedPage = fPagePool->GetPage(columnId, clusterIndex);
   if (!cachedPage.IsNull())
      return cachedPage;

   R__ASSERT(clusterId != kInvalidDescriptorId);
   const auto &clusterDescriptor = fDescriptor.GetClusterDescriptor(clusterId);
   return PopulatePageFromCluster(columnHandle, clusterDescriptor, index);
}

void ROOT::Experimental::Detail::RPageSourceFile::ReleasePage(RPage &page)
{
   fPagePool->ReturnPage(page);
}

std::unique_ptr<ROOT::Experimental::Detail::RPageSource> ROOT::Experimental::Detail::RPageSourceFile::Clone() const
{
   auto clone = new RPageSourceFile(fNTupleName, fOptions);
   clone->fFile = fFile->Clone();
   clone->fReader = Internal::RMiniFileReader(clone->fFile.get());
   return std::unique_ptr<RPageSourceFile>(clone);
}

/// \file RPageStorageRaw.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RPageStorageRaw.hxx>
#include <ROOT/RColumn.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RPageAllocator.hxx>
#include <ROOT/RPagePool.hxx>
#include <ROOT/RRawFile.hxx>

#include <Compression.h>
#include <RZip.h>
#include <TError.h>

#include <cstdio>
#include <cstring>
#include <iostream>

ROOT::Experimental::Detail::RPageSinkRaw::RPageSinkRaw(std::string_view ntupleName, std::string_view path,
   const RNTupleWriteOptions &options)
   : RPageSink(ntupleName, options)
   , fPageAllocator(std::make_unique<RPageAllocatorHeap>())
   , fZipBuffer(std::make_unique<std::array<char, kMAXZIPBUF>>())
{
   R__WARNING_HERE("NTuple") << "The RNTuple file format will change. " <<
      "Do not store real data with this version of RNTuple!";
   fFile = fopen(std::string(path).c_str(), "w");
   R__ASSERT(fFile);
}

ROOT::Experimental::Detail::RPageSinkRaw::~RPageSinkRaw()
{
   if (fFile)
      fclose(fFile);
}

void ROOT::Experimental::Detail::RPageSinkRaw::Write(const void *buffer, std::size_t nbytes)
{
   R__ASSERT(fFile);
   auto written = fwrite(buffer, 1, nbytes, fFile);
   R__ASSERT(written == nbytes);
   fFilePos += written;
}

void ROOT::Experimental::Detail::RPageSinkRaw::DoCreate(const RNTupleModel & /* model */)
{
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();
   auto szHeader = descriptor.SerializeHeader(nullptr);
   auto buffer = new unsigned char[szHeader];
   descriptor.SerializeHeader(buffer);
   Write(buffer, szHeader);
   delete[] buffer;
   fClusterStart = fFilePos;
}

ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkRaw::DoCommitPage(ColumnHandle_t columnHandle, const RPage &page)
{
   unsigned char *buffer = reinterpret_cast<unsigned char *>(page.GetBuffer());
   bool isAdoptedBuffer = true;
   auto packedBytes = page.GetSize();
   auto element = columnHandle.fColumn->GetElement();
   const auto isMappable = element->IsMappable();

   if (!isMappable) {
      packedBytes = (page.GetNElements() * element->GetBitsOnStorage() + 7) / 8;
      buffer = new unsigned char[packedBytes];
      element->Pack(buffer, page.GetBuffer(), page.GetNElements());
      isAdoptedBuffer = false;
   }

   if (fOptions.GetCompression() % 100 != 0) {
      auto level = fOptions.GetCompression() % 100;
      auto algorithm = static_cast<ROOT::RCompressionSetting::EAlgorithm::EValues>(fOptions.GetCompression() / 100);
      int szZipBuffer = kMAXZIPBUF;
      int szSource = packedBytes;
      char *source = reinterpret_cast<char *>(buffer);
      int zipBytes = 0;
      R__zipMultipleAlgorithm(level, &szSource, source, &szZipBuffer, fZipBuffer->data(), &zipBytes, algorithm);
      if ((zipBytes > 0) && (zipBytes < szSource)) {
         if (!isAdoptedBuffer)
            delete[] buffer;
         buffer = reinterpret_cast<unsigned char *>(fZipBuffer->data());
         packedBytes = zipBytes;
         isAdoptedBuffer = true;
      }
   }

   RClusterDescriptor::RLocator result;
   result.fPosition = fFilePos;
   result.fBytesOnStorage = packedBytes;
   Write(buffer, packedBytes);

   if (!isAdoptedBuffer)
      delete[] buffer;

   return result;
}

ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkRaw::DoCommitCluster(ROOT::Experimental::NTupleSize_t /* nEntries */)
{
   RClusterDescriptor::RLocator result;
   result.fPosition = fClusterStart;
   result.fBytesOnStorage = fFilePos - fClusterStart;
   fClusterStart = fFilePos;
   return result;
}

void ROOT::Experimental::Detail::RPageSinkRaw::DoCommitDataset()
{
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();
   auto szFooter = descriptor.SerializeFooter(nullptr);
   auto buffer = new unsigned char[szFooter];
   descriptor.SerializeFooter(buffer);
   Write(buffer, szFooter);
   delete[] buffer;
}

ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSinkRaw::ReservePage(ColumnHandle_t columnHandle, std::size_t nElements)
{
   if (nElements == 0)
      nElements = kDefaultElementsPerPage;
   auto elementSize = columnHandle.fColumn->GetElement()->GetSize();
   return fPageAllocator->NewPage(columnHandle.fId, elementSize, nElements);
}

void ROOT::Experimental::Detail::RPageSinkRaw::ReleasePage(RPage &page)
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
   free(page.GetBuffer());
}


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::RPageSourceRaw::RPageSourceRaw(std::string_view ntupleName,
   const RNTupleReadOptions &options)
   : RPageSource(ntupleName, options)
   , fPageAllocator(std::make_unique<RPageAllocatorFile>())
   , fPagePool(std::make_shared<RPagePool>())
   , fUnzipBuffer(std::make_unique<std::array<unsigned char, kMAXZIPBUF>>())
{
}

ROOT::Experimental::Detail::RPageSourceRaw::RPageSourceRaw(std::string_view ntupleName, std::string_view path,
   const RNTupleReadOptions &options)
   : RPageSourceRaw(ntupleName, options)
{
   fFile = std::unique_ptr<RRawFile>(RRawFile::Create(path));
   R__ASSERT(fFile);
   R__ASSERT(fFile->GetFeatures() & RRawFile::kFeatureHasSize);
}


ROOT::Experimental::Detail::RPageSourceRaw::~RPageSourceRaw()
{
}


void ROOT::Experimental::Detail::RPageSourceRaw::Read(void *buffer, std::size_t nbytes, std::uint64_t offset)
{
   auto nread = fFile->ReadAt(buffer, nbytes, offset);
   R__ASSERT(nread == nbytes);
}


ROOT::Experimental::RNTupleDescriptor ROOT::Experimental::Detail::RPageSourceRaw::DoAttach()
{
   unsigned char postscript[RNTupleDescriptor::kNBytesPostscript];
   auto fileSize = fFile->GetSize();
   R__ASSERT(fileSize != RRawFile::kUnknownFileSize);
   R__ASSERT(fileSize >= RNTupleDescriptor::kNBytesPostscript);
   auto offset = fileSize - RNTupleDescriptor::kNBytesPostscript;
   Read(postscript, RNTupleDescriptor::kNBytesPostscript, offset);

   std::uint32_t szHeader;
   std::uint32_t szFooter;
   RNTupleDescriptor::LocateMetadata(postscript, szHeader, szFooter);
   R__ASSERT(fileSize >= szHeader + szFooter);

   unsigned char *header = new unsigned char[szHeader];
   unsigned char *footer = new unsigned char[szFooter];
   Read(header, szHeader, 0);
   Read(footer, szFooter, fileSize - szFooter);

   RNTupleDescriptorBuilder descBuilder;
   descBuilder.SetFromHeader(header);
   descBuilder.AddClustersFromFooter(footer);
   delete[] header;
   delete[] footer;

   return descBuilder.GetDescriptor();
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceRaw::PopulatePageFromCluster(
   ColumnHandle_t columnHandle, const RClusterDescriptor &clusterDescriptor, ClusterSize_t::ValueType clusterIndex)
{
   auto columnId = columnHandle.fId;
   auto clusterId = clusterDescriptor.GetId();
   auto pageRange = clusterDescriptor.GetPageRange(columnId);

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

   auto element = columnHandle.fColumn->GetElement();
   auto elementSize = element->GetSize();

   auto pageSize = pageInfo.fLocator.fBytesOnStorage;
   void *pageBuffer = malloc(std::max(pageSize, static_cast<std::uint32_t>(elementSize * pageInfo.fNElements)));
   R__ASSERT(pageBuffer);
   Read(pageBuffer, pageSize, pageInfo.fLocator.fPosition);

   auto bytesOnStorage = (element->GetBitsOnStorage() * pageInfo.fNElements + 7) / 8;
   if (pageSize != bytesOnStorage) {
      // We do have the unzip information in the column range, but here we simply use the value from
      // the R__zip header
      int szUnzipBuffer = kMAXZIPBUF;
      int szSource = pageSize;
      unsigned char *source = reinterpret_cast<unsigned char *>(pageBuffer);
      int unzipBytes = 0;
      R__unzip(&szSource, source, &szUnzipBuffer, fUnzipBuffer->data(), &unzipBytes);
      R__ASSERT(unzipBytes > static_cast<int>(pageSize));
      memcpy(pageBuffer, fUnzipBuffer->data(), unzipBytes);
      pageSize = unzipBytes;
   }

   if (!element->IsMappable()) {
      pageSize = elementSize * pageInfo.fNElements;
      auto unpackedBuffer = reinterpret_cast<unsigned char *>(malloc(pageSize));
      R__ASSERT(unpackedBuffer != nullptr);
      element->Unpack(unpackedBuffer, pageBuffer, pageInfo.fNElements);
      free(pageBuffer);
      pageBuffer = unpackedBuffer;
   }

   auto indexOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
   auto newPage = fPageAllocator->NewPage(columnId, pageBuffer, elementSize, pageInfo.fNElements);
   newPage.SetWindow(indexOffset + firstInPage, RPage::RClusterInfo(clusterId, indexOffset));
   fPagePool->RegisterPage(newPage,
      RPageDeleter([](const RPage &page, void */*userData*/)
      {
         RPageAllocatorFile::DeletePage(page);
      }, nullptr));
   return newPage;
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceRaw::PopulatePage(
   ColumnHandle_t columnHandle, NTupleSize_t globalIndex)
{
   auto columnId = columnHandle.fId;
   auto cachedPage = fPagePool->GetPage(columnId, globalIndex);
   if (!cachedPage.IsNull())
      return cachedPage;

   auto clusterId = fDescriptor.FindClusterId(columnId, globalIndex);
   R__ASSERT(clusterId != kInvalidDescriptorId);
   auto clusterDescriptor = fDescriptor.GetClusterDescriptor(clusterId);
   auto selfOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
   R__ASSERT(selfOffset <= globalIndex);
   return PopulatePageFromCluster(columnHandle, clusterDescriptor, globalIndex - selfOffset);
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceRaw::PopulatePage(
   ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex)
{
   auto clusterId = clusterIndex.GetClusterId();
   auto index = clusterIndex.GetIndex();
   auto columnId = columnHandle.fId;
   auto cachedPage = fPagePool->GetPage(columnId, clusterIndex);
   if (!cachedPage.IsNull())
      return cachedPage;

   R__ASSERT(clusterId != kInvalidDescriptorId);
   auto clusterDescriptor = fDescriptor.GetClusterDescriptor(clusterId);
   return PopulatePageFromCluster(columnHandle, clusterDescriptor, index);
}

void ROOT::Experimental::Detail::RPageSourceRaw::ReleasePage(RPage &page)
{
   fPagePool->ReturnPage(page);
}

std::unique_ptr<ROOT::Experimental::Detail::RPageSource> ROOT::Experimental::Detail::RPageSourceRaw::Clone() const
{
   auto clone = new RPageSourceRaw(fNTupleName, fOptions);
   clone->fFile = fFile->Clone();
   return std::unique_ptr<RPageSourceRaw>(clone);
}

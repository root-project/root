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
#include <ROOT/RClusterPool.hxx>
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
   , fMetrics("RPageSinkRaw")
   , fPageAllocator(std::make_unique<RPageAllocatorHeap>())
   , fZipBuffer(std::make_unique<std::array<char, kMaxPageSize>>())
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
      R__ASSERT(packedBytes <= kMaxPageSize);
      auto level = fOptions.GetCompression() % 100;
      auto algorithm = static_cast<ROOT::RCompressionSetting::EAlgorithm::EValues>(fOptions.GetCompression() / 100);
      int szZipBuffer = kMaxPageSize;
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
   , fClusterPool(std::make_unique<RClusterPool>(*this))
   , fUnzipBuffer(std::make_unique<std::array<unsigned char, kMaxPageSize>>())
   , fMetrics("RPageSourceRaw")
{
   if (fOptions.GetClusterCache() == RNTupleReadOptions::EClusterCache::kDefault)
      fOptions.SetClusterCache(RNTupleReadOptions::EClusterCache::kOff);

   fCtrNRead = fMetrics.MakeCounter<decltype(fCtrNRead)>("nRead", "", "number of read() calls");
   fCtrSzRead = fMetrics.MakeCounter<decltype(fCtrSzRead)>("szRead", "B", "volume read from file");
   fCtrSzUnzip = fMetrics.MakeCounter<decltype(fCtrSzUnzip)>("szUnzip", "B", "volume after unzipping");
   fCtrNPage = fMetrics.MakeCounter<decltype(fCtrNPage)>("nPage", "", "number of populated pages");
   fCtrNPageMmap = fMetrics.MakeCounter<decltype(fCtrNPageMmap)>("nPageMmap", "", "number mmap'd pages");
   fCtrNCacheMiss = fMetrics.MakeCounter<decltype(fCtrNCacheMiss)>(
      "nCacheMiss", "", "number of pages not found in the cluster cache");
   fCtrTimeWallRead = fMetrics.MakeCounter<decltype(fCtrTimeWallRead)>(
      "timeWallRead", "ns", "wall clock time spent reading");
   fCtrTimeCpuRead = fMetrics.MakeCounter<decltype(fCtrTimeCpuRead)>("timeCpuRead", "ns", "CPU time spent reading");
   fCtrTimeWallUnzip = fMetrics.MakeCounter<decltype(fCtrTimeWallUnzip)>(
      "timeWallUnzip", "ns", "wall clock time spent decompressing");
   fCtrTimeCpuUnzip = fMetrics.MakeCounter<decltype(fCtrTimeCpuUnzip)>(
      "timeCpuUnzip", "ns", "CPU time spent decompressing");
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
   // delete cluster pool before the file
   fClusterPool = nullptr;
}


void ROOT::Experimental::Detail::RPageSourceRaw::Read(void *buffer, std::size_t nbytes, std::uint64_t offset)
{
   RNTuplePlainTimer timer(*fCtrTimeWallRead, *fCtrTimeCpuRead);
   auto nread = fFile->ReadAt(buffer, nbytes, offset);
   R__ASSERT(nread == nbytes);
   fCtrSzRead->Add(nread);
   fCtrNRead->Inc();
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

   return descBuilder.MoveDescriptor();
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceRaw::PopulatePageFromCluster(
   ColumnHandle_t columnHandle, const RClusterDescriptor &clusterDescriptor, ClusterSize_t::ValueType clusterIndex)
{
   fCtrNPage->Inc();
   auto columnId = columnHandle.fId;
   auto clusterId = clusterDescriptor.GetId();
   const auto &pageRange = clusterDescriptor.GetPageRange(columnId);

   // TODO(jblomer): binary search
   // TODO(jblomer): move to descriptor class
   RClusterDescriptor::RPageRange::RPageInfo pageInfo;
   decltype(clusterIndex) firstInPage = 0;
   NTupleSize_t pageNo = 0;
   for (const auto &pi : pageRange.fPageInfos) {
      if (firstInPage + pi.fNElements > clusterIndex) {
         pageInfo = pi;
         break;
      }
      firstInPage += pi.fNElements;
      ++pageNo;
   }
   R__ASSERT(firstInPage <= clusterIndex);
   R__ASSERT((firstInPage + pageInfo.fNElements) > clusterIndex);

   auto element = columnHandle.fColumn->GetElement();
   auto elementSize = element->GetSize();
   auto pageSize = pageInfo.fLocator.fBytesOnStorage;
   void *pageBuffer = nullptr;

   bool isAdoptedPage = false;
   std::shared_ptr<RCluster> cluster;
   if (fOptions.GetClusterCache() != RNTupleReadOptions::EClusterCache::kOff) {
      cluster = fClusterPool->GetCluster(clusterId);
      RSheetKey sheetKey(columnId, pageNo);
      const auto sheetPtr = cluster->GetSheet(sheetKey);
      if (sheetPtr == nullptr) {
         fCtrNCacheMiss->Inc();
         pageBuffer = malloc(std::max(pageSize, static_cast<std::uint32_t>(elementSize * pageInfo.fNElements)));
         R__ASSERT(pageBuffer);
         Read(pageBuffer, pageSize, pageInfo.fLocator.fPosition);
      } else {
         R__ASSERT(pageSize == sheetPtr->GetSize());
         pageBuffer = const_cast<void *>(sheetPtr->GetAddress());
         isAdoptedPage = true;
      }
   } else {
      pageBuffer = malloc(std::max(pageSize, static_cast<std::uint32_t>(elementSize * pageInfo.fNElements)));
      R__ASSERT(pageBuffer);
      Read(pageBuffer, pageSize, pageInfo.fLocator.fPosition);
   }

   auto bytesOnStorage = (element->GetBitsOnStorage() * pageInfo.fNElements + 7) / 8;
   if (pageSize != bytesOnStorage) {
      RNTuplePlainTimer timer(*fCtrTimeWallUnzip, *fCtrTimeCpuUnzip);

      R__ASSERT(bytesOnStorage <= kMaxPageSize);
      // We do have the unzip information in the column range, but here we simply use the value from
      // the R__zip header
      int szUnzipBuffer = kMaxPageSize;
      int szSource = pageSize;
      unsigned char *source = reinterpret_cast<unsigned char *>(pageBuffer);
      int unzipBytes = 0;
      R__unzip(&szSource, source, &szUnzipBuffer, fUnzipBuffer->data(), &unzipBytes);
      R__ASSERT(unzipBytes > static_cast<int>(pageSize));
      if (isAdoptedPage) {
         pageBuffer = malloc(unzipBytes);
         R__ASSERT(pageBuffer);
         isAdoptedPage = false;
      }
      memcpy(pageBuffer, fUnzipBuffer->data(), unzipBytes);
      pageSize = unzipBytes;
      fCtrSzUnzip->Add(unzipBytes);
   }

   if (!element->IsMappable()) {
      pageSize = elementSize * pageInfo.fNElements;
      auto unpackedBuffer = reinterpret_cast<unsigned char *>(malloc(pageSize));
      R__ASSERT(unpackedBuffer != nullptr);
      element->Unpack(unpackedBuffer, pageBuffer, pageInfo.fNElements);
      if (!isAdoptedPage)
         free(pageBuffer);
      pageBuffer = unpackedBuffer;
   }

   auto indexOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
   auto newPage = fPageAllocator->NewPage(columnId, pageBuffer, elementSize, pageInfo.fNElements);
   newPage.SetWindow(indexOffset + firstInPage, RPage::RClusterInfo(clusterId, indexOffset));
   if (isAdoptedPage) {
      fMmapdPages.emplace(newPage.GetBuffer(), cluster);
      fPagePool->RegisterPage(newPage,
         RPageDeleter([](const RPage &page, void *userData)
            {
               auto mmapdPages =
                  reinterpret_cast<std::unordered_multimap<void *, std::shared_ptr<RCluster>>*>(userData);
               mmapdPages->erase(mmapdPages->find(page.GetBuffer()));
            }, &fMmapdPages));
      fCtrNPageMmap->Inc();
   } else {
      fPagePool->RegisterPage(newPage,
         RPageDeleter([](const RPage &page, void */*userData*/)
         {
            RPageAllocatorFile::DeletePage(page);
         }, nullptr));
   }
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
   const auto &clusterDescriptor = fDescriptor.GetClusterDescriptor(clusterId);
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
   const auto &clusterDescriptor = fDescriptor.GetClusterDescriptor(clusterId);
   return PopulatePageFromCluster(columnHandle, clusterDescriptor, index);
}

void ROOT::Experimental::Detail::RPageSourceRaw::ReleasePage(RPage &page)
{
   fPagePool->ReturnPage(page);
}


std::unique_ptr<ROOT::Experimental::Detail::RCluster>
ROOT::Experimental::Detail::RPageSourceRaw::LoadCluster(DescriptorId_t clusterId)
{
   std::cout << "LOADING CLUSTER NUMBER " << clusterId << std::endl;
   const auto &clusterDesc = GetDescriptor().GetClusterDescriptor(clusterId);
   auto clusterLocator = clusterDesc.GetLocator();
   auto clusterSize = clusterLocator.fBytesOnStorage;
   R__ASSERT(clusterSize > 0);

   auto activeSize = 0;
   if (fOptions.GetClusterCache() == RNTupleReadOptions::EClusterCache::kMMap) {
      activeSize = clusterSize;
   } else {
      for (auto columnId : fActiveColumns) {
         const auto &pageRange = clusterDesc.GetPageRange(columnId);
         for (const auto &pageInfo : pageRange.fPageInfos) {
            const auto &pageLocator = pageInfo.fLocator;
            activeSize += pageLocator.fBytesOnStorage;
         }
      }
   }

   if ((double(activeSize) / double(clusterSize)) < 0.5) {
      std::cout << "  ... PARTIAL FILLING OF CLUSTER CACHE" << std::endl;
      auto buffer = reinterpret_cast<unsigned char *>(malloc(activeSize));
      R__ASSERT(buffer);
      auto cluster = std::make_unique<RHeapCluster>(buffer, clusterId);
      size_t bufPos = 0;
      for (auto columnId : fActiveColumns) {
         const auto &pageRange = clusterDesc.GetPageRange(columnId);
         NTupleSize_t pageNo = 0;
         for (const auto &pageInfo : pageRange.fPageInfos) {
            // TODO(jblomer): read linear
            const auto &pageLocator = pageInfo.fLocator;
            Read(buffer + bufPos, pageLocator.fBytesOnStorage, pageLocator.fPosition);
            RSheetKey key(columnId, pageNo);
            RSheet sheet(buffer + bufPos, pageLocator.fBytesOnStorage);
            cluster->InsertSheet(key, sheet);
            bufPos += pageLocator.fBytesOnStorage;
            ++pageNo;
         }
      }
      return cluster;
   }

   std::unique_ptr<RCluster> cluster;
   size_t bufferOffset = 0;
   unsigned char *buffer = nullptr;
   if (fOptions.GetClusterCache() == RNTupleReadOptions::EClusterCache::kMMap) {
      std::uint64_t mmapdOffset;
      buffer = reinterpret_cast<unsigned char *>(fFile->Map(clusterSize, clusterLocator.fPosition, mmapdOffset));
      R__ASSERT(buffer);
      bufferOffset = clusterLocator.fPosition - mmapdOffset;
      cluster = std::make_unique<RMMapCluster>(buffer, clusterId, clusterSize + bufferOffset, *fFile);
   } else {
      buffer = reinterpret_cast<unsigned char *>(malloc(clusterSize));
      R__ASSERT(buffer);
      Read(buffer, clusterSize, clusterLocator.fPosition);
      cluster = std::make_unique<RHeapCluster>(buffer, clusterId);
   }

   // TODO(jblomer): make id range
   for (unsigned int i = 0; i < fDescriptor.GetNColumns(); ++i) {
      const auto &pageRange = clusterDesc.GetPageRange(i);
      NTupleSize_t pageNo = 0;
      for (const auto &pageInfo : pageRange.fPageInfos) {
         const auto &pageLocator = pageInfo.fLocator;
         RSheetKey key(i, pageNo);
         RSheet sheet(buffer + bufferOffset + pageLocator.fPosition - clusterLocator.fPosition,
                      pageLocator.fBytesOnStorage);
         //std::cout << "REGISTER SHEET " << i << "/" << pageNo << " @ "
         //          << sheet.GetAddress() << " : " << sheet.GetSize() << std::endl;
         cluster->InsertSheet(key, sheet);
         ++pageNo;
      }
   }

   return cluster;
}

std::unique_ptr<ROOT::Experimental::Detail::RPageSource> ROOT::Experimental::Detail::RPageSourceRaw::Clone() const
{
   auto clone = new RPageSourceRaw(fNTupleName, fOptions);
   clone->fFile = fFile->Clone();
   return std::unique_ptr<RPageSourceRaw>(clone);
}

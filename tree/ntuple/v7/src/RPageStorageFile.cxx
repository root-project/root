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

#include <ROOT/RCluster.hxx>
#include <ROOT/RClusterPool.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RLogger.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleSerialize.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RPage.hxx>
#include <ROOT/RPageAllocator.hxx>
#include <ROOT/RPagePool.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/RRawFile.hxx>
#include <ROOT/RRawFileTFile.hxx>

#include <RVersion.h>
#include <TError.h>
#include <TFile.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <utility>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <queue>

ROOT::Experimental::Internal::RPageSinkFile::RPageSinkFile(std::string_view ntupleName,
                                                           const RNTupleWriteOptions &options)
   : RPagePersistentSink(ntupleName, options), fPageAllocator(std::make_unique<RPageAllocatorHeap>())
{
   static std::once_flag once;
   std::call_once(once, []() {
      R__LOG_WARNING(NTupleLog()) << "The RNTuple file format will change. "
                                  << "Do not store real data with this version of RNTuple!";
   });
   fCompressor = std::make_unique<RNTupleCompressor>();
   EnableDefaultMetrics("RPageSinkFile");
}

ROOT::Experimental::Internal::RPageSinkFile::RPageSinkFile(std::string_view ntupleName, std::string_view path,
                                                           const RNTupleWriteOptions &options)
   : RPageSinkFile(ntupleName, options)
{
   fWriter = std::unique_ptr<RNTupleFileWriter>(RNTupleFileWriter::Recreate(
      ntupleName, path, options.GetCompression(), RNTupleFileWriter::EContainerFormat::kTFile));
}

ROOT::Experimental::Internal::RPageSinkFile::RPageSinkFile(std::string_view ntupleName, TFile &file,
                                                           const RNTupleWriteOptions &options)
   : RPageSinkFile(ntupleName, options)
{
   fWriter = std::unique_ptr<RNTupleFileWriter>(RNTupleFileWriter::Append(ntupleName, file));
}

ROOT::Experimental::Internal::RPageSinkFile::~RPageSinkFile() {}

void ROOT::Experimental::Internal::RPageSinkFile::InitImpl(unsigned char *serializedHeader, std::uint32_t length)
{
   auto zipBuffer = std::make_unique<unsigned char[]>(length);
   auto szZipHeader = fCompressor->Zip(serializedHeader, length, GetWriteOptions().GetCompression(),
                                       RNTupleCompressor::MakeMemCopyWriter(zipBuffer.get()));
   fWriter->WriteNTupleHeader(zipBuffer.get(), szZipHeader, length);
}

inline ROOT::Experimental::RNTupleLocator
ROOT::Experimental::Internal::RPageSinkFile::WriteSealedPage(const RPageStorage::RSealedPage &sealedPage,
                                                             std::size_t bytesPacked)
{
   std::uint64_t offsetData;
   {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallWrite, fCounters->fTimeCpuWrite);
      offsetData = fWriter->WriteBlob(sealedPage.fBuffer, sealedPage.fSize, bytesPacked);
   }

   RNTupleLocator result;
   result.fPosition = offsetData;
   result.fBytesOnStorage = sealedPage.fSize;
   fCounters->fNPageCommitted.Inc();
   fCounters->fSzWritePayload.Add(sealedPage.fSize);
   fNBytesCurrentCluster += sealedPage.fSize;
   return result;
}

ROOT::Experimental::RNTupleLocator
ROOT::Experimental::Internal::RPageSinkFile::CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page)
{
   auto element = columnHandle.fColumn->GetElement();
   RPageStorage::RSealedPage sealedPage;
   {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallZip, fCounters->fTimeCpuZip);
      sealedPage = SealPage(page, *element, GetWriteOptions().GetCompression());
   }

   fCounters->fSzZip.Add(page.GetNBytes());
   return WriteSealedPage(sealedPage, element->GetPackedSize(page.GetNElements()));
}

ROOT::Experimental::RNTupleLocator
ROOT::Experimental::Internal::RPageSinkFile::CommitSealedPageImpl(DescriptorId_t physicalColumnId,
                                                                  const RPageStorage::RSealedPage &sealedPage)
{
   const auto bitsOnStorage = RColumnElementBase::GetBitsOnStorage(
      fDescriptorBuilder.GetDescriptor().GetColumnDescriptor(physicalColumnId).GetModel().GetType());
   const auto bytesPacked = (bitsOnStorage * sealedPage.fNElements + 7) / 8;

   return WriteSealedPage(sealedPage, bytesPacked);
}

std::vector<ROOT::Experimental::RNTupleLocator>
ROOT::Experimental::Internal::RPageSinkFile::CommitSealedPageVImpl(std::span<RPageStorage::RSealedPageGroup> ranges)
{
   size_t size = 0, bytesPacked = 0;
   for (auto &range : ranges) {
      if (range.fFirst == range.fLast) {
         // Skip empty ranges, they might not have a physical column ID!
         continue;
      }

      const auto bitsOnStorage = RColumnElementBase::GetBitsOnStorage(
         fDescriptorBuilder.GetDescriptor().GetColumnDescriptor(range.fPhysicalColumnId).GetModel().GetType());
      for (auto sealedPageIt = range.fFirst; sealedPageIt != range.fLast; ++sealedPageIt) {
         size += sealedPageIt->fSize;
         bytesPacked += (bitsOnStorage * sealedPageIt->fNElements + 7) / 8;
      }
   }
   if (size >= std::numeric_limits<std::int32_t>::max() || bytesPacked >= std::numeric_limits<std::int32_t>::max()) {
      // Cannot fit it into one key, fall back to one key per page.
      // TODO: Remove once there is support for large keys.
      return RPagePersistentSink::CommitSealedPageVImpl(ranges);
   }

   Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallWrite, fCounters->fTimeCpuWrite);
   // Reserve a blob that is large enough to hold all pages.
   std::uint64_t offset = fWriter->ReserveBlob(size, bytesPacked);

   // Now write the individual pages and record their locators.
   std::vector<ROOT::Experimental::RNTupleLocator> locators;
   for (auto &range : ranges) {
      for (auto sealedPageIt = range.fFirst; sealedPageIt != range.fLast; ++sealedPageIt) {
         fWriter->WriteIntoReservedBlob(sealedPageIt->fBuffer, sealedPageIt->fSize, offset);
         RNTupleLocator locator;
         locator.fPosition = offset;
         locator.fBytesOnStorage = sealedPageIt->fSize;
         locators.push_back(locator);
         offset += sealedPageIt->fSize;
      }
   }

   fCounters->fNPageCommitted.Add(locators.size());
   fCounters->fSzWritePayload.Add(size);
   fNBytesCurrentCluster += size;

   return locators;
}

std::uint64_t ROOT::Experimental::Internal::RPageSinkFile::CommitClusterImpl()
{
   auto result = fNBytesCurrentCluster;
   fNBytesCurrentCluster = 0;
   return result;
}

ROOT::Experimental::RNTupleLocator
ROOT::Experimental::Internal::RPageSinkFile::CommitClusterGroupImpl(unsigned char *serializedPageList,
                                                                    std::uint32_t length)
{
   auto bufPageListZip = std::make_unique<unsigned char[]>(length);
   auto szPageListZip = fCompressor->Zip(serializedPageList, length, GetWriteOptions().GetCompression(),
                                         RNTupleCompressor::MakeMemCopyWriter(bufPageListZip.get()));

   RNTupleLocator result;
   result.fBytesOnStorage = szPageListZip;
   result.fPosition = fWriter->WriteBlob(bufPageListZip.get(), szPageListZip, length);
   return result;
}

void ROOT::Experimental::Internal::RPageSinkFile::CommitDatasetImpl(unsigned char *serializedFooter,
                                                                    std::uint32_t length)
{
   auto bufFooterZip = std::make_unique<unsigned char[]>(length);
   auto szFooterZip = fCompressor->Zip(serializedFooter, length, GetWriteOptions().GetCompression(),
                                       RNTupleCompressor::MakeMemCopyWriter(bufFooterZip.get()));
   fWriter->WriteNTupleFooter(bufFooterZip.get(), szFooterZip, length);
   fWriter->Commit();
}

ROOT::Experimental::Internal::RPage
ROOT::Experimental::Internal::RPageSinkFile::ReservePage(ColumnHandle_t columnHandle, std::size_t nElements)
{
   if (nElements == 0)
      throw RException(R__FAIL("invalid call: request empty page"));
   auto elementSize = columnHandle.fColumn->GetElement()->GetSize();
   return fPageAllocator->NewPage(columnHandle.fPhysicalId, elementSize, nElements);
}

void ROOT::Experimental::Internal::RPageSinkFile::ReleasePage(RPage &page)
{
   fPageAllocator->DeletePage(page);
}


////////////////////////////////////////////////////////////////////////////////

ROOT::Experimental::Internal::RPageSourceFile::RPageSourceFile(std::string_view ntupleName,
                                                               const RNTupleReadOptions &options)
   : RPageSource(ntupleName, options),
     fPagePool(std::make_shared<RPagePool>()),
     fClusterPool(std::make_unique<RClusterPool>(*this, options.GetClusterBunchSize()))
{
   fDecompressor = std::make_unique<RNTupleDecompressor>();
   EnableDefaultMetrics("RPageSourceFile");
}

ROOT::Experimental::Internal::RPageSourceFile::RPageSourceFile(std::string_view ntupleName,
                                                               std::unique_ptr<ROOT::Internal::RRawFile> file,
                                                               const RNTupleReadOptions &options)
   : RPageSourceFile(ntupleName, options)
{
   fFile = std::move(file);
   R__ASSERT(fFile);
   fReader = RMiniFileReader(fFile.get());
}

ROOT::Experimental::Internal::RPageSourceFile::RPageSourceFile(std::string_view ntupleName, std::string_view path,
                                                               const RNTupleReadOptions &options)
   : RPageSourceFile(ntupleName, ROOT::Internal::RRawFile::Create(path), options)
{
}

void ROOT::Experimental::Internal::RPageSourceFile::InitDescriptor(const RNTuple &anchor)
{
   // TOOD(jblomer): can the epoch check be factored out across anchors?
   if (anchor.GetVersionEpoch() != RNTuple::kVersionEpoch) {
      throw RException(R__FAIL("unsupported RNTuple epoch version: " + std::to_string(anchor.GetVersionEpoch())));
   }
   if (anchor.GetVersionEpoch() == 0) {
      static std::once_flag once;
      std::call_once(once, [&anchor]() {
         R__LOG_WARNING(NTupleLog()) << "Pre-release format version: RC " << anchor.GetVersionMajor();
      });
   }

   fDescriptorBuilder.SetOnDiskHeaderSize(anchor.GetNBytesHeader());
   auto buffer = std::make_unique<unsigned char[]>(anchor.GetLenHeader());
   auto zipBuffer = std::make_unique<unsigned char[]>(anchor.GetNBytesHeader());
   fReader.ReadBuffer(zipBuffer.get(), anchor.GetNBytesHeader(), anchor.GetSeekHeader());
   fDecompressor->Unzip(zipBuffer.get(), anchor.GetNBytesHeader(), anchor.GetLenHeader(), buffer.get());
   RNTupleSerializer::DeserializeHeader(buffer.get(), anchor.GetLenHeader(), fDescriptorBuilder);

   fDescriptorBuilder.AddToOnDiskFooterSize(anchor.GetNBytesFooter());
   buffer = std::make_unique<unsigned char[]>(anchor.GetLenFooter());
   zipBuffer = std::make_unique<unsigned char[]>(anchor.GetNBytesFooter());
   fReader.ReadBuffer(zipBuffer.get(), anchor.GetNBytesFooter(), anchor.GetSeekFooter());
   fDecompressor->Unzip(zipBuffer.get(), anchor.GetNBytesFooter(), anchor.GetLenFooter(), buffer.get());
   RNTupleSerializer::DeserializeFooter(buffer.get(), anchor.GetLenFooter(), fDescriptorBuilder);
}

std::unique_ptr<ROOT::Experimental::Internal::RPageSourceFile>
ROOT::Experimental::Internal::RPageSourceFile::CreateFromAnchor(const RNTuple &anchor,
                                                                const RNTupleReadOptions &options)
{
   if (!anchor.fFile)
      throw RException(R__FAIL("This RNTuple object was not streamed from a ROOT file (TFile or descendant)"));

   std::unique_ptr<ROOT::Internal::RRawFile> rawFile;
   // For local TFiles, TDavixFile, and TNetXNGFile, we want to open a new RRawFile to take advantage of the faster
   // reading. We check the exact class name to avoid classes inheriting in ROOT (for example TMemFile) or in
   // experiment frameworks.
   std::string className = anchor.fFile->IsA()->GetName();
   auto url = anchor.fFile->GetEndpointUrl();
   auto protocol = std::string(url->GetProtocol());
   if (className == "TFile") {
      rawFile = ROOT::Internal::RRawFile::Create(url->GetFile());
   } else if (className == "TDavixFile" || className == "TNetXNGFile") {
      rawFile = ROOT::Internal::RRawFile::Create(url->GetUrl());
   } else {
      rawFile.reset(new ROOT::Internal::RRawFileTFile(anchor.fFile));
   }

   auto pageSource = std::make_unique<RPageSourceFile>("", std::move(rawFile), options);
   pageSource->InitDescriptor(anchor);
   pageSource->fNTupleName = pageSource->fDescriptorBuilder.GetDescriptor().GetName();
   return pageSource;
}

ROOT::Experimental::Internal::RPageSourceFile::~RPageSourceFile() = default;

ROOT::Experimental::RNTupleDescriptor ROOT::Experimental::Internal::RPageSourceFile::AttachImpl()
{
   // If we constructed the page source with (ntuple name, path), we need to find the anchor first.
   // Otherwise, the page source was created by OpenFromAnchor() and the header and footer are already processed.
   if (fDescriptorBuilder.GetDescriptor().GetOnDiskHeaderSize() == 0) {
      auto anchor = fReader.GetNTuple(fNTupleName).Unwrap();
      InitDescriptor(anchor);
   }

   auto desc = fDescriptorBuilder.MoveDescriptor();

   for (const auto &cgDesc : desc.GetClusterGroupIterable()) {
      auto buffer = std::make_unique<unsigned char[]>(cgDesc.GetPageListLength());
      auto zipBuffer = std::make_unique<unsigned char[]>(cgDesc.GetPageListLocator().fBytesOnStorage);
      fReader.ReadBuffer(zipBuffer.get(), cgDesc.GetPageListLocator().fBytesOnStorage,
                         cgDesc.GetPageListLocator().GetPosition<std::uint64_t>());
      fDecompressor->Unzip(zipBuffer.get(), cgDesc.GetPageListLocator().fBytesOnStorage, cgDesc.GetPageListLength(),
                           buffer.get());

      RNTupleSerializer::DeserializePageList(buffer.get(), cgDesc.GetPageListLength(), cgDesc.GetId(), desc);
   }

   // For the page reads, we rely on the I/O scheduler to define the read requests
   fFile->SetBuffering(false);

   return desc;
}

void ROOT::Experimental::Internal::RPageSourceFile::LoadSealedPage(DescriptorId_t physicalColumnId,
                                                                   RClusterIndex clusterIndex, RSealedPage &sealedPage)
{
   const auto clusterId = clusterIndex.GetClusterId();

   RClusterDescriptor::RPageRange::RPageInfo pageInfo;
   {
      auto descriptorGuard = GetSharedDescriptorGuard();
      const auto &clusterDescriptor = descriptorGuard->GetClusterDescriptor(clusterId);
      pageInfo = clusterDescriptor.GetPageRange(physicalColumnId).Find(clusterIndex.GetIndex());
   }

   const auto bytesOnStorage = pageInfo.fLocator.fBytesOnStorage;
   sealedPage.fSize = bytesOnStorage;
   sealedPage.fNElements = pageInfo.fNElements;
   if (!sealedPage.fBuffer)
      return;
   if (pageInfo.fLocator.fType != RNTupleLocator::kTypePageZero) {
      fReader.ReadBuffer(const_cast<void *>(sealedPage.fBuffer), bytesOnStorage,
                         pageInfo.fLocator.GetPosition<std::uint64_t>());
   } else {
      memcpy(const_cast<void *>(sealedPage.fBuffer), RPage::GetPageZeroBuffer(), bytesOnStorage);
   }
}

ROOT::Experimental::Internal::RPage
ROOT::Experimental::Internal::RPageSourceFile::PopulatePageFromCluster(ColumnHandle_t columnHandle,
                                                                       const RClusterInfo &clusterInfo,
                                                                       ClusterSize_t::ValueType idxInCluster)
{
   const auto columnId = columnHandle.fPhysicalId;
   const auto clusterId = clusterInfo.fClusterId;
   const auto pageInfo = clusterInfo.fPageInfo;

   const auto element = columnHandle.fColumn->GetElement();
   const auto elementSize = element->GetSize();
   const auto bytesOnStorage = pageInfo.fLocator.fBytesOnStorage;

   const void *sealedPageBuffer = nullptr; // points either to directReadBuffer or to a read-only page in the cluster
   std::unique_ptr<unsigned char []> directReadBuffer; // only used if cluster pool is turned off

   if (pageInfo.fLocator.fType == RNTupleLocator::kTypePageZero) {
      auto pageZero = RPage::MakePageZero(columnId, elementSize);
      pageZero.GrowUnchecked(pageInfo.fNElements);
      pageZero.SetWindow(clusterInfo.fColumnOffset + pageInfo.fFirstInPage,
                         RPage::RClusterInfo(clusterId, clusterInfo.fColumnOffset));
      fPagePool->RegisterPage(pageZero, RPageDeleter([](const RPage &, void *) {}, nullptr));
      return pageZero;
   }

   if (fOptions.GetClusterCache() == RNTupleReadOptions::EClusterCache::kOff) {
      directReadBuffer = std::unique_ptr<unsigned char[]>(new unsigned char[bytesOnStorage]);
      fReader.ReadBuffer(directReadBuffer.get(), bytesOnStorage, pageInfo.fLocator.GetPosition<std::uint64_t>());
      fCounters->fNPageLoaded.Inc();
      fCounters->fNRead.Inc();
      fCounters->fSzReadPayload.Add(bytesOnStorage);
      sealedPageBuffer = directReadBuffer.get();
   } else {
      if (!fCurrentCluster || (fCurrentCluster->GetId() != clusterId) || !fCurrentCluster->ContainsColumn(columnId))
         fCurrentCluster = fClusterPool->GetCluster(clusterId, fActivePhysicalColumns.ToColumnSet());
      R__ASSERT(fCurrentCluster->ContainsColumn(columnId));

      auto cachedPage = fPagePool->GetPage(columnId, RClusterIndex(clusterId, idxInCluster));
      if (!cachedPage.IsNull())
         return cachedPage;

      ROnDiskPage::Key key(columnId, pageInfo.fPageNo);
      auto onDiskPage = fCurrentCluster->GetOnDiskPage(key);
      R__ASSERT(onDiskPage && (bytesOnStorage == onDiskPage->GetSize()));
      sealedPageBuffer = onDiskPage->GetAddress();
   }

   RPage newPage;
   {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallUnzip, fCounters->fTimeCpuUnzip);
      newPage = UnsealPage({sealedPageBuffer, bytesOnStorage, pageInfo.fNElements}, *element, columnId);
      fCounters->fSzUnzip.Add(elementSize * pageInfo.fNElements);
   }

   newPage.SetWindow(clusterInfo.fColumnOffset + pageInfo.fFirstInPage,
                     RPage::RClusterInfo(clusterId, clusterInfo.fColumnOffset));
   fPagePool->RegisterPage(
      newPage, RPageDeleter([](const RPage &page, void *) { RPageAllocatorHeap::DeletePage(page); }, nullptr));
   fCounters->fNPagePopulated.Inc();
   return newPage;
}

ROOT::Experimental::Internal::RPage
ROOT::Experimental::Internal::RPageSourceFile::PopulatePage(ColumnHandle_t columnHandle, NTupleSize_t globalIndex)
{
   const auto columnId = columnHandle.fPhysicalId;
   auto cachedPage = fPagePool->GetPage(columnId, globalIndex);
   if (!cachedPage.IsNull())
      return cachedPage;

   std::uint64_t idxInCluster;
   RClusterInfo clusterInfo;
   {
      auto descriptorGuard = GetSharedDescriptorGuard();
      clusterInfo.fClusterId = descriptorGuard->FindClusterId(columnId, globalIndex);

      if (clusterInfo.fClusterId == kInvalidDescriptorId)
         throw RException(R__FAIL("entry with index " + std::to_string(globalIndex) + " out of bounds"));

      const auto &clusterDescriptor = descriptorGuard->GetClusterDescriptor(clusterInfo.fClusterId);
      clusterInfo.fColumnOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
      R__ASSERT(clusterInfo.fColumnOffset <= globalIndex);
      idxInCluster = globalIndex - clusterInfo.fColumnOffset;
      clusterInfo.fPageInfo = clusterDescriptor.GetPageRange(columnId).Find(idxInCluster);
   }

   return PopulatePageFromCluster(columnHandle, clusterInfo, idxInCluster);
}

ROOT::Experimental::Internal::RPage
ROOT::Experimental::Internal::RPageSourceFile::PopulatePage(ColumnHandle_t columnHandle, RClusterIndex clusterIndex)
{
   const auto clusterId = clusterIndex.GetClusterId();
   const auto idxInCluster = clusterIndex.GetIndex();
   const auto columnId = columnHandle.fPhysicalId;
   auto cachedPage = fPagePool->GetPage(columnId, clusterIndex);
   if (!cachedPage.IsNull())
      return cachedPage;

   if (clusterId == kInvalidDescriptorId)
      throw RException(R__FAIL("entry out of bounds"));

   RClusterInfo clusterInfo;
   {
      auto descriptorGuard = GetSharedDescriptorGuard();
      const auto &clusterDescriptor = descriptorGuard->GetClusterDescriptor(clusterId);
      clusterInfo.fClusterId = clusterId;
      clusterInfo.fColumnOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
      clusterInfo.fPageInfo = clusterDescriptor.GetPageRange(columnId).Find(idxInCluster);
   }

   return PopulatePageFromCluster(columnHandle, clusterInfo, idxInCluster);
}

void ROOT::Experimental::Internal::RPageSourceFile::ReleasePage(RPage &page)
{
   fPagePool->ReturnPage(page);
}

std::unique_ptr<ROOT::Experimental::Internal::RPageSource> ROOT::Experimental::Internal::RPageSourceFile::Clone() const
{
   auto clone = new RPageSourceFile(fNTupleName, fOptions);
   clone->fFile = fFile->Clone();
   clone->fReader = RMiniFileReader(clone->fFile.get());
   return std::unique_ptr<RPageSourceFile>(clone);
}

std::unique_ptr<ROOT::Experimental::Internal::RCluster>
ROOT::Experimental::Internal::RPageSourceFile::PrepareSingleCluster(
   const RCluster::RKey &clusterKey, std::vector<ROOT::Internal::RRawFile::RIOVec> &readRequests)
{
   struct ROnDiskPageLocator {
      ROOT::Experimental::DescriptorId_t fColumnId = 0;
      ROOT::Experimental::NTupleSize_t fPageNo = 0;
      std::uint64_t fOffset = 0;
      std::uint64_t fSize = 0;
      std::size_t fBufPos = 0;
   };

   std::vector<ROnDiskPageLocator> onDiskPages;
   auto activeSize = 0;
   auto pageZeroMap = std::make_unique<ROnDiskPageMap>();
   PrepareLoadCluster(clusterKey, *pageZeroMap,
                      [&](DescriptorId_t physicalColumnId, NTupleSize_t pageNo,
                          const RClusterDescriptor::RPageRange::RPageInfo &pageInfo) {
                         const auto &pageLocator = pageInfo.fLocator;
                         activeSize += pageLocator.fBytesOnStorage;
                         onDiskPages.push_back({physicalColumnId, pageNo, pageLocator.GetPosition<std::uint64_t>(),
                                                pageLocator.fBytesOnStorage, 0});
                      });

   // Linearize the page requests by file offset
   std::sort(onDiskPages.begin(), onDiskPages.end(),
      [](const ROnDiskPageLocator &a, const ROnDiskPageLocator &b) {return a.fOffset < b.fOffset;});

   // In order to coalesce close-by pages, we collect the sizes of the gaps between pages on disk.  We then order
   // the gaps by size, sum them up and find a cutoff for the largest gap that we tolerate when coalescing pages.
   // The size of the cutoff is given by the fraction of extra bytes we are willing to read in order to reduce
   // the number of read requests.  We thus schedule the lowest number of requests given a tolerable fraction
   // of extra bytes.
   // TODO(jblomer): Eventually we may want to select the parameter at runtime according to link latency and speed,
   // memory consumption, device block size.
   float maxOverhead = 0.25 * float(activeSize);
   std::vector<std::size_t> gaps;
   for (unsigned i = 1; i < onDiskPages.size(); ++i) {
      gaps.emplace_back(onDiskPages[i].fOffset - (onDiskPages[i-1].fSize + onDiskPages[i-1].fOffset));
   }
   std::sort(gaps.begin(), gaps.end());
   std::size_t gapCut = 0;
   std::size_t currentGap = 0;
   float szExtra = 0.0;
   for (auto g : gaps) {
      if (g != currentGap) {
         gapCut = currentGap;
         currentGap = g;
      }
      szExtra += g;
      if (szExtra  > maxOverhead)
         break;
   }

   // In a first step, we coalesce the read requests and calculate the cluster buffer size.
   // In a second step, we'll fix-up the memory destinations for the read calls given the
   // address of the allocated buffer.  We must not touch, however, the read requests from previous
   // calls to PrepareSingleCluster()
   const auto currentReadRequestIdx = readRequests.size();

   ROOT::Internal::RRawFile::RIOVec req;
   std::size_t szPayload = 0;
   std::size_t szOverhead = 0;
   for (auto &s : onDiskPages) {
      R__ASSERT(s.fSize > 0);
      auto readUpTo = req.fOffset + req.fSize;
      R__ASSERT(s.fOffset >= readUpTo);
      auto overhead = s.fOffset - readUpTo;
      szPayload += s.fSize;
      if (overhead <= gapCut) {
         szOverhead += overhead;
         s.fBufPos = reinterpret_cast<intptr_t>(req.fBuffer) + req.fSize + overhead;
         req.fSize += overhead + s.fSize;
         continue;
      }

      // close the current request and open new one
      if (req.fSize > 0)
         readRequests.emplace_back(req);

      req.fBuffer = reinterpret_cast<unsigned char *>(req.fBuffer) + req.fSize;
      s.fBufPos = reinterpret_cast<intptr_t>(req.fBuffer);

      req.fOffset = s.fOffset;
      req.fSize = s.fSize;
   }
   readRequests.emplace_back(req);
   fCounters->fSzReadPayload.Add(szPayload);
   fCounters->fSzReadOverhead.Add(szOverhead);

   // Register the on disk pages in a page map
   auto buffer = new unsigned char[reinterpret_cast<intptr_t>(req.fBuffer) + req.fSize];
   auto pageMap = std::make_unique<ROnDiskPageMapHeap>(std::unique_ptr<unsigned char []>(buffer));
   for (const auto &s : onDiskPages) {
      ROnDiskPage::Key key(s.fColumnId, s.fPageNo);
      pageMap->Register(key, ROnDiskPage(buffer + s.fBufPos, s.fSize));
   }
   fCounters->fNPageLoaded.Add(onDiskPages.size());
   for (auto i = currentReadRequestIdx; i < readRequests.size(); ++i) {
      readRequests[i].fBuffer = buffer + reinterpret_cast<intptr_t>(readRequests[i].fBuffer);
   }

   auto cluster = std::make_unique<RCluster>(clusterKey.fClusterId);
   cluster->Adopt(std::move(pageMap));
   cluster->Adopt(std::move(pageZeroMap));
   for (auto colId : clusterKey.fPhysicalColumnSet)
      cluster->SetColumnAvailable(colId);
   return cluster;
}

std::vector<std::unique_ptr<ROOT::Experimental::Internal::RCluster>>
ROOT::Experimental::Internal::RPageSourceFile::LoadClusters(std::span<RCluster::RKey> clusterKeys)
{
   fCounters->fNClusterLoaded.Add(clusterKeys.size());

   std::vector<std::unique_ptr<ROOT::Experimental::Internal::RCluster>> clusters;
   std::vector<ROOT::Internal::RRawFile::RIOVec> readRequests;

   for (auto key: clusterKeys) {
      clusters.emplace_back(PrepareSingleCluster(key, readRequests));
   }

   auto nReqs = readRequests.size();
   auto readvLimits = fFile->GetReadVLimits();

   int iReq = 0;
   while (nReqs > 0) {
      auto nBatch = std::min(nReqs, readvLimits.fMaxReqs);

      if (readvLimits.HasSizeLimit()) {
         std::uint64_t totalSize = 0;
         for (std::size_t i = 0; i < nBatch; ++i) {
            if (readRequests[iReq + i].fSize > readvLimits.fMaxSingleSize) {
               nBatch = i;
               break;
            }

            totalSize += readRequests[iReq + i].fSize;
            if (totalSize > readvLimits.fMaxTotalSize) {
               nBatch = i;
               break;
            }
         }
      }

      if (nBatch <= 1) {
         nBatch = 1;
         Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallRead, fCounters->fTimeCpuRead);
         fFile->ReadAt(readRequests[iReq].fBuffer, readRequests[iReq].fSize, readRequests[iReq].fOffset);
      } else {
         Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallRead, fCounters->fTimeCpuRead);
         fFile->ReadV(&readRequests[iReq], nBatch);
      }
      fCounters->fNReadV.Inc();
      fCounters->fNRead.Add(nBatch);

      iReq += nBatch;
      nReqs -= nBatch;
   }

   return clusters;
}

void ROOT::Experimental::Internal::RPageSourceFile::UnzipClusterImpl(RCluster *cluster)
{
   Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallUnzip, fCounters->fTimeCpuUnzip);

   const auto clusterId = cluster->GetId();
   auto descriptorGuard = GetSharedDescriptorGuard();
   const auto &clusterDescriptor = descriptorGuard->GetClusterDescriptor(clusterId);

   std::vector<std::unique_ptr<RColumnElementBase>> allElements;

   const auto &columnsInCluster = cluster->GetAvailPhysicalColumns();
   for (const auto columnId : columnsInCluster) {
      const auto &columnDesc = descriptorGuard->GetColumnDescriptor(columnId);

      allElements.emplace_back(RColumnElementBase::Generate(columnDesc.GetModel().GetType()));

      const auto &pageRange = clusterDescriptor.GetPageRange(columnId);
      std::uint64_t pageNo = 0;
      std::uint64_t firstInPage = 0;
      for (const auto &pi : pageRange.fPageInfos) {
         ROnDiskPage::Key key(columnId, pageNo);
         auto onDiskPage = cluster->GetOnDiskPage(key);
         R__ASSERT(onDiskPage && (onDiskPage->GetSize() == pi.fLocator.fBytesOnStorage));

         auto taskFunc = [this, columnId, clusterId, firstInPage, onDiskPage, element = allElements.back().get(),
                          nElements = pi.fNElements,
                          indexOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex]() {
            auto newPage = UnsealPage({onDiskPage->GetAddress(), onDiskPage->GetSize(), nElements}, *element, columnId);
            fCounters->fSzUnzip.Add(element->GetSize() * nElements);

            newPage.SetWindow(indexOffset + firstInPage, RPage::RClusterInfo(clusterId, indexOffset));
            fPagePool->PreloadPage(
               newPage, RPageDeleter([](const RPage &page, void *) { RPageAllocatorHeap::DeletePage(page); }, nullptr));
         };

         fTaskScheduler->AddTask(taskFunc);

         firstInPage += pi.fNElements;
         pageNo++;
      } // for all pages in column
   } // for all columns in cluster

   fCounters->fNPagePopulated.Add(cluster->GetNOnDiskPages());

   fTaskScheduler->Wait();
}

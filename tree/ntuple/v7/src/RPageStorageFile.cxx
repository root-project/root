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
#include <ROOT/RNTupleUtil.hxx>

#include <RVersion.h>
#include <TError.h>
#include <TFile.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <utility>

#include <functional>
#include <mutex>

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
   fFeatures.fCanMergePages = true;
}

ROOT::Experimental::Internal::RPageSinkFile::RPageSinkFile(std::string_view ntupleName, std::string_view path,
                                                           const RNTupleWriteOptions &options)
   : RPageSinkFile(ntupleName, options)
{
   fWriter = RNTupleFileWriter::Recreate(ntupleName, path, RNTupleFileWriter::EContainerFormat::kTFile, options);
}

ROOT::Experimental::Internal::RPageSinkFile::RPageSinkFile(std::string_view ntupleName, TFile &file,
                                                           const RNTupleWriteOptions &options)
   : RPageSinkFile(ntupleName, options)
{
   fWriter = RNTupleFileWriter::Append(ntupleName, file, options.GetMaxKeySize());
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
      offsetData = fWriter->WriteBlob(sealedPage.GetBuffer(), sealedPage.GetBufferSize(), bytesPacked);
   }

   RNTupleLocator result;
   result.fPosition = offsetData;
   result.fBytesOnStorage = sealedPage.GetDataSize();
   fCounters->fNPageCommitted.Inc();
   fCounters->fSzWritePayload.Add(sealedPage.GetBufferSize());
   fNBytesCurrentCluster += sealedPage.GetBufferSize();
   return result;
}

ROOT::Experimental::RNTupleLocator
ROOT::Experimental::Internal::RPageSinkFile::CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page)
{
   auto element = columnHandle.fColumn->GetElement();
   RPageStorage::RSealedPage sealedPage;
   {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallZip, fCounters->fTimeCpuZip);
      sealedPage = SealPage(page, *element);
   }

   fCounters->fSzZip.Add(page.GetNBytes());
   return WriteSealedPage(sealedPage, element->GetPackedSize(page.GetNElements()));
}

ROOT::Experimental::RNTupleLocator
ROOT::Experimental::Internal::RPageSinkFile::CommitSealedPageImpl(DescriptorId_t physicalColumnId,
                                                                  const RPageStorage::RSealedPage &sealedPage)
{
   const auto nBits = fDescriptorBuilder.GetDescriptor().GetColumnDescriptor(physicalColumnId).GetBitsOnStorage();
   const auto bytesPacked = (nBits * sealedPage.GetNElements() + 7) / 8;
   return WriteSealedPage(sealedPage, bytesPacked);
}

void ROOT::Experimental::Internal::RPageSinkFile::CommitBatchOfPages(CommitBatch &batch,
                                                                     std::vector<RNTupleLocator> &locators)
{
   Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallWrite, fCounters->fTimeCpuWrite);

   std::uint64_t offset = fWriter->ReserveBlob(batch.fSize, batch.fBytesPacked);

   locators.reserve(locators.size() + batch.fSealedPages.size());

   for (const auto *pagePtr : batch.fSealedPages) {
      fWriter->WriteIntoReservedBlob(pagePtr->GetBuffer(), pagePtr->GetBufferSize(), offset);
      RNTupleLocator locator;
      locator.fPosition = offset;
      locator.fBytesOnStorage = pagePtr->GetDataSize();
      locators.push_back(locator);
      offset += pagePtr->GetBufferSize();
   }

   fCounters->fNPageCommitted.Add(batch.fSealedPages.size());
   fCounters->fSzWritePayload.Add(batch.fSize);
   fNBytesCurrentCluster += batch.fSize;

   batch.fSize = 0;
   batch.fBytesPacked = 0;
   batch.fSealedPages.clear();
}

std::vector<ROOT::Experimental::RNTupleLocator>
ROOT::Experimental::Internal::RPageSinkFile::CommitSealedPageVImpl(std::span<RPageStorage::RSealedPageGroup> ranges,
                                                                   const std::vector<bool> &mask)
{
   const std::uint64_t maxKeySize = fOptions->GetMaxKeySize();

   CommitBatch batch{};
   std::vector<ROOT::Experimental::RNTupleLocator> locators;

   std::size_t iPage = 0;
   for (auto rangeIt = ranges.begin(); rangeIt != ranges.end(); ++rangeIt) {
      auto &range = *rangeIt;
      if (range.fFirst == range.fLast) {
         // Skip empty ranges, they might not have a physical column ID!
         continue;
      }

      const auto bitsOnStorage =
         fDescriptorBuilder.GetDescriptor().GetColumnDescriptor(range.fPhysicalColumnId).GetBitsOnStorage();

      for (auto sealedPageIt = range.fFirst; sealedPageIt != range.fLast; ++sealedPageIt, ++iPage) {
         if (!mask[iPage])
            continue;

         const auto bytesPacked = (bitsOnStorage * sealedPageIt->GetNElements() + 7) / 8;

         if (batch.fSize > 0 && batch.fSize + sealedPageIt->GetBufferSize() > maxKeySize) {
            /**
             * Adding this page would exceed maxKeySize. Since we always want to write into a single key
             * with vectorized writes, we commit the current set of pages before proceeding.
             * NOTE: we do this *before* checking if sealedPageIt->GetBufferSize() > maxKeySize to guarantee that
             * we always flush the current batch before doing an individual WriteBlob. This way we
             * preserve the assumption that a CommitBatch always contain a sequential set of pages.
             */
            CommitBatchOfPages(batch, locators);
         }

         if (sealedPageIt->GetBufferSize() > maxKeySize) {
            // This page alone is bigger than maxKeySize: save it by itself, since it will need to be
            // split into multiple keys.

            // Since this check implies the previous check on batchSize + newSize > maxSize, we should
            // already have committed the current batch before writing this page.
            assert(batch.fSize == 0);

            std::uint64_t offset =
               fWriter->WriteBlob(sealedPageIt->GetBuffer(), sealedPageIt->GetBufferSize(), bytesPacked);
            RNTupleLocator locator;
            locator.fPosition = offset;
            locator.fBytesOnStorage = sealedPageIt->GetDataSize();
            locators.push_back(locator);

            fCounters->fNPageCommitted.Inc();
            fCounters->fSzWritePayload.Add(sealedPageIt->GetBufferSize());
            fNBytesCurrentCluster += sealedPageIt->GetBufferSize();

         } else {
            batch.fSealedPages.emplace_back(&(*sealedPageIt));
            batch.fSize += sealedPageIt->GetBufferSize();
            batch.fBytesPacked += bytesPacked;
         }
      }
   }

   if (batch.fSize > 0) {
      CommitBatchOfPages(batch, locators);
   }

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
     fClusterPool(std::make_unique<RClusterPool>(*this, options.GetClusterBunchSize()))
{
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
   pageSource->fAnchor = anchor;
   pageSource->fNTupleName = pageSource->fDescriptorBuilder.GetDescriptor().GetName();
   return pageSource;
}

ROOT::Experimental::Internal::RPageSourceFile::~RPageSourceFile() = default;

void ROOT::Experimental::Internal::RPageSourceFile::LoadStructureImpl()
{
   // If we constructed the page source with (ntuple name, path), we need to find the anchor first.
   // Otherwise, the page source was created by OpenFromAnchor()
   if (!fAnchor)
      fAnchor = fReader.GetNTuple(fNTupleName).Unwrap();

   // TOOD(jblomer): can the epoch check be factored out across anchors?
   if (fAnchor->GetVersionEpoch() != RNTuple::kVersionEpoch) {
      throw RException(R__FAIL("unsupported RNTuple epoch version: " + std::to_string(fAnchor->GetVersionEpoch())));
   }
   if (fAnchor->GetVersionEpoch() == 0) {
      static std::once_flag once;
      std::call_once(once, [this]() {
         R__LOG_WARNING(NTupleLog()) << "Pre-release format version: RC " << fAnchor->GetVersionMajor();
      });
   }

   fDescriptorBuilder.SetOnDiskHeaderSize(fAnchor->GetNBytesHeader());
   fDescriptorBuilder.AddToOnDiskFooterSize(fAnchor->GetNBytesFooter());

   // Reserve enough space for the compressed and the uncompressed header/footer (see AttachImpl)
   const auto bufSize = fAnchor->GetNBytesHeader() + fAnchor->GetNBytesFooter() +
                        std::max(fAnchor->GetLenHeader(), fAnchor->GetLenFooter());
   fStructureBuffer.fBuffer = std::make_unique<unsigned char[]>(bufSize);
   fStructureBuffer.fPtrHeader = fStructureBuffer.fBuffer.get();
   fStructureBuffer.fPtrFooter = fStructureBuffer.fBuffer.get() + fAnchor->GetNBytesHeader();

   auto readvLimits = fFile->GetReadVLimits();
   // Never try to vectorize reads to a split key
   readvLimits.fMaxSingleSize = std::min<size_t>(readvLimits.fMaxSingleSize, fAnchor->GetMaxKeySize());

   if ((readvLimits.fMaxReqs < 2) ||
       (std::max(fAnchor->GetNBytesHeader(), fAnchor->GetNBytesFooter()) > readvLimits.fMaxSingleSize) ||
       (fAnchor->GetNBytesHeader() + fAnchor->GetNBytesFooter() > readvLimits.fMaxTotalSize)) {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallRead, fCounters->fTimeCpuRead);
      fReader.ReadBuffer(fStructureBuffer.fPtrHeader, fAnchor->GetNBytesHeader(), fAnchor->GetSeekHeader());
      fReader.ReadBuffer(fStructureBuffer.fPtrFooter, fAnchor->GetNBytesFooter(), fAnchor->GetSeekFooter());
      fCounters->fNRead.Add(2);
   } else {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallRead, fCounters->fTimeCpuRead);
      ROOT::Internal::RRawFile::RIOVec readRequests[2] = {
         {fStructureBuffer.fPtrHeader, fAnchor->GetSeekHeader(), fAnchor->GetNBytesHeader(), 0},
         {fStructureBuffer.fPtrFooter, fAnchor->GetSeekFooter(), fAnchor->GetNBytesFooter(), 0}};
      fFile->ReadV(readRequests, 2);
      fCounters->fNReadV.Inc();
   }
}

ROOT::Experimental::RNTupleDescriptor ROOT::Experimental::Internal::RPageSourceFile::AttachImpl()
{
   auto unzipBuf = reinterpret_cast<unsigned char *>(fStructureBuffer.fPtrFooter) + fAnchor->GetNBytesFooter();

   RNTupleDecompressor::Unzip(fStructureBuffer.fPtrHeader, fAnchor->GetNBytesHeader(), fAnchor->GetLenHeader(),
                              unzipBuf);
   RNTupleSerializer::DeserializeHeader(unzipBuf, fAnchor->GetLenHeader(), fDescriptorBuilder);

   RNTupleDecompressor::Unzip(fStructureBuffer.fPtrFooter, fAnchor->GetNBytesFooter(), fAnchor->GetLenFooter(),
                              unzipBuf);
   RNTupleSerializer::DeserializeFooter(unzipBuf, fAnchor->GetLenFooter(), fDescriptorBuilder);

   auto desc = fDescriptorBuilder.MoveDescriptor();

   std::vector<unsigned char> buffer;
   for (const auto &cgDesc : desc.GetClusterGroupIterable()) {
      buffer.resize(
         std::max<size_t>(buffer.size(), cgDesc.GetPageListLength() + cgDesc.GetPageListLocator().fBytesOnStorage));
      auto *zipBuffer = buffer.data() + cgDesc.GetPageListLength();
      fReader.ReadBuffer(zipBuffer, cgDesc.GetPageListLocator().fBytesOnStorage,
                         cgDesc.GetPageListLocator().GetPosition<std::uint64_t>());
      RNTupleDecompressor::Unzip(zipBuffer, cgDesc.GetPageListLocator().fBytesOnStorage, cgDesc.GetPageListLength(),
                                 buffer.data());

      RNTupleSerializer::DeserializePageList(buffer.data(), cgDesc.GetPageListLength(), cgDesc.GetId(), desc);
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

   sealedPage.SetBufferSize(pageInfo.fLocator.fBytesOnStorage + pageInfo.fHasChecksum * kNBytesPageChecksum);
   sealedPage.SetNElements(pageInfo.fNElements);
   sealedPage.SetHasChecksum(pageInfo.fHasChecksum);
   if (!sealedPage.GetBuffer())
      return;
   if (pageInfo.fLocator.fType != RNTupleLocator::kTypePageZero) {
      fReader.ReadBuffer(const_cast<void *>(sealedPage.GetBuffer()), sealedPage.GetBufferSize(),
                         pageInfo.fLocator.GetPosition<std::uint64_t>());
   } else {
      assert(!pageInfo.fHasChecksum);
      memcpy(const_cast<void *>(sealedPage.GetBuffer()), RPage::GetPageZeroBuffer(), sealedPage.GetBufferSize());
   }

   sealedPage.VerifyChecksumIfEnabled().ThrowOnError();
}

ROOT::Experimental::Internal::RPage
ROOT::Experimental::Internal::RPageSourceFile::LoadPageImpl(ColumnHandle_t columnHandle,
                                                            const RClusterInfo &clusterInfo,
                                                            ClusterSize_t::ValueType idxInCluster)
{
   const auto columnId = columnHandle.fPhysicalId;
   const auto clusterId = clusterInfo.fClusterId;
   const auto pageInfo = clusterInfo.fPageInfo;

   const auto element = columnHandle.fColumn->GetElement();
   const auto elementSize = element->GetSize();

   if (pageInfo.fLocator.fType == RNTupleLocator::kTypePageZero) {
      auto pageZero = RPage::MakePageZero(columnId, elementSize);
      pageZero.GrowUnchecked(pageInfo.fNElements);
      pageZero.SetWindow(clusterInfo.fColumnOffset + pageInfo.fFirstInPage,
                         RPage::RClusterInfo(clusterId, clusterInfo.fColumnOffset));
      fPagePool->RegisterPage(pageZero, RPageDeleter([](const RPage &, void *) {}, nullptr));
      return pageZero;
   }

   RSealedPage sealedPage;
   sealedPage.SetNElements(pageInfo.fNElements);
   sealedPage.SetHasChecksum(pageInfo.fHasChecksum);
   sealedPage.SetBufferSize(pageInfo.fLocator.fBytesOnStorage + pageInfo.fHasChecksum * kNBytesPageChecksum);
   std::unique_ptr<unsigned char[]> directReadBuffer; // only used if cluster pool is turned off

   if (fOptions.GetClusterCache() == RNTupleReadOptions::EClusterCache::kOff) {
      directReadBuffer = std::unique_ptr<unsigned char[]>(new unsigned char[sealedPage.GetBufferSize()]);
      fReader.ReadBuffer(directReadBuffer.get(), sealedPage.GetBufferSize(),
                         pageInfo.fLocator.GetPosition<std::uint64_t>());
      fCounters->fNPageRead.Inc();
      fCounters->fNRead.Inc();
      fCounters->fSzReadPayload.Add(sealedPage.GetBufferSize());
      sealedPage.SetBuffer(directReadBuffer.get());
   } else {
      if (!fCurrentCluster || (fCurrentCluster->GetId() != clusterId) || !fCurrentCluster->ContainsColumn(columnId))
         fCurrentCluster = fClusterPool->GetCluster(clusterId, fActivePhysicalColumns.ToColumnSet());
      R__ASSERT(fCurrentCluster->ContainsColumn(columnId));

      auto cachedPage = fPagePool->GetPage(columnId, RClusterIndex(clusterId, idxInCluster));
      if (!cachedPage.IsNull())
         return cachedPage;

      ROnDiskPage::Key key(columnId, pageInfo.fPageNo);
      auto onDiskPage = fCurrentCluster->GetOnDiskPage(key);
      R__ASSERT(onDiskPage && (sealedPage.GetBufferSize() == onDiskPage->GetSize()));
      sealedPage.SetBuffer(onDiskPage->GetAddress());
   }

   RPage newPage;
   {
      Detail::RNTupleAtomicTimer timer(fCounters->fTimeWallUnzip, fCounters->fTimeCpuUnzip);
      newPage = UnsealPage(sealedPage, *element, columnId).Unwrap();
      fCounters->fSzUnzip.Add(elementSize * pageInfo.fNElements);
   }

   newPage.SetWindow(clusterInfo.fColumnOffset + pageInfo.fFirstInPage,
                     RPage::RClusterInfo(clusterId, clusterInfo.fColumnOffset));
   fPagePool->RegisterPage(
      newPage, RPageDeleter([](const RPage &page, void *) { RPageAllocatorHeap::DeletePage(page); }, nullptr));
   fCounters->fNPageUnsealed.Inc();
   return newPage;
}

void ROOT::Experimental::Internal::RPageSourceFile::ReleasePage(RPage &page)
{
   fPagePool->ReturnPage(page);
}

std::unique_ptr<ROOT::Experimental::Internal::RPageSource>
ROOT::Experimental::Internal::RPageSourceFile::CloneImpl() const
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
   PrepareLoadCluster(
      clusterKey, *pageZeroMap,
      [&](DescriptorId_t physicalColumnId, NTupleSize_t pageNo,
          const RClusterDescriptor::RPageRange::RPageInfo &pageInfo) {
         const auto &pageLocator = pageInfo.fLocator;
         const auto nBytes = pageLocator.fBytesOnStorage + pageInfo.fHasChecksum * kNBytesPageChecksum;
         activeSize += nBytes;
         onDiskPages.push_back({physicalColumnId, pageNo, pageLocator.GetPosition<std::uint64_t>(), nBytes, 0});
      });

   // Linearize the page requests by file offset
   std::sort(onDiskPages.begin(), onDiskPages.end(),
             [](const ROnDiskPageLocator &a, const ROnDiskPageLocator &b) { return a.fOffset < b.fOffset; });

   // In order to coalesce close-by pages, we collect the sizes of the gaps between pages on disk.  We then order
   // the gaps by size, sum them up and find a cutoff for the largest gap that we tolerate when coalescing pages.
   // The size of the cutoff is given by the fraction of extra bytes we are willing to read in order to reduce
   // the number of read requests.  We thus schedule the lowest number of requests given a tolerable fraction
   // of extra bytes.
   // TODO(jblomer): Eventually we may want to select the parameter at runtime according to link latency and speed,
   // memory consumption, device block size.
   float maxOverhead = 0.25 * float(activeSize);
   std::vector<std::size_t> gaps;
   if (onDiskPages.size())
      gaps.reserve(onDiskPages.size() - 1);
   for (unsigned i = 1; i < onDiskPages.size(); ++i) {
      std::int64_t gap =
         static_cast<int64_t>(onDiskPages[i].fOffset) - (onDiskPages[i - 1].fSize + onDiskPages[i - 1].fOffset);
      gaps.emplace_back(std::max(gap, std::int64_t(0)));
      // If the pages overlap, substract the overlapped bytes from `activeSize`
      activeSize += std::min(gap, std::int64_t(0));
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
      if (szExtra > maxOverhead)
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
   const std::uint64_t maxKeySize = fReader.GetMaxKeySize();
   for (auto &s : onDiskPages) {
      R__ASSERT(s.fSize > 0);
      const std::int64_t readUpTo = req.fOffset + req.fSize;
      // Note: byte ranges of pages may overlap
      const std::uint64_t overhead = std::max(static_cast<std::int64_t>(s.fOffset) - readUpTo, std::int64_t(0));
      const std::uint64_t extent = std::max(static_cast<std::int64_t>(s.fOffset + s.fSize) - readUpTo, std::int64_t(0));
      szPayload += extent;
      if (req.fSize + extent < maxKeySize && overhead <= gapCut) {
         szOverhead += overhead;
         s.fBufPos = reinterpret_cast<intptr_t>(req.fBuffer) + s.fOffset - req.fOffset;
         req.fSize += extent;
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
   auto pageMap = std::make_unique<ROnDiskPageMapHeap>(std::unique_ptr<unsigned char[]>(buffer));
   for (const auto &s : onDiskPages) {
      ROnDiskPage::Key key(s.fColumnId, s.fPageNo);
      pageMap->Register(key, ROnDiskPage(buffer + s.fBufPos, s.fSize));
   }
   fCounters->fNPageRead.Add(onDiskPages.size());
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

   clusters.reserve(clusterKeys.size());
   for (auto key : clusterKeys) {
      clusters.emplace_back(PrepareSingleCluster(key, readRequests));
   }

   auto nReqs = readRequests.size();
   auto readvLimits = fFile->GetReadVLimits();
   // We never want to do vectorized reads of split blobs, so we limit our single size to maxKeySize.
   readvLimits.fMaxSingleSize = std::min<size_t>(readvLimits.fMaxSingleSize, fReader.GetMaxKeySize());

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
         fReader.ReadBuffer(readRequests[iReq].fBuffer, readRequests[iReq].fSize, readRequests[iReq].fOffset);
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

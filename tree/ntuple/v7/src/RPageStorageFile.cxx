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

#include <RVersion.h>
#include <TError.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <utility>

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <queue>

#ifdef MY_CODE_RPAGE_STORAGE_FILE

#include <sys/file.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <unistd.h>
#include <linux/fs.h>
#include <fstream>

#endif


ROOT::Experimental::Detail::RPageSinkFile::RPageSinkFile(std::string_view ntupleName,
   const RNTupleWriteOptions &options)
   : RPageSink(ntupleName, options)
   , fPageAllocator(std::make_unique<RPageAllocatorHeap>())
{
   R__LOG_WARNING(NTupleLog()) << "The RNTuple file format will change. " <<
      "Do not store real data with this version of RNTuple!";
   fCompressor = std::make_unique<RNTupleCompressor>();
   EnableDefaultMetrics("RPageSinkFile");
}


ROOT::Experimental::Detail::RPageSinkFile::RPageSinkFile(std::string_view ntupleName, std::string_view path,
   const RNTupleWriteOptions &options)
   : RPageSinkFile(ntupleName, options)
{
   fWriter = std::unique_ptr<Internal::RNTupleFileWriter>(Internal::RNTupleFileWriter::Recreate(
      ntupleName, path, options.GetCompression(), options.GetContainerFormat()));
}


ROOT::Experimental::Detail::RPageSinkFile::RPageSinkFile(std::string_view ntupleName, TFile &file,
   const RNTupleWriteOptions &options)
   : RPageSinkFile(ntupleName, options)
{
   fWriter = std::unique_ptr<Internal::RNTupleFileWriter>(Internal::RNTupleFileWriter::Append(ntupleName, file));
}


ROOT::Experimental::Detail::RPageSinkFile::RPageSinkFile(std::string_view ntupleName, std::string_view path,
   const RNTupleWriteOptions &options, std::unique_ptr<TFile> &file)
   : RPageSinkFile(ntupleName, options)
{
   fWriter = std::unique_ptr<Internal::RNTupleFileWriter>(
      Internal::RNTupleFileWriter::Recreate(ntupleName, path, file));
}


ROOT::Experimental::Detail::RPageSinkFile::~RPageSinkFile()
{
}

void ROOT::Experimental::Detail::RPageSinkFile::CreateImpl(const RNTupleModel & /* model */,
                                                           unsigned char *serializedHeader, std::uint32_t length)
{
   auto zipBuffer = std::make_unique<unsigned char[]>(length);
   auto szZipHeader = fCompressor->Zip(serializedHeader, length, GetWriteOptions().GetCompression(),
                                       RNTupleCompressor::MakeMemCopyWriter(zipBuffer.get()));
   fWriter->WriteNTupleHeader(zipBuffer.get(), szZipHeader, length);
}


inline ROOT::Experimental::RNTupleLocator
ROOT::Experimental::Detail::RPageSinkFile::WriteSealedPage(
   const RPageStorage::RSealedPage &sealedPage, std::size_t bytesPacked)
{
   std::uint64_t offsetData;
   {
      RNTupleAtomicTimer timer(fCounters->fTimeWallWrite, fCounters->fTimeCpuWrite);
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
ROOT::Experimental::Detail::RPageSinkFile::CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page)
{
   auto element = columnHandle.fColumn->GetElement();
   RPageStorage::RSealedPage sealedPage;
   {
      RNTupleAtomicTimer timer(fCounters->fTimeWallZip, fCounters->fTimeCpuZip);
      sealedPage = SealPage(page, *element, GetWriteOptions().GetCompression());
   }

   fCounters->fSzZip.Add(page.GetNBytes());
   return WriteSealedPage(sealedPage, element->GetPackedSize(page.GetNElements()));
}


ROOT::Experimental::RNTupleLocator
ROOT::Experimental::Detail::RPageSinkFile::CommitSealedPageImpl(
   DescriptorId_t columnId, const RPageStorage::RSealedPage &sealedPage)
{
   const auto bitsOnStorage = RColumnElementBase::GetBitsOnStorage(
      fDescriptorBuilder.GetDescriptor().GetColumnDescriptor(columnId).GetModel().GetType());
   const auto bytesPacked = (bitsOnStorage * sealedPage.fNElements + 7) / 8;

   return WriteSealedPage(sealedPage, bytesPacked);
}


std::uint64_t
ROOT::Experimental::Detail::RPageSinkFile::CommitClusterImpl(ROOT::Experimental::NTupleSize_t /* nEntries */)
{
   auto result = fNBytesCurrentCluster;
   fNBytesCurrentCluster = 0;
   return result;
}

ROOT::Experimental::RNTupleLocator
ROOT::Experimental::Detail::RPageSinkFile::CommitClusterGroupImpl(unsigned char *serializedPageList,
                                                                  std::uint32_t length)
{
   auto bufPageListZip = std::make_unique<unsigned char[]>(length);
   auto szPageListZip = fCompressor->Zip(serializedPageList, length, GetWriteOptions().GetCompression(),
                                         RNTupleCompressor::MakeMemCopyWriter(bufPageListZip.get()));

#ifdef MY_CODE
    this->fWriter->WritePadding();
#endif
   RNTupleLocator result;
   result.fBytesOnStorage = szPageListZip;
   result.fPosition = fWriter->WriteBlob(bufPageListZip.get(), szPageListZip, length);
#ifdef MY_CODE
    this->fWriter->WritePadding();
#endif
   return result;
}

void ROOT::Experimental::Detail::RPageSinkFile::CommitDatasetImpl(unsigned char *serializedFooter, std::uint32_t length)
{
   auto bufFooterZip = std::make_unique<unsigned char[]>(length);
   auto szFooterZip = fCompressor->Zip(serializedFooter, length, GetWriteOptions().GetCompression(),
                                       RNTupleCompressor::MakeMemCopyWriter(bufFooterZip.get()));
   fWriter->WriteNTupleFooter(bufFooterZip.get(), szFooterZip, length);
   fWriter->Commit();
}

ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSinkFile::ReservePage(ColumnHandle_t columnHandle, std::size_t nElements)
{
   if (nElements == 0)
      throw RException(R__FAIL("invalid call: request empty page"));
   auto elementSize = columnHandle.fColumn->GetElement()->GetSize();
   return fPageAllocator->NewPage(columnHandle.fId, elementSize, nElements);
}

void ROOT::Experimental::Detail::RPageSinkFile::ReleasePage(RPage &page)
{
   fPageAllocator->DeletePage(page);
}


#ifdef MY_CODE_RPAGE_STORAGE_FILE
   void ROOT::Experimental::Detail::RPageSinkFile::ZeroCopy( std::string_view ntupleName, std::string_view location ) {
       auto source = RPageSourceFile(ntupleName, location, RNTupleReadOptions());
       source.Attach();
       /**
        * Compute the data part starting point
        * */
       auto descSource = source.GetSharedDescriptorGuard().GetRef().Clone();

       /**
       * Update offsets
       * */
       NTupleSize_t lastColumnOffset = lastColumnOffset = fDescriptorBuilder.GetDescriptor().GetNEntries();
       NTupleSize_t lastPageOffset = 0;

       size_t nClusterGroup = fDescriptorBuilder.GetDescriptor().GetNClusterGroups();
       lastPageOffset = (nClusterGroup == 0 ? 0 :
                         fDescriptorBuilder.GetDescriptor().GetClusterGroupDescriptor(nClusterGroup-1).GetPageListLocator().fPosition + fDescriptorBuilder.GetDescriptor().GetClusterGroupDescriptor(nClusterGroup-1).GetPageListLocator().fBytesOnStorage);

       std::uint32_t nClusters = fDescriptorBuilder.GetDescriptor().GetNClusters();
       auto descDst = fDescriptorBuilder.GetDescriptor().Clone();
       size_t CGIdx = 0;
       std::uint32_t i = 0;
       std::vector<RClusterDescriptorBuilder> clusters;
       for (const auto &CG: descSource->GetClusterGroupIterable()) {


           // TODO Check this is true
           size_t startDataPosition = descSource->GetClusterDescriptor(CG.GetClusterIds().front()).GetPageRange(
                   0).fPageInfos.front().fLocator.fPosition; // anchor.fSeekHeader + source.GetSharedDescriptorGuard()->GetOnDiskHeaderSize();
           size_t alignedStartDataPosition = (startDataPosition + 4096 - 1) / 4096 * 4096;

           std::cout << "Start cluster group" << std::endl;
           std::cout << "Start data position: " << startDataPosition << " " << alignedStartDataPosition << std::endl;

           // TODO: Change this clustergroup 0
           size_t endDataClusterGroup = descSource->GetClusterGroupDescriptor(CGIdx).GetPageListLocator().fPosition;
           std::cout << "End data position: " << endDataClusterGroup << std::endl;

           size_t length = endDataClusterGroup -
                           alignedStartDataPosition;
           size_t alignedLength = length;

           /**
            * Share the data part (zero-copy)
            * */
           this->fWriter->ShareContent(location, alignedLength, alignedStartDataPosition);

            /**
            * Update metadata
            **/
           std::cout << lastColumnOffset << std::endl;
           std::cout << lastPageOffset << std::endl;
           // Round up to the block size
           lastPageOffset = lastPageOffset == 0 ? alignedStartDataPosition : (lastPageOffset + 4096 - 1) / 4096 * 4096;
           // Being the second cluster, the page locator will have as position the header size (assumed rounded up to the
           // nearest block size). Thus, subtract the header size from this offset in order to adjust the page position with
           // respect to the new cluster (in the merged tuple) and to the position in the old cluster
           lastPageOffset -= alignedStartDataPosition;


           std::uint32_t groupId = fDescriptorBuilder.GetDescriptor().GetNClusterGroups();

            for (const auto &C: CG.GetClusterIds()) {
               auto clusterDesc = descSource->GetClusterDescriptor(C).Clone();
               clusters.emplace_back(RClusterDescriptorBuilder(nClusters + i, clusterDesc.GetFirstEntryIndex() + lastColumnOffset,
                                                               clusterDesc.GetNEntries()));
               std::uint32_t nColumns = fDescriptorBuilder.GetDescriptor().GetNColumns();

               for (std::uint32_t j = 0; j < nColumns; ++j) {
                   auto nPages = clusterDesc.GetPageRange(j).fPageInfos.size();
                   RClusterDescriptor::RPageRange pageRange;
                   pageRange.fColumnId = j;
                   for (std::uint32_t k = 0; k < nPages; ++k) {
                       std::uint32_t nElements = clusterDesc.GetNEntries();
                       RNTupleLocator locator = clusterDesc.GetPageRange(j).fPageInfos[k].fLocator;
                       locator.fPosition += lastPageOffset;
                       pageRange.fPageInfos.push_back({ClusterSize_t(nElements), locator});
                   }
                   std::uint64_t columnOffset = clusterDesc.GetColumnRange(j).fFirstElementIndex;
                   columnOffset += lastColumnOffset;
                   clusters[i].CommitColumnRange(j, columnOffset, this->GetWriteOptions().GetCompression(), pageRange);
               }
               fDescriptorBuilder.AddClusterWithDetails(clusters[i].MoveDescriptor().Unwrap());
               i++;
            }
            CommitClusterGroup();
            groupId++;
            CGIdx++;
        }
   }
#endif


////////////////////////////////////////////////////////////////////////////////


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageAllocatorFile::NewPage(
   ColumnId_t columnId, void *mem, std::size_t elementSize, std::size_t nElements)
{
   RPage newPage(columnId, mem, elementSize, nElements);
   newPage.GrowUnchecked(nElements);
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
   , fPageAllocator(std::make_unique<RPageAllocatorFile>())
   , fPagePool(std::make_shared<RPagePool>())
   , fClusterPool(std::make_unique<RClusterPool>(*this, options.GetClusterBunchSize()))
{
   fDecompressor = std::make_unique<RNTupleDecompressor>();
   EnableDefaultMetrics("RPageSourceFile");
}


ROOT::Experimental::Detail::RPageSourceFile::RPageSourceFile(std::string_view ntupleName, std::string_view path,
   const RNTupleReadOptions &options)
   : RPageSourceFile(ntupleName, options)
{
   fFile = ROOT::Internal::RRawFile::Create(path);
   R__ASSERT(fFile);
   fReader = Internal::RMiniFileReader(fFile.get());
}

void ROOT::Experimental::Detail::RPageSourceFile::InitDescriptor(const Internal::RFileNTupleAnchor &anchor)
{
   fDescriptorBuilder.SetOnDiskHeaderSize(anchor.fNBytesHeader);
   auto buffer = std::make_unique<unsigned char[]>(anchor.fLenHeader);
   auto zipBuffer = std::make_unique<unsigned char[]>(anchor.fNBytesHeader);
   fReader.ReadBuffer(zipBuffer.get(), anchor.fNBytesHeader, anchor.fSeekHeader);
   fDecompressor->Unzip(zipBuffer.get(), anchor.fNBytesHeader, anchor.fLenHeader, buffer.get());
   Internal::RNTupleSerializer::DeserializeHeaderV1(buffer.get(), anchor.fLenHeader, fDescriptorBuilder);

   fDescriptorBuilder.AddToOnDiskFooterSize(anchor.fNBytesFooter);
   buffer = std::make_unique<unsigned char[]>(anchor.fLenFooter);
   zipBuffer = std::make_unique<unsigned char[]>(anchor.fNBytesFooter);
   fReader.ReadBuffer(zipBuffer.get(), anchor.fNBytesFooter, anchor.fSeekFooter);
   fDecompressor->Unzip(zipBuffer.get(), anchor.fNBytesFooter, anchor.fLenFooter, buffer.get());
   Internal::RNTupleSerializer::DeserializeFooterV1(buffer.get(), anchor.fLenFooter, fDescriptorBuilder);
}

std::unique_ptr<ROOT::Experimental::Detail::RPageSourceFile>
ROOT::Experimental::Detail::RPageSourceFile::CreateFromAnchor(const Internal::RFileNTupleAnchor &anchor,
                                                              std::string_view path, const RNTupleReadOptions &options)
{
   auto pageSource = std::make_unique<RPageSourceFile>("", path, options);
   pageSource->InitDescriptor(anchor);
   pageSource->fNTupleName = pageSource->fDescriptorBuilder.GetDescriptor().GetName();
   return pageSource;
}

ROOT::Experimental::Detail::RPageSourceFile::~RPageSourceFile() = default;


ROOT::Experimental::RNTupleDescriptor ROOT::Experimental::Detail::RPageSourceFile::AttachImpl()
{
   // If we constructed the page source with (ntuple name, path), we need to find the anchor first.
   // Otherwise, the page source was created by OpenFromAnchor() and the header and footer are already processed.
   if (fDescriptorBuilder.GetDescriptor().GetOnDiskHeaderSize() == 0) {
      auto anchor = fReader.GetNTuple(fNTupleName).Unwrap();
      InitDescriptor(anchor);
   }

   auto ntplDesc = fDescriptorBuilder.MoveDescriptor();

   for (const auto &cgDesc : ntplDesc.GetClusterGroupIterable()) {
      auto buffer = std::make_unique<unsigned char[]>(cgDesc.GetPageListLength());
      auto zipBuffer = std::make_unique<unsigned char[]>(cgDesc.GetPageListLocator().fBytesOnStorage);
      fReader.ReadBuffer(zipBuffer.get(), cgDesc.GetPageListLocator().fBytesOnStorage,
                         cgDesc.GetPageListLocator().fPosition);
      fDecompressor->Unzip(zipBuffer.get(), cgDesc.GetPageListLocator().fBytesOnStorage, cgDesc.GetPageListLength(),
                           buffer.get());

      auto clusters = RClusterGroupDescriptorBuilder::GetClusterSummaries(ntplDesc, cgDesc.GetId());
      Internal::RNTupleSerializer::DeserializePageListV1(buffer.get(), cgDesc.GetPageListLength(), clusters);
      for (std::size_t i = 0; i < clusters.size(); ++i) {
         ntplDesc.AddClusterDetails(clusters[i].MoveDescriptor().Unwrap());
      }
   }

   return ntplDesc;
}


void ROOT::Experimental::Detail::RPageSourceFile::LoadSealedPage(
   DescriptorId_t columnId, const RClusterIndex &clusterIndex, RSealedPage &sealedPage)
{
   const auto clusterId = clusterIndex.GetClusterId();

   RClusterDescriptor::RPageRange::RPageInfo pageInfo;
   {
      auto descriptorGuard = GetSharedDescriptorGuard();
      const auto &clusterDescriptor = descriptorGuard->GetClusterDescriptor(clusterId);
      pageInfo = clusterDescriptor.GetPageRange(columnId).Find(clusterIndex.GetIndex());
   }

   const auto bytesOnStorage = pageInfo.fLocator.fBytesOnStorage;
   sealedPage.fSize = bytesOnStorage;
   sealedPage.fNElements = pageInfo.fNElements;
   if (sealedPage.fBuffer)
      fReader.ReadBuffer(const_cast<void *>(sealedPage.fBuffer), bytesOnStorage, pageInfo.fLocator.fPosition);
}

ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSourceFile::PopulatePageFromCluster(ColumnHandle_t columnHandle,
                                                                     const RClusterInfo &clusterInfo,
                                                                     ClusterSize_t::ValueType idxInCluster)
{
   const auto columnId = columnHandle.fId;
   const auto clusterId = clusterInfo.fClusterId;
   const auto pageInfo = clusterInfo.fPageInfo;

   const auto element = columnHandle.fColumn->GetElement();
   const auto elementSize = element->GetSize();
   const auto bytesOnStorage = pageInfo.fLocator.fBytesOnStorage;

   const void *sealedPageBuffer = nullptr; // points either to directReadBuffer or to a read-only page in the cluster
   std::unique_ptr<unsigned char []> directReadBuffer; // only used if cluster pool is turned off

   if (fOptions.GetClusterCache() == RNTupleReadOptions::EClusterCache::kOff) {
      directReadBuffer = std::make_unique<unsigned char[]>(bytesOnStorage);
      fReader.ReadBuffer(directReadBuffer.get(), bytesOnStorage, pageInfo.fLocator.fPosition);
      fCounters->fNPageLoaded.Inc();
      fCounters->fNRead.Inc();
      fCounters->fSzReadPayload.Add(bytesOnStorage);
      sealedPageBuffer = directReadBuffer.get();
   } else {
      if (!fCurrentCluster || (fCurrentCluster->GetId() != clusterId) || !fCurrentCluster->ContainsColumn(columnId))
         fCurrentCluster = fClusterPool->GetCluster(clusterId, fActiveColumns);
      R__ASSERT(fCurrentCluster->ContainsColumn(columnId));

      auto cachedPage = fPagePool->GetPage(columnId, RClusterIndex(clusterId, idxInCluster));
      if (!cachedPage.IsNull())
         return cachedPage;

      ROnDiskPage::Key key(columnId, pageInfo.fPageNo);
      auto onDiskPage = fCurrentCluster->GetOnDiskPage(key);
      R__ASSERT(onDiskPage && (bytesOnStorage == onDiskPage->GetSize()));
      sealedPageBuffer = onDiskPage->GetAddress();
   }

   std::unique_ptr<unsigned char []> pageBuffer;
   {
      RNTupleAtomicTimer timer(fCounters->fTimeWallUnzip, fCounters->fTimeCpuUnzip);
      pageBuffer = UnsealPage({sealedPageBuffer, bytesOnStorage, pageInfo.fNElements}, *element);
      fCounters->fSzUnzip.Add(elementSize * pageInfo.fNElements);
   }

   auto newPage = fPageAllocator->NewPage(columnId, pageBuffer.release(), elementSize, pageInfo.fNElements);
   newPage.SetWindow(clusterInfo.fColumnOffset + pageInfo.fFirstInPage,
                     RPage::RClusterInfo(clusterId, clusterInfo.fColumnOffset));
   fPagePool->RegisterPage(newPage,
      RPageDeleter([](const RPage &page, void * /*userData*/)
      {
         RPageAllocatorFile::DeletePage(page);
      }, nullptr));
   fCounters->fNPagePopulated.Inc();
   return newPage;
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceFile::PopulatePage(
   ColumnHandle_t columnHandle, NTupleSize_t globalIndex)
{
   const auto columnId = columnHandle.fId;
   auto cachedPage = fPagePool->GetPage(columnId, globalIndex);
   if (!cachedPage.IsNull())
      return cachedPage;

   std::uint64_t idxInCluster;
   RClusterInfo clusterInfo;
   {
      auto descriptorGuard = GetSharedDescriptorGuard();
      clusterInfo.fClusterId = descriptorGuard->FindClusterId(columnId, globalIndex);
      R__ASSERT(clusterInfo.fClusterId != kInvalidDescriptorId);

      const auto &clusterDescriptor = descriptorGuard->GetClusterDescriptor(clusterInfo.fClusterId);
      clusterInfo.fColumnOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex;
      R__ASSERT(clusterInfo.fColumnOffset <= globalIndex);
      idxInCluster = globalIndex - clusterInfo.fColumnOffset;
      clusterInfo.fPageInfo = clusterDescriptor.GetPageRange(columnId).Find(idxInCluster);
   }

   return PopulatePageFromCluster(columnHandle, clusterInfo, idxInCluster);
}


ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSourceFile::PopulatePage(
   ColumnHandle_t columnHandle, const RClusterIndex &clusterIndex)
{
   const auto clusterId = clusterIndex.GetClusterId();
   const auto idxInCluster = clusterIndex.GetIndex();
   const auto columnId = columnHandle.fId;
   auto cachedPage = fPagePool->GetPage(columnId, clusterIndex);
   if (!cachedPage.IsNull())
      return cachedPage;

   R__ASSERT(clusterId != kInvalidDescriptorId);
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

std::unique_ptr<ROOT::Experimental::Detail::RCluster>
ROOT::Experimental::Detail::RPageSourceFile::PrepareSingleCluster(
   const RCluster::RKey &clusterKey,
   std::vector<ROOT::Internal::RRawFile::RIOVec> &readRequests)
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
   {
      auto descriptorGuard = GetSharedDescriptorGuard();
      const auto &clusterDesc = descriptorGuard->GetClusterDescriptor(clusterKey.fClusterId);

      // Collect the page necessary page meta-data and sum up the total size of the compressed and packed pages
      for (auto columnId : clusterKey.fColumnSet) {
         const auto &pageRange = clusterDesc.GetPageRange(columnId);
         NTupleSize_t pageNo = 0;
         for (const auto &pageInfo : pageRange.fPageInfos) {
            const auto &pageLocator = pageInfo.fLocator;
            activeSize += pageLocator.fBytesOnStorage;
            onDiskPages.push_back(
               {columnId, pageNo, std::uint64_t(pageLocator.fPosition), pageLocator.fBytesOnStorage, 0});
            ++pageNo;
         }
      }
   }

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
   for (auto colId : clusterKey.fColumnSet)
      cluster->SetColumnAvailable(colId);
   return cluster;
}

std::vector<std::unique_ptr<ROOT::Experimental::Detail::RCluster>>
ROOT::Experimental::Detail::RPageSourceFile::LoadClusters(std::span<RCluster::RKey> clusterKeys)
{
   fCounters->fNClusterLoaded.Add(clusterKeys.size());

   std::vector<std::unique_ptr<ROOT::Experimental::Detail::RCluster>> clusters;
   std::vector<ROOT::Internal::RRawFile::RIOVec> readRequests;

   for (auto key: clusterKeys) {
      clusters.emplace_back(PrepareSingleCluster(key, readRequests));
   }

   auto nReqs = readRequests.size();
   {
      RNTupleAtomicTimer timer(fCounters->fTimeWallRead, fCounters->fTimeCpuRead);
      fFile->ReadV(&readRequests[0], nReqs);
   }
   fCounters->fNReadV.Inc();
   fCounters->fNRead.Add(nReqs);

   return clusters;
}


void ROOT::Experimental::Detail::RPageSourceFile::UnzipClusterImpl(RCluster *cluster)
{
   RNTupleAtomicTimer timer(fCounters->fTimeWallUnzip, fCounters->fTimeCpuUnzip);
   fTaskScheduler->Reset();

   const auto clusterId = cluster->GetId();
   auto descriptorGuard = GetSharedDescriptorGuard();
   const auto &clusterDescriptor = descriptorGuard->GetClusterDescriptor(clusterId);

   std::vector<std::unique_ptr<RColumnElementBase>> allElements;

   const auto &columnsInCluster = cluster->GetAvailColumns();
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

         auto taskFunc =
            [this, columnId, clusterId, firstInPage, onDiskPage,
             element = allElements.back().get(),
             nElements = pi.fNElements,
             indexOffset = clusterDescriptor.GetColumnRange(columnId).fFirstElementIndex
            ] () {
               auto pageBuffer = UnsealPage({onDiskPage->GetAddress(), onDiskPage->GetSize(), nElements}, *element);
               fCounters->fSzUnzip.Add(element->GetSize() * nElements);

               auto newPage = fPageAllocator->NewPage(columnId, pageBuffer.release(), element->GetSize(), nElements);
               newPage.SetWindow(indexOffset + firstInPage, RPage::RClusterInfo(clusterId, indexOffset));
               fPagePool->PreloadPage(newPage,
                  RPageDeleter([](const RPage &page, void * /*userData*/)
                  {
                     RPageAllocatorFile::DeletePage(page);
                  }, nullptr));
            };

         fTaskScheduler->AddTask(taskFunc);

         firstInPage += pi.fNElements;
         pageNo++;
      } // for all pages in column
   } // for all columns in cluster

   fCounters->fNPagePopulated.Add(cluster->GetNOnDiskPages());

   fTaskScheduler->Wait();
}

/// \file RPageStorage.cxx
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

#include <ROOT/RPageStorage.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/RColumn.hxx>
#include <ROOT/RField.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RNTupleMetrics.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RPagePool.hxx>
#include <ROOT/RPageSinkBuf.hxx>
#include <ROOT/RPageStorageFile.hxx>
#include <ROOT/RStringView.hxx>
#ifdef R__ENABLE_DAOS
# include <ROOT/RPageStorageDaos.hxx>
#endif

#include <Compression.h>
#include <TError.h>

#include <utility>


ROOT::Experimental::Detail::RPageStorage::RPageStorage(std::string_view name) : fNTupleName(name)
{
}

ROOT::Experimental::Detail::RPageStorage::~RPageStorage() {}

//------------------------------------------------------------------------------

void ROOT::Experimental::Detail::RPageSource::RActivePhysicalColumns::Insert(DescriptorId_t physicalColumnID)
{
   for (unsigned i = 0; i < fIDs.size(); ++i) {
      if (fIDs[i] == physicalColumnID) {
         fRefCounters[i]++;
         return;
      }
   }
   fIDs.emplace_back(physicalColumnID);
   fRefCounters.emplace_back(1);
}

void ROOT::Experimental::Detail::RPageSource::RActivePhysicalColumns::Erase(DescriptorId_t physicalColumnID)
{
   for (unsigned i = 0; i < fIDs.size(); ++i) {
      if (fIDs[i] == physicalColumnID) {
         if (--fRefCounters[i] == 0) {
            fIDs.erase(fIDs.begin() + i);
            fRefCounters.erase(fRefCounters.begin() + i);
         }
         return;
      }
   }
}

ROOT::Experimental::Detail::RCluster::ColumnSet_t
ROOT::Experimental::Detail::RPageSource::RActivePhysicalColumns::ToColumnSet() const
{
   RCluster::ColumnSet_t result;
   for (const auto &id : fIDs)
      result.insert(id);
   return result;
}

ROOT::Experimental::Detail::RPageSource::RPageSource(std::string_view name, const RNTupleReadOptions &options)
   : RPageStorage(name), fMetrics(""), fOptions(options)
{
}

ROOT::Experimental::Detail::RPageSource::~RPageSource()
{
}

std::unique_ptr<ROOT::Experimental::Detail::RPageSource> ROOT::Experimental::Detail::RPageSource::Create(
   std::string_view ntupleName, std::string_view location, const RNTupleReadOptions &options)
{
   if (ntupleName.empty()) {
      throw RException(R__FAIL("empty RNTuple name"));
   }
   if (location.empty()) {
      throw RException(R__FAIL("empty storage location"));
   }
   if (location.find("daos://") == 0)
#ifdef R__ENABLE_DAOS
      return std::make_unique<RPageSourceDaos>(ntupleName, location, options);
#else
      throw RException(R__FAIL("This RNTuple build does not support DAOS."));
#endif

   return std::make_unique<RPageSourceFile>(ntupleName, location, options);
}

ROOT::Experimental::Detail::RPageStorage::ColumnHandle_t
ROOT::Experimental::Detail::RPageSource::AddColumn(DescriptorId_t fieldId, const RColumn &column)
{
   R__ASSERT(fieldId != kInvalidDescriptorId);
   auto physicalId = GetSharedDescriptorGuard()->FindPhysicalColumnId(fieldId, column.GetIndex());
   R__ASSERT(physicalId != kInvalidDescriptorId);
   fActivePhysicalColumns.Insert(physicalId);
   return ColumnHandle_t{physicalId, &column};
}

void ROOT::Experimental::Detail::RPageSource::DropColumn(ColumnHandle_t columnHandle)
{
   fActivePhysicalColumns.Erase(columnHandle.fPhysicalId);
}

ROOT::Experimental::NTupleSize_t ROOT::Experimental::Detail::RPageSource::GetNEntries()
{
   return GetSharedDescriptorGuard()->GetNEntries();
}

ROOT::Experimental::NTupleSize_t ROOT::Experimental::Detail::RPageSource::GetNElements(ColumnHandle_t columnHandle)
{
   return GetSharedDescriptorGuard()->GetNElements(columnHandle.fPhysicalId);
}

ROOT::Experimental::ColumnId_t ROOT::Experimental::Detail::RPageSource::GetColumnId(ColumnHandle_t columnHandle)
{
   // TODO(jblomer) distinguish trees
   return columnHandle.fPhysicalId;
}

void ROOT::Experimental::Detail::RPageSource::UnzipCluster(RCluster *cluster)
{
   if (fTaskScheduler)
      UnzipClusterImpl(cluster);
}

ROOT::Experimental::Detail::RPage ROOT::Experimental::Detail::RPageSource::UnsealPage(const RSealedPage &sealedPage,
                                                                                      const RColumnElementBase &element,
                                                                                      DescriptorId_t physicalColumnId)
{
   // Unsealing a page zero is a no-op.  `RPageRange::ExtendToFitColumnRange()` guarantees that the page zero buffer is
   // large enough to hold `sealedPage.fNElements`
   if (sealedPage.fBuffer == RPage::GetPageZeroBuffer()) {
      auto page = RPage::MakePageZero(physicalColumnId, element.GetSize());
      page.GrowUnchecked(sealedPage.fNElements);
      return page;
   }

   const auto bytesPacked = element.GetPackedSize(sealedPage.fNElements);
   using Allocator_t = RPageAllocatorHeap;
   auto page = Allocator_t::NewPage(physicalColumnId, element.GetSize(), sealedPage.fNElements);
   if (sealedPage.fSize != bytesPacked) {
      fDecompressor->Unzip(sealedPage.fBuffer, sealedPage.fSize, bytesPacked, page.GetBuffer());
   } else {
      // We cannot simply map the sealed page as we don't know its life time. Specialized page sources
      // may decide to implement to not use UnsealPage but to custom mapping / decompression code.
      // Note that usually pages are compressed.
      memcpy(page.GetBuffer(), sealedPage.fBuffer, bytesPacked);
   }

   if (!element.IsMappable()) {
      auto tmp = Allocator_t::NewPage(physicalColumnId, element.GetSize(), sealedPage.fNElements);
      element.Unpack(tmp.GetBuffer(), page.GetBuffer(), sealedPage.fNElements);
      Allocator_t::DeletePage(page);
      page = tmp;
   }

   page.GrowUnchecked(sealedPage.fNElements);
   return page;
}

void ROOT::Experimental::Detail::RPageSource::PrepareLoadCluster(
   const RCluster::RKey &clusterKey, ROnDiskPageMap &pageZeroMap,
   std::function<void(DescriptorId_t, NTupleSize_t, RClusterDescriptor::RPageRange::RPageInfo)> perPageFunc)
{
   auto descriptorGuard = GetSharedDescriptorGuard();
   const auto &clusterDesc = descriptorGuard->GetClusterDescriptor(clusterKey.fClusterId);

   for (auto physicalColumnId : clusterKey.fPhysicalColumnSet) {
      const auto &pageRange = clusterDesc.GetPageRange(physicalColumnId);
      NTupleSize_t pageNo = 0;
      for (const auto &pageInfo : pageRange.fPageInfos) {
         if (pageInfo.fLocator.fType == RNTupleLocator::kTypePageZero) {
            pageZeroMap.Register(
               ROnDiskPage::Key{physicalColumnId, pageNo},
               ROnDiskPage(const_cast<void *>(RPage::GetPageZeroBuffer()), pageInfo.fLocator.fBytesOnStorage));
         } else {
            perPageFunc(physicalColumnId, pageNo, pageInfo);
         }
         ++pageNo;
      }
   }
}

void ROOT::Experimental::Detail::RPageSource::EnableDefaultMetrics(const std::string &prefix)
{
   fMetrics = RNTupleMetrics(prefix);
   fCounters = std::unique_ptr<RCounters>(new RCounters{
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("nReadV", "", "number of vector read requests"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("nRead", "", "number of byte ranges read"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("szReadPayload", "B", "volume read from storage (required)"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("szReadOverhead", "B", "volume read from storage (overhead)"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("szUnzip", "B", "volume after unzipping"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("nClusterLoaded", "",
                                                   "number of partial clusters preloaded from storage"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("nPageLoaded", "", "number of pages loaded from storage"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("nPagePopulated", "", "number of populated pages"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("timeWallRead", "ns", "wall clock time spent reading"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("timeWallUnzip", "ns", "wall clock time spent decompressing"),
      *fMetrics.MakeCounter<RNTupleTickCounter<RNTupleAtomicCounter>*>("timeCpuRead", "ns", "CPU time spent reading"),
      *fMetrics.MakeCounter<RNTupleTickCounter<RNTupleAtomicCounter>*> ("timeCpuUnzip", "ns",
                                                                        "CPU time spent decompressing"),
      *fMetrics.MakeCounter<RNTupleCalcPerf*> ("bwRead", "MB/s", "bandwidth compressed bytes read per second",
         fMetrics, [](const RNTupleMetrics &metrics) -> std::pair<bool, double> {
            if (const auto szReadPayload = metrics.GetLocalCounter("szReadPayload")) {
               if (const auto szReadOverhead = metrics.GetLocalCounter("szReadOverhead")) {
                  if (const auto timeWallRead = metrics.GetLocalCounter("timeWallRead")) {
                     if (auto walltime = timeWallRead->GetValueAsInt()) {
                        double payload = szReadPayload->GetValueAsInt();
                        double overhead = szReadOverhead->GetValueAsInt();
                        // unit: bytes / nanosecond = GB/s
                        return {true, (1000. * (payload + overhead) / walltime)};
                     }
                  }
               }
            }
            return {false, -1.};
         }
      ),
      *fMetrics.MakeCounter<RNTupleCalcPerf*> ("bwReadUnzip", "MB/s", "bandwidth uncompressed bytes read per second",
         fMetrics, [](const RNTupleMetrics &metrics) -> std::pair<bool, double> {
            if (const auto szUnzip = metrics.GetLocalCounter("szUnzip")) {
               if (const auto timeWallRead = metrics.GetLocalCounter("timeWallRead")) {
                  if (auto walltime = timeWallRead->GetValueAsInt()) {
                     double unzip = szUnzip->GetValueAsInt();
                     // unit: bytes / nanosecond = GB/s
                     return {true, 1000. * unzip / walltime};
                  }
               }
            }
            return {false, -1.};
         }
      ),
      *fMetrics.MakeCounter<RNTupleCalcPerf*> ("bwUnzip", "MB/s", "decompression bandwidth of uncompressed bytes per second",
         fMetrics, [](const RNTupleMetrics &metrics) -> std::pair<bool, double> {
            if (const auto szUnzip = metrics.GetLocalCounter("szUnzip")) {
               if (const auto timeWallUnzip = metrics.GetLocalCounter("timeWallUnzip")) {
                  if (auto walltime = timeWallUnzip->GetValueAsInt()) {
                     double unzip = szUnzip->GetValueAsInt();
                     // unit: bytes / nanosecond = GB/s
                     return {true, 1000. * unzip / walltime};
                  }
               }
            }
            return {false, -1.};
         }
      ),
      *fMetrics.MakeCounter<RNTupleCalcPerf*> ("rtReadEfficiency", "", "ratio of payload over all bytes read",
         fMetrics, [](const RNTupleMetrics &metrics) -> std::pair<bool, double> {
            if (const auto szReadPayload = metrics.GetLocalCounter("szReadPayload")) {
               if (const auto szReadOverhead = metrics.GetLocalCounter("szReadOverhead")) {
                  if (auto payload = szReadPayload->GetValueAsInt()) {
                     // r/(r+o) = 1/((r+o)/r) = 1/(1 + o/r)
                     return {true, 1./(1. + (1. * szReadOverhead->GetValueAsInt()) / payload)};
                  }
               }
            }
            return {false, -1.};
         }
      ),
      *fMetrics.MakeCounter<RNTupleCalcPerf*> ("rtCompression", "", "ratio of compressed bytes / uncompressed bytes",
         fMetrics, [](const RNTupleMetrics &metrics) -> std::pair<bool, double> {
            if (const auto szReadPayload = metrics.GetLocalCounter("szReadPayload")) {
               if (const auto szUnzip = metrics.GetLocalCounter("szUnzip")) {
                  if (auto unzip = szUnzip->GetValueAsInt()) {
                     return {true, (1. * szReadPayload->GetValueAsInt()) / unzip};
                  }
               }
            }
            return {false, -1.};
         }
      )
   });
}


//------------------------------------------------------------------------------


ROOT::Experimental::Detail::RPageSink::RPageSink(std::string_view name, const RNTupleWriteOptions &options)
   : RPageStorage(name), fMetrics(""), fOptions(options.Clone())
{
}

ROOT::Experimental::Detail::RPageSink::~RPageSink()
{
}

std::unique_ptr<ROOT::Experimental::Detail::RPageSink> ROOT::Experimental::Detail::RPageSink::Create(
   std::string_view ntupleName, std::string_view location, const RNTupleWriteOptions &options)
{
   if (ntupleName.empty()) {
      throw RException(R__FAIL("empty RNTuple name"));
   }
   if (location.empty()) {
      throw RException(R__FAIL("empty storage location"));
   }
   std::unique_ptr<ROOT::Experimental::Detail::RPageSink> realSink;
   if (location.find("daos://") == 0) {
#ifdef R__ENABLE_DAOS
      realSink = std::make_unique<RPageSinkDaos>(ntupleName, location, options);
#else
      throw RException(R__FAIL("This RNTuple build does not support DAOS."));
#endif
   } else {
      realSink = std::make_unique<RPageSinkFile>(ntupleName, location, options);
   }

   if (options.GetUseBufferedWrite())
      return std::make_unique<RPageSinkBuf>(std::move(realSink));
   return realSink;
}

ROOT::Experimental::Detail::RPageStorage::ColumnHandle_t
ROOT::Experimental::Detail::RPageSink::AddColumn(DescriptorId_t fieldId, const RColumn &column)
{
   auto columnId = fDescriptorBuilder.GetDescriptor().GetNPhysicalColumns();
   fDescriptorBuilder.AddColumn(columnId, columnId, fieldId, column.GetModel(), column.GetIndex(),
                                column.GetFirstElementIndex());
   return ColumnHandle_t{columnId, &column};
}

void ROOT::Experimental::Detail::RPageSink::UpdateSchema(const RNTupleModelChangeset &changeset,
                                                         NTupleSize_t firstEntry)
{
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();
   auto addField = [&](RFieldBase &f) {
      auto fieldId = descriptor.GetNFields();
      fDescriptorBuilder.AddField(RFieldDescriptorBuilder::FromField(f).FieldId(fieldId).MakeDescriptor().Unwrap());
      fDescriptorBuilder.AddFieldLink(f.GetParent()->GetOnDiskId(), fieldId);
      f.SetOnDiskId(fieldId);
      f.ConnectPageSink(*this, firstEntry); // issues in turn one or several calls to `AddColumn()`
   };
   auto addProjectedField = [&](RFieldBase &f) {
      auto fieldId = descriptor.GetNFields();
      fDescriptorBuilder.AddField(RFieldDescriptorBuilder::FromField(f).FieldId(fieldId).MakeDescriptor().Unwrap());
      fDescriptorBuilder.AddFieldLink(f.GetParent()->GetOnDiskId(), fieldId);
      f.SetOnDiskId(fieldId);
      auto sourceFieldId = changeset.fModel.GetProjectedFields().GetSourceField(&f)->GetOnDiskId();
      for (const auto &source : descriptor.GetColumnIterable(sourceFieldId)) {
         auto targetId = descriptor.GetNLogicalColumns();
         fDescriptorBuilder.AddColumn(targetId, source.GetLogicalId(), fieldId, source.GetModel(), source.GetIndex());
      }
   };

   R__ASSERT(firstEntry >= fPrevClusterNEntries);
   const auto nColumnsBeforeUpdate = descriptor.GetNPhysicalColumns();
   for (auto f : changeset.fAddedFields) {
      addField(*f);
      for (auto &descendant : *f)
         addField(descendant);
   }
   for (auto f : changeset.fAddedProjectedFields) {
      addProjectedField(*f);
      for (auto &descendant : *f)
         addProjectedField(descendant);
   }

   const auto nColumns = descriptor.GetNPhysicalColumns();
   for (DescriptorId_t i = nColumnsBeforeUpdate; i < nColumns; ++i) {
      RClusterDescriptor::RColumnRange columnRange;
      columnRange.fPhysicalColumnId = i;
      // We set the first element index in the current cluster to the first element that is part of a materialized page
      // (i.e., that is part of a page list). For deferred columns, however, the column range is fixed up as needed by
      // `RClusterDescriptorBuilder::AddDeferredColumnRanges()` on read back.
      columnRange.fFirstElementIndex = descriptor.GetColumnDescriptor(i).GetFirstElementIndex();
      columnRange.fNElements = 0;
      columnRange.fCompressionSettings = GetWriteOptions().GetCompression();
      fOpenColumnRanges.emplace_back(columnRange);
      RClusterDescriptor::RPageRange pageRange;
      pageRange.fPhysicalColumnId = i;
      fOpenPageRanges.emplace_back(std::move(pageRange));
   }

   // Mapping of memory to on-disk column IDs usually happens during serialization of the ntuple header. If the
   // header was already serialized, this has to be done manually as it is required for page list serialization.
   if (fSerializationContext.GetHeaderSize() > 0)
      fSerializationContext.MapSchema(descriptor, /*forHeaderExtension=*/true);
}

void ROOT::Experimental::Detail::RPageSink::Create(RNTupleModel &model)
{
   fDescriptorBuilder.SetNTuple(fNTupleName, model.GetDescription());
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();

   auto &fieldZero = *model.GetFieldZero();
   fDescriptorBuilder.AddField(RFieldDescriptorBuilder::FromField(fieldZero).FieldId(0).MakeDescriptor().Unwrap());
   fieldZero.SetOnDiskId(0);
   model.GetProjectedFields().GetFieldZero()->SetOnDiskId(0);

   RNTupleModelChangeset initialChangeset{model};
   for (auto f : fieldZero.GetSubFields())
      initialChangeset.fAddedFields.emplace_back(f);
   for (auto f : model.GetProjectedFields().GetFieldZero()->GetSubFields())
      initialChangeset.fAddedProjectedFields.emplace_back(f);
   UpdateSchema(initialChangeset, 0U);

   fSerializationContext = Internal::RNTupleSerializer::SerializeHeaderV1(nullptr, descriptor);
   auto buffer = std::make_unique<unsigned char[]>(fSerializationContext.GetHeaderSize());
   fSerializationContext = Internal::RNTupleSerializer::SerializeHeaderV1(buffer.get(), descriptor);
   CreateImpl(model, buffer.get(), fSerializationContext.GetHeaderSize());

   fDescriptorBuilder.BeginHeaderExtension();
}

void ROOT::Experimental::Detail::RPageSink::CommitPage(ColumnHandle_t columnHandle, const RPage &page)
{
   fOpenColumnRanges.at(columnHandle.fPhysicalId).fNElements += page.GetNElements();

   RClusterDescriptor::RPageRange::RPageInfo pageInfo;
   pageInfo.fNElements = page.GetNElements();
   pageInfo.fLocator = CommitPageImpl(columnHandle, page);
   fOpenPageRanges.at(columnHandle.fPhysicalId).fPageInfos.emplace_back(pageInfo);
}

void ROOT::Experimental::Detail::RPageSink::CommitSealedPage(
   ROOT::Experimental::DescriptorId_t physicalColumnId,
   const ROOT::Experimental::Detail::RPageStorage::RSealedPage &sealedPage)
{
   fOpenColumnRanges.at(physicalColumnId).fNElements += sealedPage.fNElements;

   RClusterDescriptor::RPageRange::RPageInfo pageInfo;
   pageInfo.fNElements = sealedPage.fNElements;
   pageInfo.fLocator = CommitSealedPageImpl(physicalColumnId, sealedPage);
   fOpenPageRanges.at(physicalColumnId).fPageInfos.emplace_back(pageInfo);
}

std::vector<ROOT::Experimental::RNTupleLocator>
ROOT::Experimental::Detail::RPageSink::CommitSealedPageVImpl(std::span<RPageStorage::RSealedPageGroup> ranges)
{
   std::vector<ROOT::Experimental::RNTupleLocator> locators;
   for (auto &range : ranges) {
      for (auto sealedPageIt = range.fFirst; sealedPageIt != range.fLast; ++sealedPageIt)
         locators.push_back(CommitSealedPageImpl(range.fPhysicalColumnId, *sealedPageIt));
   }
   return locators;
}

void ROOT::Experimental::Detail::RPageSink::CommitSealedPageV(std::span<RPageStorage::RSealedPageGroup> ranges)
{
   auto locators = CommitSealedPageVImpl(ranges);
   unsigned i = 0;

   for (auto &range : ranges) {
      for (auto sealedPageIt = range.fFirst; sealedPageIt != range.fLast; ++sealedPageIt) {
         fOpenColumnRanges.at(range.fPhysicalColumnId).fNElements += sealedPageIt->fNElements;

         RClusterDescriptor::RPageRange::RPageInfo pageInfo;
         pageInfo.fNElements = sealedPageIt->fNElements;
         pageInfo.fLocator = locators[i++];
         fOpenPageRanges.at(range.fPhysicalColumnId).fPageInfos.emplace_back(pageInfo);
      }
   }
}

std::uint64_t ROOT::Experimental::Detail::RPageSink::CommitCluster(ROOT::Experimental::NTupleSize_t nEntries)
{
   auto nbytes = CommitClusterImpl(nEntries);

   R__ASSERT((nEntries - fPrevClusterNEntries) < ClusterSize_t(-1));
   auto nEntriesInCluster = ClusterSize_t(nEntries - fPrevClusterNEntries);
   RClusterDescriptorBuilder clusterBuilder(fDescriptorBuilder.GetDescriptor().GetNClusters(), fPrevClusterNEntries,
                                            nEntriesInCluster);
   for (unsigned int i = 0; i < fOpenColumnRanges.size(); ++i) {
      RClusterDescriptor::RPageRange fullRange;
      fullRange.fPhysicalColumnId = i;
      std::swap(fullRange, fOpenPageRanges[i]);
      clusterBuilder.CommitColumnRange(i, fOpenColumnRanges[i].fFirstElementIndex,
                                       fOpenColumnRanges[i].fCompressionSettings, fullRange);
      fOpenColumnRanges[i].fFirstElementIndex += fOpenColumnRanges[i].fNElements;
      fOpenColumnRanges[i].fNElements = 0;
   }
   fDescriptorBuilder.AddClusterWithDetails(clusterBuilder.MoveDescriptor().Unwrap());
   fPrevClusterNEntries = nEntries;
   return nbytes;
}

void ROOT::Experimental::Detail::RPageSink::CommitClusterGroup()
{
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();

   const auto nClusters = descriptor.GetNClusters();
   std::vector<DescriptorId_t> physClusterIDs;
   for (auto i = fNextClusterInGroup; i < nClusters; ++i) {
      physClusterIDs.emplace_back(fSerializationContext.MapClusterId(i));
   }

   auto szPageList =
      Internal::RNTupleSerializer::SerializePageListV1(nullptr, descriptor, physClusterIDs, fSerializationContext);
   auto bufPageList = std::make_unique<unsigned char[]>(szPageList);
   Internal::RNTupleSerializer::SerializePageListV1(bufPageList.get(), descriptor, physClusterIDs,
                                                    fSerializationContext);

   const auto clusterGroupId = descriptor.GetNClusterGroups();
   const auto locator = CommitClusterGroupImpl(bufPageList.get(), szPageList);
   RClusterGroupDescriptorBuilder cgBuilder;
   cgBuilder.ClusterGroupId(clusterGroupId).PageListLocator(locator).PageListLength(szPageList);
   for (auto i = fNextClusterInGroup; i < nClusters; ++i) {
      cgBuilder.AddCluster(i);
   }
   fDescriptorBuilder.AddClusterGroup(std::move(cgBuilder));
   fSerializationContext.MapClusterGroupId(clusterGroupId);

   fNextClusterInGroup = nClusters;
}

void ROOT::Experimental::Detail::RPageSink::CommitDataset()
{
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();

   auto szFooter = Internal::RNTupleSerializer::SerializeFooterV1(nullptr, descriptor, fSerializationContext);
   auto bufFooter = std::make_unique<unsigned char[]>(szFooter);
   Internal::RNTupleSerializer::SerializeFooterV1(bufFooter.get(), descriptor, fSerializationContext);

   CommitDatasetImpl(bufFooter.get(), szFooter);
}

ROOT::Experimental::Detail::RPageStorage::RSealedPage
ROOT::Experimental::Detail::RPageSink::SealPage(const RPage &page,
   const RColumnElementBase &element, int compressionSetting, void *buf)
{
   unsigned char *pageBuf = reinterpret_cast<unsigned char *>(page.GetBuffer());
   bool isAdoptedBuffer = true;
   auto packedBytes = page.GetNBytes();

   if (!element.IsMappable()) {
      packedBytes = element.GetPackedSize(page.GetNElements());
      pageBuf = new unsigned char[packedBytes];
      isAdoptedBuffer = false;
      element.Pack(pageBuf, page.GetBuffer(), page.GetNElements());
   }
   auto zippedBytes = packedBytes;

   if ((compressionSetting != 0) || !element.IsMappable()) {
      zippedBytes = RNTupleCompressor::Zip(pageBuf, packedBytes, compressionSetting, buf);
      if (!isAdoptedBuffer)
         delete[] pageBuf;
      pageBuf = reinterpret_cast<unsigned char *>(buf);
      isAdoptedBuffer = true;
   }

   R__ASSERT(isAdoptedBuffer);

   return RSealedPage{pageBuf, static_cast<std::uint32_t>(zippedBytes), page.GetNElements()};
}

ROOT::Experimental::Detail::RPageStorage::RSealedPage
ROOT::Experimental::Detail::RPageSink::SealPage(
   const RPage &page, const RColumnElementBase &element, int compressionSetting)
{
   R__ASSERT(fCompressor);
   return SealPage(page, element, compressionSetting, fCompressor->GetZipBuffer());
}

void ROOT::Experimental::Detail::RPageSink::EnableDefaultMetrics(const std::string &prefix)
{
   fMetrics = RNTupleMetrics(prefix);
   fCounters = std::unique_ptr<RCounters>(new RCounters{
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("nPageCommitted", "", "number of pages committed to storage"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("szWritePayload", "B", "volume written for committed pages"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("szZip", "B", "volume before zipping"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("timeWallWrite", "ns", "wall clock time spent writing"),
      *fMetrics.MakeCounter<RNTupleAtomicCounter*>("timeWallZip", "ns", "wall clock time spent compressing"),
      *fMetrics.MakeCounter<RNTupleTickCounter<RNTupleAtomicCounter>*>("timeCpuWrite", "ns", "CPU time spent writing"),
      *fMetrics.MakeCounter<RNTupleTickCounter<RNTupleAtomicCounter>*> ("timeCpuZip", "ns",
                                                                        "CPU time spent compressing")
   });
}

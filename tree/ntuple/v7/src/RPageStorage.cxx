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
#include <memory>
#include <string_view>
#ifdef R__ENABLE_DAOS
# include <ROOT/RPageStorageDaos.hxx>
#endif

#include <Compression.h>
#include <TError.h>

#include <utility>

ROOT::Experimental::Internal::RPageStorage::RPageStorage(std::string_view name) : fMetrics(""), fNTupleName(name) {}

ROOT::Experimental::Internal::RPageStorage::~RPageStorage() {}

//------------------------------------------------------------------------------

void ROOT::Experimental::Internal::RPageSource::RActivePhysicalColumns::Insert(DescriptorId_t physicalColumnID)
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

void ROOT::Experimental::Internal::RPageSource::RActivePhysicalColumns::Erase(DescriptorId_t physicalColumnID)
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

ROOT::Experimental::Internal::RCluster::ColumnSet_t
ROOT::Experimental::Internal::RPageSource::RActivePhysicalColumns::ToColumnSet() const
{
   RCluster::ColumnSet_t result;
   for (const auto &id : fIDs)
      result.insert(id);
   return result;
}

bool ROOT::Experimental::Internal::RPageSource::REntryRange::IntersectsWith(const RClusterDescriptor &clusterDesc) const
{
   if (fFirstEntry == kInvalidNTupleIndex) {
      /// Entry range unset, we assume that the entry range covers the complete source
      return true;
   }

   if (clusterDesc.GetNEntries() == 0)
      return true;
   if ((clusterDesc.GetFirstEntryIndex() + clusterDesc.GetNEntries()) <= fFirstEntry)
      return false;
   if (clusterDesc.GetFirstEntryIndex() >= (fFirstEntry + fNEntries))
      return false;
   return true;
}

ROOT::Experimental::Internal::RPageSource::RPageSource(std::string_view name, const RNTupleReadOptions &options)
   : RPageStorage(name), fOptions(options)
{
}

ROOT::Experimental::Internal::RPageSource::~RPageSource() {}

std::unique_ptr<ROOT::Experimental::Internal::RPageSource>
ROOT::Experimental::Internal::RPageSource::Create(std::string_view ntupleName, std::string_view location,
                                                  const RNTupleReadOptions &options)
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

ROOT::Experimental::Internal::RPageStorage::ColumnHandle_t
ROOT::Experimental::Internal::RPageSource::AddColumn(DescriptorId_t fieldId, const RColumn &column)
{
   R__ASSERT(fieldId != kInvalidDescriptorId);
   auto physicalId = GetSharedDescriptorGuard()->FindPhysicalColumnId(fieldId, column.GetIndex());
   R__ASSERT(physicalId != kInvalidDescriptorId);
   fActivePhysicalColumns.Insert(physicalId);
   return ColumnHandle_t{physicalId, &column};
}

void ROOT::Experimental::Internal::RPageSource::DropColumn(ColumnHandle_t columnHandle)
{
   fActivePhysicalColumns.Erase(columnHandle.fPhysicalId);
}

void ROOT::Experimental::Internal::RPageSource::SetEntryRange(const REntryRange &range)
{
   if ((range.fFirstEntry + range.fNEntries) > GetNEntries()) {
      throw RException(R__FAIL("invalid entry range"));
   }
   fEntryRange = range;
}

ROOT::Experimental::NTupleSize_t ROOT::Experimental::Internal::RPageSource::GetNEntries()
{
   return GetSharedDescriptorGuard()->GetNEntries();
}

ROOT::Experimental::NTupleSize_t ROOT::Experimental::Internal::RPageSource::GetNElements(ColumnHandle_t columnHandle)
{
   return GetSharedDescriptorGuard()->GetNElements(columnHandle.fPhysicalId);
}

ROOT::Experimental::ColumnId_t ROOT::Experimental::Internal::RPageSource::GetColumnId(ColumnHandle_t columnHandle)
{
   // TODO(jblomer) distinguish trees
   return columnHandle.fPhysicalId;
}

void ROOT::Experimental::Internal::RPageSource::UnzipCluster(RCluster *cluster)
{
   if (fTaskScheduler)
      UnzipClusterImpl(cluster);
}

void ROOT::Experimental::Internal::RPageSource::PrepareLoadCluster(
   const RCluster::RKey &clusterKey, ROnDiskPageMap &pageZeroMap,
   std::function<void(DescriptorId_t, NTupleSize_t, const RClusterDescriptor::RPageRange::RPageInfo &)> perPageFunc)
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

void ROOT::Experimental::Internal::RPageSource::EnableDefaultMetrics(const std::string &prefix)
{
   fMetrics = Detail::RNTupleMetrics(prefix);
   fCounters = std::make_unique<RCounters>(RCounters{
      *fMetrics.MakeCounter<Detail::RNTupleAtomicCounter *>("nReadV", "", "number of vector read requests"),
      *fMetrics.MakeCounter<Detail::RNTupleAtomicCounter *>("nRead", "", "number of byte ranges read"),
      *fMetrics.MakeCounter<Detail::RNTupleAtomicCounter *>("szReadPayload", "B",
                                                            "volume read from storage (required)"),
      *fMetrics.MakeCounter<Detail::RNTupleAtomicCounter *>("szReadOverhead", "B",
                                                            "volume read from storage (overhead)"),
      *fMetrics.MakeCounter<Detail::RNTupleAtomicCounter *>("szUnzip", "B", "volume after unzipping"),
      *fMetrics.MakeCounter<Detail::RNTupleAtomicCounter *>("nClusterLoaded", "",
                                                            "number of partial clusters preloaded from storage"),
      *fMetrics.MakeCounter<Detail::RNTupleAtomicCounter *>("nPageLoaded", "", "number of pages loaded from storage"),
      *fMetrics.MakeCounter<Detail::RNTupleAtomicCounter *>("nPagePopulated", "", "number of populated pages"),
      *fMetrics.MakeCounter<Detail::RNTupleAtomicCounter *>("timeWallRead", "ns", "wall clock time spent reading"),
      *fMetrics.MakeCounter<Detail::RNTupleAtomicCounter *>("timeWallUnzip", "ns",
                                                            "wall clock time spent decompressing"),
      *fMetrics.MakeCounter<Detail::RNTupleTickCounter<Detail::RNTupleAtomicCounter> *>("timeCpuRead", "ns",
                                                                                        "CPU time spent reading"),
      *fMetrics.MakeCounter<Detail::RNTupleTickCounter<Detail::RNTupleAtomicCounter> *>("timeCpuUnzip", "ns",
                                                                                        "CPU time spent decompressing"),
      *fMetrics.MakeCounter<Detail::RNTupleCalcPerf *>(
         "bwRead", "MB/s", "bandwidth compressed bytes read per second", fMetrics,
         [](const Detail::RNTupleMetrics &metrics) -> std::pair<bool, double> {
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
         }),
      *fMetrics.MakeCounter<Detail::RNTupleCalcPerf *>(
         "bwReadUnzip", "MB/s", "bandwidth uncompressed bytes read per second", fMetrics,
         [](const Detail::RNTupleMetrics &metrics) -> std::pair<bool, double> {
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
         }),
      *fMetrics.MakeCounter<Detail::RNTupleCalcPerf *>(
         "bwUnzip", "MB/s", "decompression bandwidth of uncompressed bytes per second", fMetrics,
         [](const Detail::RNTupleMetrics &metrics) -> std::pair<bool, double> {
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
         }),
      *fMetrics.MakeCounter<Detail::RNTupleCalcPerf *>(
         "rtReadEfficiency", "", "ratio of payload over all bytes read", fMetrics,
         [](const Detail::RNTupleMetrics &metrics) -> std::pair<bool, double> {
            if (const auto szReadPayload = metrics.GetLocalCounter("szReadPayload")) {
               if (const auto szReadOverhead = metrics.GetLocalCounter("szReadOverhead")) {
                  if (auto payload = szReadPayload->GetValueAsInt()) {
                     // r/(r+o) = 1/((r+o)/r) = 1/(1 + o/r)
                     return {true, 1./(1. + (1. * szReadOverhead->GetValueAsInt()) / payload)};
                  }
               }
            }
            return {false, -1.};
         }),
      *fMetrics.MakeCounter<Detail::RNTupleCalcPerf *>(
         "rtCompression", "", "ratio of compressed bytes / uncompressed bytes", fMetrics,
         [](const Detail::RNTupleMetrics &metrics) -> std::pair<bool, double> {
            if (const auto szReadPayload = metrics.GetLocalCounter("szReadPayload")) {
               if (const auto szUnzip = metrics.GetLocalCounter("szUnzip")) {
                  if (auto unzip = szUnzip->GetValueAsInt()) {
                     return {true, (1. * szReadPayload->GetValueAsInt()) / unzip};
                  }
               }
            }
            return {false, -1.};
         })});
}

ROOT::Experimental::Internal::RPage
ROOT::Experimental::Internal::RPageSource::UnsealPage(const RSealedPage &sealedPage, const RColumnElementBase &element,
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

//------------------------------------------------------------------------------

ROOT::Experimental::Internal::RPageSink::RPageSink(std::string_view name, const RNTupleWriteOptions &options)
   : RPageStorage(name), fOptions(options.Clone())
{
}

ROOT::Experimental::Internal::RPageSink::~RPageSink() {}

ROOT::Experimental::Internal::RPageStorage::RSealedPage
ROOT::Experimental::Internal::RPageSink::SealPage(const RPage &page, const RColumnElementBase &element,
                                                  int compressionSetting, void *buf, bool allowAlias)
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

   if ((compressionSetting != 0) || !element.IsMappable() || !allowAlias) {
      zippedBytes = RNTupleCompressor::Zip(pageBuf, packedBytes, compressionSetting, buf);
      if (!isAdoptedBuffer)
         delete[] pageBuf;
      pageBuf = reinterpret_cast<unsigned char *>(buf);
      isAdoptedBuffer = true;
   }

   R__ASSERT(isAdoptedBuffer);

   return RSealedPage{pageBuf, static_cast<std::uint32_t>(zippedBytes), page.GetNElements()};
}

ROOT::Experimental::Internal::RPageStorage::RSealedPage
ROOT::Experimental::Internal::RPageSink::SealPage(const RPage &page, const RColumnElementBase &element,
                                                  int compressionSetting)
{
   R__ASSERT(fCompressor);
   return SealPage(page, element, compressionSetting, fCompressor->GetZipBuffer());
}

//------------------------------------------------------------------------------

std::unique_ptr<ROOT::Experimental::Internal::RPageSink>
ROOT::Experimental::Internal::RPagePersistentSink::Create(std::string_view ntupleName, std::string_view location,
                                                          const RNTupleWriteOptions &options)
{
   if (ntupleName.empty()) {
      throw RException(R__FAIL("empty RNTuple name"));
   }
   if (location.empty()) {
      throw RException(R__FAIL("empty storage location"));
   }
   if (location.find("daos://") == 0) {
#ifdef R__ENABLE_DAOS
      return std::make_unique<RPageSinkDaos>(ntupleName, location, options);
#else
      throw RException(R__FAIL("This RNTuple build does not support DAOS."));
#endif
   }

   // Otherwise assume that the user wants us to create a file.
   return std::make_unique<RPageSinkFile>(ntupleName, location, options);
}

ROOT::Experimental::Internal::RPagePersistentSink::RPagePersistentSink(std::string_view name,
                                                                       const RNTupleWriteOptions &options)
   : RPageSink(name, options)
{
}

ROOT::Experimental::Internal::RPagePersistentSink::~RPagePersistentSink() {}

ROOT::Experimental::Internal::RPageStorage::ColumnHandle_t
ROOT::Experimental::Internal::RPagePersistentSink::AddColumn(DescriptorId_t fieldId, const RColumn &column)
{
   auto columnId = fDescriptorBuilder.GetDescriptor().GetNPhysicalColumns();
   fDescriptorBuilder.AddColumn(columnId, columnId, fieldId, column.GetModel(), column.GetIndex(),
                                column.GetFirstElementIndex());
   return ColumnHandle_t{columnId, &column};
}

void ROOT::Experimental::Internal::RPagePersistentSink::UpdateSchema(const RNTupleModelChangeset &changeset,
                                                                     NTupleSize_t firstEntry)
{
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();
   auto addField = [&](RFieldBase &f) {
      auto fieldId = descriptor.GetNFields();
      fDescriptorBuilder.AddField(RFieldDescriptorBuilder::FromField(f).FieldId(fieldId).MakeDescriptor().Unwrap());
      fDescriptorBuilder.AddFieldLink(f.GetParent()->GetOnDiskId(), fieldId);
      f.SetOnDiskId(fieldId);
      CallConnectPageSinkOnField(f, *this, firstEntry); // issues in turn calls to `AddColumn()`
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

void ROOT::Experimental::Internal::RPagePersistentSink::InitImpl(RNTupleModel &model)
{
   fDescriptorBuilder.SetNTuple(fNTupleName, model.GetDescription());
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();

   auto &fieldZero = model.GetFieldZero();
   fDescriptorBuilder.AddField(RFieldDescriptorBuilder::FromField(fieldZero).FieldId(0).MakeDescriptor().Unwrap());
   fieldZero.SetOnDiskId(0);
   model.GetProjectedFields().GetFieldZero()->SetOnDiskId(0);

   RNTupleModelChangeset initialChangeset{model};
   for (auto f : fieldZero.GetSubFields())
      initialChangeset.fAddedFields.emplace_back(f);
   for (auto f : model.GetProjectedFields().GetFieldZero()->GetSubFields())
      initialChangeset.fAddedProjectedFields.emplace_back(f);
   UpdateSchema(initialChangeset, 0U);

   fSerializationContext = RNTupleSerializer::SerializeHeader(nullptr, descriptor);
   auto buffer = std::make_unique<unsigned char[]>(fSerializationContext.GetHeaderSize());
   fSerializationContext = RNTupleSerializer::SerializeHeader(buffer.get(), descriptor);
   InitImpl(buffer.get(), fSerializationContext.GetHeaderSize());

   fDescriptorBuilder.BeginHeaderExtension();
}

void ROOT::Experimental::Internal::RPagePersistentSink::InitFromDescriptor(const RNTupleDescriptor &descriptor)
{
   {
      auto model = descriptor.CreateModel();
      Init(*model.get());
   }

   auto clusterId = descriptor.FindClusterId(0, 0);

   while (clusterId != ROOT::Experimental::kInvalidDescriptorId) {
      auto &cluster = descriptor.GetClusterDescriptor(clusterId);
      auto nEntries = cluster.GetNEntries();

      RClusterDescriptorBuilder clusterBuilder;
      clusterBuilder.ClusterId(fDescriptorBuilder.GetDescriptor().GetNActiveClusters())
         .FirstEntryIndex(fPrevClusterNEntries)
         .NEntries(nEntries);

      for (unsigned int i = 0; i < fOpenColumnRanges.size(); ++i) {
         R__ASSERT(fOpenColumnRanges[i].fPhysicalColumnId == i);
         const auto &columnRange = cluster.GetColumnRange(i);
         R__ASSERT(columnRange.fPhysicalColumnId == i);
         const auto &pageRange = cluster.GetPageRange(i);
         R__ASSERT(pageRange.fPhysicalColumnId == i);
         clusterBuilder.CommitColumnRange(i, fOpenColumnRanges[i].fFirstElementIndex, columnRange.fCompressionSettings,
                                          pageRange);
         fOpenColumnRanges[i].fFirstElementIndex += columnRange.fNElements;
      }
      fDescriptorBuilder.AddCluster(clusterBuilder.MoveDescriptor().Unwrap());
      fPrevClusterNEntries += nEntries;

      clusterId = descriptor.FindNextClusterId(clusterId);
   }
}

void ROOT::Experimental::Internal::RPagePersistentSink::CommitPage(ColumnHandle_t columnHandle, const RPage &page)
{
   fOpenColumnRanges.at(columnHandle.fPhysicalId).fNElements += page.GetNElements();

   RClusterDescriptor::RPageRange::RPageInfo pageInfo;
   pageInfo.fNElements = page.GetNElements();
   pageInfo.fLocator = CommitPageImpl(columnHandle, page);
   fOpenPageRanges.at(columnHandle.fPhysicalId).fPageInfos.emplace_back(pageInfo);
}

void ROOT::Experimental::Internal::RPagePersistentSink::CommitSealedPage(DescriptorId_t physicalColumnId,
                                                                         const RPageStorage::RSealedPage &sealedPage)
{
   fOpenColumnRanges.at(physicalColumnId).fNElements += sealedPage.fNElements;

   RClusterDescriptor::RPageRange::RPageInfo pageInfo;
   pageInfo.fNElements = sealedPage.fNElements;
   pageInfo.fLocator = CommitSealedPageImpl(physicalColumnId, sealedPage);
   fOpenPageRanges.at(physicalColumnId).fPageInfos.emplace_back(pageInfo);
}

std::vector<ROOT::Experimental::RNTupleLocator>
ROOT::Experimental::Internal::RPagePersistentSink::CommitSealedPageVImpl(
   std::span<RPageStorage::RSealedPageGroup> ranges)
{
   std::vector<ROOT::Experimental::RNTupleLocator> locators;
   for (auto &range : ranges) {
      for (auto sealedPageIt = range.fFirst; sealedPageIt != range.fLast; ++sealedPageIt)
         locators.push_back(CommitSealedPageImpl(range.fPhysicalColumnId, *sealedPageIt));
   }
   return locators;
}

void ROOT::Experimental::Internal::RPagePersistentSink::CommitSealedPageV(
   std::span<RPageStorage::RSealedPageGroup> ranges)
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

std::uint64_t
ROOT::Experimental::Internal::RPagePersistentSink::CommitCluster(ROOT::Experimental::NTupleSize_t nNewEntries)
{
   auto nbytes = CommitClusterImpl();

   RClusterDescriptorBuilder clusterBuilder;
   clusterBuilder.ClusterId(fDescriptorBuilder.GetDescriptor().GetNActiveClusters())
      .FirstEntryIndex(fPrevClusterNEntries)
      .NEntries(nNewEntries);
   for (unsigned int i = 0; i < fOpenColumnRanges.size(); ++i) {
      RClusterDescriptor::RPageRange fullRange;
      fullRange.fPhysicalColumnId = i;
      std::swap(fullRange, fOpenPageRanges[i]);
      clusterBuilder.CommitColumnRange(i, fOpenColumnRanges[i].fFirstElementIndex,
                                       fOpenColumnRanges[i].fCompressionSettings, fullRange);
      fOpenColumnRanges[i].fFirstElementIndex += fOpenColumnRanges[i].fNElements;
      fOpenColumnRanges[i].fNElements = 0;
   }
   fDescriptorBuilder.AddCluster(clusterBuilder.MoveDescriptor().Unwrap());
   fPrevClusterNEntries += nNewEntries;
   return nbytes;
}

void ROOT::Experimental::Internal::RPagePersistentSink::CommitClusterGroup()
{
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();

   const auto nClusters = descriptor.GetNActiveClusters();
   std::vector<DescriptorId_t> physClusterIDs;
   for (auto i = fNextClusterInGroup; i < nClusters; ++i) {
      physClusterIDs.emplace_back(fSerializationContext.MapClusterId(i));
   }

   auto szPageList = RNTupleSerializer::SerializePageList(nullptr, descriptor, physClusterIDs, fSerializationContext);
   auto bufPageList = std::make_unique<unsigned char[]>(szPageList);
   RNTupleSerializer::SerializePageList(bufPageList.get(), descriptor, physClusterIDs, fSerializationContext);

   const auto clusterGroupId = descriptor.GetNClusterGroups();
   const auto locator = CommitClusterGroupImpl(bufPageList.get(), szPageList);
   RClusterGroupDescriptorBuilder cgBuilder;
   cgBuilder.ClusterGroupId(clusterGroupId).PageListLocator(locator).PageListLength(szPageList);
   if (fNextClusterInGroup == nClusters) {
      cgBuilder.MinEntry(0).EntrySpan(0).NClusters(0);
   } else {
      const auto &firstClusterDesc = descriptor.GetClusterDescriptor(fNextClusterInGroup);
      const auto &lastClusterDesc = descriptor.GetClusterDescriptor(nClusters - 1);
      cgBuilder.MinEntry(firstClusterDesc.GetFirstEntryIndex())
         .EntrySpan(lastClusterDesc.GetFirstEntryIndex() + lastClusterDesc.GetNEntries() -
                    firstClusterDesc.GetFirstEntryIndex())
         .NClusters(nClusters - fNextClusterInGroup);
   }
   std::vector<DescriptorId_t> clusterIds;
   for (auto i = fNextClusterInGroup; i < nClusters; ++i) {
      clusterIds.emplace_back(i);
   }
   cgBuilder.AddClusters(clusterIds);
   fDescriptorBuilder.AddClusterGroup(cgBuilder.MoveDescriptor().Unwrap());
   fSerializationContext.MapClusterGroupId(clusterGroupId);

   fNextClusterInGroup = nClusters;
}

void ROOT::Experimental::Internal::RPagePersistentSink::CommitDataset()
{
   const auto &descriptor = fDescriptorBuilder.GetDescriptor();

   auto szFooter = RNTupleSerializer::SerializeFooter(nullptr, descriptor, fSerializationContext);
   auto bufFooter = std::make_unique<unsigned char[]>(szFooter);
   RNTupleSerializer::SerializeFooter(bufFooter.get(), descriptor, fSerializationContext);

   CommitDatasetImpl(bufFooter.get(), szFooter);
}

void ROOT::Experimental::Internal::RPagePersistentSink::EnableDefaultMetrics(const std::string &prefix)
{
   fMetrics = Detail::RNTupleMetrics(prefix);
   fCounters = std::make_unique<RCounters>(RCounters{
      *fMetrics.MakeCounter<Detail::RNTupleAtomicCounter *>("nPageCommitted", "",
                                                            "number of pages committed to storage"),
      *fMetrics.MakeCounter<Detail::RNTupleAtomicCounter *>("szWritePayload", "B",
                                                            "volume written for committed pages"),
      *fMetrics.MakeCounter<Detail::RNTupleAtomicCounter *>("szZip", "B", "volume before zipping"),
      *fMetrics.MakeCounter<Detail::RNTupleAtomicCounter *>("timeWallWrite", "ns", "wall clock time spent writing"),
      *fMetrics.MakeCounter<Detail::RNTupleAtomicCounter *>("timeWallZip", "ns", "wall clock time spent compressing"),
      *fMetrics.MakeCounter<Detail::RNTupleTickCounter<Detail::RNTupleAtomicCounter> *>("timeCpuWrite", "ns",
                                                                                        "CPU time spent writing"),
      *fMetrics.MakeCounter<Detail::RNTupleTickCounter<Detail::RNTupleAtomicCounter> *>("timeCpuZip", "ns",
                                                                                        "CPU time spent compressing")});
}

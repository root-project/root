/// \file RPageSinkBuf.cxx
/// \ingroup NTuple
/// \author Jakob Blomer <jblomer@cern.ch>
/// \author Max Orok <maxwellorok@gmail.com>
/// \author Javier Lopez-Gomez <javier.lopez.gomez@cern.ch>
/// \date 2021-03-17

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleUtils.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RPageSinkBuf.hxx>

#include <algorithm>
#include <memory>

using ROOT::Experimental::Detail::RNTupleAtomicCounter;
using ROOT::Experimental::Detail::RNTupleAtomicTimer;
using ROOT::Experimental::Detail::RNTupleMetrics;
using ROOT::Experimental::Detail::RNTuplePlainCounter;
using ROOT::Experimental::Detail::RNTuplePlainTimer;
using ROOT::Experimental::Detail::RNTupleTickCounter;
using ROOT::Internal::MakeUninitArray;

void ROOT::Internal::RPageSinkBuf::RColumnBuf::DropBufferedPages()
{
   fBufferedPages.clear();
   // Each RSealedPage points to the same region as `fBuf` for some element in `fBufferedPages`; thus, no further
   // clean-up is required
   fSealedPages.clear();
}

ROOT::Internal::RPageSinkBuf::RPageSinkBuf(std::unique_ptr<RPageSink> inner)
   : RPageSink(inner->GetNTupleName(), inner->GetWriteOptions()), fInnerSink(std::move(inner))
{
   fMetrics = RNTupleMetrics("RPageSinkBuf");
   fCounters = std::make_unique<RCounters>(
      RCounters{*fMetrics.MakeCounter<RNTuplePlainCounter *>("ParallelZip", "", "compressing pages in parallel"),
                *fMetrics.MakeCounter<RNTupleAtomicCounter *>("timeWallZip", "ns", "wall clock time spent compressing"),
                *fMetrics.MakeCounter<RNTuplePlainCounter *>("timeWallCriticalSection", "ns",
                                                             "wall clock time spent in critical sections"),
                *fMetrics.MakeCounter<RNTupleTickCounter<RNTupleAtomicCounter> *>("timeCpuZip", "ns",
                                                                                  "CPU time spent compressing"),
                *fMetrics.MakeCounter<RNTupleTickCounter<RNTuplePlainCounter> *>(
                   "timeCpuCriticalSection", "ns", "CPU time spent in critical section")});
   fMetrics.ObserveMetrics(fInnerSink->GetMetrics());
}

ROOT::Internal::RPageSinkBuf::~RPageSinkBuf()
{
   // Wait for unterminated tasks, if any, as they may still hold a reference to `this`.
   // This cannot be moved to the base class destructor, given non-static members have been destroyed by the time the
   // base class destructor is invoked.
   WaitForAllTasks();
}

ROOT::Internal::RPageStorage::ColumnHandle_t
ROOT::Internal::RPageSinkBuf::AddColumn(ROOT::DescriptorId_t /*fieldId*/, ROOT::Internal::RColumn &column)
{
   return ColumnHandle_t{fNColumns++, &column};
}

void ROOT::Internal::RPageSinkBuf::ConnectFields(const std::vector<ROOT::RFieldBase *> &fields,
                                                 ROOT::NTupleSize_t firstEntry)
{
   auto connectField = [&](ROOT::RFieldBase &f) {
      // Field Zero would have id 0.
      ++fNFields;
      f.SetOnDiskId(fNFields);
      CallConnectPageSinkOnField(f, *this, firstEntry); // issues in turn calls to `AddColumn()`
   };
   for (auto *f : fields) {
      connectField(*f);
      for (auto &descendant : *f) {
         connectField(descendant);
      }
   }
   fBufferedColumns.resize(fNColumns);
}

const ROOT::RNTupleDescriptor &ROOT::Internal::RPageSinkBuf::GetDescriptor() const
{
   return fInnerSink->GetDescriptor();
}

void ROOT::Internal::RPageSinkBuf::InitImpl(ROOT::RNTupleModel &model)
{
   ConnectFields(GetFieldZeroOfModel(model).GetMutableSubfields(), 0U);

   fInnerModel = model.Clone();
   fInnerSink->Init(*fInnerModel);
}

void ROOT::Internal::RPageSinkBuf::UpdateSchema(const ROOT::Internal::RNTupleModelChangeset &changeset,
                                                ROOT::NTupleSize_t firstEntry)
{
   ConnectFields(changeset.fAddedFields, firstEntry);

   // The buffered page sink maintains a copy of the RNTupleModel for the inner sink; replicate the changes there
   // TODO(jalopezg): we should be able, in general, to simplify the buffered sink.
   auto cloneAddField = [&](const ROOT::RFieldBase *field) {
      auto cloned = field->Clone(field->GetFieldName());
      auto p = &(*cloned);
      fInnerModel->AddField(std::move(cloned));
      return p;
   };
   auto cloneAddProjectedField = [&](ROOT::RFieldBase *field) {
      auto cloned = field->Clone(field->GetFieldName());
      auto p = &(*cloned);
      auto &projectedFields = GetProjectedFieldsOfModel(changeset.fModel);
      RProjectedFields::FieldMap_t fieldMap;
      fieldMap[p] = &fInnerModel->GetConstField(projectedFields.GetSourceField(field)->GetQualifiedFieldName());
      auto targetIt = cloned->begin();
      for (auto &f : *field)
         fieldMap[&(*targetIt++)] =
            &fInnerModel->GetConstField(projectedFields.GetSourceField(&f)->GetQualifiedFieldName());
      GetProjectedFieldsOfModel(*fInnerModel).Add(std::move(cloned), fieldMap);
      return p;
   };
   RNTupleModelChangeset innerChangeset{*fInnerModel};
   fInnerModel->Unfreeze();
   std::transform(changeset.fAddedFields.cbegin(), changeset.fAddedFields.cend(),
                  std::back_inserter(innerChangeset.fAddedFields), cloneAddField);
   std::transform(changeset.fAddedProjectedFields.cbegin(), changeset.fAddedProjectedFields.cend(),
                  std::back_inserter(innerChangeset.fAddedProjectedFields), cloneAddProjectedField);
   fInnerModel->Freeze();
   fInnerSink->UpdateSchema(innerChangeset, firstEntry);
}

void ROOT::Internal::RPageSinkBuf::UpdateExtraTypeInfo(const ROOT::RExtraTypeInfoDescriptor &extraTypeInfo)
{
   RPageSink::RSinkGuard g(fInnerSink->GetSinkGuard());
   RNTuplePlainTimer timer(fCounters->fTimeWallCriticalSection, fCounters->fTimeCpuCriticalSection);
   fInnerSink->UpdateExtraTypeInfo(extraTypeInfo);
}

void ROOT::Internal::RPageSinkBuf::CommitSuppressedColumn(ColumnHandle_t columnHandle)
{
   fSuppressedColumns.emplace_back(columnHandle);
}

void ROOT::Internal::RPageSinkBuf::CommitPage(ColumnHandle_t columnHandle, const RPage &page)
{
   auto colId = columnHandle.fPhysicalId;
   const auto &element = *columnHandle.fColumn->GetElement();

   // Safety: References are guaranteed to be valid until the element is destroyed. In other words, all buffered page
   // elements are valid until DropBufferedPages().
   auto &zipItem = fBufferedColumns.at(colId).BufferPage(columnHandle);
   std::size_t maxSealedPageBytes = page.GetNBytes() + GetWriteOptions().GetEnablePageChecksums() * kNBytesPageChecksum;
   // Do not allocate the buffer yet, in case of IMT we only need it once the task is started.
   auto &sealedPage = fBufferedColumns.at(colId).RegisterSealedPage();

   auto allocateBuf = [&zipItem, maxSealedPageBytes]() {
      zipItem.fBuf = MakeUninitArray<unsigned char>(maxSealedPageBytes);
      R__ASSERT(zipItem.fBuf);
   };
   auto shrinkSealedPage = [&zipItem, maxSealedPageBytes, &sealedPage]() {
      // If the sealed page is smaller than the maximum size (with compression), allocate what is needed and copy the
      // sealed page content to save memory.
      auto sealedBufferSize = sealedPage.GetBufferSize();
      if (sealedBufferSize < maxSealedPageBytes) {
         auto buf = MakeUninitArray<unsigned char>(sealedBufferSize);
         memcpy(buf.get(), sealedPage.GetBuffer(), sealedBufferSize);
         zipItem.fBuf = std::move(buf);
         sealedPage.SetBuffer(zipItem.fBuf.get());
      }
   };

   if (!fTaskScheduler) {
      allocateBuf();
      // Seal the page right now, avoiding the allocation and copy, but making sure that the page buffer is not aliased.
      RSealPageConfig config;
      config.fPage = &page;
      config.fElement = &element;
      config.fCompressionSettings = GetWriteOptions().GetCompression();
      config.fWriteChecksum = GetWriteOptions().GetEnablePageChecksums();
      config.fAllowAlias = false;
      config.fBuffer = zipItem.fBuf.get();
      {
         RNTupleAtomicTimer timer(fCounters->fTimeWallZip, fCounters->fTimeCpuZip);
         sealedPage = SealPage(config);
      }
      shrinkSealedPage();
      zipItem.fSealedPage = &sealedPage;
      return;
   }

   // TODO avoid frequent (de)allocations by holding on to allocated buffers in RColumnBuf
   zipItem.fPage = fPageAllocator->NewPage(page.GetElementSize(), page.GetNElements());
   // make sure the page is aware of how many elements it will have
   zipItem.fPage.GrowUnchecked(page.GetNElements());
   memcpy(zipItem.fPage.GetBuffer(), page.GetBuffer(), page.GetNBytes());

   fCounters->fParallelZip.SetValue(1);
   // Thread safety: Each thread works on a distinct zipItem which owns its
   // compression buffer.
   fTaskScheduler->AddTask([this, &zipItem, &sealedPage, &element, allocateBuf, shrinkSealedPage] {
      allocateBuf();
      RSealPageConfig config;
      config.fPage = &zipItem.fPage;
      config.fElement = &element;
      config.fCompressionSettings = GetWriteOptions().GetCompression();
      config.fWriteChecksum = GetWriteOptions().GetEnablePageChecksums();
      // Make sure the page buffer is not aliased so that we can free the uncompressed page.
      config.fAllowAlias = false;
      config.fBuffer = zipItem.fBuf.get();
      // TODO: Somehow expose the time spent in zipping via the metrics. Wall time is tricky because the tasks run
      // in parallel...
      sealedPage = SealPage(config);
      shrinkSealedPage();
      zipItem.fSealedPage = &sealedPage;
      // Release the uncompressed page. This works because the "page allocator must be thread-safe."
      zipItem.fPage = RPage();
   });
}

void ROOT::Internal::RPageSinkBuf::CommitSealedPage(ROOT::DescriptorId_t /*physicalColumnId*/,
                                                    const RSealedPage & /*sealedPage*/)
{
   throw RException(R__FAIL("should never commit sealed pages to RPageSinkBuf"));
}

void ROOT::Internal::RPageSinkBuf::CommitSealedPageV(
   std::span<ROOT::Internal::RPageStorage::RSealedPageGroup> /*ranges*/)
{
   throw RException(R__FAIL("should never commit sealed pages to RPageSinkBuf"));
}

// We implement both StageCluster() and CommitCluster() because we can call CommitCluster() on the inner sink more
// efficiently in a single critical section. For parallel writing, it also guarantees that we produce a fully sequential
// file.
void ROOT::Internal::RPageSinkBuf::FlushClusterImpl(std::function<void(void)> FlushClusterFn)
{
   WaitForAllTasks();

   std::vector<RSealedPageGroup> toCommit;
   toCommit.reserve(fBufferedColumns.size());
   for (auto &bufColumn : fBufferedColumns) {
      R__ASSERT(bufColumn.HasSealedPagesOnly());
      const auto &sealedPages = bufColumn.GetSealedPages();
      toCommit.emplace_back(bufColumn.GetHandle().fPhysicalId, sealedPages.cbegin(), sealedPages.cend());
   }

   {
      RPageSink::RSinkGuard g(fInnerSink->GetSinkGuard());
      RNTuplePlainTimer timer(fCounters->fTimeWallCriticalSection, fCounters->fTimeCpuCriticalSection);
      fInnerSink->CommitSealedPageV(toCommit);

      for (auto handle : fSuppressedColumns)
         fInnerSink->CommitSuppressedColumn(handle);
      fSuppressedColumns.clear();

      FlushClusterFn();
   }

   for (auto &bufColumn : fBufferedColumns)
      bufColumn.DropBufferedPages();
}

std::uint64_t ROOT::Internal::RPageSinkBuf::CommitCluster(ROOT::NTupleSize_t nNewEntries)
{
   std::uint64_t nbytes;
   FlushClusterImpl([&] { nbytes = fInnerSink->CommitCluster(nNewEntries); });
   return nbytes;
}

ROOT::Internal::RPageSink::RStagedCluster ROOT::Internal::RPageSinkBuf::StageCluster(ROOT::NTupleSize_t nNewEntries)
{
   RPageSink::RStagedCluster stagedCluster;
   FlushClusterImpl([&] { stagedCluster = fInnerSink->StageCluster(nNewEntries); });
   return stagedCluster;
}

void ROOT::Internal::RPageSinkBuf::CommitStagedClusters(std::span<RStagedCluster> clusters)
{
   RPageSink::RSinkGuard g(fInnerSink->GetSinkGuard());
   RNTuplePlainTimer timer(fCounters->fTimeWallCriticalSection, fCounters->fTimeCpuCriticalSection);
   fInnerSink->CommitStagedClusters(clusters);
}

void ROOT::Internal::RPageSinkBuf::CommitClusterGroup()
{
   RPageSink::RSinkGuard g(fInnerSink->GetSinkGuard());
   RNTuplePlainTimer timer(fCounters->fTimeWallCriticalSection, fCounters->fTimeCpuCriticalSection);
   fInnerSink->CommitClusterGroup();
}

void ROOT::Internal::RPageSinkBuf::CommitDatasetImpl()
{
   RPageSink::RSinkGuard g(fInnerSink->GetSinkGuard());
   RNTuplePlainTimer timer(fCounters->fTimeWallCriticalSection, fCounters->fTimeCpuCriticalSection);
   fInnerSink->CommitDataset();
}

ROOT::Internal::RPage ROOT::Internal::RPageSinkBuf::ReservePage(ColumnHandle_t columnHandle, std::size_t nElements)
{
   return fInnerSink->ReservePage(columnHandle, nElements);
}

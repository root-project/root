/// \file RPageSinkBuf.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \author Max Orok <maxwellorok@gmail.com>
/// \author Javier Lopez-Gomez <javier.lopez.gomez@cern.ch>
/// \date 2021-03-17
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleWriteOptions.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RPageSinkBuf.hxx>

#include <algorithm>
#include <memory>

void ROOT::Experimental::Internal::RPageSinkBuf::RColumnBuf::DropBufferedPages()
{
   for (auto &bufPage : fBufferedPages) {
      if (!bufPage.fPage.IsNull()) {
         fCol.fColumn->GetPageSink()->ReleasePage(bufPage.fPage);
      }
   }
   fBufferedPages.clear();
   // Each RSealedPage points to the same region as `fBuf` for some element in `fBufferedPages`; thus, no further
   // clean-up is required
   fSealedPages.clear();
}

ROOT::Experimental::Internal::RPageSinkBuf::RPageSinkBuf(std::unique_ptr<RPageSink> inner)
   : RPageSink(inner->GetNTupleName(), inner->GetWriteOptions()), fInnerSink(std::move(inner))
{
   fMetrics = Detail::RNTupleMetrics("RPageSinkBuf");
   fCounters = std::make_unique<RCounters>(RCounters{
      *fMetrics.MakeCounter<Detail::RNTuplePlainCounter *>("ParallelZip", "", "compressing pages in parallel"),
      *fMetrics.MakeCounter<Detail::RNTuplePlainCounter *>("timeWallCriticalSection", "ns",
                                                           "wall clock time spent in critical sections"),
      *fMetrics.MakeCounter<Detail::RNTupleTickCounter<Detail::RNTuplePlainCounter> *>(
         "timeCpuCriticalSection", "ns", "CPU time spent in critical section")});
   fMetrics.ObserveMetrics(fInnerSink->GetMetrics());
}

ROOT::Experimental::Internal::RPageSinkBuf::~RPageSinkBuf()
{
   // Wait for unterminated tasks, if any, as they may still hold a reference to `this`.
   // This cannot be moved to the base class destructor, given non-static members have been destroyed by the time the
   // base class destructor is invoked.
   WaitForAllTasks();
}

ROOT::Experimental::Internal::RPageStorage::ColumnHandle_t
ROOT::Experimental::Internal::RPageSinkBuf::AddColumn(DescriptorId_t /*fieldId*/, const RColumn &column)
{
   return ColumnHandle_t{fNColumns++, &column};
}

void ROOT::Experimental::Internal::RPageSinkBuf::ConnectFields(const std::vector<RFieldBase *> &fields,
                                                               NTupleSize_t firstEntry)
{
   auto connectField = [&](RFieldBase &f) {
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

void ROOT::Experimental::Internal::RPageSinkBuf::Init(RNTupleModel &model)
{
   ConnectFields(model.GetFieldZero().GetSubFields(), 0U);

   fInnerModel = model.Clone();
   fInnerSink->Init(*fInnerModel);
}

void ROOT::Experimental::Internal::RPageSinkBuf::UpdateSchema(const RNTupleModelChangeset &changeset,
                                                              NTupleSize_t firstEntry)
{
   ConnectFields(changeset.fAddedFields, firstEntry);

   // The buffered page sink maintains a copy of the RNTupleModel for the inner sink; replicate the changes there
   // TODO(jalopezg): we should be able, in general, to simplify the buffered sink.
   auto cloneAddField = [&](const RFieldBase *field) {
      auto cloned = field->Clone(field->GetFieldName());
      auto p = &(*cloned);
      fInnerModel->AddField(std::move(cloned));
      return p;
   };
   auto cloneAddProjectedField = [&](RFieldBase *field) {
      auto cloned = field->Clone(field->GetFieldName());
      auto p = &(*cloned);
      auto &projectedFields = changeset.fModel.GetProjectedFields();
      RNTupleModel::RProjectedFields::FieldMap_t fieldMap;
      fieldMap[p] = projectedFields.GetSourceField(field);
      auto targetIt = cloned->begin();
      for (auto &f : *field)
         fieldMap[&(*targetIt++)] = projectedFields.GetSourceField(&f);
      const_cast<RNTupleModel::RProjectedFields &>(fInnerModel->GetProjectedFields()).Add(std::move(cloned), fieldMap);
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

void ROOT::Experimental::Internal::RPageSinkBuf::CommitPage(ColumnHandle_t columnHandle, const RPage &page)
{
   auto colId = columnHandle.fPhysicalId;
   const auto &element = *columnHandle.fColumn->GetElement();

   // Safety: References are guaranteed to be valid until the
   // element is destroyed. In other words, all buffered page elements are
   // valid until the return value of DrainBufferedPages() goes out of scope in
   // CommitCluster().
   auto &zipItem = fBufferedColumns.at(colId).BufferPage(columnHandle);
   zipItem.AllocateSealedPageBuf(page.GetNBytes());
   R__ASSERT(zipItem.fBuf);
   auto &sealedPage = fBufferedColumns.at(colId).RegisterSealedPage();

   if (!fTaskScheduler) {
      // Seal the page right now, avoiding the allocation and copy, but making sure that the page buffer is not aliased.
      sealedPage =
         SealPage(page, element, GetWriteOptions().GetCompression(), zipItem.fBuf.get(), /*allowAlias=*/false);
      zipItem.fSealedPage = &sealedPage;
      return;
   }

   // TODO avoid frequent (de)allocations by holding on to allocated buffers in RColumnBuf
   zipItem.fPage = ReservePage(columnHandle, page.GetNElements());
   // make sure the page is aware of how many elements it will have
   zipItem.fPage.GrowUnchecked(page.GetNElements());
   memcpy(zipItem.fPage.GetBuffer(), page.GetBuffer(), page.GetNBytes());

   fCounters->fParallelZip.SetValue(1);
   // Thread safety: Each thread works on a distinct zipItem which owns its
   // compression buffer.
   fTaskScheduler->AddTask([this, &zipItem, &sealedPage, &element] {
      sealedPage = SealPage(zipItem.fPage, element, GetWriteOptions().GetCompression(), zipItem.fBuf.get());
      zipItem.fSealedPage = &sealedPage;
   });
}

void ROOT::Experimental::Internal::RPageSinkBuf::CommitSealedPage(DescriptorId_t /*physicalColumnId*/,
                                                                  const RSealedPage & /*sealedPage*/)
{
   throw RException(R__FAIL("should never commit sealed pages to RPageSinkBuf"));
}

void ROOT::Experimental::Internal::RPageSinkBuf::CommitSealedPageV(std::span<RPageStorage::RSealedPageGroup> /*ranges*/)
{
   throw RException(R__FAIL("should never commit sealed pages to RPageSinkBuf"));
}

std::uint64_t ROOT::Experimental::Internal::RPageSinkBuf::CommitCluster(ROOT::Experimental::NTupleSize_t nNewEntries)
{
   WaitForAllTasks();

   std::vector<RSealedPageGroup> toCommit;
   toCommit.reserve(fBufferedColumns.size());
   for (auto &bufColumn : fBufferedColumns) {
      R__ASSERT(bufColumn.HasSealedPagesOnly());
      const auto &sealedPages = bufColumn.GetSealedPages();
      toCommit.emplace_back(bufColumn.GetHandle().fPhysicalId, sealedPages.cbegin(), sealedPages.cend());
   }

   std::uint64_t nbytes;
   {
      RPageSink::RSinkGuard g(fInnerSink->GetSinkGuard());
      Detail::RNTuplePlainTimer timer(fCounters->fTimeWallCriticalSection, fCounters->fTimeCpuCriticalSection);
      fInnerSink->CommitSealedPageV(toCommit);

      nbytes = fInnerSink->CommitCluster(nNewEntries);
   }

   for (auto &bufColumn : fBufferedColumns)
      bufColumn.DropBufferedPages();
   return nbytes;
}

void ROOT::Experimental::Internal::RPageSinkBuf::CommitClusterGroup()
{
   RPageSink::RSinkGuard g(fInnerSink->GetSinkGuard());
   Detail::RNTuplePlainTimer timer(fCounters->fTimeWallCriticalSection, fCounters->fTimeCpuCriticalSection);
   fInnerSink->CommitClusterGroup();
}

void ROOT::Experimental::Internal::RPageSinkBuf::CommitDataset()
{
   RPageSink::RSinkGuard g(fInnerSink->GetSinkGuard());
   Detail::RNTuplePlainTimer timer(fCounters->fTimeWallCriticalSection, fCounters->fTimeCpuCriticalSection);
   fInnerSink->CommitDataset();
}

ROOT::Experimental::Internal::RPage
ROOT::Experimental::Internal::RPageSinkBuf::ReservePage(ColumnHandle_t columnHandle, std::size_t nElements)
{
   return fInnerSink->ReservePage(columnHandle, nElements);
}

void ROOT::Experimental::Internal::RPageSinkBuf::ReleasePage(RPage &page)
{
   fInnerSink->ReleasePage(page);
}

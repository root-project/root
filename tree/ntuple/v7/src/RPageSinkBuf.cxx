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

#include <ROOT/RNTupleOptions.hxx>
#include <ROOT/RNTupleModel.hxx>
#include <ROOT/RNTupleZip.hxx>
#include <ROOT/RPageSinkBuf.hxx>

#include <algorithm>
#include <memory>

void ROOT::Experimental::Detail::RPageSinkBuf::RColumnBuf::DropBufferedPages()
{
   for (auto &bufPage : fBufferedPages) {
      fCol.fColumn->GetPageSink()->ReleasePage(bufPage.fPage);
   }
   fBufferedPages.clear();
   // Each RSealedPage points to the same region as `fBuf` for some element in `fBufferedPages`; thus, no further
   // clean-up is required
   fSealedPages.clear();
}

ROOT::Experimental::Detail::RPageSinkBuf::RPageSinkBuf(std::unique_ptr<RPageSink> inner)
   : RPageSink(inner->GetNTupleName(), inner->GetWriteOptions()), fInnerSink(std::move(inner))
{
   fMetrics = RNTupleMetrics("RPageSinkBuf");
   fCounters = std::make_unique<RCounters>(RCounters{
      *fMetrics.MakeCounter<RNTuplePlainCounter*>("ParallelZip", "",
         "compressing pages in parallel")
   });
   fMetrics.ObserveMetrics(fInnerSink->GetMetrics());
}

ROOT::Experimental::Detail::RPageSinkBuf::~RPageSinkBuf()
{
   // Wait for unterminated tasks, if any, as they may still hold a reference to `this`.
   // This cannot be moved to the base class destructor, given non-static members have been destroyed by the time the
   // base class destructor is invoked.
   WaitForAllTasks();
}

ROOT::Experimental::Detail::RPageStorage::ColumnHandle_t
ROOT::Experimental::Detail::RPageSinkBuf::AddColumn(DescriptorId_t /*fieldId*/, const RColumn &column)
{
   return ColumnHandle_t{fNColumns++, &column};
}

void ROOT::Experimental::Detail::RPageSinkBuf::ConnectFields(const std::vector<RFieldBase *> &fields,
                                                             NTupleSize_t firstEntry)
{
   auto connectField = [&](RFieldBase &f) {
      // Field Zero would have id 0.
      ++fNFields;
      f.SetOnDiskId(fNFields);
      f.ConnectPageSink(*this, firstEntry); // issues in turn one or several calls to `AddColumn()`
   };
   for (auto *f : fields) {
      connectField(*f);
      for (auto &descendant : *f) {
         connectField(descendant);
      }
   }
   fBufferedColumns.resize(fNColumns);
}

void ROOT::Experimental::Detail::RPageSinkBuf::Create(RNTupleModel &model)
{
   auto &fieldZero = *model.GetFieldZero();
   ConnectFields(fieldZero.GetSubFields(), 0U);

   fInnerModel = model.Clone();
   fInnerSink->Create(*fInnerModel);
}

void ROOT::Experimental::Detail::RPageSinkBuf::UpdateSchema(const RNTupleModelChangeset &changeset,
                                                            NTupleSize_t firstEntry)
{
   ConnectFields(changeset.fAddedFields, firstEntry);

   // The buffered page sink maintains a copy of the RNTupleModel for the inner sink; replicate the changes there
   // TODO(jalopezg): we should be able, in general, to simplify the buffered sink.
   auto cloneAddField = [&](const RFieldBase *field) {
      auto cloned = field->Clone(field->GetName());
      auto p = &(*cloned);
      fInnerModel->AddField(std::move(cloned));
      return p;
   };
   auto cloneAddProjectedField = [&](RFieldBase *field) {
      auto cloned = field->Clone(field->GetName());
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

void ROOT::Experimental::Detail::RPageSinkBuf::CommitPage(ColumnHandle_t columnHandle, const RPage &page)
{
   // TODO avoid frequent (de)allocations by holding on to allocated buffers in RColumnBuf
   RPage bufPage = ReservePage(columnHandle, page.GetNElements());
   // make sure the page is aware of how many elements it will have
   bufPage.GrowUnchecked(page.GetNElements());
   memcpy(bufPage.GetBuffer(), page.GetBuffer(), page.GetNBytes());
   // Safety: References are guaranteed to be valid until the
   // element is destroyed. In other words, all buffered page elements are
   // valid until the return value of DrainBufferedPages() goes out of scope in
   // CommitCluster().
   auto &zipItem = fBufferedColumns.at(columnHandle.fPhysicalId).BufferPage(columnHandle, bufPage);
   if (!fTaskScheduler) {
      return;
   }
   fCounters->fParallelZip.SetValue(1);
   // Thread safety: Each thread works on a distinct zipItem which owns its
   // compression buffer.
   zipItem.AllocateSealedPageBuf();
   R__ASSERT(zipItem.fBuf);
   auto &sealedPage = fBufferedColumns.at(columnHandle.fPhysicalId).RegisterSealedPage();
   fTaskScheduler->AddTask([this, &zipItem, &sealedPage, colId = columnHandle.fPhysicalId] {
      sealedPage = SealPage(zipItem.fPage, *fBufferedColumns.at(colId).GetHandle().fColumn->GetElement(),
                            GetWriteOptions().GetCompression(), zipItem.fBuf.get());
      zipItem.fSealedPage = &sealedPage;
   });
}

void ROOT::Experimental::Detail::RPageSinkBuf::CommitSealedPage(DescriptorId_t /*physicalColumnId*/,
                                                                const RSealedPage & /*sealedPage*/)
{
   throw RException(R__FAIL("should never commit sealed pages to RPageSinkBuf"));
}

void ROOT::Experimental::Detail::RPageSinkBuf::CommitSealedPageV(std::span<RPageStorage::RSealedPageGroup> /*ranges*/)
{
   throw RException(R__FAIL("should never commit sealed pages to RPageSinkBuf"));
}

std::uint64_t ROOT::Experimental::Detail::RPageSinkBuf::CommitCluster(ROOT::Experimental::NTupleSize_t nEntries)
{
   WaitForAllTasks();

   // If we have only sealed pages in all buffered columns, commit them in a single `CommitSealedPageV()` call
   bool singleCommitCall = std::all_of(fBufferedColumns.begin(), fBufferedColumns.end(),
                                       [](auto &bufColumn) { return bufColumn.HasSealedPagesOnly(); });
   if (singleCommitCall) {
      std::vector<RSealedPageGroup> toCommit;
      toCommit.reserve(fBufferedColumns.size());
      for (auto &bufColumn : fBufferedColumns) {
         const auto &sealedPages = bufColumn.GetSealedPages();
         toCommit.emplace_back(bufColumn.GetHandle().fPhysicalId, sealedPages.cbegin(), sealedPages.cend());
      }
      fInnerSink->CommitSealedPageV(toCommit);

      for (auto &bufColumn : fBufferedColumns)
         bufColumn.DropBufferedPages();
      return fInnerSink->CommitCluster(nEntries);
   }

   // Otherwise, try to do it per column
   for (auto &bufColumn : fBufferedColumns) {
      // In practice, either all (see above) or none of the buffered pages have been sealed, depending on whether
      // a task scheduler is available. The rare condition of a few columns consisting only of sealed pages should
      // not happen unless the API is misused.
      if (!bufColumn.IsEmpty() && bufColumn.HasSealedPagesOnly())
         throw RException(R__FAIL("only a few columns have all pages sealed"));

      // Slow path: if the buffered column contains both sealed and unsealed pages, commit them one by one.
      // TODO(jalopezg): coalesce contiguous sealed pages and commit via `CommitSealedPageV()`.
      auto drained = bufColumn.DrainBufferedPages();
      for (auto &bufPage : std::get<std::deque<RColumnBuf::RPageZipItem>>(drained)) {
         if (bufPage.IsSealed()) {
            fInnerSink->CommitSealedPage(bufColumn.GetHandle().fPhysicalId, *bufPage.fSealedPage);
         } else {
            fInnerSink->CommitPage(bufColumn.GetHandle(), bufPage.fPage);
         }
         ReleasePage(bufPage.fPage);
      }
   }
   return fInnerSink->CommitCluster(nEntries);
}

void ROOT::Experimental::Detail::RPageSinkBuf::CommitClusterGroup()
{
   fInnerSink->CommitClusterGroup();
}

void ROOT::Experimental::Detail::RPageSinkBuf::CommitDataset()
{
   fInnerSink->CommitDataset();
}

ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSinkBuf::ReservePage(ColumnHandle_t columnHandle, std::size_t nElements)
{
   return fInnerSink->ReservePage(columnHandle, nElements);
}

void ROOT::Experimental::Detail::RPageSinkBuf::ReleasePage(RPage &page)
{
   fInnerSink->ReleasePage(page);
}

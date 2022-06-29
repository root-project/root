/// \file RPageSinkBuf.cxx
/// \ingroup NTuple ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \author Max Orok <maxwellorok@gmail.com>
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

ROOT::Experimental::Detail::RPageSinkBuf::RPageSinkBuf(std::unique_ptr<RPageSink> inner)
   : RPageSink(inner->GetNTupleName(), inner->GetWriteOptions())
   , fMetrics("RPageSinkBuf")
   , fInnerSink(std::move(inner))
{
   fCounters = std::unique_ptr<RCounters>(new RCounters{
      *fMetrics.MakeCounter<RNTuplePlainCounter*>("ParallelZip", "",
         "compressing pages in parallel")
   });
   fMetrics.ObserveMetrics(fInnerSink->GetMetrics());
}

void ROOT::Experimental::Detail::RPageSinkBuf::CreateImpl(const RNTupleModel &model,
                                                          unsigned char * /* serializedHeader */,
                                                          std::uint32_t /* length */)
{
   fBufferedColumns.resize(fDescriptorBuilder.GetDescriptor().GetNColumns());
   fInnerModel = model.Clone();
   fInnerSink->Create(*fInnerModel);
}

ROOT::Experimental::RNTupleLocator
ROOT::Experimental::Detail::RPageSinkBuf::CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page)
{
   // TODO avoid frequent (de)allocations by holding on to allocated buffers in RColumnBuf
   RPage bufPage = ReservePage(columnHandle, page.GetNElements());
   // make sure the page is aware of how many elements it will have
   bufPage.GrowUnchecked(page.GetNElements());
   memcpy(bufPage.GetBuffer(), page.GetBuffer(), page.GetNBytes());
   // Safety: RColumnBuf::iterators are guaranteed to be valid until the
   // element is destroyed. In other words, all buffered page iterators are
   // valid until the return value of DrainBufferedPages() goes out of scope in
   // CommitCluster().
   RColumnBuf::iterator zipItem =
      fBufferedColumns.at(columnHandle.fId).BufferPage(columnHandle, bufPage);
   if (!fTaskScheduler) {
      return RNTupleLocator{};
   }
   fCounters->fParallelZip.SetValue(1);
   // Thread safety: Each thread works on a distinct zipItem which owns its
   // compression buffer.
   zipItem->AllocateSealedPageBuf();
   R__ASSERT(zipItem->fBuf);
   auto sealedPage = fBufferedColumns.at(columnHandle.fId).RegisterSealedPage();
   fTaskScheduler->AddTask([this, zipItem, sealedPage, colId = columnHandle.fId] {
      *sealedPage = SealPage(zipItem->fPage, *fBufferedColumns.at(colId).GetHandle().fColumn->GetElement(),
                             GetWriteOptions().GetCompression(), zipItem->fBuf.get());
      zipItem->fSealedPage = &(*sealedPage);
   });

   // we're feeding bad locators to fOpenPageRanges but it should not matter
   // because they never get written out
   return RNTupleLocator{};
}

ROOT::Experimental::RNTupleLocator
ROOT::Experimental::Detail::RPageSinkBuf::CommitSealedPageImpl(
   DescriptorId_t columnId, const RSealedPage &sealedPage)
{
   fInnerSink->CommitSealedPage(columnId, sealedPage);
   // we're feeding bad locators to fOpenPageRanges but it should not matter
   // because they never get written out
   return RNTupleLocator{};
}

std::uint64_t
ROOT::Experimental::Detail::RPageSinkBuf::CommitClusterImpl(ROOT::Experimental::NTupleSize_t nEntries)
{
   if (fTaskScheduler) {
      fTaskScheduler->Wait();
      fTaskScheduler->Reset();
   }

   // If we have only sealed pages in all buffered columns, commit them in a single `CommitSealedPageV()` call
   bool singleCommitCall = std::all_of(fBufferedColumns.begin(), fBufferedColumns.end(),
                                       [](auto &bufColumn) { return bufColumn.HasSealedPagesOnly(); });
   if (singleCommitCall) {
      std::vector<RSealedPageGroup> toCommit;
      toCommit.reserve(fBufferedColumns.size());
      for (auto &bufColumn : fBufferedColumns) {
         const auto &sealedPages = bufColumn.GetSealedPages();
         toCommit.emplace_back(bufColumn.GetHandle().fId, sealedPages.cbegin(), sealedPages.cend());
      }
      fInnerSink->CommitSealedPageV(toCommit);

      for (auto &bufColumn : fBufferedColumns)
         bufColumn.DrainBufferedPages();
      return fInnerSink->CommitCluster(nEntries);
   }

   // Otherwise, try to do it per column
   for (auto &bufColumn : fBufferedColumns) {
      // In practice, either all (see above) or none of the buffered pages have been sealed, depending on whether
      // a task scheduler is available. The rare condition of a few columns consisting only of sealed pages should
      // not happen unless the API is misused.
      if (bufColumn.HasSealedPagesOnly())
         throw RException(R__FAIL("only a few columns have all pages sealed"));

      // Slow path: if the buffered column contains both sealed and unsealed pages, commit them one by one.
      // TODO(jalopezg): coalesce contiguous sealed pages and commit via `CommitSealedPageV()`.
      auto drained = bufColumn.DrainBufferedPages();
      for (auto &bufPage : std::get<std::deque<RColumnBuf::RPageZipItem>>(drained)) {
         if (bufPage.IsSealed()) {
            fInnerSink->CommitSealedPage(bufColumn.GetHandle().fId, *bufPage.fSealedPage);
         } else {
            fInnerSink->CommitPage(bufColumn.GetHandle(), bufPage.fPage);
         }
         ReleasePage(bufPage.fPage);
      }
   }
   return fInnerSink->CommitCluster(nEntries);
}

ROOT::Experimental::RNTupleLocator
ROOT::Experimental::Detail::RPageSinkBuf::CommitClusterGroupImpl(unsigned char * /* serializedPageList */,
                                                                 std::uint32_t /* length */)
{
   fInnerSink->CommitClusterGroup();
   // We're not using that locator any further, so it is safe to return a dummy one
   return RNTupleLocator{};
}

void ROOT::Experimental::Detail::RPageSinkBuf::CommitDatasetImpl(unsigned char * /* serializedFooter */,
                                                                 std::uint32_t /* length */)
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

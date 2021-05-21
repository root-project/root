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

void ROOT::Experimental::Detail::RPageSinkBuf::CreateImpl(const RNTupleModel &model)
{
   fBufferedColumns.resize(fLastColumnId);
   fInnerModel = model.Clone();
   fInnerSink->Create(*fInnerModel);
}

ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkBuf::CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page)
{
   // TODO avoid frequent (de)allocations by holding on to allocated buffers in RColumnBuf
   RPage bufPage = ReservePage(columnHandle, page.GetNElements());
   // make sure the page is aware of how many elements it will have
   R__ASSERT(bufPage.TryGrow(page.GetNElements()));
   memcpy(bufPage.GetBuffer(), page.GetBuffer(), page.GetSize());
   // Safety: RColumnBuf::iterators are guaranteed to be valid until the
   // element is destroyed. In other words, all buffered page iterators are
   // valid until the return value of DrainBufferedPages() goes out of scope in
   // CommitCluster().
   RColumnBuf::iterator zipItem =
      fBufferedColumns.at(columnHandle.fId).BufferPage(columnHandle, bufPage);
   if (!fTaskScheduler) {
      return RClusterDescriptor::RLocator{};
   }
   fCounters->fParallelZip.SetValue(1);
   // Thread safety: Each thread works on a distinct zipItem which owns its
   // compression buffer.
   zipItem->AllocateSealedPageBuf();
   R__ASSERT(zipItem->fBuf);
   fTaskScheduler->AddTask([this, zipItem, colId = columnHandle.fId] {
      zipItem->fSealedPage = SealPage(zipItem->fPage,
         *fBufferedColumns.at(colId).GetHandle().fColumn->GetElement(),
         fOptions.GetCompression(), zipItem->fBuf.get()
      );
   });

   // we're feeding bad locators to fOpenPageRanges but it should not matter
   // because they never get written out
   return RClusterDescriptor::RLocator{};
}

ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkBuf::CommitSealedPageImpl(
   DescriptorId_t columnId, const RSealedPage &sealedPage)
{
   fInnerSink->CommitSealedPage(columnId, sealedPage);
   // we're feeding bad locators to fOpenPageRanges but it should not matter
   // because they never get written out
   return RClusterDescriptor::RLocator{};
}

ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkBuf::CommitClusterImpl(ROOT::Experimental::NTupleSize_t nEntries)
{
   if (fTaskScheduler) {
      fTaskScheduler->Wait();
      fTaskScheduler->Reset();
   }

   for (auto &bufColumn : fBufferedColumns) {
      for (auto &bufPage : bufColumn.DrainBufferedPages()) {
         if (bufPage.IsSealed()) {
            fInnerSink->CommitSealedPage(bufColumn.GetHandle().fId, bufPage.fSealedPage);
         } else {
            fInnerSink->CommitPage(bufColumn.GetHandle(), bufPage.fPage);
         }
         ReleasePage(bufPage.fPage);
      }
   }
   fInnerSink->CommitCluster(nEntries);
   // we're feeding bad locators to fOpenPageRanges but it should not matter
   // because they never get written out
   return RClusterDescriptor::RLocator{};
}

void ROOT::Experimental::Detail::RPageSinkBuf::CommitDatasetImpl()
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

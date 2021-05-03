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
   : RPageSink(inner->GetNTupleName(), inner->GetWriteOptions()), fInnerSink(std::move(inner)) {}

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
   fBufferedColumns.at(columnHandle.fId).BufferPage(columnHandle, bufPage);
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
      ParallelClusterZip(nEntries);
      return RClusterDescriptor::RLocator{};
   }

   for (auto &bufColumn : fBufferedColumns) {
      for (auto &bufPage : bufColumn.DrainBufferedPages()) {
         fInnerSink->CommitPage(bufColumn.GetHandle(), bufPage);
         ReleasePage(bufPage);
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

namespace {
   using namespace ROOT::Experimental::Detail;
   struct RPageZipItem {
      RPage fPage;
      // Compression scratch buffer for fSealedPage.
      std::unique_ptr<unsigned char[]> fBuf;
      RPageStorage::RSealedPage fSealedPage;
      explicit RPageZipItem(RPage page)
         : fPage(page), fBuf(std::make_unique<unsigned char[]>(fPage.GetSize())) {}
   };
} // anonymous namespace

void ROOT::Experimental::Detail::RPageSinkBuf::ParallelClusterZip(
   ROOT::Experimental::NTupleSize_t nEntries)
{
   // TODO(max) add timers like in RPageSourceFile::UnzipClusterImpl
   R__ASSERT(fTaskScheduler);
   fTaskScheduler->Reset();
   // zipItems[nColumns][nColumnPages]
   std::vector<std::vector<RPageZipItem>> zipItems;
   for (auto &col : fBufferedColumns) {
      std::vector<RPageZipItem> zipCol;
      for (const auto &page : col.DrainBufferedPages()) {
         zipCol.push_back(RPageZipItem(page));
      }
      zipItems.push_back(std::move(zipCol));
   }
   // Thread safety: Each task works on a distinct RPageZipItem `zi`.
   // Task (i,j) seals RPage (i,j) -- zi.fPage -- using the scratch buffer
   // zi.fBuf.
   for (std::size_t i = 0; i < fBufferedColumns.size(); i++) {
      for (std::size_t j = 0; j < zipItems.at(i).size(); j++) {
         fTaskScheduler->AddTask([this, &zipItems, i, j] {
            auto &zi = zipItems.at(i).at(j);
            zi.fSealedPage = SealPage(zi.fPage,
               *fBufferedColumns.at(i).GetHandle().fColumn->GetElement(),
               fOptions.GetCompression(), zi.fBuf.get()
            );
         });
      }
   }
   fTaskScheduler->Wait();

   for (std::size_t i = 0; i < fBufferedColumns.size(); i++) {
      for (auto &zi : zipItems.at(i)) {
         fInnerSink->CommitSealedPage(
            fBufferedColumns.at(i).GetHandle().fId, zi.fSealedPage
         );
         ReleasePage(zi.fPage);
      }
   }
   fInnerSink->CommitCluster(nEntries);
}

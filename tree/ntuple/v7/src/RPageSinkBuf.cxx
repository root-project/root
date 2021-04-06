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
#include <ROOT/RPageSinkBuf.hxx>

ROOT::Experimental::Detail::RPageSinkBuf::RPageSinkBuf(std::unique_ptr<RPageSink> inner)
   : RPageSink(inner->GetNTupleName(), inner->GetWriteOptions()), fInner(std::move(inner)) {}

void ROOT::Experimental::Detail::RPageSinkBuf::CreateImpl(const RNTupleModel &model)
{
   fBufferedColumns.resize(fLastColumnId);
   fInnerModel = model.Clone();
   fInner->Create(*fInnerModel);
}

ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkBuf::CommitPageImpl(ColumnHandle_t columnHandle, const RPage &page)
{
   fBufferedColumns.at(columnHandle.fId).BufferPage(columnHandle, page);
   // at this point we're feeding bad locators to fOpenPageRanges
   // but it may not matter if they are never written out
   return RClusterDescriptor::RLocator{};
}

ROOT::Experimental::RClusterDescriptor::RLocator
ROOT::Experimental::Detail::RPageSinkBuf::CommitClusterImpl(ROOT::Experimental::NTupleSize_t nEntries)
{
   for (auto &bufColumn : fBufferedColumns) {
      for (const auto &bufPage : bufColumn.DrainBufferedPages()) {
         fInner->CommitPage(bufColumn.GetHandle(), bufPage);
      }
   }
   fInner->CommitCluster(nEntries);
   // at this point we're feeding bad locators to fOpenColumnRanges
   // but it may not matter if they are never written out
   return RClusterDescriptor::RLocator{};
}

void ROOT::Experimental::Detail::RPageSinkBuf::CommitDatasetImpl()
{
   fInner->CommitDataset();
}

ROOT::Experimental::Detail::RPage
ROOT::Experimental::Detail::RPageSinkBuf::ReservePage(ColumnHandle_t columnHandle, std::size_t nElements)
{
   return fInner->ReservePage(columnHandle, nElements);
}

void ROOT::Experimental::Detail::RPageSinkBuf::ReleasePage(RPage &page)
{
   fInner->ReleasePage(page);
}

/// \file RColumn.cxx
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

#include <ROOT/RColumn.hxx>
#include <ROOT/RColumnModel.hxx>
#include <ROOT/RPageStorage.hxx>

#include <TError.h>

#include <iostream>

ROOT::Experimental::Detail::RColumn::RColumn(const RColumnModel& model, std::uint32_t index)
   : fModel(model), fIndex(index)
{
}

ROOT::Experimental::Detail::RColumn::~RColumn()
{
   if (!fWritePage[0].IsNull())
      fPageSink->ReleasePage(fWritePage[0]);
   if (!fWritePage[1].IsNull())
      fPageSink->ReleasePage(fWritePage[1]);
   if (!fReadPage.IsNull())
      fPageSource->ReleasePage(fReadPage);
   if (fHandleSink)
      fPageSink->DropColumn(fHandleSink);
   if (fHandleSource)
      fPageSource->DropColumn(fHandleSource);
}

void ROOT::Experimental::Detail::RColumn::Connect(DescriptorId_t fieldId, RPageStorage *pageStorage)
{
   switch (pageStorage->GetType()) {
   case EPageStorageType::kSink:
      fPageSink = static_cast<RPageSink*>(pageStorage); // the page sink initializes fWritePage on AddColumn
      fHandleSink = fPageSink->AddColumn(fieldId, *this);
      fApproxNElementsPerPage = fPageSink->GetWriteOptions().GetApproxUnzippedPageSize() / fElement->GetSize();
      if (fApproxNElementsPerPage < 2)
         throw RException(R__FAIL("page size too small for writing"));
      // We now have 0 < fApproxNElementsPerPage / 2 < fApproxNElementsPerPage
      fWritePage[0] = fPageSink->ReservePage(fHandleSink, fApproxNElementsPerPage + fApproxNElementsPerPage / 2);
      fWritePage[1] = fPageSink->ReservePage(fHandleSink, fApproxNElementsPerPage + fApproxNElementsPerPage / 2);
      break;
   case EPageStorageType::kSource:
      fPageSource = static_cast<RPageSource*>(pageStorage);
      fHandleSource = fPageSource->AddColumn(fieldId, *this);
      fNElements = fPageSource->GetNElements(fHandleSource);
      fColumnIdSource = fPageSource->GetColumnId(fHandleSource);
      break;
   default:
      R__ASSERT(false);
   }
}

void ROOT::Experimental::Detail::RColumn::Flush()
{
   auto otherIdx = 1 - fWritePageIdx;
   if (fWritePage[fWritePageIdx].IsEmpty() && fWritePage[otherIdx].IsEmpty())
      return;

   if ((fWritePage[fWritePageIdx].GetNElements() < fApproxNElementsPerPage / 2) && !fWritePage[otherIdx].IsEmpty()) {
      // Small tail page: merge with previously used page; we know that there is enough space in the shadow page
      void *dst = fWritePage[otherIdx].GrowUnchecked(fWritePage[fWritePageIdx].GetNElements());
      RColumnElementBase elem(fWritePage[fWritePageIdx].GetBuffer(), fWritePage[fWritePageIdx].GetElementSize());
      elem.WriteTo(dst, fWritePage[fWritePageIdx].GetNElements());
      fWritePage[fWritePageIdx].Reset(0);
      std::swap(fWritePageIdx, otherIdx);
   }

   R__ASSERT(fWritePage[otherIdx].IsEmpty());
   fPageSink->CommitPage(fHandleSink, fWritePage[fWritePageIdx]);
   fWritePage[fWritePageIdx].Reset(fNElements);
}

void ROOT::Experimental::Detail::RColumn::MapPage(const NTupleSize_t index)
{
   fPageSource->ReleasePage(fReadPage);
   fReadPage = fPageSource->PopulatePage(fHandleSource, index);
}

void ROOT::Experimental::Detail::RColumn::MapPage(const RClusterIndex &clusterIndex)
{
   fPageSource->ReleasePage(fReadPage);
   fReadPage = fPageSource->PopulatePage(fHandleSource, clusterIndex);
}

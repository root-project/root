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
   if (!fHeadPage[0].IsNull())
      fPageSink->ReleasePage(fHeadPage[0]);
   if (!fHeadPage[1].IsNull())
      fPageSink->ReleasePage(fHeadPage[1]);
   if (!fCurrentPage.IsNull())
      fPageSource->ReleasePage(fCurrentPage);
   if (fHandleSink)
      fPageSink->DropColumn(fHandleSink);
   if (fHandleSource)
      fPageSource->DropColumn(fHandleSource);
}

void ROOT::Experimental::Detail::RColumn::Connect(DescriptorId_t fieldId, RPageStorage *pageStorage)
{
   switch (pageStorage->GetType()) {
   case EPageStorageType::kSink:
      fPageSink = static_cast<RPageSink*>(pageStorage); // the page sink initializes fHeadPage on AddColumn
      fHandleSink = fPageSink->AddColumn(fieldId, *this);
      fApproxNElementsPerPage = fPageSink->GetWriteOptions().GetApproxUnzippedPageSize() / fElement->GetSize();
      fHeadPage[0] = fPageSink->ReservePage(fHandleSink, fApproxNElementsPerPage + fApproxNElementsPerPage / 2);
      fHeadPage[1] = fPageSink->ReservePage(fHandleSink, fApproxNElementsPerPage + fApproxNElementsPerPage / 2);
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
   auto otherIdx = (fHeadPageIdx + 1) % 2;
   if (fHeadPage[fHeadPageIdx].GetSize() == 0 && fHeadPage[otherIdx].GetSize() == 0)
      return;

   if ((fHeadPage[fHeadPageIdx].GetNElements() < fApproxNElementsPerPage / 2) &&
       fHeadPage[otherIdx].GetNElements())
   {
      // Small tail page: merge with previously used page
      void *dst = fHeadPage[otherIdx].TryGrow(fHeadPage[fHeadPageIdx].GetNElements());
      R__ASSERT(dst != nullptr);
      RColumnElementBase elem(fHeadPage[fHeadPageIdx].GetBuffer(), fHeadPage[fHeadPageIdx].GetElementSize());
      elem.WriteTo(dst, fHeadPage[fHeadPageIdx].GetNElements());
      fHeadPage[fHeadPageIdx].Reset(0);
      fHeadPageIdx = otherIdx;
   } else {
      fHeadPage[otherIdx].Reset(0);
   }
   fPageSink->CommitPage(fHandleSink, fHeadPage[fHeadPageIdx]);
   fHeadPage[fHeadPageIdx].Reset(fNElements);
}

void ROOT::Experimental::Detail::RColumn::MapPage(const NTupleSize_t index)
{
   fPageSource->ReleasePage(fCurrentPage);
   fCurrentPage = fPageSource->PopulatePage(fHandleSource, index);
}

void ROOT::Experimental::Detail::RColumn::MapPage(const RClusterIndex &clusterIndex)
{
   fPageSource->ReleasePage(fCurrentPage);
   fCurrentPage = fPageSource->PopulatePage(fHandleSource, clusterIndex);
}

/// \file RColumn.cxx
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04

/*************************************************************************
 * Copyright (C) 1995-2019, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RColumn.hxx>
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RPageStorage.hxx>

#include <TError.h>

#include <algorithm>
#include <cassert>
#include <utility>

ROOT::Internal::RWriteColumn::~RWriteColumn()
{
   if (fPageSink)
      fPageSink->DropColumn(MakeHandle());
}

void ROOT::Internal::RWriteColumn::ConnectPageSink(ROOT::DescriptorId_t fieldId, RPageSink &pageSink,
                                                   ROOT::NTupleSize_t firstElementIndex)
{
   fInitialNElements = pageSink.GetWriteOptions().GetInitialUnzippedPageSize() / fElement->GetSize();
   if (fInitialNElements < 1) {
      throw RException(R__FAIL("initial page size is too small for at least one element"));
   }

   fPageSink = &pageSink;
   fFirstElementIndex = firstElementIndex;
   fOnDiskId = fPageSink->AddColumn(fieldId, *this).fPhysicalId;
   fWritePage = fPageSink->ReservePage(MakeHandle(), fInitialNElements);
   if (fWritePage.IsNull())
      throw RException(R__FAIL("page buffer memory budget too small"));
}

void ROOT::Internal::RWriteColumn::Flush()
{
   if (fWritePage.GetNElements() == 0)
      return;

   fPageSink->CommitPage(MakeHandle(), fWritePage);
   fWritePage = fPageSink->ReservePage(MakeHandle(), fInitialNElements);
   R__ASSERT(!fWritePage.IsNull());
   fWritePage.Reset(fNElements);
}

void ROOT::Internal::RWriteColumn::CommitSuppressed()
{
   fPageSink->CommitSuppressedColumn(MakeHandle());
}

void ROOT::Internal::RReadColumn::ConnectPageSource(ROOT::DescriptorId_t fieldId, RPageSource &pageSource)
{
   fPageSource = &pageSource;
   fOnDiskId = fPageSource->AddColumn(fieldId, *this).fPhysicalId;
   {
      auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();
      fFirstElementIndex = descriptorGuard->GetColumnDescriptor(fOnDiskId).GetFirstElementIndex();
   }
}

ROOT::Internal::RReadColumn::~RReadColumn()
{
   if (fPageSource)
      fPageSource->DropColumn(MakeHandle());
}

bool ROOT::Internal::RReadColumn::TryMapPage(ROOT::NTupleSize_t globalIndex)
{
   const auto nTeam = fTeam.size();
   std::size_t iTeam = 1;
   do {
      fReadPageRef = fPageSource->LoadPage(fTeam.at(fLastGoodTeamIdx)->MakeHandle(), globalIndex);
      if (!fReadPageRef.Get().IsNull())
         break;
      fLastGoodTeamIdx = (fLastGoodTeamIdx + 1) % nTeam;
      iTeam++;
   } while (iTeam <= nTeam);

   return fReadPageRef.Get().Contains(globalIndex);
}

bool ROOT::Internal::RReadColumn::TryMapPage(RNTupleLocalIndex localIndex)
{
   const auto nTeam = fTeam.size();
   std::size_t iTeam = 1;
   do {
      fReadPageRef = fPageSource->LoadPage(fTeam.at(fLastGoodTeamIdx)->MakeHandle(), localIndex);
      if (!fReadPageRef.Get().IsNull())
         break;
      fLastGoodTeamIdx = (fLastGoodTeamIdx + 1) % nTeam;
      iTeam++;
   } while (iTeam <= nTeam);

   return fReadPageRef.Get().Contains(localIndex);
}

void ROOT::Internal::RReadColumn::MergeTeams(RReadColumn &other)
{
   // We are working on very small vectors here, so quadratic complexity works
   for (auto *c : other.fTeam) {
      if (std::find(fTeam.begin(), fTeam.end(), c) == fTeam.end())
         fTeam.emplace_back(c);
   }

   for (auto c : fTeam) {
      if (c == this)
         continue;
      c->fTeam = fTeam;
   }
}

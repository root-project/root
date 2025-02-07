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
#include <ROOT/RNTupleDescriptor.hxx>
#include <ROOT/RPageStorage.hxx>

#include <TError.h>

#include <algorithm>
#include <cassert>
#include <utility>

ROOT::Experimental::Internal::RColumn::RColumn(ENTupleColumnType type, std::uint32_t columnIndex,
                                               std::uint16_t representationIndex)
   : fType(type), fIndex(columnIndex), fRepresentationIndex(representationIndex), fTeam({this})
{
}

ROOT::Experimental::Internal::RColumn::~RColumn()
{
   if (fHandleSink)
      fPageSink->DropColumn(fHandleSink);
   if (fHandleSource)
      fPageSource->DropColumn(fHandleSource);
}

void ROOT::Experimental::Internal::RColumn::ConnectPageSink(ROOT::DescriptorId_t fieldId, RPageSink &pageSink,
                                                            ROOT::NTupleSize_t firstElementIndex)
{
   fInitialNElements = pageSink.GetWriteOptions().GetInitialUnzippedPageSize() / fElement->GetSize();
   if (fInitialNElements < 1) {
      throw RException(R__FAIL("initial page size is too small for at least one element"));
   }

   fPageSink = &pageSink;
   fFirstElementIndex = firstElementIndex;
   fHandleSink = fPageSink->AddColumn(fieldId, *this);
   fOnDiskId = fPageSink->GetColumnId(fHandleSink);
   fWritePage = fPageSink->ReservePage(fHandleSink, fInitialNElements);
   if (fWritePage.IsNull())
      throw RException(R__FAIL("page buffer memory budget too small"));
}

void ROOT::Experimental::Internal::RColumn::ConnectPageSource(ROOT::DescriptorId_t fieldId, RPageSource &pageSource)
{
   fPageSource = &pageSource;
   fHandleSource = fPageSource->AddColumn(fieldId, *this);
   fNElements = fPageSource->GetNElements(fHandleSource);
   fOnDiskId = fPageSource->GetColumnId(fHandleSource);
   {
      auto descriptorGuard = fPageSource->GetSharedDescriptorGuard();
      fFirstElementIndex = descriptorGuard->GetColumnDescriptor(fOnDiskId).GetFirstElementIndex();
   }
}

void ROOT::Experimental::Internal::RColumn::Flush()
{
   if (fWritePage.GetNElements() == 0)
      return;

   fPageSink->CommitPage(fHandleSink, fWritePage);
   fWritePage = fPageSink->ReservePage(fHandleSink, fInitialNElements);
   R__ASSERT(!fWritePage.IsNull());
   fWritePage.Reset(fNElements);
}

void ROOT::Experimental::Internal::RColumn::CommitSuppressed()
{
   fPageSink->CommitSuppressedColumn(fHandleSink);
}

bool ROOT::Experimental::Internal::RColumn::TryMapPage(ROOT::NTupleSize_t globalIndex)
{
   const auto nTeam = fTeam.size();
   std::size_t iTeam = 1;
   do {
      fReadPageRef = fPageSource->LoadPage(fTeam.at(fLastGoodTeamIdx)->GetHandleSource(), globalIndex);
      if (!fReadPageRef.Get().IsNull())
         break;
      fLastGoodTeamIdx = (fLastGoodTeamIdx + 1) % nTeam;
      iTeam++;
   } while (iTeam <= nTeam);

   return fReadPageRef.Get().Contains(globalIndex);
}

bool ROOT::Experimental::Internal::RColumn::TryMapPage(RNTupleLocalIndex localIndex)
{
   const auto nTeam = fTeam.size();
   std::size_t iTeam = 1;
   do {
      fReadPageRef = fPageSource->LoadPage(fTeam.at(fLastGoodTeamIdx)->GetHandleSource(), localIndex);
      if (!fReadPageRef.Get().IsNull())
         break;
      fLastGoodTeamIdx = (fLastGoodTeamIdx + 1) % nTeam;
      iTeam++;
   } while (iTeam <= nTeam);

   return fReadPageRef.Get().Contains(localIndex);
}

void ROOT::Experimental::Internal::RColumn::MergeTeams(RColumn &other)
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

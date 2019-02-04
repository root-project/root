/// \file RColumn.cxx
/// \ingroup Forest ROOT7
/// \author Jakob Blomer <jblomer@cern.ch>
/// \date 2018-10-04
/// \warning This is part of the ROOT 7 prototype! It will change without notice. It might trigger earthquakes. Feedback
/// is welcome!

/*************************************************************************
 * Copyright (C) 1995-2015, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <ROOT/RColumn.hxx>
#include <ROOT/RColumnModel.hxx>
#include <ROOT/RPageStorage.hxx>

#include <TError.h>


ROOT::Experimental::Detail::RColumn::RColumn(const RColumnModel &model, RPageStorage *pageStorage)
   : fModel(model), fPageSink(nullptr), fPageSource(nullptr), fHeadPage(), fRangeEnd(0)
{
   switch (pageStorage->GetType()) {
   case EPageStorageType::kSink:
      R__ASSERT(fPageSink == nullptr);
      fPageSink = static_cast<RPageSink*>(pageStorage); // the page sink initializes fHeadPage on AddColumn
      break;
   case EPageStorageType::kSource:
      R__ASSERT(fPageSource == nullptr);
      fPageSource = static_cast<RPageSource*>(pageStorage);
      // TODO(jblomer): set fRangeEnd
      break;
   default:
      R__ASSERT(false);
   }

   pageStorage->AddColumn(this);
}

void ROOT::Experimental::Detail::RColumn::Flush()
{
   if (fHeadPage.GetSize() == 0) return;

   fPageSink->CommitPage(fHeadPage, this);
   fHeadPage.Reset(fRangeEnd);
}

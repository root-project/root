// @(#)root/table:$Id$
// Author: Valery Fine(fine@bnl.gov)   13/03/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TColumnView.h"
#include "TVirtualPad.h"
//______________________________________________________________________________
//
//  TColumnView
//
//  It is a helper class to present TTable object view TBrowser
////////////////////////////////////////////////////////////////////////////////

ClassImp(TColumnView);

////////////////////////////////////////////////////////////////////////////////
///constructor

TColumnView::TColumnView(const char *colName,TTable *table):TChair(table)
{
   SetName(colName);
}
////////////////////////////////////////////////////////////////////////////////
///destructor

TColumnView::~TColumnView()
{
}
////////////////////////////////////////////////////////////////////////////////
/// Create a column histogram for the simple column

void TColumnView::Browse(TBrowser *)
{
   if (!IsFolder())
   {
      Draw(GetName(),"");
      if (gPad) {
         gPad->Modified();
         gPad->Update();
      }
   }
}
////////////////////////////////////////////////////////////////////////////////
/// Create a histogram from the context menu

TH1 *TColumnView::Histogram(const char *selection)
{
   TH1 *h = Draw(GetName(),selection);
   if (gPad) {
      gPad->Modified();
      gPad->Update();
   }
   return h;
}

////////////////////////////////////////////////////////////////////////////////
/// Treat the column with the pointer to the "Ptr" as a "folder"

Bool_t  TColumnView::IsFolder() const
{
   Bool_t isFolder = kFALSE;
   const TTable *thisTable = Table();
   if (thisTable) {
      Int_t cIndx = thisTable->GetColumnIndex(GetName());
      if ((thisTable->GetColumnType(cIndx)) == TTable::kPtr ) isFolder = kTRUE;
   }
   return isFolder;
}

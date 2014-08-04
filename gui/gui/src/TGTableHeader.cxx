// Author: Roel Aaij 21/07/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGTableCell.h"
#include "TClass.h"
#include "TGWindow.h"
#include "TGResourcePool.h"
#include "Riostream.h"
#include "TSystem.h"
#include "TImage.h"
#include "TEnv.h"
#include "TGToolTip.h"
#include "TGWidget.h"
#include "TGPicture.h"
#include "TGTable.h"
#include "TRandom3.h"

ClassImp(TGTableHeader)

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTableHeader                                                        //
//                                                                      //
// TGTableHeader is the class that implements a header for a row or     //
// column. Interactivity on a per column or row basis is implemented    //
// using this header.                                                   //
//                                                                      //
// TGTableHeader is for internal use in TGTable only.                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

//______________________________________________________________________________
TGTableHeader::TGTableHeader(const TGWindow *p, TGTable *table, TGString *label,
                             UInt_t position, EHeaderType type, UInt_t width,
                             UInt_t height, GContext_t norm, FontStruct_t font,
                             UInt_t option)
   : TGTableCell(p, table, label, 0, 0, width, height, norm, font, option,
                 kFALSE), fType(type), fReadOnly(kFALSE), fEnabled(kTRUE),
     fHasOwnLabel(kFALSE)
{
   // TGTableHeader constuctor.

   if (type == kColumnHeader) {
      fWidth = (table) ? table->GetTableHeader()->GetWidth() : 80;
      fHeight = 25;
      fRow = 0;
      fColumn = position;
   } else if (type == kRowHeader) {
      fWidth = 80;
      fHeight = (table) ? table->GetTableHeader()->GetHeight() : 25;
      fRow = position;
      fColumn = 0;
   } else {
      fWidth = 80;
      fHeight = 25;
   }

   if (!label) {
      SetDefaultLabel();
   } else {
      fHasOwnLabel = kTRUE;
   }

   Init();
}

//______________________________________________________________________________
TGTableHeader::TGTableHeader(const TGWindow *p, TGTable *table,
                             const char *label, UInt_t position,
                             EHeaderType type, UInt_t width, UInt_t height,
                             GContext_t norm, FontStruct_t font, UInt_t option)
   : TGTableCell(p, table, label, 0, 0, width, height, norm, font,
                 option, kFALSE), fType(type), fReadOnly(kFALSE), fEnabled(kTRUE),
     fHasOwnLabel(kFALSE)
{
   // TGTableHeader constuctor.

   if (type == kColumnHeader) {
      fWidth = table->GetTableHeader()->GetWidth();
      fHeight = 25;
      fRow = 0;
      fColumn = position;
   } else if (type == kRowHeader) {
      fWidth = 80;
      fHeight = table->GetTableHeader()->GetHeight();
      fRow = position;
      fColumn = 0;
   } else {
      fWidth = 80;
      fHeight = 25;
   }

   if (!label) {
      SetDefaultLabel();
   } else {
      fHasOwnLabel = kTRUE;
   }

   Init();
}


//______________________________________________________________________________
TGTableHeader::~TGTableHeader()
{
   // TGTableHeader destructor.
}

//______________________________________________________________________________
void TGTableHeader::Init()
{
   // Initialize the TGTableHeader

   if (fType == kTableHeader) {
      SetBackgroundColor(fTable->GetBackground());
   } else {
      SetBackgroundColor(fTable->GetHeaderBackground());
   }

   Resize(fWidth, fHeight);

   Int_t max_ascent = 0, max_descent = 0;

   fTWidth  = gVirtualX->TextWidth(fFontStruct, fLabel->GetString(), fLabel->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;

}

//______________________________________________________________________________
void TGTableHeader::SetWidth(UInt_t width)
{
   // Resize the TGTableHeader.

   Resize(width, GetDefaultHeight());
}

//______________________________________________________________________________
void TGTableHeader::SetHeight(UInt_t height)
{
   // Resize the TGTableHeader.

   Resize(GetDefaultWidth(), height);
}

//______________________________________________________________________________
void TGTableHeader::SetLabel(const char *label)
{
   // Set the label of the TGTableHeader to label.

   if(label) {
      TGTableCell::SetLabel(label);
   } else {
      SetDefaultLabel();
   }
}

//______________________________________________________________________________
void TGTableHeader::SetDefaultLabel()
{
   // Set the label of the TGTableHeader to the default label, "Row #"
   // or "Col #".

   fHasOwnLabel = kFALSE;
   if (fLabel) delete fLabel;
   fLabel = new TGString();
   if (fType == kRowHeader) {
      *fLabel += "Row ";
      *fLabel += fRow;
   } else if (fType == kColumnHeader) {
      *fLabel += "Col ";
      *fLabel += fColumn;
   } else {
      *fLabel += fTable->GetNTableRows();
      *fLabel += "x";
      *fLabel += fTable->GetNTableColumns();
      *fLabel += " Table";
   }
}

//______________________________________________________________________________
void TGTableHeader::SetPosition(UInt_t pos)
{
   // Set the position of the TGTableHeader to pos.

   // Verify functionality

   if (fType == kRowHeader) {
      fRow = pos;
      fColumn = 0;
   } else if (fType == kColumnHeader) {
      fRow = 0;
      fColumn = pos;
   } else {
      fRow = 0;
      fColumn = 0;
   }
}

//______________________________________________________________________________
void TGTableHeader::Resize(TGDimension newsize)
{
   // Resize the TGTableHeader.

   Resize(newsize.fWidth, newsize.fHeight);
}

//______________________________________________________________________________
void TGTableHeader::Resize(UInt_t width, UInt_t height)
{
   // Resize the TGTableHeader.

   // Implementation of resizing of an entire row of columns probably goes here.
   TGTableCell::Resize(width, height);
}

//______________________________________________________________________________
void TGTableHeader::Sort(Bool_t order)
{
   // Sort the contents of this row or column in given order.

   // Note: not implemented yet.

   if (order == kSortAscending) {
   } else {
   }
}

//______________________________________________________________________________
void TGTableHeader::UpdatePosition()
{
   // Update the positon of the TGTableHeader.

   // Verify functionality. If rows are inserted or removed, internal
   // column numbers are no longer consistent.

   UInt_t nhdr = 0;
   if (fType == kColumnHeader) {
      while(fTable->GetColumnHeader(nhdr) != this) {
         nhdr++;
      }
      fColumn = nhdr;
   } else if (fType == kRowHeader) {
      while(fTable->GetRowHeader(nhdr) != this) {
         nhdr++;
      }
      fRow = nhdr;
   } else {
      fRow = 0;
      fColumn = 0;
   }
}

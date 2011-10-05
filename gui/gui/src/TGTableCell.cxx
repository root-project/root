// Author: Roel Aaij 21/07/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TGFrame.h"
#include "TClass.h"
#include "TGWidget.h"
#include "TGWindow.h"
#include "TGResourcePool.h"
#include "Riostream.h"
#include "TSystem.h"
#include "TImage.h"
#include "TEnv.h"
#include "TGToolTip.h"
#include "TGPicture.h"
#include "TGTable.h"
#include "TVirtualTableInterface.h"
#include "TColor.h"

ClassImp(TGTableCell)

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTableCell                                                          //
//                                                                      //
// TGTableCell is the class that represents a single cell in a TGTable. //
//                                                                      //
// This class is for internal use in TGTable only.                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

const TGGC *TGTableCell::fgDefaultGC = 0;
const TGFont *TGTableCell::fgDefaultFont = 0;

//______________________________________________________________________________
TGTableCell::TGTableCell(const TGWindow *p, TGTable *table, TGString *label,
                         UInt_t row, UInt_t column, UInt_t width, UInt_t height, 
                         GContext_t norm, FontStruct_t font, UInt_t option, 
                         Bool_t resize) 
   : TGFrame(p, width, height, option), fTip(0), fReadOnly(kFALSE), 
     fEnabled(kTRUE), fTMode(kTextRight | kTextCenterY), fImage(0), 
     fFontStruct(font), fHasOwnFont(kFALSE), fColumn(column), fRow(row), 
     fTable(table)
{
   // TGTableCell constructor.
   
   if (label) {
      fLabel = label;
   } else {
      fLabel = new TGString("0");
   }
   
   fNormGC = norm;
   Init(resize);
}

//______________________________________________________________________________
TGTableCell::TGTableCell(const TGWindow *p, TGTable *table, const char *label, 
                         UInt_t row, UInt_t column, UInt_t width, UInt_t height,
                         GContext_t norm, FontStruct_t font, UInt_t option, 
                         Bool_t resize) 
   : TGFrame(p, width, height, option), fTip(0), fReadOnly(kFALSE), 
     fEnabled(kTRUE), fTMode(kTextRight | kTextCenterY), fImage(0), 
     fFontStruct(font), fHasOwnFont(kFALSE), fColumn(column), fRow(row), 
     fTable(table)
{
   // TGTableCell constructor
   
   if (label) {
      fLabel = new TGString(label);
   } else {
      fLabel = new TGString("0");
   }

   fNormGC = norm;
   Init(resize);
}

// //______________________________________________________________________________
// TGTableCell::TGTableCell(const TGWindow *p, TGTable *table, TGPicture *image, 
//                          TGString *label, UInt_t row, UInt_t column, 
//                          GContext_t norm, FontStruct_t font, UInt_t option, 
//                          Bool_t resize)
//    : TGFrame(p, 80, 25, option), fTip(0), fReadOnly(kFALSE), fEnabled(kTRUE), 
//      fTMode(kTextRight | kTextCenterY), fImage(image), fFontStruct(font), 
//      fHasOwnFont(kFALSE), fColumn(column), fRow(row), fTable(table)
// {
//    if (label) {
//       fLabel = label;
//    } else {
//       fLabel = new TGString("0");
//    }
//    fNormGC = norm;

//    Init(resize);
// }

// //______________________________________________________________________________
// TGTableCell::TGTableCell(const TGWindow *p, TGTable *table, TGPicture *image, 
//                          const char *label, UInt_t row, UInt_t column, 
//                          GContext_t norm, FontStruct_t font, UInt_t option,
//                          Bool_t resize) 
//    : TGFrame(p, 80, 25, option), fTip(0), fReadOnly(kFALSE), fEnabled(kTRUE), 
//      fTMode(kTextRight | kTextCenterY), fImage(image), fFontStruct(font), 
//      fHasOwnFont(kFALSE), fColumn(column), fRow(row), fTable(table)
// {
   
//    if (label) {
//       fLabel = new TGString(label);
//    } else {
//       fLabel = new TGString("0");
//    }
   
//    fNormGC = norm;

//    Init(resize);
// }


//______________________________________________________________________________
TGTableCell::~TGTableCell()
{
   // TGTableCell destructor.

   if (fImage) delete fImage;
   if (fLabel) delete fLabel;
   if (fTip) delete fTip;
}

//______________________________________________________________________________
void TGTableCell::Init(Bool_t resize)
{
   // Initialise the TGTableCell.

   Int_t max_ascent = 0, max_descent = 0;
   
   fTWidth  = gVirtualX->TextWidth(fFontStruct, fLabel->GetString(), fLabel->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;

   // Modifications for individual cell drawing test, original block is marked
   
   if (fTable) {
      // Original from here
      TGTableHeader *chdr = 0;
      TGTableHeader *rhdr = 0;
      if(resize) {
         chdr = fTable->GetColumnHeader(fColumn);
         rhdr = fTable->GetRowHeader(fRow);
         if (rhdr) {
            SetBackgroundColor(rhdr->GetBackground());
            if (chdr) Resize(chdr->GetWidth(), rhdr->GetHeight());
         }
      }
      SetBackgroundColor(fTable->GetRowBackground(fRow));
      // Upto here
   } else {
      fWidth = 80;
      fHeight = 25;
      Resize(fWidth, fHeight);
      SetBackgroundColor(fgWhitePixel);
   }
   // End of modifications

}

//______________________________________________________________________________
void TGTableCell::DoRedraw()
{
   // Redraw the TGTableCell.

   TGFrame::DoRedraw();

   Int_t x = 0, y = 0;

   // To be done: Add a tooltip with the complete label when it
   // doesn't fit in the cell.
   if (fTWidth > fWidth - 4) fTMode = kTextLeft;
   
   if (fTMode & kTextLeft) {
      x = 4;
   } else if (fTMode & kTextRight) {
      x = fWidth - fTWidth - 4;
   } else {
      x = (fWidth - fTWidth) / 2;
   }

   if (fTMode & kTextTop) {
      y = 3;
   } else if (fTMode & kTextBottom) {
      y = fHeight - fTHeight - 3;
   } else {
      y = (fHeight - fTHeight - 4) / 2;
   }
   
   y += fTHeight;

   fLabel->Draw(fId, fNormGC, x, y);
}

//______________________________________________________________________________
void TGTableCell::MoveDraw(Int_t x, Int_t y)
{
   // Move the TGTableCell and redraw it.

   // Note, this method is unused.

   TGFrame::Move(x, y);
   DoRedraw();
}

//______________________________________________________________________________
void TGTableCell::Resize(UInt_t width, UInt_t height)
{
   // Resize the TGTableCell.

   fWidth = width;
   fHeight = height;
   TGFrame::Resize(width, height);
   Layout();
}

//______________________________________________________________________________
void TGTableCell::Resize(TGDimension newsize)
{
   // Resize the TGTableCell.

   Resize(newsize.fWidth, newsize.fHeight);
}

//______________________________________________________________________________
FontStruct_t TGTableCell::GetDefaultFontStruct()
{
   // Return default font structure.

   if (!fgDefaultFont)
      fgDefaultFont = gClient->GetResourcePool()->GetDefaultFont();
   return fgDefaultFont->GetFontStruct();
}

//______________________________________________________________________________
const TGGC &TGTableCell::GetDefaultGC()
{
   // Return default graphics context.

   if (!fgDefaultGC)
      fgDefaultGC = gClient->GetResourcePool()->GetFrameGC();
   return *fgDefaultGC;
}

//______________________________________________________________________________
void TGTableCell::DrawBorder()
{
   // Draw the cell border.

   // FIXME Borders are drawn very crudely. There is much room for improvement.
   gVirtualX->DrawRectangle(fId, fNormGC, 0, 0, fWidth - 1, fHeight - 1);
}

//______________________________________________________________________________
void TGTableCell::DrawBorder(Handle_t id, Int_t x, Int_t y)
{
   // DrawBorder called from DrawCopy.

   gVirtualX->DrawRectangle(id, fNormGC, x, y, x + fWidth - 1, y +fHeight - 1);
}

//______________________________________________________________________________
void TGTableCell::Highlight()
{
   // Highlight the this TGTableCell.

   // Currently not implemented.
}

// //______________________________________________________________________________
// void TGTableCell::SetRow(UInt_t row) 
// {
//    fRow = row;
// }

// //______________________________________________________________________________
// void TGTableCell::SetColumn(UInt_t column)
// {
//    fColumn = column;
// }

//______________________________________________________________________________
void TGTableCell::DrawCopy(Handle_t id, Int_t x, Int_t y)
{
   // Draw list view item in other window.
   // List view item is placed and layout in the container frame,
   // but is drawn in viewport.
   
   // FIXME this method is only needed if the table container frame is a 
   // TGContainer. It is broken and not used in the current implementation.

   Int_t lx = 0, ly = 0;

   if (fTMode & kTextLeft) {
      lx = 4;
   } else if (fTMode & kTextRight) {
      lx = fWidth - fTWidth - 4;
   } else {
      lx = (fWidth - fTWidth) / 2;
   }

   if (fTMode & kTextTop) {
      ly = 3;
   } else if (fTMode & kTextBottom) {
      ly = fHeight - fTHeight - 3;
   } else {
      ly = (fHeight - fTHeight - 4) / 2;
   }
   
   ly += fTHeight;
   
   //    if (fActive) {
   //       gVirtualX->SetForeground(fNormGC, fgDefaultSelectedBackground);
   //       gVirtualX->FillRectangle(id, fNormGC, x + lx, y + ly, fTWidth, fTHeight + 1);
   //       gVirtualX->SetForeground(fNormGC, fClient->GetResourcePool()->GetSelectedFgndColor());
   //    } else {

   gVirtualX->SetForeground(fNormGC, fgWhitePixel);
   gVirtualX->FillRectangle(id, fNormGC, x, y, fWidth, fHeight);
   gVirtualX->SetForeground(fNormGC, fgBlackPixel);
   DrawBorder(id, x, y);

   //    }
   
   fLabel->Draw(id, fNormGC, x + lx, y + ly);
}

//______________________________________________________________________________
void TGTableCell::SetLabel(const char *label)
{
   // Set the label of this cell to label.

   fLabel->SetString(label);

   Int_t max_ascent = 0, max_descent = 0;

   fTWidth  = gVirtualX->TextWidth(fFontStruct, fLabel->GetString(), fLabel->GetLength());
   gVirtualX->GetFontProperties(fFontStruct, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;

}

//______________________________________________________________________________
void TGTableCell::SetImage(TGPicture *image)
{
   // Set the image that this cell contains to image.

   // Note: currently not used.
   if (fImage) delete fImage;
   fImage = image;
}

// //______________________________________________________________________________
// void TGTableCell::SetBckgndGC(TGGC *gc)
// {
// }

//______________________________________________________________________________
void TGTableCell::SetFont(FontStruct_t font)
{
   // Changes text font.
   // If global is kTRUE font is changed globally, otherwise - locally.

   if (font != fFontStruct) {
      FontH_t v = gVirtualX->GetFontHandle(font);
      if (!v) return;

      fFontStruct = font;
      TGGCPool *pool =  fClient->GetResourcePool()->GetGCPool();
      TGGC *gc = pool->FindGC(fNormGC);

      gc = pool->GetGC((GCValues_t*)gc->GetAttributes(), kTRUE); // copy
      fHasOwnFont = kTRUE;

      gc->SetFont(v);

      fNormGC = gc->GetGC();
      gClient->NeedRedraw(this);
   }
}

//______________________________________________________________________________
void TGTableCell::SetFont(const char *fontName)
{
   // Changes text font specified by name.
   // If global is true color is changed globally, otherwise - locally.

   TGFont *font = fClient->GetFont(fontName);
   if (font) {
      SetFont(font->GetFontStruct());
   }
}

//______________________________________________________________________________
void TGTableCell::SetTextJustify(Int_t tmode)
{
   // Set the text justify mode of the cell to mode.

   fTMode = tmode;
}

//______________________________________________________________________________
void TGTableCell::Select()
{
   // Select this TGTableCell.

   // Note: currently not implemented.
}

//______________________________________________________________________________
void TGTableCell::SelectRow()
{
   // Select the row that this TGTableCell belongs to.

   // Note: currently not implemented.
}

//______________________________________________________________________________
void TGTableCell::SelectColumn()
{
   // Select the column that this TGTableCell belongs to.

   // Note: currently not implemented.
}

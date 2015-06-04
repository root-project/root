// @(#)root/gui:$Id$
// Author: Fons Rademakers   1/7/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
/**************************************************************************

    This source is based on Xclass95, a Win95-looking GUI toolkit.
    Copyright (C) 1996, 1997 David Barth, Ricky Ralston, Hector Peraza.

    Xclass95 is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

**************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTextView                                                           //
//                                                                      //
// A TGTextView is a text viewer widget. It is a specialization of      //
// TGView. It uses the TGText class (which contains all text            //
// manipulation code, i.e. loading a file in memory, changing,          //
// removing lines, etc.). Use a TGTextView to view non-editable text.   //
// For supported messages see TGView.                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGTextView.h"
#include "TGScrollBar.h"
#include "TGResourcePool.h"
#include "TSystem.h"
#include "TGDNDManager.h"
#include "TBufferFile.h"
#include "TSystemFile.h"
#include "TObjString.h"
#include "TMacro.h"
#include "TGMsgBox.h"
#include "TUrl.h"
#include "Riostream.h"


const TGFont *TGTextView::fgDefaultFont = 0;
TGGC         *TGTextView::fgDefaultGC = 0;
TGGC         *TGTextView::fgDefaultSelectedGC = 0;
const TGGC   *TGTextView::fgDefaultSelectedBackgroundGC = 0;


//______________________________________________________________________________
Bool_t TViewTimer::Notify()
{
   // Notify when timer times out and reset the timer.

   fView->HandleTimer(this);
   Reset();
   return kFALSE;
}

ClassImp(TGTextView)


//______________________________________________________________________________
void TGTextView::Init(ULong_t back)
{
   // Initialize a text view widget.

   // set in TGResourcePool via .rootrc
   fFont        = GetDefaultFontStruct();
   fNormGC      = GetDefaultGC();
   fSelGC       = GetDefaultSelectedGC();
   fSelbackGC   = GetDefaultSelectedBackgroundGC();

   fWhiteGC = *fClient->GetResourcePool()->GetDocumentBckgndGC();
   fWhiteGC.SetGraphicsExposures(kTRUE);
   fWhiteGC.SetBackground(back);
   fWhiteGC.SetForeground(back);

   fMarkedFromX = 0;
   fMarkedFromY = 0;
   fReadOnly    = kFALSE;
   fIsMarked    = kFALSE;

   fText = new TGText();
   TGView::Clear();

   fClipText = new TGText();

   gVirtualX->GetFontProperties(fFont, fMaxAscent, fMaxDescent);
   fScrollVal.fY = fMaxAscent + fMaxDescent;
   fScrollVal.fX = fMaxWidth = gVirtualX->TextWidth(fFont, "@", 1);

   fScrollTimer = new TViewTimer(this, 75);
   gSystem->AddTimer(fScrollTimer);

   // define DND types
   fDNDTypeList = new Atom_t[3];
   fDNDTypeList[0] = gVirtualX->InternAtom("application/root", kFALSE);
   fDNDTypeList[1] = gVirtualX->InternAtom("text/uri-list", kFALSE);
   fDNDTypeList[2] = 0;
   gVirtualX->SetDNDAware(fId, fDNDTypeList);
   SetDNDTarget(kTRUE);

   gVirtualX->ClearWindow(fCanvas->GetId());
   Layout();
}

//______________________________________________________________________________
TGTextView::TGTextView(const TGWindow *parent, UInt_t w, UInt_t h, Int_t id,
                       UInt_t sboptions, ULong_t back) :
     TGView(parent, w, h, id, 3, 3, kSunkenFrame | kDoubleBorder, sboptions, back)
{
   // Create a text view widget.

   Init(back);
}

//______________________________________________________________________________
TGTextView::TGTextView(const TGWindow *parent, UInt_t w, UInt_t h, TGText *text,
                       Int_t id, UInt_t sboptions, ULong_t back) :
     TGView(parent, w, h, id, 3, 3, kSunkenFrame | kDoubleBorder, sboptions, back)
{
   // Create a text view widget.

   Init(back);
   TGLongPosition pos, srcStart, srcEnd;
   pos.fX = pos.fY = 0;
   srcStart.fX = srcStart.fY = 0;
   srcEnd.fY = text->RowCount()-1;
   srcEnd.fX = text->GetLineLength(srcEnd.fY)-1;
   fText->InsText(pos, text, srcStart, srcEnd);
}

//______________________________________________________________________________
TGTextView::TGTextView(const TGWindow *parent, UInt_t w, UInt_t h,
                       const char *string, Int_t id, UInt_t sboptions,
                       ULong_t back) :
     TGView(parent, w, h, id, 3, 3, kSunkenFrame | kDoubleBorder, sboptions, back)
{
   // Create a text view widget.

   Init(back);
   TGLongPosition pos;
   pos.fX = pos.fY = 0;
   fText->InsText(pos, string);
}

//______________________________________________________________________________
TGTextView::~TGTextView()
{
   // Cleanup text view widget.

   delete fScrollTimer;
   delete fText;
   delete fClipText;
   delete [] fDNDTypeList;
}

//______________________________________________________________________________
void TGTextView::SetBackground(Pixel_t p)
{
   // set background  color

   fCanvas->SetBackgroundColor(p);
   fWhiteGC.SetBackground(p);
   fWhiteGC.SetForeground(p);
}

//______________________________________________________________________________
void TGTextView::SetSelectBack(Pixel_t p)
{
   // set selected text background color

   fSelbackGC.SetBackground(p);
   fSelbackGC.SetForeground(p);
}

//______________________________________________________________________________
void TGTextView::SetSelectFore(Pixel_t p)
{
   // set selected text color 

   fSelGC.SetBackground(p);
   fSelGC.SetForeground(p);
}

//______________________________________________________________________________
void TGTextView::SetText(TGText *text)
{
   // Adopt a new text buffer. The text will be deleted by this object.

   Clear();
   delete fText;
   fText = text;
   Layout();
}

//______________________________________________________________________________
void TGTextView::AddText(TGText *text)
{
   // Add text to the view widget.

   UInt_t h1 = (UInt_t)ToScrYCoord(fText->RowCount());

   fText->AddText(text);
   Layout();

   UInt_t h2 = (UInt_t)ToScrYCoord(fText->RowCount());

   if (h2 <= h1) {
      return;
   }

   if (h2 < fCanvas->GetHeight()) {
      UpdateRegion(0, h1, fCanvas->GetWidth(), h2 - h1);
   }
}

//______________________________________________________________________________
void TGTextView::AddLine(const char *string)
{
   // Add a line of text to the view widget.

   UInt_t h1 = (UInt_t)ToScrYCoord(fText->RowCount());

   AddLineFast(string);
   Layout();

   UInt_t h2 = (UInt_t)ToScrYCoord(fText->RowCount());

   if (h2 <= h1) {
      return;
   }
   if (h2 < fCanvas->GetHeight()) {
      UpdateRegion(0, h1, fCanvas->GetWidth(), h2 - h1);
   }
}

//______________________________________________________________________________
void TGTextView::AddLineFast(const char *string)
{
   // Add a line of text to the view widget.
   // Fast version. Use it if you are going to add
   // several lines, than call Update().

   TGLongPosition pos;
   pos.fX = 0;
   pos.fY = fText->RowCount();
   fText->InsText(pos, string);
}

//______________________________________________________________________________
void TGTextView::Update()
{
   // update the whole window of text view

   Layout();
   fExposedRegion.Empty();
   UpdateRegion(0, 0, fCanvas->GetWidth(), fCanvas->GetHeight());
}

//______________________________________________________________________________
Long_t TGTextView::ReturnLongestLineWidth()
{
   // Return width of longest line.

   Long_t count = 0, longest = 0, width;
   Long_t rows = fText->RowCount();
   while (count < rows) {
      width = ToScrXCoord(fText->GetLineLength(count), count) + fVisible.fX;
      if (width > longest) {
         longest = width;
      }
      count++;
   }
   return longest;
}

//______________________________________________________________________________
Bool_t TGTextView::Search(const char *string, Bool_t direction, Bool_t caseSensitive)
{
   // Search for string in text. If direction is true search forward.
   // Returns true if string is found.

   TGLongPosition pos, pos2;
   pos2.fX = pos2.fY = 0;
   if (fIsMarked) {
      if (!direction) {
         pos2.fX = fMarkedStart.fX;
         pos2.fY = fMarkedStart.fY;
      } else {
         pos2.fX = fMarkedEnd.fX + 1;
         pos2.fY = fMarkedEnd.fY;
      }
   }
   if (!fText->Search(&pos, pos2, string, direction, caseSensitive)) {
      return kFALSE;
   }
   UnMark();
   fIsMarked = kTRUE;
   fMarkedStart.fY = fMarkedEnd.fY = pos.fY;
   fMarkedStart.fX = pos.fX;
   fMarkedEnd.fX = fMarkedStart.fX + strlen(string) - 1;
   pos.fY = ToObjYCoord(fVisible.fY);
   if ((fMarkedStart.fY < pos.fY) ||
       (ToScrYCoord(fMarkedStart.fY) >= (Int_t)fCanvas->GetHeight())) {
      pos.fY = fMarkedStart.fY;
   }
   pos.fX = ToObjXCoord(fVisible.fX, pos.fY);
   if ((fMarkedStart.fX < pos.fX) ||
       (ToScrXCoord(fMarkedStart.fX, pos.fY) >= (Int_t)fCanvas->GetWidth())) {
      pos.fX = fMarkedStart.fX;
   }

   SetVsbPosition((ToScrYCoord(pos.fY) + fVisible.fY)/fScrollVal.fY);
   SetHsbPosition((ToScrXCoord(pos.fX, pos.fY) + fVisible.fX)/fScrollVal.fX);
   UpdateRegion(0, (Int_t)ToScrYCoord(fMarkedStart.fY), fCanvas->GetWidth(),
              UInt_t(ToScrYCoord(fMarkedEnd.fY+1) - ToScrYCoord(fMarkedEnd.fY)));

   return kTRUE;
}

//______________________________________________________________________________
void TGTextView::SetFont(FontStruct_t font)
{
   // Changes text entry font.

   if (font != fFont) {
      fFont = font;
      fNormGC.SetFont(gVirtualX->GetFontHandle(fFont));
      fSelGC.SetFont(gVirtualX->GetFontHandle(fFont));
      fClient->NeedRedraw(this);
   }
}

//______________________________________________________________________________
Long_t TGTextView::ToScrYCoord(Long_t yCoord)
{
   // Convert line number to screen coordinate.

   if (yCoord * (fMaxAscent + fMaxDescent) <= 0) {
      return 0;
   }
   if (yCoord > fText->RowCount()) {
      return fText->RowCount() * (fMaxAscent + fMaxDescent);
   }
   return yCoord * (fMaxAscent + fMaxDescent) - fVisible.fY;
}

//______________________________________________________________________________
Long_t TGTextView::ToScrXCoord(Long_t xCoord, Long_t line)
{
   // Convert column number in specified line to screen coordinate.

   TGLongPosition pos;
   char *buffer;

   pos.fX = 0;
   pos.fY = line;
   Long_t width = fText->GetLineLength(line);
   if (xCoord <= 0 || pos.fY < 0 || width <= 0) {
      return 0;
   }
   if (xCoord > width) {
      xCoord = width;
   }
   buffer = fText->GetLine(pos, xCoord);
   width = gVirtualX->TextWidth(fFont, buffer, (Int_t)xCoord) - fVisible.fX;
   delete [] buffer;

   return width;
}

//______________________________________________________________________________
Long_t TGTextView::ToObjYCoord(Long_t yCoord)
{
   // Convert y screen coordinate to line number.

   return  yCoord / (fMaxAscent + fMaxDescent);
}

//______________________________________________________________________________
Long_t TGTextView::ToObjXCoord(Long_t xCoord, Long_t line)
{
   // Convert x screen coordinate to column in specified line.

   TGLongPosition pos;
   char *buffer, *travelBuffer;
   char charBuffer;

   if (line < 0 || line >= fText->RowCount()) {
      return 0;
   }

   Long_t len = fText->GetLineLength(line);
   pos.fX = 0;
   pos.fY = line;
   if (len <= 0 || xCoord < 0) {
      return 0;
   }

   Long_t viscoord =  xCoord;
   buffer = fText->GetLine(pos, len);
   if (!buffer) return 0;
   travelBuffer = buffer;
   charBuffer = *travelBuffer++;
   int cw = gVirtualX->TextWidth(fFont, &charBuffer, 1);

   while (viscoord - cw >= 0 && pos.fX < len) {
      viscoord -= cw;
      pos.fX++;
      charBuffer = *travelBuffer++;
      cw = gVirtualX->TextWidth(fFont, &charBuffer, 1);
   }

   delete [] buffer;
   return pos.fX;
}

//______________________________________________________________________________
void TGTextView::Clear(Option_t *)
{
   // Clear text view widget.

   TGView::Clear();
   fIsMarked  = kFALSE;
   fIsSaved   = kTRUE;
   fMarkedStart.fX = fMarkedStart.fY = 0;
   fMarkedEnd.fX   = fMarkedEnd.fY   = 0;
   fIsMarking = kFALSE;

   delete fText;
   fText = new TGText();
   fText->Clear();
   SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_ISMARKED), fWidgetId, kFALSE);
   Marked(kFALSE);
   gVirtualX->ClearWindow(fCanvas->GetId());
   SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_DATACHANGE), fWidgetId, 0);
   DataChanged();
   Layout();
}

//______________________________________________________________________________
Bool_t TGTextView::LoadFile(const char *filename, Long_t startpos, Long_t length)
{
   // Load a file in the text view widget. Return false in case file does not
   // exist.

   FILE *fp;
   if (!(fp = fopen(filename, "r")))
      return kFALSE;
   fclose(fp);

   ShowTop();
   Clear();
   fText->Load(filename, startpos, length);
   Update();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextView::LoadBuffer(const char *txtbuf)
{
   // Load text from a text buffer. Return false in case of failure.

   if (!txtbuf || !strlen(txtbuf)) {
      return kFALSE;
   }

   Clear();
   fText->LoadBuffer(txtbuf);
   Update();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextView::Copy()
{
   // Copy selected text to clipboard.

   TGLongPosition insPos, startPos, endPos;

   if (!fIsMarked) {
      return kFALSE;
   }
   delete fClipText;
   fClipText   = new TGText();
   insPos.fY   = insPos.fX = 0;
   startPos.fX = fMarkedStart.fX;
   startPos.fY = fMarkedStart.fY;
   endPos.fX   = fMarkedEnd.fX-1;
   endPos.fY   = fMarkedEnd.fY;
   if (endPos.fX == -1) {
      if (endPos.fY > 0) {
         endPos.fY--;
      }
      endPos.fX = fText->GetLineLength(endPos.fY);
      if (endPos.fX < 0) {
         endPos.fX = 0;
      }
   }
   fClipText->InsText(insPos, fText, startPos, endPos);
   gVirtualX->SetPrimarySelectionOwner(fId);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextView::SelectAll()
{
   // Select all text in the viewer.

   if (fText->RowCount() == 1 && fText->GetLineLength(0) == 0) {
      return kFALSE;
   }
   fIsMarked = kTRUE;
   fMarkedStart.fY = 0;
   fMarkedStart.fX = 0;
   fMarkedEnd.fY = fText->RowCount()-1;
   fMarkedEnd.fX = fText->GetLineLength(fMarkedEnd.fY);
   if (fMarkedEnd.fX < 0) {
      fMarkedEnd.fX = 0;
   }
   UpdateRegion(0, 0, fCanvas->GetWidth(), fCanvas->GetHeight());
   Copy();

   return kTRUE;
}

//______________________________________________________________________________
void TGTextView::DrawRegion(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Draw lines in exposed region.

   char *buffer;

   TGLongPosition pos;
   Long_t xoffset, len, len1, len2;
   Long_t line_count = fText->RowCount();
   Rectangle_t rect;
   rect.fX = x;
   rect.fY = y;
   pos.fY = ToObjYCoord(fVisible.fY + h);
   rect.fHeight = UShort_t(h + ToScrYCoord(pos.fY + 1) - ToScrYCoord(pos.fY));
   pos.fX = ToObjXCoord(fVisible.fX + w, pos.fY);
   rect.fWidth = UShort_t(w + ToScrXCoord(pos.fX + 1, pos.fY) - ToScrXCoord(pos.fX, pos.fY));
   Int_t yloc = rect.fY + (Int_t)fScrollVal.fY;
   pos.fY = ToObjYCoord(fVisible.fY + rect.fY);

   while (pos.fY <= line_count &&
          yloc - fScrollVal.fY <= (Int_t)fCanvas->GetHeight() &&
          yloc <= rect.fY + rect.fHeight) {

      pos.fX = ToObjXCoord(fVisible.fX + rect.fX, pos.fY);
      xoffset = ToScrXCoord(pos.fX, pos.fY);
      len = fText->GetLineLength(pos.fY) - pos.fX;

      gVirtualX->ClearArea(fCanvas->GetId(), x, Int_t(ToScrYCoord(pos.fY)),
                           rect.fWidth, UInt_t(ToScrYCoord(pos.fY+1)-ToScrYCoord(pos.fY)));


      if (len > 0) {
         if (len > ToObjXCoord(fVisible.fX + rect.fX + rect.fWidth, pos.fY) - pos.fX) {
            len = ToObjXCoord(fVisible.fX + rect.fX + rect.fWidth, pos.fY) - pos.fX + 1;
         }
         if (pos.fX == 0) {
            xoffset = -fVisible.fX;
         }
         if (pos.fY >= ToObjYCoord(fVisible.fY)) {
            buffer = fText->GetLine(pos, len);
            if (!buffer) // skip next lines and continue the while() loop
               continue;
            Int_t i = 0;
            while (buffer[i] != '\0') {
               if (buffer[i] == '\t') {
                  buffer[i] = ' ';
                  Int_t j = i+1;
                  while (buffer[j] == 16 && buffer[j] != '\0') {
                     buffer[j++] = ' ';
                  }
               }
               i++;
            }

            if (!fIsMarked ||
                pos.fY < fMarkedStart.fY || pos.fY > fMarkedEnd.fY ||
               (pos.fY == fMarkedStart.fY &&
                fMarkedStart.fX >= pos.fX+len &&
                fMarkedStart.fY != fMarkedEnd.fY) ||
               (pos.fY == fMarkedEnd.fY &&
                fMarkedEnd.fX < pos.fX &&
                fMarkedStart.fY != fMarkedEnd.fY) ||
               (fMarkedStart.fY == fMarkedEnd.fY &&
                (fMarkedEnd.fX < pos.fX ||
                 fMarkedStart.fX > pos.fX+len))) {

               gVirtualX->DrawString(fCanvas->GetId(), fNormGC(), Int_t(xoffset),
                                     Int_t(ToScrYCoord(pos.fY+1) - fMaxDescent),
                                     buffer, Int_t(len));
            } else {
               if (pos.fY > fMarkedStart.fY && pos.fY < fMarkedEnd.fY) {
                  len1 = 0;
                  len2 = len;
               } else {
                  if (fMarkedStart.fY == fMarkedEnd.fY) {
                     if (fMarkedStart.fX >= pos.fX &&
                         fMarkedStart.fX <= pos.fX + len) {
                        len1 = fMarkedStart.fX - pos.fX;
                     } else {
                        len1 = 0;
                     }
                     if (fMarkedEnd.fX >= pos.fX &&
                         fMarkedEnd.fX <= pos.fX + len) {
                        len2 = fMarkedEnd.fX - pos.fX - len1;  // +1
                     } else {
                        len2 = len - len1;
                     }
                  } else {
                     if (pos.fY == fMarkedStart.fY) {
                        if (fMarkedStart.fX < pos.fX) {
                           len1 = 0;
                           len2 = len;
                        } else {
                           len1 = fMarkedStart.fX - pos.fX;
                           len2 = len - len1;
                        }
                     } else {
                        if (fMarkedEnd.fX > pos.fX+len) {
                           len1 = 0;
                           len2 = len;
                        } else {
                           len1 = 0 ;
                           len2 = fMarkedEnd.fX - pos.fX;  // +1
                        }
                     }
                  }
               }
               gVirtualX->DrawString(fCanvas->GetId(), fNormGC(),
                                     Int_t(ToScrXCoord(pos.fX, pos.fY)),
                                     Int_t(ToScrYCoord(pos.fY+1) - fMaxDescent),
                                     buffer, Int_t(len1));
               gVirtualX->FillRectangle(fCanvas->GetId(), fSelbackGC(),
                                     Int_t(ToScrXCoord(pos.fX+len1, pos.fY)),
                                     Int_t(ToScrYCoord(pos.fY)),
                                     UInt_t(ToScrXCoord(pos.fX+len1+len2, pos.fY) -
                                     ToScrXCoord(pos.fX+len1, pos.fY)),
                                     UInt_t(ToScrYCoord(pos.fY+1)-ToScrYCoord(pos.fY)));
               gVirtualX->DrawString(fCanvas->GetId(), fSelGC(),
                                     Int_t(ToScrXCoord(pos.fX+len1, pos.fY)),
                                     Int_t(ToScrYCoord(pos.fY+1) - fMaxDescent),
                                     buffer+len1, Int_t(len2));
               gVirtualX->DrawString(fCanvas->GetId(), fNormGC(),
                                     Int_t(ToScrXCoord(pos.fX+len1+len2, pos.fY)),
                                     Int_t(ToScrYCoord(pos.fY+1) - fMaxDescent),
                                     buffer+len1+len2, Int_t(len-(len1+len2)));
            }
            delete [] buffer;
         }
      }
      pos.fY++;
      yloc += Int_t(ToScrYCoord(pos.fY) - ToScrYCoord(pos.fY-1));
   }
}

//______________________________________________________________________________
Bool_t TGTextView::HandleCrossing(Event_t *event)
{
   // Handle mouse crossing event.

   if (event->fWindow != fCanvas->GetId())
      return kTRUE;

   fMousePos.fY = ToObjYCoord(fVisible.fY + event->fY);
   if (ToScrYCoord(fMousePos.fY+1) >= (Int_t)fCanvas->GetHeight()) {
      fMousePos.fY--;
   }
   fMousePos.fX = ToObjXCoord(fVisible.fX + event->fX, fMousePos.fY);
   if (fMousePos.fX >= ReturnLineLength(fMousePos.fY)) {
      fMousePos.fX--;
   }
   if ((event->fState & kButton1Mask) && fIsMarked && fIsMarking) {
      if (event->fType == kLeaveNotify) {
         if (event->fX < 0) {
            fScrolling = 0;
            return kFALSE;
         }
         if (event->fX >= (Int_t)fCanvas->GetWidth()) {
            fScrolling = 1;
            return kFALSE;
         }
         if (event->fY < 0) {
            fScrolling = 2;
            return kFALSE;
         }
         if (event->fY >= (Int_t)fCanvas->GetHeight()) {
            fScrolling = 3;
            return kFALSE;
         }
      } else {
         fScrolling = -1;
         Mark(fMousePos.fX, fMousePos.fY);
      }
   } else {
      fIsMarking = kFALSE;
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextView::HandleTimer(TTimer *)
{
   // Handle scroll timer.

   static const Int_t kAutoScrollFudge = 10;
   static const Int_t kAcceleration[kAutoScrollFudge + 1] = {1, 1, 1, 1, 2, 3, 4, 6, 8, 12, 16};

   TGLongPosition size;
   Window_t  dum1, dum2;
   Event_t   ev;
   ev.fType = kButtonPress;
   Int_t x, y;
   Int_t dy = 0;

   if (fMarkedStart.fY == fMarkedEnd.fY) {
      return kFALSE;
   }
   if (fIsMarked && (fScrolling != -1)) {
      // where cursor
      gVirtualX->QueryPointer(fId, dum1, dum2, ev.fXRoot, ev.fYRoot, x, y, ev.fState);

      fMousePos.fY = ToObjYCoord(fVisible.fY + y);

      if (fMousePos.fY >= ReturnLineCount()) {
         fMousePos.fY = ReturnLineCount() - 1;
      }
      if (fMousePos.fY < 0) {
         fMousePos.fY = 0;
      }
      if (ev.fState & kButton1Mask) {

         // Figure scroll amount y
         if (y < kAutoScrollFudge) {
            dy = kAutoScrollFudge - y;
         } else if ((Int_t)fCanvas->GetHeight() - kAutoScrollFudge <= y) {
            dy = fCanvas->GetHeight() - kAutoScrollFudge - y;
         }
         Int_t ady = TMath::Abs(dy) >> 3;

         if (dy) {    
            if (ady > kAutoScrollFudge) ady = kAutoScrollFudge;
            dy = kAcceleration[ady];
         } else {
            dy = 1;
         }

         if (y > (Int_t)fCanvas->GetHeight()) {
            fScrolling = 3;
         }
         if (y < 0) {
            fScrolling = 2;
         }
      } else {
         fScrolling = -1;
      }

      size.fY = ToObjYCoord(fVisible.fY + fCanvas->GetHeight()) - 1;
      size.fX = ToObjXCoord(fVisible.fX + fCanvas->GetWidth(), fMousePos.fY) - 1;
      switch (fScrolling) {
         case -1:
            break;
         case 0:
            if (fVisible.fX == 0) {
               fScrolling = -1;
               break;
            } else {
               SetHsbPosition(fVisible.fX / fScrollVal.fX - 1);
               Mark(ToObjXCoord(fVisible.fX, fMousePos.fY) - 1, fMousePos.fY);
            }
            break;
         case 1:
            if ((Int_t)fCanvas->GetWidth() >= ToScrXCoord(ReturnLineLength(fMousePos.fY), fMousePos.fY)) {
               fScrolling = -1;
               break;
            } else {
               SetHsbPosition(fVisible.fX / fScrollVal.fX + 1);
               Mark(size.fX+1, fMousePos.fY);
            }
            break;
         case 2:
            if (fVisible.fY == 0) {
               fScrolling = -1;
               break;
            } else {
               SetVsbPosition(fVisible.fY/fScrollVal.fY - dy);
               Mark(fMousePos.fX, fMarkedStart.fY - 1);
            }
            break;
         case 3:
            if ((Int_t)fCanvas->GetHeight() >= ToScrYCoord(ReturnLineCount())) {
               fScrolling = -1;
               break;
            } else {
               SetVsbPosition(fVisible.fY/fScrollVal.fY + dy);
               Mark(fMousePos.fX, size.fY + 1);
            }
            break;
         default:
            break;
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextView::HandleButton(Event_t *event)
{
   // Handle mouse button event in text editor.

   if (event->fWindow != fCanvas->GetId()) {
      return kFALSE;
   }

   if (event->fCode == kButton1) {
      if (event->fType == kButtonPress) {
         if (fIsMarked) {
            if (event->fState & kKeyShiftMask) {
               fIsMarking = kTRUE;
               HandleMotion(event);
               return kTRUE;
            }

            UnMark();
         }
         fIsMarked = kTRUE;
         fIsMarking = kTRUE;
         fMousePos.fY = ToObjYCoord(fVisible.fY + event->fY);
         fMousePos.fX = ToObjXCoord(fVisible.fX + event->fX, fMousePos.fY);
         fMarkedStart.fX = fMarkedEnd.fX = fMousePos.fX;
         fMarkedStart.fY = fMarkedEnd.fY = fMousePos.fY;
      } else {
         fScrolling = -1;
         if ((fMarkedStart.fX == fMarkedEnd.fX) &&
             (fMarkedStart.fY == fMarkedEnd.fY)) {
            fIsMarked = kFALSE;
            SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_ISMARKED),
                        fWidgetId, kFALSE);
            Marked(kFALSE);
         } else {
            SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_ISMARKED),
                        fWidgetId, kTRUE);
            Marked(kTRUE);
         }
         fIsMarking = kFALSE;
      }
   } else if (event->fCode == kButton4) {
      // move three lines up
      if (fVisible.fY > 0) {
         Long_t amount = fVisible.fY / fScrollVal.fY - 3;
         SetVsbPosition((amount >= 0) ? amount : 0);
         //Mark(fMousePos.fX, fMarkedStart.fY - 3);
      }
   } else if (event->fCode == kButton5) {
      // move three lines down
      if ((Int_t)fCanvas->GetHeight() < ToScrYCoord(ReturnLineCount())) {
         TGLongPosition size;
         size.fY = ToObjYCoord(fVisible.fY + fCanvas->GetHeight()) - 1;
         SetVsbPosition(fVisible.fY / fScrollVal.fY + 3);
         //Mark(fMousePos.fX, size.fY + 3);
      }
   } else if (event->fType == kButtonPress) {
      if (event->fCode == kButton2) {
         SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_CLICK2),
                     fWidgetId, (event->fYRoot << 16) | event->fXRoot);
         UnMark();
      } else if (event->fCode == kButton3) {
         SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_CLICK3),
                     fWidgetId, (event->fYRoot << 16) | event->fXRoot);
      }
   }

   if (event->fType == kButtonRelease) {
      if (event->fCode == kButton1) {
         if (fIsMarked) {
            Copy();
         }
      }
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextView::HandleDoubleClick(Event_t *)
{
   // handle double click

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGTextView::HandleMotion(Event_t *event)
{
   // Handle mouse motion event in the text editor widget.

   if ((ToObjYCoord(fVisible.fY+event->fY) == fMousePos.fY) &&
       (ToObjXCoord(fVisible.fX+event->fX, ToObjYCoord(fVisible.fY + event->fY)) == fMousePos.fX)) {
      return kTRUE;
   }

   if (fScrolling != -1) {
      return kTRUE;
   }

   fMousePos.fY = ToObjYCoord(fVisible.fY + event->fY);
   if (fMousePos.fY >= ReturnLineCount()) {
      fMousePos.fY = ReturnLineCount()-1;
   }
   fMousePos.fX = ToObjXCoord(fVisible.fX + event->fX, fMousePos.fY);

   if (fMousePos.fX > ReturnLineLength(fMousePos.fY)) {
      fMousePos.fX = ReturnLineLength(fMousePos.fY);
   }
   if (event->fWindow != fCanvas->GetId()) {
      return kTRUE;
   }

   if (!fIsMarking) {
      return kTRUE;
   }
   if (event->fX < 0) {
      return kTRUE;
   }
   if (event->fX >= (Int_t)fCanvas->GetWidth()) {
      return kTRUE;
   }
   if (event->fY < 0) {
      return kTRUE;
   }
   if (event->fY >= (Int_t)fCanvas->GetHeight()) {
      return kTRUE;
   }
   Mark(fMousePos.fX, fMousePos.fY);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextView::HandleSelectionClear(Event_t * /*event*/)
{
   // Handle selection clear event.

   if (fIsMarked) {
      UnMark();
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextView::HandleSelectionRequest(Event_t *event)
{
   // Handle request to send current clipboard contents to requestor window.

   Event_t reply;
   char *buffer, *temp_buffer;
   Long_t len, prev_len, temp_len, count;
   TGLongPosition pos;
   Atom_t targets[2];
   Atom_t type;

   reply.fType    = kSelectionNotify;
   reply.fTime    = event->fTime;
   reply.fUser[0] = event->fUser[0];     // requestor
   reply.fUser[1] = event->fUser[1];     // selection
   reply.fUser[2] = event->fUser[2];     // target
   reply.fUser[3] = event->fUser[3];     // property

   targets[0] = gVirtualX->InternAtom("TARGETS", kFALSE);
   targets[1] = gVirtualX->InternAtom("XA_STRING", kFALSE);

   if ((Atom_t)event->fUser[2] == targets[0]) {
      type = gVirtualX->InternAtom("XA_ATOM", kFALSE);
      gVirtualX->ChangeProperty((Window_t) event->fUser[0], (Atom_t) event->fUser[3],
                                type, (UChar_t*) targets, (Int_t) 2);

      gVirtualX->SendEvent((Window_t)event->fUser[0], &reply);
      return kTRUE;
   }

   len = 0;
   for (count = 0; count < fClipText->RowCount(); count++) {
      len += fClipText->GetLineLength(count)+1;
   }
   len--;  // remove \n for last line

   pos.fY = pos.fX = 0;
   buffer = new char[len+1];
   prev_len = temp_len = 0;
   for (pos.fY = 0; pos.fY < fClipText->RowCount(); pos.fY++) {
      temp_len = fClipText->GetLineLength(pos.fY);
      if (temp_len < 0) break;
      temp_buffer = fClipText->GetLine(pos, temp_len);
      strncpy(buffer+prev_len, temp_buffer, (UInt_t)temp_len);
      if (pos.fY < fClipText->RowCount()-1) {
         buffer[prev_len+temp_len] = 10;   // \n
         prev_len += temp_len+1;
      } else
         prev_len += temp_len;
      delete [] temp_buffer;
   }
   buffer[len] = '\0';

   // get rid of special tab fillers
   ULong_t i = 0;
   while (buffer[i]) {
      if (buffer[i] == '\t') {
         ULong_t j = i + 1;
         while (buffer[j] == 16 && buffer[j]) {
            j++;
         }
         // coverity[secure_coding]
         strcpy(buffer+i+1, buffer+j);
         len -= j - i - 1;
      }
      i++;
   }

   gVirtualX->ChangeProperty((Window_t) event->fUser[0], (Atom_t) event->fUser[3],
                             (Atom_t) event->fUser[2], (UChar_t*) buffer,
                             (Int_t) len);

   delete [] buffer;

   gVirtualX->SendEvent((Window_t)event->fUser[0], &reply);

   return kTRUE;
}

//______________________________________________________________________________
static Bool_t IsTextFile(const char *candidate)
{
   // Returns true if given a text file
   // Uses the specification given on p86 of the Camel book
   // - Text files have no NULLs in the first block
   // - and less than 30% of characters with high bit set

   Int_t i;
   Int_t nchars;
   Int_t weirdcount = 0;
   char buffer[512];
   FILE *infile;
   FileStat_t buf;

   if (gSystem->GetPathInfo(candidate, buf) || !(buf.fMode & kS_IFREG))
      return kFALSE;

   infile = fopen(candidate, "r");
   if (infile) {
      // Read a block
      nchars = fread(buffer, 1, 512, infile);
      fclose (infile);
      // Examine the block
      for (i = 0; i < nchars; i++) {
         if (buffer[i] & 128)
            weirdcount++;
         if (buffer[i] == '\0')
            // No NULLs in text files
            return kFALSE;
      }
      if ((nchars > 0) && ((weirdcount * 100 / nchars) > 30))
         return kFALSE;
   } else {
      // Couldn't open it. Not a text file then
      return kFALSE;
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextView::HandleDNDDrop(TDNDData *data)
{
   // Handle Drop event

   static Atom_t rootObj = gVirtualX->InternAtom("application/root", kFALSE);
   static Atom_t uriObj  = gVirtualX->InternAtom("text/uri-list", kFALSE);

   if (fText->RowCount() > 1) {
      Int_t ret;
      new TGMsgBox(fClient->GetRoot(), GetMainFrame(),
                   "Overvrite", "Do you want to replace existing text?",
                   kMBIconExclamation, kMBYes | kMBNo, &ret);
      if (ret == kMBNo)
         return kTRUE;
   }
   if (data->fDataType == rootObj) {
      TBufferFile buf(TBuffer::kRead, data->fDataLength, (void *)data->fData);
      buf.SetReadMode();
      TObject *obj = (TObject *)buf.ReadObjectAny(TObject::Class());
      if (obj && obj->InheritsFrom("TMacro")) {
         TMacro *macro = (TMacro *)obj;
         TIter next(macro->GetListOfLines());
         TObjString *objs;
         while ((objs = (TObjString*) next())) {
            AddLine(objs->GetName());
         }
      }
      else if (obj && obj->InheritsFrom("TSystemFile")) {
         TSystemFile *sfile = (TSystemFile *)obj;
         LoadFile(sfile->GetName());
         DataDropped(sfile->GetName());
      }
      return kTRUE;
   }
   else if (data->fDataType == uriObj) {
      TString sfname((char *)data->fData);
      if (sfname.Length() > 7) {
         sfname.ReplaceAll("\r\n", "");
         TUrl uri(sfname.Data());
         if (IsTextFile(uri.GetFile())) {
            LoadFile(uri.GetFile());
            DataDropped(uri.GetFile());
         }
      }
   }
   return kFALSE;
}

//______________________________________________________________________________
Atom_t TGTextView::HandleDNDPosition(Int_t /*x*/, Int_t /*y*/, Atom_t action,
                                      Int_t /*xroot*/, Int_t /*yroot*/)
{
   // Handle Drag position event

   return action;
}

//______________________________________________________________________________
Atom_t TGTextView::HandleDNDEnter(Atom_t *typelist)
{
   // Handle Drag Enter event

   static Atom_t rootObj  = gVirtualX->InternAtom("application/root", kFALSE);
   static Atom_t uriObj  = gVirtualX->InternAtom("text/uri-list", kFALSE);
   Atom_t ret = kNone;
   for (int i = 0; typelist[i] != kNone; ++i) {
      if (typelist[i] == rootObj)
         ret = rootObj;
      if (typelist[i] == uriObj)
         ret = uriObj;
   }
   return ret;
}

//______________________________________________________________________________
Bool_t TGTextView::HandleDNDLeave()
{
   // Handle Drag Leave event

   return kTRUE;
}

//______________________________________________________________________________
void TGTextView::Mark(Long_t xPos, Long_t yPos)
{
   // Mark a text region from xPos to yPos.

   TGLongPosition posStart, posEnd, pos;

   pos.fY = yPos;
   pos.fX = xPos;
   if (pos.fY > fText->RowCount()-1) {
      pos.fY = fText->RowCount()-1;
   }
   if (pos.fX > fText->GetLineLength(pos.fY)) {
      pos.fX = fText->GetLineLength(pos.fY);
   }
   if (pos.fY < fMarkedStart.fY) {
      posEnd.fY = fMarkedStart.fY;
      if (fMarkedFromY == 1 || fMarkedFromX == 1) {
         posEnd.fY = fMarkedEnd.fY;
         fMarkedEnd.fX = fMarkedStart.fX;
         fMarkedEnd.fY = fMarkedStart.fY;
      }
      posStart.fY = pos.fY;
      fMarkedStart.fY = pos.fY;
      fMarkedStart.fX = pos.fX;
      fMarkedFromY = 0;
      fMarkedFromX = 0;
   } else if (pos.fY > fMarkedEnd.fY) {
      posStart.fY = fMarkedEnd.fY;
      if (fMarkedFromY == 0 || fMarkedFromX == 0) {
         if (fMarkedStart.fY != fMarkedEnd.fY) {
            posStart.fY = fMarkedStart.fY;
            fMarkedStart.fX = fMarkedEnd.fX;
            fMarkedStart.fY = fMarkedEnd.fY;
         }
      }
      fMarkedEnd.fY = pos.fY;
      fMarkedEnd.fX = pos.fX;  // -1
      fMarkedFromY = 1;
      fMarkedFromX = 1;

      posEnd.fY = fMarkedEnd.fY;
   } else {
      if (pos.fX <= fMarkedStart.fX && pos.fY == fMarkedStart.fY) {
         posEnd.fY = fMarkedStart.fY;
         if (fMarkedFromY == 1 || fMarkedFromX == 1) {
            posEnd.fY = fMarkedEnd.fY;
            fMarkedEnd.fX = fMarkedStart.fX;
            fMarkedEnd.fY = fMarkedStart.fY;
         }
         fMarkedStart.fX = pos.fX;
         fMarkedFromY = 0;
         fMarkedFromX = 0;
         posStart.fY = fMarkedStart.fY;
      } else {
         if (pos.fX > fMarkedEnd.fX && pos.fY == fMarkedEnd.fY) {
            posStart.fY = fMarkedEnd.fY;
            if (fMarkedFromY == 0 || fMarkedFromX == 0) {
               posStart.fY = fMarkedStart.fY;
               fMarkedStart.fX = fMarkedEnd.fX;
               fMarkedStart.fY = fMarkedEnd.fY;
            }
            fMarkedEnd.fX = pos.fX;   // -1
            fMarkedFromY = 1;
            fMarkedFromX = 1;
            posEnd.fY = fMarkedEnd.fY;
         } else {
            if (fMarkedFromY == 0 || fMarkedFromX == 0) {
               posStart.fY = fMarkedStart.fY;
               fMarkedStart.fY = pos.fY;
               fMarkedStart.fX = pos.fX;
               posEnd.fY = fMarkedStart.fY;
               fMarkedFromX = 0;
               if (fMarkedStart.fY == fMarkedEnd.fY &&
                   fMarkedStart.fX > fMarkedEnd.fX) {
                  fMarkedStart.fX = fMarkedEnd.fX;
                  fMarkedEnd.fX = pos.fX;  // -1
                  fMarkedFromX  = 1;
               }
            } else if (fMarkedFromX == 1 || fMarkedFromY == 1) {
               posStart.fY = pos.fY;
               posEnd.fY = fMarkedEnd.fY;
               fMarkedEnd.fY = pos.fY;
               fMarkedEnd.fX = pos.fX;  // -1
               fMarkedFromY = 1;
               fMarkedFromX = 1;
               if (fMarkedEnd.fX == -1) {
                  fMarkedEnd.fY = pos.fY-1;
                  fMarkedEnd.fX = fText->GetLineLength(fMarkedEnd.fY); // -1
                  if (fMarkedEnd.fX < 0) {
                     fMarkedEnd.fX = 0;
                  }
               }
               fMarkedFromX = 1;
               if (fMarkedStart.fY == fMarkedEnd.fY &&
                   fMarkedStart.fX > fMarkedEnd.fX) {
                  fMarkedEnd.fX = fMarkedStart.fX;
                  fMarkedStart.fX = pos.fX;
                  fMarkedFromX = 0;
               }
            }
         }
      }
   }

   if (fMarkedEnd.fX == -1) {
      if (fMarkedEnd.fY > 0) {
         fMarkedEnd.fY--;
      }
      fMarkedEnd.fX = fText->GetLineLength(fMarkedEnd.fY);  // -1
      if (fMarkedEnd.fX < 0) {
         fMarkedEnd.fX = 0;
      }
   }
   fIsMarked = kTRUE;

   Int_t yy = (Int_t)ToScrYCoord(posStart.fY);
   UInt_t hh = UInt_t(ToScrYCoord(posEnd.fY + 1) - ToScrYCoord(posStart.fY));

   DrawRegion(0, yy, fCanvas->GetWidth(), hh);
   return;
}

//______________________________________________________________________________
void TGTextView::UnMark()
{
   // Clear marked region.

   if (!fIsMarked || 
       ((fMarkedEnd.fY == fMarkedStart.fY) && 
       (fMarkedEnd.fX == fMarkedStart.fX))) {
      return;
   }
   fIsMarked = kFALSE;

   Int_t y = (Int_t)ToScrYCoord(fMarkedStart.fY);
   UInt_t h = UInt_t(ToScrYCoord(fMarkedEnd.fY + 1) - y);

   // update marked region
   UpdateRegion(0, y, fCanvas->GetWidth(), h);
}

//______________________________________________________________________________
void TGTextView::AdjustWidth()
{
   // Adjust widget width to longest line.

   Long_t line = fText->GetLongestLine();
   if (line <= 0) {
      return;
   }
   Long_t size = ToScrXCoord(fText->GetLineLength(line), line) + fVisible.fX;
   if (fVsb->IsMapped()) {
      size += fVsb->GetDefaultWidth();
   }
   size += (fBorderWidth << 1) + fXMargin+1;
   Resize((UInt_t)size, fHeight);
}

//______________________________________________________________________________
void TGTextView::Layout()
{
   // Layout the components of view.

   VLayout();
   HLayout();
}

//______________________________________________________________________________
void TGTextView::HLayout()
{
   // Horizontal layout of widgets (canvas, scrollbar).

   if (!fHsb) return;

   Int_t tcw, tch;
   Long_t cols;
   tch = fHeight - (fBorderWidth << 1) - fYMargin-1;
   tcw = fWidth - (fBorderWidth << 1) - fXMargin-1;

   if (fVsb && fVsb->IsMapped()) {
      tcw -= fVsb->GetDefaultWidth();
      if (tcw < 0) tcw = 0;
   }
   fCanvas->SetHeight(tch);
   fCanvas->SetWidth(tcw);
   cols = ReturnLongestLineWidth();
   if (cols <= tcw) {
      if (fHsb && fHsb->IsMapped()) {
         SetVisibleStart(0, kHorizontal);
         fHsb->UnmapWindow();
         VLayout();
      }
      fCanvas->MoveResize(fBorderWidth + fXMargin, fBorderWidth + fYMargin, tcw, tch);
   } else {
      if (fHsb) {
         tch -= fHsb->GetDefaultHeight();
         if (tch < 0) tch = 0;
         fHsb->MoveResize(fBorderWidth, fHeight - fHsb->GetDefaultHeight()-fBorderWidth,
                          tcw+1+fBorderWidth, fHsb->GetDefaultHeight());
         fHsb->MapWindow();
         fHsb->SetRange(Int_t(cols/fScrollVal.fX), Int_t(tcw/fScrollVal.fX));
      }
      fCanvas->MoveResize(fBorderWidth + fXMargin, fBorderWidth + fYMargin, tcw, tch);
   }
}

//______________________________________________________________________________
void TGTextView::VLayout()
{
   // Vertical layout of widgets (canvas, scrollbar).

   Int_t  tcw, tch;
   Long_t lines;

   tch = fHeight - (fBorderWidth << 1) - fYMargin-1;
   tcw = fWidth - (fBorderWidth << 1) - fXMargin-1;
   if (fHsb && fHsb->IsMapped()) {
      tch -= fHsb->GetDefaultHeight();
      if (tch < 0) tch = 0;
   }
   fCanvas->SetHeight(tch);
   fCanvas->SetWidth(tcw);
   lines = ReturnHeighestColHeight();
   if (lines <= tch) {
      if (fVsb && fVsb->IsMapped()) {
         SetVisibleStart(0, kVertical);
         fVsb->UnmapWindow();
         HLayout();
      }
      fCanvas->MoveResize(fBorderWidth + fXMargin, fBorderWidth + fYMargin, tcw, tch);
   } else {
      if (fVsb) {
         tcw -= fVsb->GetDefaultWidth();
         if (tcw < 0) tcw = 0;
         fVsb->MoveResize(fWidth - fVsb->GetDefaultWidth() - fBorderWidth, fBorderWidth,
                          fVsb->GetDefaultWidth(), tch+1+fBorderWidth);
         fVsb->MapWindow();
         fVsb->SetRange(Int_t(lines/fScrollVal.fY), Int_t(tch/fScrollVal.fY));
      }
      fCanvas->MoveResize(fBorderWidth + fXMargin, fBorderWidth + fYMargin, tcw, tch);
   }
}

//______________________________________________________________________________
void TGTextView::SetSBRange(Int_t direction)
{
   // Set the range for the kVertical or kHorizontal scrollbar.

   if (direction == kVertical) {
      if (!fVsb) {
         return;
      }
      if (ReturnHeighestColHeight() <= (Int_t)fCanvas->GetHeight()) {
         if (fVsb->IsMapped()) {
            VLayout();
         } else {
            return;
         }
      }
      if (!fVsb->IsMapped()) {
         VLayout();
      }
      fVsb->SetRange(Int_t(ReturnHeighestColHeight()/fScrollVal.fY),
                     Int_t(fCanvas->GetHeight()/fScrollVal.fY));
      HLayout();
   } else {
      if (!fHsb) {
         return;
      }
      if (ReturnLongestLineWidth() <= (Int_t)fCanvas->GetWidth()) {
         if (fHsb->IsMapped()) {
            HLayout();
         } else {
            return;
         }
      }
      if (!fHsb->IsMapped()) {
         HLayout();
      }
      fHsb->SetRange(Int_t(ReturnLongestLineWidth()/fScrollVal.fX),
                     Int_t(fCanvas->GetWidth()/fScrollVal.fX));
      VLayout();
   }
}

//______________________________________________________________________________
void TGTextView::SetHsbPosition(Long_t newPos)
{
   // Set position of horizontal scrollbar.

   if (fHsb && fHsb->IsMapped()) {
      fHsb->SetPosition(Int_t(newPos));
   } else {
      SetVisibleStart(Int_t(newPos * fScrollVal.fX), kHorizontal);
   }
}

//______________________________________________________________________________
void TGTextView::SetVsbPosition(Long_t newPos)
{
   // Set position of vertical scrollbar.

   if (fVsb && fVsb->IsMapped()) {
      fVsb->SetPosition(Int_t(newPos));
   } else {
      SetVisibleStart(Int_t(newPos * fScrollVal.fY), kVertical);
   }
}

//______________________________________________________________________________
FontStruct_t TGTextView::GetDefaultFontStruct()
{
   // Return default font structure in use.

   if (!fgDefaultFont) {
      fgDefaultFont = gClient->GetResourcePool()->GetDocumentFixedFont();
   }
   return fgDefaultFont->GetFontStruct();
}

//______________________________________________________________________________
void TGTextView::ShowBottom()
{
   // Show bottom of the page.

   Int_t  tch;
   Long_t lines, newPos;

   tch = fCanvas->GetHeight();
   lines = ReturnHeighestColHeight();
   if (lines > tch) {
      newPos = lines / fScrollVal.fY;
      SetVsbPosition(newPos);
   }
   Layout();
}

//______________________________________________________________________________
void TGTextView::ShowTop()
{
   // Show top of the page.

   SetVsbPosition(0);
   Layout();
}

//______________________________________________________________________________
void TGTextView::SetForegroundColor(Pixel_t col)
{
   // Set text color.

   fNormGC.SetBackground(col);
   fNormGC.SetForeground(col);
}

//______________________________________________________________________________
const TGGC &TGTextView::GetDefaultGC()
{
   // Return default graphics context in use.

   if (!fgDefaultGC) {
      fgDefaultGC = new TGGC(*gClient->GetResourcePool()->GetFrameGC());
      fgDefaultGC->SetFont(fgDefaultFont->GetFontHandle());
   }
   return *fgDefaultGC;
}

//______________________________________________________________________________
const TGGC &TGTextView::GetDefaultSelectedGC()
{
   // Return selection graphics context in use.

   if (!fgDefaultSelectedGC) {
      fgDefaultSelectedGC = new TGGC(*gClient->GetResourcePool()->GetSelectedGC());
      fgDefaultSelectedGC->SetFont(fgDefaultFont->GetFontHandle());
   }
   return *fgDefaultSelectedGC;
}

//______________________________________________________________________________
const TGGC &TGTextView::GetDefaultSelectedBackgroundGC()
{
   // Return graphics context for highlighted frame background.

   if (!fgDefaultSelectedBackgroundGC) {
      fgDefaultSelectedBackgroundGC = gClient->GetResourcePool()->GetSelectedBckgndGC();
   }
   return *fgDefaultSelectedBackgroundGC;
}

//______________________________________________________________________________
void TGTextView::SavePrimitive(ostream &out, Option_t *option /*= ""*/)
{
   // Save a text edit widget as a C++ statement(s) on output stream out

   char quote = '"';
   out << "   TGTextView *";
   out << GetName() << " = new TGTextView(" << fParent->GetName()
       << "," << GetWidth() << "," << GetHeight()
       << ");"<< endl;

   if (option && strstr(option, "keep_names"))
      out << "   " << GetName() << "->SetName(\"" << GetName() << "\");" << endl;

   if (fCanvas->GetBackground() != TGFrame::fgWhitePixel) {
      out << "   " << GetName() << "->ChangeBackground(" << fCanvas->GetBackground() << ");" << endl;
   }

   TGText *txt = GetText();
   Bool_t fromfile = strlen(txt->GetFileName()) ? kTRUE : kFALSE;
   TString fn;

   if (fromfile) {
      const char *filename = txt->GetFileName();
      fn = gSystem->ExpandPathName(gSystem->UnixPathName(filename));
   } else {
      fn = TString::Format("Txt%s", GetName()+5);
      txt->Save(fn.Data());
   }
   out << "   " << GetName() << "->LoadFile(" << quote << fn.Data() << quote << ");" << endl;
}

// @(#)root/gui:$Name:  $:$Id: TGTextView.cxx,v 1.4 2000/07/06 16:47:54 rdm Exp $
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


ClassImp(TGTextView)


//______________________________________________________________________________
void TGTextView::Init(ULong_t /*back*/)
{
   // Initialize a text view widget.

   // set in TGClient via fgDefaultFontStruct and select font via .rootrc
   fFont      = fgDefaultFontStruct;
   fNormGC    = fgDefaultGC;
   fSelGC     = fgDefaultSelectedGC;
   fSelbackGC = fgDefaultSelectedBackgroundGC;
   fDeleteGC  = kFALSE;

   fText = new TGText();
   TGView::Clear();

   fClipText = new TGText();

   gVirtualX->GetFontProperties(fFont, fMaxAscent, fMaxDescent);
   fScrollVal.fY = fMaxAscent + fMaxDescent;
   fScrollVal.fX = fMaxWidth = gVirtualX->TextWidth(fFont, "W", 1);

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

   if (fDeleteGC) {
      gVirtualX->DeleteGC(fNormGC);
      gVirtualX->DeleteGC(fSelGC);
   }
   delete fText;
   delete fClipText;
}

//______________________________________________________________________________
void TGTextView::SetText(TGText *text)
{
   // Adopt a new text buffer. The text will be deleted by this object.

   Clear();
   delete fText;
   fText = text;
   Layout();
   DrawRegion(0, 0, fCanvas->GetWidth(), fCanvas->GetHeight());
}

//______________________________________________________________________________
void TGTextView::AddText(TGText *text)
{
   // Add text to the view widget.

   fText->AddText(text);
   Layout();
   DrawRegion(0, 0, fCanvas->GetWidth(), fCanvas->GetHeight());
}

//______________________________________________________________________________
void TGTextView::AddLine(const char *string)
{
   // Add a line of text to the view widget.

   fText->InsLine(fText->RowCount(), string);
   Layout();
   DrawRegion(0, 0, fCanvas->GetWidth(), fCanvas->GetHeight());
}

//______________________________________________________________________________
Long_t TGTextView::ReturnLongestLineWidth()
{
   // Return width of longest line.

   Long_t count = 0, longest = 0, width;
   Long_t rows = fText->RowCount();
   while (count < rows) {
      width = ToScrXCoord(fText->GetLineLength(count), count) + fVisible.fX;
      if (width > longest)
         longest = width;
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
      pos2.fX = fMarkedEnd.fX + 1;
      pos2.fY = fMarkedEnd.fY;
   }
   if (!fText->Search(&pos, pos2, string, direction, caseSensitive))
      return kFALSE;
   UnMark();
   fIsMarked = kTRUE;
   fMarkedStart.fY = fMarkedEnd.fY = pos.fY;
   fMarkedStart.fX = pos.fX;
   fMarkedEnd.fX = fMarkedStart.fX + strlen(string) - 1;
   fMarkedEnd.fX++;
   fMarkedEnd.fX--;
   pos.fY = ToObjYCoord(fVisible.fY);
   if ((fMarkedStart.fY < pos.fY) ||
       (ToScrYCoord(fMarkedStart.fY) >= (Int_t)fCanvas->GetHeight()))
      pos.fY = fMarkedStart.fY;
   pos.fX = ToObjXCoord(fVisible.fX, pos.fY);
   if ((fMarkedStart.fX < pos.fX) ||
       (ToScrXCoord(fMarkedStart.fX, pos.fY) >= (Int_t)fCanvas->GetWidth()))
      pos.fX = fMarkedStart.fX;

   SetVsbPosition((ToScrYCoord(pos.fY) + fVisible.fY)/fScrollVal.fY);
   SetHsbPosition((ToScrXCoord(pos.fX, pos.fY) + fVisible.fX)/fScrollVal.fX);
   DrawRegion(0, (Int_t)ToScrYCoord(fMarkedStart.fY), fCanvas->GetWidth(),
              UInt_t(ToScrYCoord(fMarkedEnd.fY+1) - ToScrYCoord(fMarkedEnd.fY)));

   return kTRUE;
}

//______________________________________________________________________________
void TGTextView::SetFont(FontStruct_t font)
{
   // Changes text entry font.

   if (font != fFont) {
      GCValues_t gval;
      fFont = font;

      // get unique copies of the norm GC and the sel GC (if not already copied)
      if (!fDeleteGC) {
         GContext_t norm = gVirtualX->CreateGC(fCanvas->GetId(), 0);
         gVirtualX->CopyGC(fNormGC, norm, 0);
         fNormGC = norm;
         GContext_t sel  = gVirtualX->CreateGC(fCanvas->GetId(), 0);
         gVirtualX->CopyGC(fSelGC, sel, 0);
         fSelGC = sel;
      }
      gval.fMask = kGCFont;
      gval.fFont = gVirtualX->GetFontHandle(fFont);
      gVirtualX->ChangeGC(fNormGC, &gval);
      gVirtualX->ChangeGC(fSelGC, &gval);
      fClient->NeedRedraw(this);
      fDeleteGC = kTRUE;
   }
}

//______________________________________________________________________________
Long_t TGTextView::ToScrYCoord(Long_t yCoord)
{
   // Convert line number to screen coordinate.

   if (yCoord * (fMaxAscent + fMaxDescent) <= 0)
      return 0;
   if (yCoord > fText->RowCount())
      return fText->RowCount() * (fMaxAscent + fMaxDescent);
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
   if (xCoord <= 0 || pos.fY < 0 || width <= 0)
      return 0;
   if (xCoord > width)
      xCoord = width;
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

   if (line < 0 || line >= fText->RowCount())
      return 0;
   Long_t len = fText->GetLineLength(line);
   pos.fX = 0;
   pos.fY = line;
   if (len <= 0 || xCoord < 0)
      return 0;
   Long_t viscoord =  xCoord;
   buffer = fText->GetLine(pos, len);
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
   delete fText;
   fText = new TGText();
   fText->Clear();
   SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_ISMARKED), fWidgetId, kFALSE);
   gVirtualX->ClearWindow(fCanvas->GetId());
   SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_DATACHANGE), fWidgetId, 0);
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

   Clear();
   fText->Load(filename, startpos, length);
   Layout();
   DrawRegion(0, 0, fCanvas->GetWidth(), fCanvas->GetHeight());
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextView::LoadBuffer(const char *txtbuf)
{
   // Load text from a text buffer. Return false in case of failure.

   if (!txtbuf || !strlen(txtbuf))
      return kFALSE;

   Clear();
   fText->LoadBuffer(txtbuf);
   Layout();
   DrawRegion(0, 0, fCanvas->GetWidth(), fCanvas->GetHeight());
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextView::Copy()
{
   // Copy selected text to clipboard.

   TGLongPosition insPos, startPos, endPos;

   if (!fIsMarked)
      return kFALSE;
   delete fClipText;
   fClipText = new TGText();
   insPos.fY = insPos.fX = 0;
   startPos.fX = fMarkedStart.fX;
   startPos.fY = fMarkedStart.fY;
   endPos.fX   = fMarkedEnd.fX;
   endPos.fY   = fMarkedEnd.fY;
   fClipText->InsText(insPos, fText, startPos, endPos);
   gVirtualX->SetPrimarySelectionOwner(fId);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextView::SelectAll()
{
   // Select all text in the viewer.

   if (fText->RowCount() == 1 && fText->GetLineLength(0) == 0)
      return kFALSE;
   fIsMarked = kTRUE;
   fMarkedStart.fY = 0;
   fMarkedStart.fX = 0;
   fMarkedEnd.fY = fText->RowCount()-1;
   fMarkedEnd.fX = fText->GetLineLength(fMarkedEnd.fY)-1;
   if (fMarkedEnd.fX < 0)
      fMarkedEnd.fX = 0;
   DrawRegion(0, 0, fCanvas->GetWidth(), fCanvas->GetHeight());
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
   rect.fHeight = UShort_t(h + ToScrYCoord(pos.fY+1) - ToScrYCoord(pos.fY));
   pos.fX = ToObjXCoord(fVisible.fX+w, pos.fY);
   rect.fWidth = UShort_t(w + ToScrXCoord(pos.fX+1,pos.fY) - ToScrXCoord(pos.fX, pos.fY));
   Int_t yloc = rect.fY + (Int_t)fScrollVal.fY;
   pos.fY = ToObjYCoord(fVisible.fY + rect.fY);
   while (pos.fY < line_count &&
          yloc - fScrollVal.fY <= (Int_t)fCanvas->GetHeight() &&
          yloc - fScrollVal.fY <= rect.fY + rect.fHeight) {
      pos.fX = ToObjXCoord(fVisible.fX + rect.fX, pos.fY);
      xoffset = ToScrXCoord(pos.fX, pos.fY);
      len = fText->GetLineLength(pos.fY) - pos.fX;
      gVirtualX->ClearArea(fCanvas->GetId(), x, Int_t(ToScrYCoord(pos.fY)),
                           rect.fWidth, UInt_t(ToScrYCoord(pos.fY+1)-ToScrYCoord(pos.fY)));
      if (len > 0) {
         if (len > ToObjXCoord(fVisible.fX + rect.fX + rect.fWidth, pos.fY) - pos.fX)
            len = ToObjXCoord(fVisible.fX + rect.fX + rect.fWidth, pos.fY) - pos.fX + 1;
         if (pos.fX == 0)
            xoffset = -fVisible.fX;
         if (pos.fY >= ToObjYCoord(fVisible.fY)) {
            buffer = fText->GetLine(pos, len);
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
               gVirtualX->DrawString(fCanvas->GetId(), fNormGC, Int_t(xoffset),
                                     Int_t(ToScrYCoord(pos.fY+1) - fMaxDescent),
                                     buffer, Int_t(len));
            } else {
               if (pos.fY > fMarkedStart.fY && pos.fY < fMarkedEnd.fY) {
                  len1 = 0;
                  len2 = len;
               } else {
                  if (fMarkedStart.fY == fMarkedEnd.fY) {
                     if (fMarkedStart.fX >= pos.fX &&
                         fMarkedStart.fX <= pos.fX + len)
                        len1 = fMarkedStart.fX - pos.fX;
                     else
                        len1 = 0;
                     if (fMarkedEnd.fX >= pos.fX &&
                         fMarkedEnd.fX <= pos.fX+len)
                        len2 = fMarkedEnd.fX+1 - pos.fX - len1;
                     else
                        len2 = len - len1;
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
                           len2 = fMarkedEnd.fX+1 - pos.fX;
                        }
                     }
                  }
               }
               gVirtualX->DrawString(fCanvas->GetId(), fNormGC,
                                     Int_t(ToScrXCoord(pos.fX, pos.fY)),
                                     Int_t(ToScrYCoord(pos.fY+1) - fMaxDescent),
                                     buffer, Int_t(len1));
               gVirtualX->FillRectangle(fCanvas->GetId(), fSelbackGC,
                                     Int_t(ToScrXCoord(pos.fX+len1, pos.fY)),
                                     Int_t(ToScrYCoord(pos.fY)),
                                     UInt_t(ToScrXCoord(pos.fX+len1+len2, pos.fY) -
                                     ToScrXCoord(pos.fX+len1, pos.fY)),
//                                     len-(len1+len2) ?
//                                     ToScrXCoord(pos.fX+len1+len2, pos.fY) -
//                                     ToScrXCoord(pos.fX+len1, pos.fY) :
//                                     fCanvas->GetWidth(),
                                     UInt_t(ToScrYCoord(pos.fY+1)-ToScrYCoord(pos.fY)));
               gVirtualX->DrawString(fCanvas->GetId(), fSelGC,
                                     Int_t(ToScrXCoord(pos.fX+len1, pos.fY)),
                                     Int_t(ToScrYCoord(pos.fY+1) - fMaxDescent),
                                     buffer+len1, Int_t(len2));
               gVirtualX->DrawString(fCanvas->GetId(), fNormGC,
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
Bool_t TGTextView::HandleSelectionRequest(Event_t *event)
{
   // Handle request to send current clipboard contents to requestor window.

   Event_t reply;
   char *buffer, *temp_buffer;
   Long_t len, prev_len, temp_len, count;
   TGLongPosition pos;

   reply.fType    = kSelectionNotify;
   reply.fTime    = event->fTime;
   reply.fUser[0] = event->fUser[0];     // requestor
   reply.fUser[1] = event->fUser[1];     // selection
   reply.fUser[2] = event->fUser[2];     // target
   reply.fUser[3] = event->fUser[3];     // property

   len = 0;
   for (count = 0; count < fClipText->RowCount(); count++)
      len += fClipText->GetLineLength(count)+1;

   pos.fY = pos.fX = 0;
   buffer = new char[len+1];
   prev_len = temp_len = 0;
   for (pos.fY = 0; pos.fY < fClipText->RowCount(); pos.fY++) {
      temp_len = fClipText->GetLineLength(pos.fY);
      temp_buffer = fClipText->GetLine(pos, temp_len);
      strncpy(buffer+prev_len, temp_buffer, (UInt_t)temp_len);
      buffer[prev_len+temp_len] = 10;   // \n
      delete [] temp_buffer;
      prev_len += temp_len+1;
   }
   buffer[len] = '\0';

   gVirtualX->ChangeProperty((Window_t) event->fUser[0], (Atom_t) event->fUser[3],
                             (Atom_t) event->fUser[2], (UChar_t*) buffer,
                             (Int_t) len-1);

   delete [] buffer;

   gVirtualX->SendEvent((Window_t)event->fUser[0], &reply);

   return kTRUE;
}

//______________________________________________________________________________
void TGTextView::Mark(Long_t xPos, Long_t yPos)
{
   // Mark a text region from xPos to yPos.

   TGLongPosition posStart, posEnd, pos;

   pos.fY = yPos;
   pos.fX = xPos;
   if (pos.fY > fText->RowCount()-1)
      pos.fY = fText->RowCount()-1;
   if (pos.fX > fText->GetLineLength(pos.fY))
      pos.fX = fText->GetLineLength(pos.fY);
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
      fMarkedEnd.fX = pos.fX - 1;
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
            fMarkedEnd.fX = pos.fX-1;
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
                  fMarkedEnd.fX = pos.fX - 1;
                  fMarkedFromX  = 1;
               }
            } else if (fMarkedFromX == 1 || fMarkedFromY == 1) {
               posStart.fY = pos.fY;
               posEnd.fY = fMarkedEnd.fY;
               fMarkedEnd.fY = pos.fY;
               fMarkedEnd.fX = pos.fX - 1;
               fMarkedFromY = 1;
               fMarkedFromX = 1;
               if (fMarkedEnd.fX == -1) {
                  fMarkedEnd.fY = pos.fY-1;
                  fMarkedEnd.fX = fText->GetLineLength(fMarkedEnd.fY)-1;
                  if (fMarkedEnd.fX < 0)
                     fMarkedEnd.fX = 0;
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
   if (fMarkedEnd.fX == fText->GetLineLength(fMarkedEnd.fY))
      fMarkedEnd.fX--;

   if (fMarkedEnd.fX == -1) {
      if (fMarkedEnd.fY > 0)
         fMarkedEnd.fY--;
      fMarkedEnd.fX = fText->GetLineLength(fMarkedEnd.fY)-1;
      if (fMarkedEnd.fX < 0)
         fMarkedEnd.fX = 0;
   }
   DrawRegion(0, (Int_t)ToScrYCoord(posStart.fY), fCanvas->GetWidth(),
              UInt_t(ToScrYCoord(posEnd.fY+1)-ToScrYCoord(posStart.fY)));
   return;
}

//______________________________________________________________________________
void TGTextView::UnMark()
{
   // Clear marked region.

   fIsMarked = kFALSE;
   gVirtualX->ClearWindow(fCanvas->GetId());
   DrawRegion(0, 0, fCanvas->GetWidth(), fCanvas->GetHeight());
}

//______________________________________________________________________________
void TGTextView::AdjustWidth()
{
   // Adjust widget width to longest line.

   Long_t line = fText->GetLongestLine();
   if (line <= 0)
      return;
   Long_t size = ToScrXCoord(fText->GetLineLength(line), line) + fVisible.fX;
   if (fVsb->IsMapped())
      size += fVsb->GetDefaultWidth();
   size += (fBorderWidth << 1) + fXMargin+1;
   Resize((UInt_t)size, fHeight);
}

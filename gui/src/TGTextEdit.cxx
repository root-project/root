// @(#)root/gui:$Name:  $:$Id: TGTextEdit.cxx,v 1.10 2000/10/22 19:28:58 rdm Exp $
// Author: Fons Rademakers   3/7/2000

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
// TGTextEdit                                                           //
//                                                                      //
// A TGTextEdit is a specialization of TGTextView. It provides the      //
// text edit functionality to the static text viewing widget.           //
// For the messages supported by this widget see the TGView class.      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGTextEdit.h"
#include "TGTextEditDialogs.h"
#include "TSystem.h"
#include "TMath.h"
#include "TTimer.h"
#include "TGMenu.h"
#include "TGMsgBox.h"
#include "TGFileDialog.h"
#include "KeySymbols.h"


const char *gFiletypes[] = { "All files",     "*",
                             "Text files",    "*.txt",
                             "ROOT macros",   "*.C",
                             0,               0 };
static char *gPrinter      = 0;
static char *gPrintCommand = 0;


ClassImp(TGTextEdit)

//______________________________________________________________________________
TGTextEdit::TGTextEdit(const TGWindow *parent, UInt_t w, UInt_t h, Int_t id,
                       UInt_t sboptions, ULong_t back) :
     TGTextView(parent, w, h, id, sboptions, back)
{
   // Create a text edit widget.

   Init();
}

//______________________________________________________________________________
TGTextEdit::TGTextEdit(const TGWindow *parent, UInt_t w, UInt_t h, TGText *text,
                       Int_t id, UInt_t sboptions, ULong_t back) :
     TGTextView(parent, w, h, text, id, sboptions, back)
{
   // Create a text edit widget. Initialize it with the specified text buffer.

   Init();
}

//______________________________________________________________________________
TGTextEdit::TGTextEdit(const TGWindow *parent, UInt_t w, UInt_t h,
                       const char *string, Int_t id, UInt_t sboptions,
                       ULong_t back) :
     TGTextView(parent, w, h, string, id, sboptions, back)
{
   // Create a text edit widget. Initialize it with the specified string.

   Init();
}

//______________________________________________________________________________
TGTextEdit::~TGTextEdit()
{
   // Cleanup text edit widget.

   gVirtualX->DeleteGC(fCursor0GC);
   gVirtualX->DeleteGC(fCursor1GC);

   delete fCurBlink;
   delete fMenu;
   delete fSearch;
}

//______________________________________________________________________________
void TGTextEdit::Init()
{
   // Initiliaze a text edit widget.

   GCValues_t gval;
   fCursor1GC = gVirtualX->CreateGC(fCanvas->GetId(), 0);
   gVirtualX->CopyGC(fNormGC, fCursor1GC, 0);

   gval.fMask = kGCFunction;
   gval.fFunction = kGXand;
   gVirtualX->ChangeGC(fCursor1GC, &gval);

   fCursor0GC = gVirtualX->CreateGC(fCanvas->GetId(), 0);
   gVirtualX->CopyGC(fSelGC, fCursor0GC, 0);
   gval.fFunction = kGXxor;
   gVirtualX->ChangeGC(fCursor0GC, &gval);

   gVirtualX->SetCursor(fCanvas->GetId(), fgDefaultCursor);

   fCursorState = 1;
   fCurrent.fY  = fCurrent.fX = 0;
   fInsertMode  = kInsert;
   fCurBlink    = 0;
   fSearch      = 0;

   // create popup menu with default editor actions
   fMenu = new TGPopupMenu(fClient->GetRoot());
   fMenu->AddEntry("New", kM_FILE_NEW);
   fMenu->AddEntry("Open...", kM_FILE_OPEN);
   fMenu->AddSeparator();
   fMenu->AddEntry("Close", kM_FILE_CLOSE);
   fMenu->AddEntry("Save", kM_FILE_SAVE);
   fMenu->AddEntry("Save As...", kM_FILE_SAVEAS);
   fMenu->AddSeparator();
   fMenu->AddEntry("Print...", kM_FILE_PRINT);
   fMenu->AddSeparator();
   fMenu->AddEntry("Cut", kM_EDIT_CUT);
   fMenu->AddEntry("Copy", kM_EDIT_COPY);
   fMenu->AddEntry("Paste", kM_EDIT_PASTE);
   fMenu->AddEntry("Select All", kM_EDIT_SELECTALL);
   fMenu->AddSeparator();
   fMenu->AddEntry("Find...", kM_SEARCH_FIND);
   fMenu->AddEntry("Find Again", kM_SEARCH_FINDAGAIN);
   fMenu->AddEntry("Goto...", kM_SEARCH_GOTO);

   fMenu->Associate(this);
}

//______________________________________________________________________________
void TGTextEdit::SetMenuState()
{
   // Enable/disable menu items in function of what is possible.

   if (fText->RowCount() == 1 && fText->GetLineLength(0) <= 0) {
      fMenu->DisableEntry(kM_FILE_CLOSE);
      fMenu->DisableEntry(kM_FILE_SAVE);
      fMenu->DisableEntry(kM_FILE_SAVEAS);
      fMenu->DisableEntry(kM_FILE_PRINT);
      fMenu->DisableEntry(kM_EDIT_SELECTALL);
      fMenu->DisableEntry(kM_SEARCH_FIND);
      fMenu->DisableEntry(kM_SEARCH_FINDAGAIN);
      fMenu->DisableEntry(kM_SEARCH_GOTO);
   } else {
      fMenu->EnableEntry(kM_FILE_CLOSE);
      fMenu->EnableEntry(kM_FILE_SAVE);
      fMenu->EnableEntry(kM_FILE_SAVEAS);
      fMenu->EnableEntry(kM_FILE_PRINT);
      fMenu->EnableEntry(kM_EDIT_SELECTALL);
      fMenu->EnableEntry(kM_SEARCH_FIND);
      fMenu->EnableEntry(kM_SEARCH_FINDAGAIN);
      fMenu->EnableEntry(kM_SEARCH_GOTO);
   }

   if (IsSaved())
      fMenu->DisableEntry(kM_FILE_SAVE);
   else
      fMenu->EnableEntry(kM_FILE_SAVE);

   if (fIsMarked) {
      fMenu->EnableEntry(kM_EDIT_CUT);
      fMenu->EnableEntry(kM_EDIT_COPY);
   } else {
      fMenu->DisableEntry(kM_EDIT_CUT);
      fMenu->DisableEntry(kM_EDIT_COPY);
   }
}

//______________________________________________________________________________
Long_t TGTextEdit::ReturnLongestLineWidth()
{
   // Return width of longest line in widget.

   Long_t linewidth = TGTextView::ReturnLongestLineWidth();
   linewidth += 3*fScrollVal.fX;
   return linewidth;
}

//______________________________________________________________________________
void TGTextEdit::Clear(Option_t *)
{
   // Clear text edit widget.

   fCursorState = 1;
   fCurrent.fY = fCurrent.fX = 0;
   TGTextView::Clear();
}

//______________________________________________________________________________
Bool_t TGTextEdit::SaveFile(const char *filename, Bool_t saveas)
{
   // Save file. If filename==0 ask user via dialog for a filename, if in
   // addition saveas==kTRUE always ask for new filename. Returns
   // kTRUE if file was correctly saved, kFALSE otherwise.

   if (!filename) {
      Bool_t untitled = !strlen(fText->GetFileName()) ? kTRUE : kFALSE;
      if (untitled || saveas) {
         TGFileInfo fi;
         fi.fFileTypes = (char **)gFiletypes;
         new TGFileDialog(fClient->GetRoot(), this, kFDSave, &fi);
         if (fi.fFilename && strlen(fi.fFilename))
            return fText->Save(fi.fFilename);
         return kFALSE;
      }
      return fText->Save(fText->GetFileName());
   }

   return fText->Save(filename);
}

//______________________________________________________________________________
Bool_t TGTextEdit::Copy()
{
   // Copy text.

   TGTextView::Copy();
   if (fCurrent.fX == 0 && fCurrent.fY == fMarkedEnd.fY) {
      TGLongPosition pos;
      pos.fY = fClipText->RowCount();
      pos.fX = 0;
      fClipText->InsText(pos, 0);
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::Cut()
{
   // Cut text.

   if (!Copy())
      return kFALSE;
   Delete();

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::Paste()
{
   // Paste text into widget.

    gVirtualX->ConvertPrimarySelection(fId, fClipboard, 0);
    return kTRUE;
}

//______________________________________________________________________________
void TGTextEdit::Print(Option_t *) const
{
   // Send current buffer to printer.

   char msg[512];

   sprintf(msg, "%s -P%s\n", gPrintCommand, gPrinter);
   FILE *p = gSystem->OpenPipe(msg, "w");
   if (p) {
      char   *buf1, *buf2;
      Long_t  len;
      ULong_t i = 0;
      TGLongPosition pos;

      pos.fX = pos.fY = 0;
      while (pos.fY < fText->RowCount()) {
         len = fText->GetLineLength(pos.fY);
         buf1 = fText->GetLine(pos, len);
         buf2 = new char[len + 2];
         strncpy(buf2, buf1, (UInt_t)len);
         buf2[len]   = '\n';
         buf2[len+1] = '\0';
         while (buf2[i] != '\0') {
            if (buf2[i] == '\t') {
               ULong_t j = i+1;
               while (buf2[j] == 16 && buf2[j] != '\0')
                  j++;
               strcpy(buf2+i+1, buf2+j);
            }
            i++;
         }
         fwrite(buf2, sizeof(char), strlen(buf2)+1, p);

         delete [] buf1;
         delete [] buf2;
         pos.fY++;
      }
      gSystem->ClosePipe(p);

      Bool_t untitled = !strlen(fText->GetFileName()) ? kTRUE : kFALSE;
      sprintf(msg, "Printed: %s\nLines: %ld\nUsing: %s -P%s",
              untitled ? "Untitled" : fText->GetFileName(),
              fText->RowCount() - 1, gPrintCommand, gPrinter);
      new TGMsgBox(fClient->GetRoot(), this, "Editor", msg,
                   kMBIconAsterisk, kMBOk, 0);
   } else {
      sprintf(msg, "Could not execute: %s -P%s\n", gPrintCommand, gPrinter);
      new TGMsgBox(fClient->GetRoot(), this, "Editor", msg,
                   kMBIconExclamation, kMBOk, 0);
   }
}

//______________________________________________________________________________
void TGTextEdit::Delete(Option_t *)
{
   // Delete selection.

   if (!fIsMarked)
      return;

   if (fMarkedStart.fX == fMarkedEnd.fX &&
       fMarkedStart.fY == fMarkedEnd.fY) {
      Long_t len = fText->GetLineLength(fCurrent.fY);
      if (fCurrent.fY == fText->RowCount()-1 && fCurrent.fX == len) {
         gVirtualX->Bell(0);
         return;
      }
      NextChar();
      DelChar();
      return;
   }

   TGLongPosition pos, endPos;
   Bool_t dellast = kFALSE;

   endPos.fX = fMarkedEnd.fX-1;
   endPos.fY = fMarkedEnd.fY;
   if (endPos.fX == -1) {
      if (endPos.fY > 0)
         endPos.fY--;
      endPos.fX = fText->GetLineLength(endPos.fY);
      if (endPos.fX < 0)
         endPos.fX = 0;
      dellast = kTRUE;
   }

   fText->DelText(fMarkedStart, endPos);
   if (dellast)
      fText->DelLine(endPos.fY);

   pos.fY = ToObjYCoord(fVisible.fY);
   UnMark();
   if (fMarkedStart.fY < pos.fY)
      pos.fY = fMarkedStart.fY;
   pos.fX = ToObjXCoord(fVisible.fX, pos.fY);
   if (fMarkedStart.fX < pos.fX)
      pos.fX = fMarkedStart.fX;
   SetVsbPosition((ToScrYCoord(pos.fY)+fVisible.fY)/fScrollVal.fY);
   SetHsbPosition((ToScrXCoord(pos.fX, pos.fY)+fVisible.fX)/fScrollVal.fX);
   SetSBRange(kHorizontal);
   SetSBRange(kVertical);
   SetCurrent(fMarkedStart);

   SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_ISMARKED), fWidgetId, kFALSE);
   Marked(kFALSE);

   // only to make sure that IsSaved() returns true in case everything has
   // been deleted
   if (fText->RowCount() == 1 && fText->GetLineLength(0) == 0) {
      delete fText;
      fText = new TGText();
      fText->Clear();
   }
}

//______________________________________________________________________________
Bool_t TGTextEdit::Search(const char *string, Bool_t direction,
                          Bool_t caseSensitive)
{
   // Search for string in the specified direction. If direction is true
   // the search will be in forward direction.

   TGLongPosition pos;
   if (!fText->Search(&pos, fCurrent, string, direction, caseSensitive))
      return kFALSE;
   UnMark();
   fIsMarked = kTRUE;
   fMarkedStart.fY = fMarkedEnd.fY = pos.fY;
   fMarkedStart.fX = pos.fX;
   fMarkedEnd.fX = fMarkedStart.fX + strlen(string);

   if (direction)
      SetCurrent(fMarkedEnd);
   else
      SetCurrent(fMarkedStart);

   pos.fY = ToObjYCoord(fVisible.fY);
   if (fCurrent.fY < pos.fY ||
       ToScrYCoord(fCurrent.fY) >= (Int_t)fCanvas->GetHeight())
      pos.fY = fMarkedStart.fY;
   pos.fX = ToObjXCoord(fVisible.fX, pos.fY);
   if (fCurrent.fX < pos.fX ||
       ToScrXCoord(fCurrent.fX, pos.fY) >= (Int_t)fCanvas->GetWidth())
      pos.fX = fMarkedStart.fX;

   SetVsbPosition((ToScrYCoord(pos.fY)+fVisible.fY)/fScrollVal.fY);
   SetHsbPosition((ToScrXCoord(pos.fX, pos.fY)+fVisible.fX)/fScrollVal.fX);
   DrawRegion(0, (Int_t)ToScrYCoord(fMarkedStart.fY), fCanvas->GetWidth(),
              UInt_t(ToScrYCoord(fMarkedEnd.fY+1)-ToScrYCoord(fMarkedEnd.fY)));

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::Replace(TGLongPosition textPos, const char *oldText,
                           const char *newText, Bool_t direction, Bool_t caseSensitive)
{
   // Replace text starting at textPos.

   TGLongPosition pos;
   if (!fText->Replace(textPos, oldText, newText, direction, caseSensitive))
      return kFALSE;
   UnMark();
   fIsMarked = kTRUE;
   fMarkedStart.fY = fMarkedEnd.fY = textPos.fY;
   fMarkedStart.fX = textPos.fX;
   fMarkedEnd.fX = fMarkedStart.fX + strlen(newText);

   if (direction)
      SetCurrent(fMarkedEnd);
   else
      SetCurrent(fMarkedStart);

   pos.fY = ToObjYCoord(fVisible.fY);
   if (fCurrent.fY < pos.fY ||
       ToScrYCoord(fCurrent.fY) >= (Int_t)fCanvas->GetHeight())
      pos.fY = fMarkedStart.fY;
   pos.fX = ToObjXCoord(fVisible.fX, pos.fY);
   if (fCurrent.fX < pos.fX ||
       ToScrXCoord(fCurrent.fX, pos.fY) >= (Int_t)fCanvas->GetWidth())
      pos.fX = fMarkedStart.fX;

   SetVsbPosition((ToScrYCoord(pos.fY)+fVisible.fY)/fScrollVal.fY);
   SetHsbPosition((ToScrXCoord(pos.fX, pos.fY)+fVisible.fX)/fScrollVal.fX);
   DrawRegion(0, (Int_t)ToScrYCoord(fMarkedStart.fY), fCanvas->GetWidth(),
              UInt_t(ToScrYCoord(fMarkedEnd.fY+1)-ToScrYCoord(fMarkedEnd.fY)));

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::Goto(Long_t line, Long_t column)
{
   // Goto the specified line.

   if (line < 0)
      line = 0;
   if (line >= fText->RowCount())
      line = fText->RowCount() - 1;
   if (column < 0)
      column = 0;
   if (column > fText->GetLineLength(line))
      column = fText->GetLineLength(line);

   TGLongPosition gotopos, pos;
   gotopos.fY = line;
   gotopos.fX = column;
   SetCurrent(gotopos);

   pos.fY = ToObjYCoord(fVisible.fY);
   if (fCurrent.fY < pos.fY ||
       ToScrYCoord(fCurrent.fY) >= (Int_t)fCanvas->GetHeight())
      pos.fY = gotopos.fY;

   SetVsbPosition((ToScrYCoord(pos.fY)+fVisible.fY)/fScrollVal.fY);
   SetHsbPosition(0);

   return kTRUE;
}

//______________________________________________________________________________
void TGTextEdit::SetInsertMode(EInsertMode mode)
{
   // Sets the mode how characters are entered.

   if (fInsertMode == mode) return;

   fInsertMode = mode;
}

//______________________________________________________________________________
void TGTextEdit::CursorOff()
{
   // If cursor if on, turn it off.

   if (fCursorState == 1)
      DrawCursor(2);
   fCursorState = 2;
}

//______________________________________________________________________________
void TGTextEdit::CursorOn()
{
   // Turn cursor on.

   DrawCursor(1);
   fCursorState = 1;

   if (fCurBlink)
      fCurBlink->Reset();
}

//______________________________________________________________________________
void TGTextEdit::SetCurrent(TGLongPosition new_coord)
{
   // Make the specified position the current position.

   CursorOff();

   fCurrent.fY = new_coord.fY;
   fCurrent.fX = new_coord.fX;

   CursorOn();

   SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_DATACHANGE), fWidgetId, 0);
   DataChanged();
}

//______________________________________________________________________________
void TGTextEdit::DrawCursor(Int_t mode)
{
   // Draw cursor. If mode = 1 draw cursor, if mode = 2 erase cursor.

   char count = -1;
   char cursor = ' ';
   if (fCurrent.fY >= fText->RowCount() || fCurrent.fX > fText->GetLineLength(fCurrent.fY))
      return;

   if (fCurrent.fY >= ToObjYCoord(fVisible.fY) &&
       fCurrent.fY <= ToObjYCoord(fVisible.fY+fCanvas->GetHeight()) &&
       fCurrent.fX >= ToObjXCoord(fVisible.fX, fCurrent.fY) &&
       fCurrent.fX <= ToObjXCoord(fVisible.fX+fCanvas->GetWidth(),fCurrent.fY)) {
      if (fCurrent.fY < fText->RowCount())
         count = fText->GetChar(fCurrent);
      if (count == -1)
         cursor =' ';
      else
         cursor = count;
      if (mode == 2) {
         if (fIsMarked && count != -1) {
            if ((fCurrent.fY > fMarkedStart.fY && fCurrent.fY < fMarkedEnd.fY) ||
                (fCurrent.fY == fMarkedStart.fY && fCurrent.fX >= fMarkedStart.fX &&
                 fCurrent.fY < fMarkedEnd.fY) ||
                (fCurrent.fY == fMarkedEnd.fY && fCurrent.fX < fMarkedEnd.fX &&
                 fCurrent.fY > fMarkedStart.fY) ||
                (fCurrent.fY == fMarkedStart.fY && fCurrent.fY == fMarkedEnd.fY &&
                 fCurrent.fX >= fMarkedStart.fX && fCurrent.fX < fMarkedEnd.fX &&
                 fMarkedStart.fX != fMarkedEnd.fX)) {
               // back ground fillrectangle
               gVirtualX->FillRectangle(fCanvas->GetId(), fSelbackGC,
                                     Int_t(ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                                     Int_t(ToScrYCoord(fCurrent.fY)),
                                     UInt_t(ToScrXCoord(fCurrent.fX+1, fCurrent.fY) -
                                     ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                                     UInt_t(ToScrYCoord(fCurrent.fY+1)-ToScrYCoord(fCurrent.fY)));
               if (count != -1)
                  gVirtualX->DrawString(fCanvas->GetId(), fSelGC, (Int_t)ToScrXCoord(fCurrent.fX,fCurrent.fY),
                       Int_t(ToScrYCoord(fCurrent.fY+1) - fMaxDescent), &cursor, 1);
            } else {
               gVirtualX->ClearArea(fCanvas->GetId(),
                                    Int_t(ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                                    Int_t(ToScrYCoord(fCurrent.fY)),
                                    UInt_t(ToScrXCoord(fCurrent.fX+1, fCurrent.fY) -
                                    ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                                    UInt_t(ToScrYCoord(fCurrent.fY+1)-ToScrYCoord(fCurrent.fY)));
               if (count != -1)
                  gVirtualX->DrawString(fCanvas->GetId(), fNormGC, (Int_t)ToScrXCoord(fCurrent.fX,fCurrent.fY),
                       Int_t(ToScrYCoord(fCurrent.fY+1) - fMaxDescent), &cursor, 1);
            }
         } else {
//            gVirtualX->DrawLine(fCanvas->GetId(), fCursor0GC,
//                                ToScrXCoord(fCurrent.fX, fCurrent.fY),
//                                ToScrYCoord(fCurrent.fY),
//                                ToScrXCoord(fCurrent.fX, fCurrent.fY),
//                                ToScrYCoord(fCurrent.fY+1)-1);
            gVirtualX->FillRectangle(fCanvas->GetId(), fCursor0GC,
                                     Int_t(ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                                     Int_t(ToScrYCoord(fCurrent.fY)),
                                     2,
                                     UInt_t(ToScrYCoord(fCurrent.fY+1)-ToScrYCoord(fCurrent.fY)));
            gVirtualX->DrawString(fCanvas->GetId(), fNormGC, (Int_t)ToScrXCoord(fCurrent.fX,fCurrent.fY),
                       Int_t(ToScrYCoord(fCurrent.fY+1) - fMaxDescent), &cursor, 1);
         }
      } else
         if (mode == 1) {
//            gVirtualX->DrawLine(fCanvas->GetId(), fCursor1GC,
//                                ToScrXCoord(fCurrent.fX, fCurrent.fY),
//                                ToScrYCoord(fCurrent.fY),
//                                ToScrXCoord(fCurrent.fX, fCurrent.fY),
//                                ToScrYCoord(fCurrent.fY+1)-1);
            gVirtualX->FillRectangle(fCanvas->GetId(), fCursor1GC,
                                     Int_t(ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                                     Int_t(ToScrYCoord(fCurrent.fY)),
                                     2,
                                     UInt_t(ToScrYCoord(fCurrent.fY+1)-ToScrYCoord(fCurrent.fY)));
         }
   }
}

//______________________________________________________________________________
void TGTextEdit::AdjustPos()
{
   // Adjust current position.

   TGLongPosition pos;
   pos.fY = fCurrent.fY;
   pos.fX = fCurrent.fX;

   if (pos.fY < ToObjYCoord(fVisible.fY))
      pos.fY = ToObjYCoord(fVisible.fY);
   else if (ToScrYCoord(pos.fY+1) >= (Int_t) fCanvas->GetHeight())
      pos.fY = ToObjYCoord(fVisible.fY + fCanvas->GetHeight())-1;
   if (pos.fX < ToObjXCoord(fVisible.fX, pos.fY))
      pos.fX = ToObjXCoord(fVisible.fX, pos.fY);
   else if (ToScrXCoord(pos.fX, pos.fY) >= (Int_t) fCanvas->GetWidth())
      pos.fX = ToObjXCoord(fVisible.fX + fCanvas->GetWidth(), pos.fY)-1;
   if (pos.fY != fCurrent.fY || pos.fX != fCurrent.fX)
      SetCurrent(pos);
}

//______________________________________________________________________________
Bool_t TGTextEdit::HandleTimer(TTimer *t)
{
   // Handle timer cursor blink timer.

   if (t != fCurBlink) {
      TGTextView::HandleTimer(t);
      return kTRUE;
   }

   if (fCursorState == 1)
      fCursorState = 2;
   else
      fCursorState = 1;

   DrawCursor(fCursorState);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::HandleSelection(Event_t *event)
{
   // Handle selection notify event.

   TString data;
   Int_t   nchar;

   gVirtualX->GetPasteBuffer((Window_t)event->fUser[0], (Atom_t)event->fUser[3],
                             data, nchar, kFALSE);

   if (!nchar) return kTRUE;

   delete fClipText;

   fClipText = new TGText;
   fClipText->LoadBuffer(data.Data());

   TGLongPosition start_src, end_src, pos;

   pos.fX = pos.fY = 0;
   start_src.fY = start_src.fX = 0;
   end_src.fY = fClipText->RowCount()-1;
   end_src.fX = fClipText->GetLineLength(end_src.fY)-1;

   if (end_src.fX < 0)
      end_src.fX = 0;
   fText->InsText(fCurrent, fClipText, start_src, end_src);
   UnMark();
   pos.fY = fCurrent.fY + fClipText->RowCount()-1;
   pos.fX = fClipText->GetLineLength(fClipText->RowCount()-1);
   if (start_src.fY == end_src.fY)
      pos.fX = pos.fX + fCurrent.fX;
   SetCurrent(pos);
   if (ToScrYCoord(pos.fY) >= (Int_t)fCanvas->GetHeight())
      pos.fY = ToScrYCoord(pos.fY) + fVisible.fY - fCanvas->GetHeight()/2;
   else
      pos.fY = fVisible.fY;
   if (ToScrXCoord(pos.fX, fCurrent.fY) >= (Int_t) fCanvas->GetWidth())
      pos.fX = ToScrXCoord(pos.fX, fCurrent.fY) + fVisible.fX + fCanvas->GetWidth()/2;
   else if (ToScrXCoord(pos.fX, fCurrent.fY < 0) && pos.fX != 0) {
      if (fVisible.fX - (Int_t)fCanvas->GetWidth()/2 > 0)
         pos.fX = fVisible.fX - fCanvas->GetWidth()/2;
      else
         pos.fX = 0;
   } else
      pos.fX = fVisible.fX;

   SetSBRange(kHorizontal);
   SetSBRange(kVertical);
   SetVsbPosition(pos.fY/fScrollVal.fY);
   SetHsbPosition(pos.fX/fScrollVal.fX);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::HandleButton(Event_t *event)
{
   // Handle mouse button event in text edit widget.

   if (event->fWindow != fCanvas->GetId())
      return kTRUE;

   TGLongPosition pos;

   TGTextView::HandleButton(event);

   if (event->fType == kButtonPress) {
      if (event->fCode == kButton1 || event->fCode == kButton2) {
         pos.fY = ToObjYCoord(fVisible.fY + event->fY);
         if (pos.fY >= fText->RowCount())
            pos.fY = fText->RowCount()-1;
         pos.fX = ToObjXCoord(fVisible.fX+event->fX, pos.fY);
         if (pos.fX >= fText->GetLineLength(pos.fY))
            pos.fX = fText->GetLineLength(pos.fY);
         while (fText->GetChar(pos) == 16)
            pos.fX++;

         SetCurrent(pos);
      }
      if (event->fCode == kButton2) {
         if (gVirtualX->GetPrimarySelectionOwner() != kNone)
            gVirtualX->ConvertPrimarySelection(fId, fClipboard, event->fTime);
      }
      if (event->fCode == kButton3) {
         SetMenuState();
         fMenu->PlaceMenu(event->fXRoot, event->fYRoot, kFALSE, kTRUE);
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::HandleMotion(Event_t *event)
{
   // Handle mouse motion event in text edit widget.

   TGLongPosition pos;
   if (event->fWindow != fCanvas->GetId())
      return kTRUE;

   if (fScrolling == -1) {
      pos.fY = ToObjYCoord(fVisible.fY+event->fY);
      if (pos.fY >= fText->RowCount())
         pos.fY = fText->RowCount()-1;
      pos.fX = ToObjXCoord(fVisible.fX+event->fX, pos.fY);
      if (pos.fX > fText->GetLineLength(pos.fY))
         pos.fX = fText->GetLineLength(pos.fY);
      if (fText->GetChar(pos) == 16) {
         if (pos.fX < fCurrent.fX)
            pos.fX = fCurrent.fX;
         if (pos.fX > fCurrent.fX)
            do
               pos.fX++;
            while (fText->GetChar(pos) == 16);
      }
      event->fY = (Int_t)ToScrYCoord(pos.fY);
      event->fX = (Int_t)ToScrXCoord(pos.fX, pos.fY);
      if (pos.fY != fCurrent.fY || pos.fX != fCurrent.fX) {
         TGTextView::HandleMotion(event);
         SetCurrent(pos);
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::HandleKey(Event_t *event)
{
   // The key press event handler converts a key press to some line editor
   // action.

   Bool_t mark_ok = kFALSE;
   char   input[10];
   Int_t  n;
   UInt_t keysym;

   if (event->fType == kGKeyPress) {
      gVirtualX->LookupString(event, input, sizeof(input), keysym);
      n = strlen(input);

      AdjustPos();

      switch ((EKeySym)keysym) {   // ignore these keys
         case kKey_Shift:
         case kKey_Control:
         case kKey_Meta:
         case kKey_Alt:
         case kKey_CapsLock:
         case kKey_NumLock:
         case kKey_ScrollLock:
            return kTRUE;
         default:
            break;
      }
      if (event->fState & kKeyControlMask) {   // Cntrl key modifier pressed
         switch((EKeySym)keysym & ~0x20) {   // treat upper and lower the same
            case kKey_A:
               mark_ok = kTRUE;
               Home();
               break;
            case kKey_B:
               mark_ok = kTRUE;
               PrevChar();
               break;
            case kKey_C:
               Copy();
               return kTRUE;
            case kKey_D:
               if (fIsMarked)
                  Cut();
               else {
                  Long_t len = fText->GetLineLength(fCurrent.fY);
                  if (fCurrent.fY == fText->RowCount()-1 && fCurrent.fX == len) {
                     gVirtualX->Bell(0);
                     return kTRUE;
                  }
                  NextChar();
                  DelChar();
               }
               break;
            case kKey_E:
               mark_ok = kTRUE;
               End();
               break;
            case kKey_F:
               mark_ok = kTRUE;
               NextChar();
               break;
            case kKey_H:
               DelChar();
               break;
            case kKey_K:
               End();
               fIsMarked = kTRUE;
               Mark(fCurrent.fX, fCurrent.fY);
               Cut();
               break;
            case kKey_U:
               Home();
               UnMark();
               fMarkedStart.fY = fMarkedEnd.fY = fCurrent.fY;
               fMarkedStart.fX = fMarkedEnd.fX = fCurrent.fX;
               End();
               fIsMarked = kTRUE;
               Mark(fCurrent.fX, fCurrent.fY);
               Cut();
               break;
            case kKey_V:
            case kKey_Y:
               Paste();
               return kTRUE;
            case kKey_X:
               Cut();
               return kTRUE;
            default:
               return kTRUE;
         }
      }
      if (n && keysym >= 32 && keysym < 127 &&     // printable keys
          !(event->fState & kKeyControlMask) &&
          (EKeySym)keysym != kKey_Delete &&
          (EKeySym)keysym != kKey_Backspace) {

         if (fIsMarked)
            Cut();
         InsChar(input[0]);

      } else {

         switch ((EKeySym)keysym) {
            case kKey_F3:
               // typically FindAgain action
               SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_F3), fWidgetId,
                           kTRUE);
               SetMenuState();
               if (fMenu->IsEntryEnabled(kM_SEARCH_FINDAGAIN)) {
                  SendMessage(this, MK_MSG(kC_COMMAND, kCM_MENU),
                              kM_SEARCH_FINDAGAIN, 0);
                  FindAgain();
               }
               break;
            case kKey_Delete:
               if (fIsMarked)
                  Cut();
               else {
                  Long_t len = fText->GetLineLength(fCurrent.fY);
                  if (fCurrent.fY == fText->RowCount()-1 && fCurrent.fX == len) {
                     gVirtualX->Bell(0);
                     return kTRUE;
                  }
                  NextChar();
                  DelChar();
               }
               break;
            case kKey_Return:
            case kKey_Enter:
               BreakLine();
               break;
            case kKey_Tab:
               InsChar('\t');
               break;
            case kKey_Backspace:
               if (fIsMarked)
                  Cut();
               else
                  DelChar();
               break;
            case kKey_Left:
               mark_ok = kTRUE;
               PrevChar();
               break;
            case kKey_Right:
               mark_ok = kTRUE;
               NextChar();
               break;
            case kKey_Up:
               mark_ok = kTRUE;
               LineUp();
               break;
            case kKey_Down:
               mark_ok = kTRUE;
               LineDown();
               break;
            case kKey_PageUp:
               mark_ok = kTRUE;
               ScreenUp();
               break;
            case kKey_PageDown:
               mark_ok = kTRUE;
               ScreenDown();
               break;
            case kKey_Home:
               mark_ok = kTRUE;
               Home();
               break;
            case kKey_End:
               mark_ok = kTRUE;
               End();
               break;
            case kKey_Insert:           // switch on/off insert mode
               SetInsertMode(GetInsertMode() == kInsert ? kReplace : kInsert);
               break;
            default:
               break;
         }
      }
      if ((event->fState & kKeyShiftMask) && mark_ok) {
         fIsMarked = kTRUE;
         Mark(fCurrent.fX, fCurrent.fY);
         Copy();
         SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_ISMARKED), fWidgetId,
                     kTRUE);
         Marked(kTRUE);
      } else {
         if (fIsMarked) {
            fIsMarked = kFALSE;
            UnMark();
            SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_ISMARKED),
                        fWidgetId, kFALSE);
            Marked(kFALSE);
         }
         fMarkedStart.fY = fMarkedEnd.fY = fCurrent.fY;
         fMarkedStart.fX = fMarkedEnd.fX = fCurrent.fX;
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::HandleCrossing(Event_t *event)
{
   // Handle mouse crossing event.

   if (event->fWindow != fCanvas->GetId())
      return kTRUE;

   if (event->fType == kEnterNotify) {
      if (!fCurBlink)
         fCurBlink = new TViewTimer(this, 500);
      fCurBlink->Reset();
      gSystem->AddTimer(fCurBlink);
   } else {
      if (fCurBlink) fCurBlink->Remove();
      if (fCursorState == 2) {
         DrawCursor(1);
         fCursorState = 1;
      }
   }

   TGTextView::HandleCrossing(event);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextEdit::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   // Process context menu messages.

   TGTextView::ProcessMessage(msg, parm1, parm2);

   switch(GET_MSG(msg)) {
      case kC_COMMAND:
         switch(GET_SUBMSG(msg)) {
            case kCM_MENU:
               switch (parm1) {
                  case kM_FILE_NEW:
                  case kM_FILE_CLOSE:
                  case kM_FILE_OPEN:
                     if (!IsSaved()) {
                        Int_t retval;
                        Bool_t untitled = !strlen(fText->GetFileName()) ? kTRUE : kFALSE;
                        char msg[512];

                        sprintf(msg, "Save \"%s\"?",
                                untitled ? "Untitled" : fText->GetFileName());
                        new TGMsgBox(fClient->GetRoot(), this, "Editor", msg,
                           kMBIconExclamation, kMBYes|kMBNo|kMBCancel, &retval);

                        if (retval == kMBCancel)
                           return kTRUE;
                        if (retval == kMBYes)
                           if (!SaveFile(0))
                              return kTRUE;
                     }
                     Clear();
                     if (parm1 == kM_FILE_CLOSE) {
                        SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_CLOSE),
                                    fWidgetId, 0);
                        Closed();
                     }
                     if (parm1 == kM_FILE_OPEN) {
                        TGFileInfo fi;
                        fi.fFileTypes = (char **)gFiletypes;
                        new TGFileDialog(fClient->GetRoot(), this, kFDOpen, &fi);
                        if (fi.fFilename && strlen(fi.fFilename)) {
                           LoadFile(fi.fFilename);
                           SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_OPEN),
                                       fWidgetId, 0);
                           Opened();
                        }
                     }
                     break;
                  case kM_FILE_SAVE:
                     if (SaveFile(0)) {
                        SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_SAVE),
                                    fWidgetId, 0);
                        Saved();
                     }
                     break;
                  case kM_FILE_SAVEAS:
                     if (SaveFile(0, kTRUE)) {
                        SendMessage(fMsgWindow, MK_MSG(kC_TEXTVIEW, kTXT_SAVE),
                                    fWidgetId, 0);
                        SavedAs();
                     }
                     break;
                  case kM_FILE_PRINT:
                     {
                        Int_t ret;
                        if (!gPrinter) {
                           gPrinter = StrDup("892_2_cor"); // use gEnv
                           gPrintCommand = StrDup("xprint");
                        }
                        new TGPrintDialog(fClient->GetRoot(), this, 400, 150,
                                          &gPrinter, &gPrintCommand, &ret);
                        if (ret)
                           Print();
                     }
                     break;
                  case kM_EDIT_CUT:
                     Cut();
                     break;
                  case kM_EDIT_COPY:
                     Copy();
                     break;
                  case kM_EDIT_PASTE:
                     Paste();
                     break;
                  case kM_EDIT_SELECTALL:
                     SelectAll();
                     break;
                  case kM_SEARCH_FIND:
                     {
                        Int_t ret = 0;
                        if (!fSearch)
                           fSearch = new TGSearchType;
                        new TGSearchDialog(fClient->GetRoot(), this, 400, 150,
                                           fSearch, &ret);
                        if (ret) {
                           if (!Search(fSearch->fBuffer, fSearch->fDirection,
                                       fSearch->fCaseSensitive)) {
                              char msg[256];
                              sprintf(msg, "Couldn't find \"%s\"", fSearch->fBuffer);
                              new TGMsgBox(fClient->GetRoot(), this, "Editor", msg,
                                           kMBIconExclamation, kMBOk, 0);
                           }
                        }
                     }
                     break;
                  case kM_SEARCH_FINDAGAIN:
                     if (!Search(fSearch->fBuffer, fSearch->fDirection,
                                 fSearch->fCaseSensitive)) {
                        char msg[256];
                        sprintf(msg, "Couldn't find \"%s\"", fSearch->fBuffer);
                        new TGMsgBox(fClient->GetRoot(), this, "Editor", msg,
                                     kMBIconExclamation, kMBOk, 0);
                     }
                     break;
                  case kM_SEARCH_GOTO:
                     {
                        Long_t ret = fCurrent.fY+1;
                        new TGGotoDialog(fClient->GetRoot(), this, 400, 150, &ret);
                        if (ret > -1) {
                           ret--;   // user specifies lines starting at 1
                           Goto(ret);
                        }
                     }
                     break;
                  default:
                     printf("No action implemented for menu id %ld\n", parm1);
                     break;
               }
            default:
               break;
         }
         break;

      default:
         break;
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGTextEdit::InsChar(char character)
{
   // Insert a character in the text edit widget.

   char *charstring = 0;
   TGLongPosition pos;

   if (character == '\t') {
      pos.fX = fCurrent.fX;
      pos.fY = fCurrent.fY;
      fText->InsChar(pos, '\t');
      pos.fX++;
      while (pos.fX & 0x7)
         pos.fX++;
      fText->ReTab(pos.fY);
      DrawRegion(0, (Int_t)ToScrYCoord(pos.fY), fCanvas->GetWidth(),
                 UInt_t(ToScrYCoord(pos.fY+1) - ToScrYCoord(pos.fY)));
      SetSBRange(kHorizontal);
      if (ToScrXCoord(pos.fX, pos.fY) >= (Int_t)fCanvas->GetWidth()) {
         if (pos.fX != fText->GetLineLength(fCurrent.fY))
            SetHsbPosition((fVisible.fX+fCanvas->GetWidth()/2)/fScrollVal.fX);
         else
            SetHsbPosition(fVisible.fX/fScrollVal.fX+strlen(charstring));
      }
      SetCurrent(pos);
      return;
   } else {
      fText->InsChar(fCurrent, character);
      pos.fX = fCurrent.fX + 1;
      pos.fY = fCurrent.fY;
      charstring = new char[2];
      charstring[1] = '\0';
      charstring[0] = character;
   }
   SetSBRange(kHorizontal);
   if (ToScrXCoord(pos.fX, pos.fY) >= (Int_t)fCanvas->GetWidth()) {
      if (pos.fX != fText->GetLineLength(fCurrent.fY))
         SetHsbPosition((fVisible.fX+fCanvas->GetWidth()/2)/fScrollVal.fX);
      else
         SetHsbPosition(fVisible.fX/fScrollVal.fX+strlen(charstring));
      if (!fHsb)
         gVirtualX->DrawString(fCanvas->GetId(), fNormGC,
                               (Int_t)ToScrXCoord(fCurrent.fX, fCurrent.fY),
                               Int_t(ToScrYCoord(fCurrent.fY+1) - fMaxDescent),
                               charstring, strlen(charstring));
   } else {
      gVirtualX->CopyArea(fCanvas->GetId(), fCanvas->GetId(), fNormGC,
                          (Int_t)ToScrXCoord(fCurrent.fX, fCurrent.fY),
                          (Int_t)ToScrYCoord(fCurrent.fY), fCanvas->GetWidth(),
                          UInt_t(ToScrYCoord(fCurrent.fY+1)-ToScrYCoord(fCurrent.fY)),
                          (Int_t)ToScrXCoord(pos.fX, fCurrent.fY),
                          (Int_t)ToScrYCoord(fCurrent.fY));
      gVirtualX->ClearArea(fCanvas->GetId(),
                           Int_t(ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                           Int_t(ToScrYCoord(fCurrent.fY)),
                           UInt_t(ToScrXCoord(fCurrent.fX+strlen(charstring), fCurrent.fY) -
                           ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                           UInt_t(ToScrYCoord(fCurrent.fY+1)-ToScrYCoord(fCurrent.fY)));
      gVirtualX->DrawString(fCanvas->GetId(), fNormGC,
                            Int_t(ToScrXCoord(fCurrent.fX, fCurrent.fY)),
                            Int_t(ToScrYCoord(fCurrent.fY+1) - fMaxDescent),
                            charstring, strlen(charstring));
      fCursorState = 2;  // the ClearArea effectively turned off the cursor
   }
   delete [] charstring;
   SetCurrent(pos);
}

//______________________________________________________________________________
void TGTextEdit::DelChar()
{
   // Delete a character from the text edit widget.

   if (fCurrent.fY == 0 && fCurrent.fX == 0) {
      gVirtualX->Bell(0);
      return;
   }

   char *buffer;
   TGLongPosition pos, pos2;
   Long_t len;

   pos.fY = fCurrent.fY;
   pos.fX = fCurrent.fX;

   if (fCurrent.fX > 0) {
      pos.fX--;
      if (fText->GetChar(pos) == 16) {
         do {
            pos.fX++;
            fText->DelChar(pos);
            pos.fX -= 2;
         } while (fText->GetChar(pos) != '\t');
         pos.fX++;
         fText->DelChar(pos);
         pos.fX--;
         fText->ReTab(pos.fY);
         DrawRegion(0, (Int_t)ToScrYCoord(pos.fY), fCanvas->GetWidth(),
                    UInt_t(ToScrYCoord(pos.fY+2)-ToScrYCoord(pos.fY)));
      } else {
         pos.fX = fCurrent.fX;
         fText->DelChar(fCurrent);
         pos.fX = fCurrent.fX - 1;
      }
      if (ToScrXCoord(fCurrent.fX-1, fCurrent.fY) < 0)
         SetHsbPosition((fVisible.fX-fCanvas->GetWidth()/2)/fScrollVal.fX);
      SetSBRange(kHorizontal);
      DrawRegion(0, (Int_t)ToScrYCoord(pos.fY), fCanvas->GetWidth(),
                 UInt_t(ToScrYCoord(pos.fY+2)-ToScrYCoord(pos.fY)));
   } else {
      if (fCurrent.fY > 0) {
         len = fText->GetLineLength(fCurrent.fY);
         if (len > 0) {
            buffer = fText->GetLine(fCurrent, len);
            pos.fY--;
            pos.fX = fText->GetLineLength(fCurrent.fY-1);
            fText->InsText(pos, buffer);
            pos.fY++;
            delete [] buffer;
         } else
            pos.fX = fText->GetLineLength(fCurrent.fY-1);
         pos2.fY = ToScrYCoord(fCurrent.fY+1);
         pos.fY = fCurrent.fY - 1;
         fText->DelLine(fCurrent.fY);
         len = fText->GetLineLength(fCurrent.fY-1);
         if (ToScrXCoord(pos.fX, fCurrent.fY-1) >= (Int_t)fCanvas->GetWidth())
            SetHsbPosition((ToScrXCoord(pos.fX, pos.fY)+fVisible.fX-fCanvas->GetWidth()/2)/fScrollVal.fX);

         gVirtualX->CopyArea(fCanvas->GetId(), fCanvas->GetId(), fNormGC, 0,
                             Int_t(pos2.fY), fWidth,
                             UInt_t(fCanvas->GetHeight() - ToScrYCoord(fCurrent.fY)),
                             0, (Int_t)ToScrYCoord(fCurrent.fY));
         if (ToScrYCoord(pos.fY) < 0)
            SetVsbPosition(fVisible.fY/fScrollVal.fY-1);
         DrawRegion(0, (Int_t)ToScrYCoord(pos.fY), fCanvas->GetWidth(),
                    UInt_t(ToScrYCoord(pos.fY+1) - ToScrYCoord(pos.fY)));
         SetSBRange(kVertical);
         SetSBRange(kHorizontal);
      }
   }
   SetCurrent(pos);
}

//______________________________________________________________________________
void TGTextEdit::BreakLine()
{
   // Break a line.

   TGLongPosition pos;
   fText->BreakLine(fCurrent);
   if (ToScrYCoord(fCurrent.fY+2) <= (Int_t)fCanvas->GetHeight()) {
      gVirtualX->CopyArea(fCanvas->GetId(), fCanvas->GetId(), fNormGC, 0,
                          (Int_t)ToScrYCoord(fCurrent.fY+1), fCanvas->GetWidth(),
                          UInt_t(fCanvas->GetHeight()-(ToScrYCoord(fCurrent.fY+2)-
                          ToScrYCoord(fCurrent.fY))),
                          0, (Int_t)ToScrYCoord(fCurrent.fY+2));
      DrawRegion(0, (Int_t)ToScrYCoord(fCurrent.fY), fCanvas->GetWidth(),
                 UInt_t(ToScrYCoord(fCurrent.fY+2) - ToScrYCoord(fCurrent.fY)));

      if (fVisible.fX != 0)
         SetHsbPosition(0);
      SetSBRange(kHorizontal);
      SetSBRange(kVertical);
   } else {
      SetSBRange(kHorizontal);
      SetSBRange(kVertical);
      SetVsbPosition(fVisible.fY/fScrollVal.fY + 1);
      DrawRegion(0, (Int_t)ToScrYCoord(fCurrent.fY),
                 fCanvas->GetWidth(),
                 UInt_t(ToScrYCoord(fCurrent.fY+1) - ToScrYCoord(fCurrent.fY)));
   }
   pos.fY = fCurrent.fY+1;
   pos.fX = 0;
   SetCurrent(pos);
}

//______________________________________________________________________________
void TGTextEdit::ScrollCanvas(Int_t new_top, Int_t direction)
{
   // Scroll the canvas to new_top in the kVertical or kHorizontal direction.

   CursorOff();

   TGTextView::ScrollCanvas(new_top, direction);

   CursorOn();
}

//______________________________________________________________________________
void TGTextEdit::DrawRegion(Int_t x, Int_t y, UInt_t width, UInt_t height)
{
   // Redraw the text edit widget.

   CursorOff();

   TGTextView::DrawRegion(x, y, width, height);

   CursorOn();
}

//______________________________________________________________________________
void TGTextEdit::PrevChar()
{
   // Go to the previous character.

   if (fCurrent.fY == 0 && fCurrent.fX == 0) {
      gVirtualX->Bell(0);
      return;
   }

   TGLongPosition pos;
   Long_t len;

   pos.fY = fCurrent.fY;
   pos.fX = fCurrent.fX;
   if (fCurrent.fX > 0) {
      pos.fX--;
      while (fText->GetChar(pos) == 16)
         pos.fX--;
      if (ToScrXCoord(pos.fX, pos.fY) < 0) {
         if (fVisible.fX-(Int_t)fCanvas->GetWidth()/2 >= 0)
            SetHsbPosition((fVisible.fX-fCanvas->GetWidth()/2)/fScrollVal.fX);
         else
            SetHsbPosition(0);
      }
   } else {
      if (fCurrent.fY > 0) {
         pos.fY = fCurrent.fY - 1;
         len = fText->GetLineLength(pos.fY);
         if (ToScrYCoord(fCurrent.fY) <= 0)
            SetVsbPosition(fVisible.fY/fScrollVal.fY-1);
         if (ToScrXCoord(len, pos.fY) >= (Int_t)fCanvas->GetWidth())
            SetHsbPosition((ToScrXCoord(len, pos.fY)+fVisible.fX -
                            fCanvas->GetWidth()/2)/fScrollVal.fX);
         pos.fX = len;
      }
   }
   SetCurrent(pos);
}

//______________________________________________________________________________
void TGTextEdit::NextChar()
{
   // Go to next character.

   Long_t len = fText->GetLineLength(fCurrent.fY);

   if (fCurrent.fY == fText->RowCount()-1 && fCurrent.fX == len) {
      gVirtualX->Bell(0);
      return;
   }

   TGLongPosition pos;
   pos.fY = fCurrent.fY;
   if (fCurrent.fX < len) {
      if (fText->GetChar(fCurrent) == '\t')
         pos.fX = fCurrent.fX + 8 - (fCurrent.fX & 0x7);
      else
         pos.fX = fCurrent.fX + 1;

      if (ToScrXCoord(pos.fX, pos.fY) >= (Int_t)fCanvas->GetWidth())
         SetHsbPosition(fVisible.fX/fScrollVal.fX+(fCanvas->GetWidth()/2)/fScrollVal.fX);
   } else {
      if (fCurrent.fY < fText->RowCount()-1) {
         pos.fY = fCurrent.fY + 1;
         if (ToScrYCoord(pos.fY+1) >= (Int_t)fCanvas->GetHeight())
            SetVsbPosition(fVisible.fY/fScrollVal.fY+1);
         SetHsbPosition(0);
         pos.fX = 0;
      }
   }
   SetCurrent(pos);
}

//______________________________________________________________________________
void TGTextEdit::LineUp()
{
   // Make current position first line in window by scrolling up.

   TGLongPosition pos;
   Long_t len;
   if (fCurrent.fY > 0) {
      pos.fY = fCurrent.fY - 1;
      if (ToScrYCoord(fCurrent.fY) <= 0)
         SetVsbPosition(fVisible.fY/fScrollVal.fY-1);
      len = fText->GetLineLength(fCurrent.fY-1);
      if (fCurrent.fX > len) {
         if (ToScrXCoord(len, pos.fY) <= 0) {
            if (ToScrXCoord(len, pos.fY) < 0)
               SetHsbPosition(ToScrXCoord(len, pos.fY)+
                            (fVisible.fX-fCanvas->GetWidth()/2)/fScrollVal.fX);
            else
               SetHsbPosition(0);
         }
         pos.fX = len;
      } else
         pos.fX = ToObjXCoord(ToScrXCoord(fCurrent.fX, fCurrent.fY)+fVisible.fX, pos.fY);
      while (fText->GetChar(pos) == 16)
         pos.fX++;
      SetCurrent(pos);
   }
}

//______________________________________________________________________________
void TGTextEdit::LineDown()
{
   // Move one line down.

   TGLongPosition pos;
   Long_t len;
   if (fCurrent.fY < fText->RowCount()-1) {
      len = fText->GetLineLength(fCurrent.fY+1);
      pos.fY = fCurrent.fY + 1;
      if (ToScrYCoord(pos.fY+1) > (Int_t)fCanvas->GetHeight())
         SetVsbPosition(fVisible.fY/fScrollVal.fY+1);
      if (fCurrent.fX > len) {
         if (ToScrXCoord(len, pos.fY) <= 0) {
            if (ToScrXCoord(len, pos.fY) < 0)
               SetHsbPosition((ToScrXCoord(len, pos.fY)+fVisible.fX-fCanvas->GetWidth()/2)/fScrollVal.fX);
            else
               SetHsbPosition(0);
         }
         pos.fX = len;
      } else
         pos.fX = ToObjXCoord(ToScrXCoord(fCurrent.fX, fCurrent.fY)+fVisible.fX, pos.fY);
      while (fText->GetChar(pos) == 16)
         pos.fX++;
      SetCurrent(pos);
   }
}

//______________________________________________________________________________
void TGTextEdit::ScreenUp()
{
   // Move one screen up.

   TGLongPosition pos;
   pos.fX = fCurrent.fX;
   pos.fY = fCurrent.fY - (ToObjYCoord(fCanvas->GetHeight())-ToObjYCoord(0))-1;
   if (fVisible.fY - (Int_t)fCanvas->GetHeight() >= 0) { // +1
      SetVsbPosition((fVisible.fY - fCanvas->GetHeight())/fScrollVal.fY);
   } else {
      pos.fY = 0;
      SetVsbPosition(0);
   }
   while (fText->GetChar(pos) == 16)
      pos.fX++;
   SetCurrent(pos);
}

//______________________________________________________________________________
void TGTextEdit::ScreenDown()
{
   // Move one screen down.

   TGLongPosition pos;
   pos.fX = fCurrent.fX;
   pos.fY = fCurrent.fY + (ToObjYCoord(fCanvas->GetHeight()) - ToObjYCoord(0));
   Long_t count = fText->RowCount()-1;
   if ((Int_t)fCanvas->GetHeight() < ToScrYCoord(count)) {
      SetVsbPosition((fVisible.fY+fCanvas->GetHeight())/fScrollVal.fY);
   } else
      pos.fY = count;
   while (fText->GetChar(pos) == 16)
      pos.fX++;
   SetCurrent(pos);
}

//______________________________________________________________________________
void TGTextEdit::Home()
{
   // Move to beginning of line.

   TGLongPosition pos;
   pos.fY = fCurrent.fY;
   pos.fX = 0;
   SetHsbPosition(0);
   SetCurrent(pos);
}

//______________________________________________________________________________
void TGTextEdit::End()
{
   // Move to end of line.

   TGLongPosition pos;
   pos.fY = fCurrent.fY;
   pos.fX = fText->GetLineLength(pos.fY);
   if (ToScrXCoord(pos.fX, pos.fY) >= (Int_t)fCanvas->GetWidth())
      SetHsbPosition((ToScrXCoord(pos.fX, pos.fY)+fVisible.fX-fCanvas->GetWidth()/2)/fScrollVal.fX);
   SetCurrent(pos);
}

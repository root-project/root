// @(#)root/gui:$Name$:$Id$
// Author: Fons Rademakers   26/04/98

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
// TGText                                                               //
//                                                                      //
// A TGText is a multi line text buffer. It allows the text to be       //
// loaded from file, saved to file and edited. It is used in the        //
// TGTextEdit widget. Single line text is handled by TGTextBuffer.      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGText.h"
#include "TList.h"



ClassImp(TGText)

//______________________________________________________________________________
void TGText::TGTextP()
{
   // Common initialization method.

   fLines      = new TList;
   fCurrent    = 0;
   fCurrentRow = 0;
   fColCount   = 0;
   fRowCount   = 1;
   fIsSaved    = kTRUE;
}

//______________________________________________________________________________
TGText::TGText()
{
   // Create default (empty) object.

   TGTextP();
}

//______________________________________________________________________________
TGText::TGText(TGText *text)
{
   // Create text object and initialize with other text object.

   TGPosition pos, end;

   pos.fX = pos.fY = 0;
   end.fY = text->RowCount() - 1;
   end.fX = text->GetLineLength(end.fY) - 1;
   TGTextP();
   InsText(pos, text, pos, end);
}

//______________________________________________________________________________
TGText::TGText(const char *string)
{
   // Create text object and initialize with string.

   TGPosition pos;

   pos.fX = pos.fY = 0;
   TGTextP();
   InsLine(pos, TString(string));
}

//______________________________________________________________________________
TGText::~TGText()
{
   // Destroy text object.

   Clear();
   delete fLines;
}

//______________________________________________________________________________
void TGText::Clear()
{
   // Clear text object.

   fLines->Delete();
   fCurrent    = 0;
   fCurrentRow = 0;
   fColCount   = 0;
   fRowCount   = 1;
   fIsSaved    = kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::Load(const char *fn, Int_t startpos, Int_t length)
{
   // Load text from file fn. Startpos is the begin from where to
   // load the file and length is the number of characters to read
   // from the file.

   const int kMaxlen = 8000;

   Bool_t      finished = kFALSE;
   Int_t       count, charcount, i, cnt;
   FILE       *fp;
   char        buf[kMaxlen], c, *src, *dst, *buf2;

   if (!(fp = fopen(fn, "r"))) return kFALSE;
   i = 0;
   fseek(fp, startpos, SEEK_SET);
   charcount = 0;
   while (fgets(buf, kMaxlen, fp)) {
      if ((length != -1) && (charcount+(Int_t)strlen(buf) > length)) {
         count = length - charcount;
         finished = kTRUE;
      } else
         count = kMaxlen;
      charcount += strlen(buf);
      buf2 = new char[count+1];
      buf2[count] = '\0';	
      src = buf;
      dst = buf2;
      cnt = 0;
      while ((c = *src++)) {
         // Don't put CR or NL in buffer
         if (c == 0x0D || c == 0x0A)
            break;
         // Expand tabs
         else if (c == 0x09)
            do
               *dst++ = ' ';
            while (((dst-buf2) & 0x7) && (cnt++ < count-1));
         else
            *dst++ = c;
         if (cnt++ >= count-1) break;
      }
      *dst = '\0';
      fLines->Add(new TObjString(buf2));
      ++i;
      delete [] buf2;
      if (finished)
         break;
   }
   fclose(fp);

   // Remember the number of lines
   fRowCount = i;
   if (fRowCount == 0)
      fRowCount++;
   fIsSaved = kTRUE;
   LongestLine();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::Save(const char *fn)
{
   // Save text buffer to file fn.

   FILE *fp;
   if (!(fp = fopen(fn, "w"))) return kFALSE;

   TIter next(fLines);
   TObjString *str;

   while ((str = (TObjString *) next())) {
      int   len = str->GetString().Length();
      char *buffer = new char[len + 2];
      strcpy(buffer, str->GetName());
      buffer[len]   = '\n';
      buffer[len+1] = '\0';
      if (fputs(buffer, fp) == EOF) {
         delete [] buffer;
         fclose(fp);
         return kFALSE;
      }
      delete [] buffer;
   }
   fIsSaved = kTRUE;
   fclose(fp);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::Append(const char *fn)
{
   // Append buffer to file fn.

   FILE *fp;
   if (!(fp = fopen(fn, "a"))) return kFALSE;

   TIter next(fLines);
   TObjString *str;

   while ((str = (TObjString *) next())) {
      int   len = str->GetString().Length();
      char *buffer = new char[len + 2];
      strcpy(buffer, str->GetName());
      buffer[len]   = '\n';
      buffer[len+1] = '\0';
      if (fputs(buffer, fp) == EOF) {
         delete [] buffer;
         fclose(fp);
         return kFALSE;
      }
      delete [] buffer;
   }
   fIsSaved = kTRUE;
   fclose(fp);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::DelChar(TGPosition pos)
{
   // Delete character at specified position pos.

   if ((pos.fY >= fRowCount) || (pos.fY < 0))
      return kFALSE;

   SetCurrentRow(pos.fY);
   if (!fCurrent) return kFALSE;
   TString &str = fCurrent->String();

   if ((pos.fX > str.Length()) || (pos.fX < 0))
      return kFALSE;

   if (pos.fX > 0) {
      if (str.Length() > 0) {
         str.Remove(pos.fX, 1);
         fIsSaved = kFALSE;
         LongestLine();
         return kTRUE;
      }
   } else if (fCurrentRow > 0) {
      TObjString *strb = (TObjString *)fLines->Before(fCurrent);
      strb->String().Append(str);
      fLines->Remove(fCurrent);
      fIsSaved = kFALSE;
      LongestLine();
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::InsChar(TGPosition pos, char c)
{
   // Insert character c at the specified position pos.

   if ((pos.fY >= fRowCount) || (pos.fY < 0) || (pos.fX < 0))
      return kFALSE;
   SetCurrentRow(pos.fY);
   if (!fCurrent) return kFALSE;
   TString &str = fCurrent->String();

   char s[2];
   s[0] = c;
   s[1] = 0;
   str.Insert(pos.fX, s);

   fIsSaved = kFALSE;
   LongestLine();
   return kTRUE;
}

//______________________________________________________________________________
char TGText::GetChar(TGPosition pos)
{
   // Get character a position pos.

   SetCurrentRow(pos.fY);
   if (!fCurrent) return -1;
   TString &str = fCurrent->String();
   if (str.Length() <= pos.fX)
      return -1;
   else
      return str[pos.fX];
}

//______________________________________________________________________________
Bool_t TGText::DelText(TGPosition start, TGPosition end)
{
   // Delete text between start and end positions.

   if ((start.fY < 0) || (start.fY >= fRowCount) ||
       (end.fY < 0)   || (end.fY >= fRowCount))
      return kFALSE;
   if ((end.fX < 0) || (end.fX > GetLineLength(end.fY)))
      return kFALSE;
   SetCurrentRow(start.fY);
   if (!fCurrent) return kFALSE;
   TString &str = fCurrent->String();

   if ((start.fX < 0) || (start.fX > str.Length()))
      return kFALSE;

   str.Remove(start.fX);
   Int_t temp_row = fCurrentRow;
   SetCurrentRow(end.fY);
   if (!fCurrent) return kFALSE;
   TString &str2 = fCurrent->String();
   if ((str2.Length() > 0) && (str2.Length() != end.fX+1)) {
      str2.Remove(0, end.fX);
   }

   while (fCurrentRow > temp_row)
      DelLine(fCurrentRow);
   SetCurrentRow(temp_row);
   fIsSaved = kFALSE;
   LongestLine();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::InsText(TGPosition ins_pos, TGText *src, TGPosition start_src, TGPosition end_src)
{
   // Insert src text from start_src to end_src into text at position ins_pos.

   if ((start_src.fY < 0) || (start_src.fY >= src->RowCount()) ||
       (end_src.fY < 0)   || (end_src.fY >= src->RowCount()))
      return kFALSE;
   if ((start_src.fX < 0) || (start_src.fX >= src->GetLineLength(start_src.fY)) ||
       (end_src.fX < 0)   || (end_src.fX >= src->GetLineLength(end_src.fY)))
      return kFALSE;
   if ((ins_pos.fY < 0) || (ins_pos.fY >= fRowCount))
      return kFALSE;

   SetCurrentRow(ins_pos.fY);
   if (!fCurrent) return kFALSE;
   TString &str = fCurrent->String();
   if ((ins_pos.fX < 0) || (ins_pos.fX > str.Length()))
      return kFALSE;

   TString str2;
   if (!src->GetLine(start_src, str2))
      return kFALSE;

   str.Insert(ins_pos.fX, str2);

   ins_pos.fX = 0;
   ins_pos.fY++;
   start_src.fX = 0;
   start_src.fY++;
   while (start_src.fY <= end_src.fY) {
      src->GetLine(start_src, str2);
      if (start_src.fY == end_src.fY)
         InsLine(ins_pos, str2(0, end_src.fX+1));
      else
         InsLine(ins_pos, str2);
      ins_pos.fY++;
      start_src.fY++;
   }
   LongestLine();
   fIsSaved = kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::InsLine(TGPosition pos, const TString &str)
{
   // Insert string before specified position. Returns false if insertion failed.

   if (!SetCurrentRow(pos.fY))
      fLines->Add(new TObjString(str));
   else
      fLines->AddBefore(fCurrent, new TObjString(str));
   if (fCurrent)
      fCurrentRow++;
   fRowCount++;
   fIsSaved = kFALSE;
   LongestLine();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::DelLine(Int_t row)
{
   // Delete specified row. Returns false if row does not exist.

   if (!SetCurrentRow(row))
      return kFALSE;

   if (fCurrent) {
      TObjString *cur = (TObjString *) fLines->After(fCurrent);
      fLines->Remove(fCurrent);
      delete fCurrent;
      if (cur)
         fCurrent = cur;
      else {
         fCurrent = (TObjString *)fLines->Last();
         fCurrentRow--;
      }
      fRowCount--;
      LongestLine();
   }

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::GetLine(TGPosition pos, TString &str)
{
   // Return string at position pos. Returns false in case pos does not exist.

   SetCurrentRow(pos.fY);
   if (!fCurrent)
      return kFALSE;
   TString &s = fCurrent->String();
   if ((pos.fX < 0) || (pos.fX > s.Length()))
      return kFALSE;
   str = s(pos.fX, s.Length()-pos.fX);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::BreakLine(TGPosition pos)
{
   // Break line at position pos. Returns false if pos does not exist.

   SetCurrentRow(pos.fY);
   if (!fCurrent)
      return kFALSE;
   TString &s = fCurrent->String();
   if ((pos.fX < 0) || (pos.fX > s.Length()))
      return kFALSE;
   TString ns = s(pos.fX, s.Length()-pos.fX);
   s.Remove(pos.fX);
   fLines->AddAfter(fCurrent, new TObjString(ns));
   fIsSaved = kFALSE;
   fRowCount++;
   LongestLine();
   return kTRUE;
}

//______________________________________________________________________________
Int_t TGText::GetLineLength(Int_t row)
{
   // Get length of specified line. Returns -1 if row does not exist.

   SetCurrentRow(row);
   if (!fCurrent)
      return -1;
   else
      return fCurrent->String().Length();
}

//______________________________________________________________________________
Bool_t TGText::SetCurrentRow(Int_t row)
{
   // Make specified row the current row. Returns false if row does not exist.

   if (row < 0) {
      fCurrent = 0;
      fCurrentRow = 0;
      return kFALSE;
   }

   if (row == fCurrentRow && fCurrent)
      return kTRUE;

   int count = 0;

   TObjLink *lnk = fLines->FirstLink();
   while (lnk) {
      if (count == row) break;
      lnk = lnk->Next();
      count++;
   }

   if (!lnk) {
      fCurrent = 0;
      fCurrentRow = 0;
      if (row == 0)
         return kTRUE;
      else
         return kFALSE;
   } else {
      fCurrent = (TObjString *) lnk->GetObject();
      fCurrentRow = row;
      return kTRUE;
   }
}

//______________________________________________________________________________
Bool_t TGText::Search(TGPosition *foundPos, TGPosition start, const char *searchString,
                      Bool_t direction, Bool_t caseSensitive)
{
   // Search for string searchString starting at the specified position going
   // in forward (direction = true) or backward direction. Returns true if
   // found and foundPos is set accordingly.

   SetCurrentRow(start.fY);
   if (!fCurrent)
      return kFALSE;

   if (direction) {
      TString s = fCurrent->GetString();
      if (start.fX > 0)
         s = s(start.fX, s.Length()-start.fX);
      while (1) {
         foundPos->fX = DownSearchBM(s, searchString, caseSensitive);
         if (foundPos->fX != kNPOS) {
            foundPos->fX += start.fX;
            foundPos->fY  = fCurrentRow;
            return kTRUE;
         }
         SetCurrentRow(fCurrentRow+1);
         if (!fCurrent) break;
         s = fCurrent->GetString();
         start.fX = 0;
      }
   } else {
      TString s = fCurrent->GetString();
      if (start.fX > 0)
         s = s(0, start.fX+1);
      while (1) {
         foundPos->fX = UpSearchBM(s, searchString, caseSensitive);
         if (foundPos->fX != kNPOS) {
            foundPos->fY = fCurrentRow;
            return kTRUE;
         }
         SetCurrentRow(fCurrentRow-1);
         if (!fCurrent) break;
         s = fCurrent->GetString();
      }
   }
   return kFALSE;
}

//______________________________________________________________________________
Int_t TGText::DownSearchBM(const TString &s, const char *searchPattern, Bool_t cs)
{
   // Search down in string s for pattern. If cs is true be case sensitive.
   // Returns x position in string or kNPOS if not found.

   if (cs)
      return s.Index(searchPattern, 0);
   else
      return s.Index(searchPattern, 0, TString::kIgnoreCase);
}


//______________________________________________________________________________
Int_t TGText::UpSearchBM(const TString &s, const char *searchPattern, Bool_t cs)
{
   // Search up in string s for pattern. If cs is true be case sensitive.
   // Returns x position in string or kNPOS if not found.

   TString::ECaseCompare cmp;
   if (cs)
      cmp = TString::kExact;
   else
      cmp = TString::kIgnoreCase;

   int iret, ibeg = 0;

   while (1) {
      iret = s.Index(searchPattern, ibeg, cmp);
      if (iret == kNPOS && ibeg == 0) return iret;
      if (iret == kNPOS && ibeg != 0) return ibeg-1;
      ibeg = iret + 1;
   }
}

//______________________________________________________________________________
Bool_t TGText::Replace(TGPosition pos, const char *oldText, const char *newText,
                       Bool_t direction, Bool_t caseSensitive)
{
   // Replace oldText by newText. Returns false if nothing replaced.

   TGPosition delBeg;

   if (Search(&delBeg, pos, oldText, direction, caseSensitive)) {
      TGPosition delEnd;
      delEnd.fX = delBeg.fX + strlen(oldText) - 1;
      delEnd.fY = delBeg.fY;
      DelText(delBeg, delEnd);

      TGText newT(newText);
      TGPosition start, end;
      start.fX = 0;
      start.fY = 0;
      end.fX = strlen(newText) - 1;
      end.fY = 0;
      InsText(delBeg, &newT, start, end);
      return kTRUE;
   }
   return kFALSE;
}

//______________________________________________________________________________
void TGText::LongestLine()
{
   // Set fLongestLine.

   TIter next(fLines);
   TObjString *os;
   int cols, colmax = 0;

   while ((os = (TObjString *) next())) {
      cols = os->String().Length();
      if (cols > colmax)
         colmax = cols;
   }
   fLongestLine = colmax;
}


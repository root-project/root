// @(#)root/gui:$Name:  $:$Id: TGText.cxx,v 1.2 2000/07/03 18:50:57 rdm Exp $
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
// TGTextEdit widget. Single line text is handled by TGTextBuffer       //
// and the TGTextEntry widget.                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGText.h"


const Int_t kMaxLen = 8000;


ClassImp(TGTextLine)

//______________________________________________________________________________
TGTextLine::TGTextLine()
{
   // Create empty line of text (default ctor).

   fLength = 0;
   fString = 0;
   fPrev = fNext = 0;
}

//______________________________________________________________________________
TGTextLine::TGTextLine(TGTextLine *line)
{
   // Initialize line of text with other line of text (not copy ctor).

   fLength = line->GetLineLength();
   fString = 0;
   if (fLength > 0)
      fString = line->GetText(0, line->GetLineLength());
   fPrev = fNext = 0;
}

//______________________________________________________________________________
TGTextLine::TGTextLine(const char *string)
{
   // Initialize line of text with a const char*.

   if (string) {
      fLength = strlen(string);
      fString = new char[fLength+1];
      strcpy(fString, string);
   } else {
      fLength = 0;
      fString = 0;
   }
   fPrev = fNext = 0;
}

//______________________________________________________________________________
TGTextLine::~TGTextLine()
{
   // Delete a line of text.

   if (fString)
      delete fString;
}

//______________________________________________________________________________
void TGTextLine::Clear()
{
   // Clear a line of text.

   if (fString)
      delete fString;
   fString = 0;
   fLength = 0;
}

//______________________________________________________________________________
void TGTextLine::DelText(ULong_t pos, ULong_t length)
{
   // Delete length chars from line starting at position pos.

   char *newstring;

   if ((fLength == 0) || (pos >= fLength) || (pos+length > fLength))
      return;

   if (fLength - length <= 0) {
      delete fString;
      fLength = 0;
      fString = 0;
      return;
   }
   newstring = new char[fLength - length+1];
   strncpy(newstring, fString, pos);
   strncpy(newstring+pos, fString+pos+length, fLength-pos-length);
   delete fString;
   fString = newstring;
   fLength = fLength - length;
   fString[fLength] = '\0';
}

//______________________________________________________________________________
void TGTextLine::InsText(ULong_t pos, const char *text)
{
   // Insert text in line starting at position pos.

   char *newstring;
   if (pos > fLength || !text)
      return;
   newstring = new char[strlen(text)+fLength+1];
   if (fString != 0)
      strncpy(newstring, fString, pos);
   strcpy(newstring+pos, text);
   if (fLength - pos  > 0)
      strncpy(newstring+pos+strlen(text), fString+pos, fLength-pos);
   fLength = fLength + strlen(text);
   delete fString;
   fString = newstring;
   fString[fLength] ='\0';
}

//______________________________________________________________________________
char *TGTextLine::GetText(ULong_t pos, ULong_t length)
{
   // Get length characters from line starting at pos. Returns 0
   // in case pos and length are out of range. The returned string
   // must be freed by the user.

   char *retstring;

   if ((pos >= fLength) || (length > fLength) || (pos+length > fLength))
      return 0;
   retstring = new char[length+1];
   retstring[length] = '\0';
   strncpy(retstring, fString+pos, length);

   return retstring;
}

//______________________________________________________________________________
void TGTextLine::DelChar(ULong_t pos)
{
   // Delete a character from the line.

   char *newstring;
   if ((fLength <= 0) || (pos > fLength))
      return;
   newstring = new char[fLength -1];
   strncpy(newstring, fString, pos-1);
   if (pos < fLength)
      strncpy(newstring+pos-1, fString+pos, fLength-pos);
   delete fString;
   fString = newstring;
   fLength--;
}

//______________________________________________________________________________
void TGTextLine::InsChar(ULong_t pos, char character)
{
   // Insert a character at the specified position.

   char *newstring;
   if (pos > fLength)
      return;
   newstring = new char[fLength+2];
   newstring[fLength+1] = '\0';
   if (fLength > 0)
     strncpy (newstring, fString, pos);
   newstring[pos] = character;
   if (fLength - pos > 0)
      strncpy(newstring+pos+1, fString+pos, fLength-pos);
   delete fString;
   fString = newstring;
   fLength++;
}

//______________________________________________________________________________
char TGTextLine::GetChar(ULong_t pos)
{
   // Get a character at the specified position from the line.
   // Returns -1 if pos is out of range.

   if ((fLength <= 0) || (pos >= fLength))
      return -1;
   return fString[pos];
}


ClassImp(TGText)

//______________________________________________________________________________
void TGText::TGTextP()
{
   // Common initialization method.

   fFirst       = new TGTextLine;
   fCurrent     = fFirst;
   fCurrentRow  = 0;
   fColCount    = 0;
   fRowCount    = 1;
   fLongestLine = 0;
   fIsSaved     = kTRUE;
}

//______________________________________________________________________________
TGText::TGText()
{
   // Create default (empty) text buffer.

   TGTextP();
}

//______________________________________________________________________________
TGText::TGText(TGText *text)
{
   // Create text buffer and initialize with other text buffer.

   TGLongPosition pos, end;

   pos.fX = pos.fY = 0;
   end.fY = text->RowCount() - 1;
   end.fX = text->GetLineLength(end.fY) - 1;
   TGTextP();
   InsText(pos, text, pos, end);
}

//______________________________________________________________________________
TGText::TGText(const char *string)
{
   // Create text buffer and initialize with single line string.

   TGLongPosition pos;

   pos.fX = pos.fY = 0;
   TGTextP();
   InsText(pos, string);
}

//______________________________________________________________________________
TGText::~TGText()
{
   // Destroy text buffer.

   Clear();
   delete fFirst;
}

//______________________________________________________________________________
void TGText::Clear()
{
   // Clear text buffer.

   TGTextLine *travel = fFirst->fNext;
   TGTextLine *toDelete;
   while (travel != 0) {
      toDelete = travel;
      travel = travel->fNext;
      delete toDelete;
   }
   fFirst->Clear();
   fCurrent     = fFirst;
   fCurrentRow  = 0;
   fColCount    = 0;
   fRowCount    = 1;
   fLongestLine = 0;
   fIsSaved     = kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::Load(const char *fn, Long_t startpos, Long_t length)
{
   // Load text from file fn. Startpos is the begin from where to
   // load the file and length is the number of characters to read
   // from the file.

   Bool_t      isFirst = kTRUE;
   Bool_t      finished = kFALSE;
   Long_t      count, charcount, i, cnt;
   FILE       *fp;
   char        buf[kMaxLen], c, *src, *dst, *buffer, *buf2;
   TGTextLine *travel, *temp;

   travel = fFirst;

   if (!(fp = fopen(fn, "r"))) return kFALSE;
   i = 0;
   fseek(fp, startpos, SEEK_SET);
   charcount = 0;
   while (fgets(buf, kMaxLen, fp)) {
      if ((length != -1) && (charcount+(Int_t)strlen(buf) > length)) {
         count = length - charcount;
         finished = kTRUE;
      } else
         count = kMaxLen;
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
         else if (c == 0x09) {
            *dst++ = '\t';
            do
               *dst++ = 16;  //*dst++ = ' ';
            while (((dst-buf2) & 0x7) && (cnt++ < count-1));
         } else
            *dst++ = c;
         if (cnt++ >= count-1) break;
      }
      *dst = '\0';
      temp = new TGTextLine;
      buffer = new char[strlen(buf2)+1];
      strcpy(buffer, buf2);
      temp->fLength = strlen(buf2);
      temp->fString = buffer;
      temp->fNext = temp->fPrev = 0;
      if (isFirst) {
         delete fFirst;
         fFirst   = temp;
         fCurrent = temp;
         travel   = fFirst;
         isFirst  = kFALSE;
      } else {
         travel->fNext = temp;
         temp->fPrev   = travel;
         travel        = travel->fNext;
      }
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
Bool_t TGText::LoadBuffer(const char *txtbuf)
{
   // Load a 0 terminated buffer. Lines will be split a '\n'.

   Bool_t      isFirst = kTRUE;
   Bool_t      finished = kFALSE;
   Long_t      i, cnt;
   TGTextLine *travel, *temp;
   char        buf[kMaxLen], c, *src, *dst, *buffer, *buf2, *s;
   const char *tbuf = txtbuf;

   travel = fFirst;

   if (!tbuf || !strlen(tbuf))
      return kFALSE;

   i = 0;
next:
   if ((s = (char*)strchr(tbuf, '\n'))) {
      if (s-tbuf+1 >= kMaxLen-1) {
         strncpy(buf, tbuf, kMaxLen-2);
         buf[kMaxLen-2] = '\n';
         buf[kMaxLen-1] = 0;
      } else {
         strncpy(buf, tbuf, s-tbuf+1);
         buf[s-tbuf+1] = 0;
      }
      tbuf = s+1;
   } else {
      if ((Int_t)strlen(tbuf) >= kMaxLen) {
         strncpy(buf, tbuf, kMaxLen-1);
          buf[kMaxLen-1] = 0;
      } else
         strcpy(buf, tbuf);
      finished = kTRUE;
   }

   buf2 = new char[kMaxLen+1];
   buf2[kMaxLen] = '\0';
   src = buf;
   dst = buf2;
   cnt = 0;
   while ((c = *src++)) {
      // Don't put CR or NL in buffer
      if (c == 0x0D || c == 0x0A)
         break;
      // Expand tabs
      else if (c == 0x09) {
         *dst++ = '\t';
         do
            *dst++ = 16;  //*dst++ = ' ';
         while (((dst-buf2) & 0x7) && (cnt++ < kMaxLen-1));
      } else
         *dst++ = c;
      if (cnt++ >= kMaxLen-1) break;
   }
   *dst = '\0';
   temp = new TGTextLine;
   buffer = new char[strlen(buf2)+1];
   strcpy(buffer, buf2);
   temp->fLength = strlen(buf2);
   temp->fString = buffer;
   temp->fNext = temp->fPrev = 0;
   if (isFirst) {
      delete fFirst;
      fFirst   = temp;
      fCurrent = temp;
      travel   = fFirst;
      isFirst  = kFALSE;
   } else {
      travel->fNext = temp;
      temp->fPrev   = travel;
      travel        = travel->fNext;
   }
   ++i;
   delete [] buf2;

   if (!finished && tbuf && strlen(tbuf))
      goto next;

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

   char *buffer;
   TGTextLine *travel = fFirst;
   FILE *fp;
   if (!(fp = fopen(fn, "w"))) return kFALSE;

   while (travel) {
      ULong_t i = 0;
      buffer = new char[travel->fLength+2];
      strncpy(buffer, travel->fString, travel->fLength);
      buffer[travel->fLength]   = '\n';
      buffer[travel->fLength+1] = '\0';
      while (buffer[i] != '\0') {
         if (buffer[i] == '\t') {
            ULong_t j = i+1;
            while (buffer[j] == 16 && buffer[j] != '\0')
               j++;
            strcpy(buffer+i+1, buffer+j);
         }
         i++;
      }
      if (fputs(buffer, fp) == EOF) {
         delete [] buffer;
         fclose(fp);
         return kFALSE;
      }
      delete [] buffer;
      travel = travel->fNext;
   }
   fIsSaved = kTRUE;
   fclose(fp);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::Append(const char *fn)
{
   // Append buffer to file fn.

   char *buffer;
   TGTextLine *travel = fFirst;
   FILE *fp;
   if (!(fp = fopen(fn, "a"))) return kFALSE;

   while (travel) {
      ULong_t i = 0;
      buffer = new char[travel->fLength+2];
      strncpy(buffer, travel->fString, travel->fLength);
      buffer[travel->fLength]   = '\n';
      buffer[travel->fLength+1] = '\0';
      while (buffer[i] != '\0') {
         if (buffer[i] == '\t') {
            ULong_t j = i+1;
            while (buffer[j] == 16 && buffer[j] != '\0')
               j++;
            strcpy(buffer+i+1, buffer+j);
         }
         i++;
      }
      if (fputs(buffer, fp) == EOF) {
         delete [] buffer;
         fclose(fp);
         return kFALSE;
      }
      delete [] buffer;
      travel = travel->fNext;
   }
   fIsSaved = kTRUE;
   fclose(fp);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::DelChar(TGLongPosition pos)
{
   // Delete character at specified position pos.

   if ((pos.fY >= fRowCount) || (pos.fY < 0))
      return kFALSE;

   if (!SetCurrentRow(pos.fY)) return kFALSE;
   fCurrent->DelChar(pos.fX);

   fIsSaved = kFALSE;
   LongestLine();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::InsChar(TGLongPosition pos, char c)
{
   // Insert character c at the specified position pos.

   if ((pos.fY >= fRowCount) || (pos.fY < 0) || (pos.fX < 0))
      return kFALSE;

   if (!SetCurrentRow(pos.fY)) return kFALSE;
   fCurrent->InsChar(pos.fX, c);

   fIsSaved = kFALSE;
   LongestLine();
   return kTRUE;
}

//______________________________________________________________________________
char TGText::GetChar(TGLongPosition pos)
{
   // Get character a position pos. If charcater not valid return -1.

   if (pos.fY >= fRowCount)
      return -1;

   if (!SetCurrentRow(pos.fY)) return -1;
   return fCurrent->GetChar(pos.fX);
}

//______________________________________________________________________________
Bool_t TGText::DelText(TGLongPosition start, TGLongPosition end)
{
   // Delete text between start and end positions. Returns false in
   // case of failure (start and end not being within bounds).

   if ((start.fY < 0) || (start.fY >= fRowCount) ||
       (end.fY < 0)   || (end.fY >= fRowCount))
      return kFALSE;
   if ((end.fX < 0) || (end.fX >= GetLineLength(end.fY)))
      return kFALSE;

   char *tempbuffer;

   if (!SetCurrentRow(start.fY)) return kFALSE;

   if (start.fY == end.fY) {
      fCurrent->DelText(start.fX, end.fX-start.fX+1);
      return kTRUE;
   }
   fCurrent->DelText(start.fX, fCurrent->fLength-start.fX);
   SetCurrentRow(fCurrentRow+1);
   for (long i = start.fY+1; i < end.fY; i++)
      DelLine(fCurrentRow);
   tempbuffer = fCurrent->GetText(end.fX+1, fCurrent->fLength-end.fX-1);
   DelLine(fCurrentRow);
   SetCurrentRow(start.fY);
   if (tempbuffer)
      fCurrent->InsText(fCurrent->GetLineLength(), tempbuffer);
   else {
      if (fCurrent->fNext) {
         fCurrent->InsText(fCurrent->fLength, fCurrent->fNext->fString);
         DelLine(fCurrentRow+1);
         SetCurrentRow(start.fY);
      }
   }

   fIsSaved = kFALSE;
   LongestLine();
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::InsText(TGLongPosition ins_pos, TGText *src,
                       TGLongPosition start_src, TGLongPosition end_src)
{
   // Insert src text from start_src to end_src into text at position ins_pos.
   // Returns false in case of failure (start_src, end_src out of range for
   // src, and ins_pos out of range for this).

   if ((start_src.fY < 0) || (start_src.fY >= src->RowCount()) ||
       (end_src.fY < 0)   || (end_src.fY >= src->RowCount()))
      return kFALSE;
   if ((start_src.fX < 0) || (start_src.fX > src->GetLineLength(start_src.fY)) ||
       (end_src.fX < 0)   || (end_src.fX > src->GetLineLength(end_src.fY)))
      return kFALSE;
   if ((ins_pos.fY < 0) || (ins_pos.fY > fRowCount))
      return kFALSE;
   if ((ins_pos.fX < 0) || (ins_pos.fX > GetLineLength(ins_pos.fY)))
      return kFALSE;

   TGLongPosition pos;
   ULong_t len;
   char *lineString;
   char *restString;
   TGTextLine *following;

   if (ins_pos.fY == fRowCount) {  // for appending text
      pos.fY = fRowCount - 1;
      pos.fX = GetLineLength(pos.fY);
      BreakLine(pos);  // current row is set by this
   } else {
      // otherwise going to the desired row
      if (!SetCurrentRow(ins_pos.fY)) return kFALSE;
   }

   // preparing first line to be inserted
   restString = fCurrent->GetText(ins_pos.fX, fCurrent->fLength - ins_pos.fX);
   fCurrent->DelText(ins_pos.fX, fCurrent->fLength - ins_pos.fX);
   following = fCurrent->fNext;
   // inserting first line
   if (start_src.fY == end_src.fY)
      len = end_src.fX - start_src.fX+1;
   else
      len = src->GetLineLength(start_src.fY) - start_src.fX;
   if (len > 0) {
      lineString = src->GetLine(start_src, len);
      fCurrent->InsText(ins_pos.fX, lineString);
      delete [] lineString;
   } //else
      //BreakLine(ins_pos);
   // [...] inserting possible lines
   pos.fY = start_src.fY+1;
   pos.fX = 0;
   for ( ; pos.fY < end_src.fY; pos.fY++) {
      lineString = src->GetLine(pos, src->GetLineLength(pos.fY));
      fCurrent->fNext = new TGTextLine(lineString);
      fCurrent->fNext->fPrev = fCurrent;
      fCurrent = fCurrent->fNext;
      fRowCount++;
      fCurrentRow++;
      delete [] lineString;
   }
   // last line of inserted text is as special as first line
   if (start_src.fY != end_src.fY) {
      pos.fY = end_src.fY;
      pos.fX = 0;
      lineString = src->GetLine(pos, end_src.fX+1);
//      if (lineString) {
         fCurrent->fNext = new TGTextLine(lineString);
         fCurrent->fNext->fPrev = fCurrent;
         fCurrent = fCurrent->fNext;
         fRowCount++;
         fCurrentRow++;
         delete [] lineString;
//      } else {
//         pos.fY = fCurrentRow;
//         pos.fX = 0;
//         BreakLine(pos);
//      }
   }
   // ok, now we have to add the rest of the first destination line
   if (restString) {
      if (ins_pos.fX == 0) {
         fCurrent->fNext = new TGTextLine(restString);
         fCurrent->fNext->fPrev = fCurrent;
         fCurrent = fCurrent->fNext;
         fRowCount++;
         fCurrentRow++;
      } else
         fCurrent->InsText(fCurrent->fLength, restString);
      delete [] restString;
   }
   // now re-linking the rest of the origin text
   fCurrent->fNext = following;
   if (fCurrent->fNext)
      fCurrent->fNext->fPrev = fCurrent;

   LongestLine();
   fIsSaved = kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::InsText(TGLongPosition pos, const char *buffer)
{
   // Insert single line at specified position. Return false in case position
   // is out of bounds.

   if ((pos.fY < 0) || (pos.fY > fRowCount))
      return kFALSE;
   if ((pos.fX < 0) || (pos.fX > GetLineLength(pos.fY)))
      return kFALSE;

   if (pos.fY == fRowCount) {
      SetCurrentRow(fRowCount-1);
      fCurrent->fNext = new TGTextLine(buffer);
      fCurrent->fNext->fPrev = fCurrent;
      fRowCount++;
   } else {
      SetCurrentRow(pos.fY);
      fCurrent->InsText(pos.fX, buffer);
   }
   LongestLine();
   fIsSaved = kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::AddText(TGText *text)
{
   // Add another text buffer to this buffer.

   TGLongPosition end, start_src, end_src;

   end.fY = fRowCount;
   end.fX = 0;
   start_src.fX = start_src.fY = 0;
   end_src.fY   = text->RowCount()-1;
   end_src.fX   = text->GetLineLength(end_src.fY)-1;
   fIsSaved     = kFALSE;
   return InsText(end, text, start_src, end_src);
}

//______________________________________________________________________________
Bool_t TGText::InsLine(ULong_t pos, const char *string)
{
   // Insert string before specified position.
   // Returns false if insertion failed.

   TGTextLine *previous, *newline;
   if ((Long_t)pos >= fRowCount)
      return kFALSE;
   if ((Long_t)pos < fRowCount)
      SetCurrentRow(pos);
   else
      SetCurrentRow(fRowCount-1);
   if (!fCurrent) return kFALSE;

   previous = fCurrent->fPrev;
   newline = new TGTextLine(string);
   newline->fPrev = previous;
   if (previous)
      previous->fNext = newline;
   else
      fFirst = newline;
   newline->fNext = fCurrent;
   fCurrent->fPrev = newline;
   fRowCount++;
   fCurrentRow++;

   LongestLine();
   fIsSaved = kFALSE;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::DelLine(ULong_t pos)
{
   // Delete specified row. Returns false if row does not exist.

   if (!SetCurrentRow(pos) || (fRowCount == 1))
      return kFALSE;

   TGTextLine *travel = fCurrent;
   if (travel == fFirst) {
      fFirst = fFirst->fNext;
      fFirst->fPrev = 0;
   } else {
      travel->fPrev->fNext = travel->fNext;
      if (travel->fNext) {
         travel->fNext->fPrev = travel->fPrev;
         fCurrent = fCurrent->fNext;
      } else {
         fCurrent = fCurrent->fPrev;
         fCurrentRow--;
      }
   }
   delete travel;

   fRowCount--;
   fIsSaved = kFALSE;
   LongestLine();

   return kTRUE;
}

//______________________________________________________________________________
char *TGText::GetLine(TGLongPosition pos, ULong_t length)
{
   // Return string at position pos. Returns 0 in case pos is not valid.
   // The returned string must be deleted by the user.

   if (SetCurrentRow(pos.fY))
      return fCurrent->GetText(pos.fX, length);
   return 0;
}

//______________________________________________________________________________
Bool_t TGText::BreakLine(TGLongPosition pos)
{
   // Break line at position pos. Returns false if pos is not valid.

   if (!SetCurrentRow(pos.fY))
      return kFALSE;
   if ((pos.fX < 0) || (pos.fX > (Long_t)fCurrent->fLength))
      return kFALSE;

   TGTextLine *temp;
   char *tempbuffer;
   if (pos.fX < (Long_t)fCurrent->fLength) {
      tempbuffer = fCurrent->GetText(pos.fX, fCurrent->fLength-pos.fX);
      temp = new TGTextLine(tempbuffer);
      fCurrent->DelText(pos.fX, fCurrent->fLength - pos.fX);
      delete [] tempbuffer;
   } else
      temp = new TGTextLine;
   temp->fPrev = fCurrent;
   temp->fNext = fCurrent->fNext;
   fCurrent->fNext = temp;
   if (temp->fNext)
      temp->fNext->fPrev = temp;
   fIsSaved = kFALSE;
   fRowCount++;
   fCurrentRow++;
   fCurrent = fCurrent->fNext;
   LongestLine();
   return kTRUE;
}

//______________________________________________________________________________
Long_t TGText::GetLineLength(Long_t row)
{
   // Get length of specified line. Returns -1 if row does not exist.

   if (!SetCurrentRow(row))
      return -1;
   else
      return (Long_t)fCurrent->GetLineLength();
}

//______________________________________________________________________________
Bool_t TGText::SetCurrentRow(Long_t row)
{
   // Make specified row the current row. Returns false if row does not exist.
   // In which case fCurrent is not changed or set to the last valid line.

   Long_t count;
   if ((row < 0) || (row >= fRowCount))
      return kFALSE;
   if (row > fCurrentRow) {
      for (count = fCurrentRow; count < row; count++) {
         if (!fCurrent->fNext) {
            fCurrentRow = count;
            return kFALSE;
         }
         fCurrent = fCurrent->fNext;
      }
   } else {
      if (fCurrentRow == row)
         return kTRUE;
      for (count = fCurrentRow; count > row; count--) {
         if (!fCurrent->fPrev) {
            fCurrentRow = count;
            return kFALSE;
         }
         fCurrent = fCurrent->fPrev;
      }
   }
   fCurrentRow = row;
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGText::Search(TGLongPosition *foundPos, TGLongPosition start,
                      const char *searchString,
                      Bool_t direction, Bool_t caseSensitive)
{
   // Search for string searchString starting at the specified position going
   // in forward (direction = true) or backward direction. Returns true if
   // found and foundPos is set accordingly.

   if (!SetCurrentRow(start.fY))
      return kFALSE;

   if (direction) {
      while(1) {
         foundPos->fX = DownSearchBM(fCurrent->fString+start.fX,
                                     searchString, fCurrent->fLength - start.fX,
                                     caseSensitive);
         if (foundPos->fX != -1) {
            foundPos->fX += start.fX;
            foundPos->fY  = fCurrentRow;
            return kTRUE;
         }
         if (!SetCurrentRow(fCurrentRow+1))
            break;
         start.fX = 0;
      }
   } else {
      while(1) {
         foundPos->fX = UpSearchBM(fCurrent->fString, searchString, start.fX,
                                   caseSensitive);
         if (foundPos->fX != -1) {
            foundPos->fY = fCurrentRow;
            return kTRUE;
         }
         if (!SetCurrentRow(fCurrentRow-1))
            break;
         start.fX = fCurrent->fLength;
      }
   }
   return kFALSE;
}

//______________________________________________________________________________
Long_t TGText::DownSearchBM(const char *actualLine, const char *searchPattern,
                            Long_t len, Bool_t cs)
{
   // Search down in string for pattern. If cs is true be case sensitive.
   // Returns x position in string or -1 if not found.

   Long_t d[95], i, j, k, m, n;

   m = strlen(searchPattern);
   n = len;
   if ((len <= 0) || (m > n))
      return -1;

   for (i = 0; i < 95; i++)
      d[i] = m;
   for (i = 0; i < m - 1; i++) {
      d[searchPattern[i]-32] = m - i - 1;
      if (!cs) {
         if ((searchPattern[i] >= 65) && (searchPattern[i] <= 90))
            d[searchPattern[i]] = m - i - 1;
      } else {
         if ((searchPattern[i] >= 97) && (searchPattern[i] <= 122))
            d[searchPattern[i]-64] = m - i - 1;
      }
   }
   i = m;
   do {
      j = m;
      k = i;
      do {
         k--;
         j--;
      } while ((j >= 0) && CharEqual(searchPattern[j], actualLine[k], cs));
      i = i + d[actualLine[i-1]-32];
   } while ((j >= 0) && (i <= n));

   if (j < 0)
      return i - m - d[searchPattern[m-1] - 32];

   return -1;
}

//______________________________________________________________________________
Long_t TGText::UpSearchBM(const char *actualLine, const char *searchPattern,
                          Long_t len, Bool_t cs)
{
   // Search up in string for pattern. If cs is true be case sensitive.
   // Returns x position in string or -1 if not found.

   Long_t d[95], i, j, k, m, n;

   m = strlen(searchPattern);
   n = len;
   if ((len <= 0) || (m > n))
      return -1;
   for (i = 0; i < 95; i++)
      d[i] = m;
   for (i = m - 1; i > 0; i--) {
      d[searchPattern[i]-32] = i;
      if (!cs) {
         if ((searchPattern[i] >= 65) && (searchPattern[i] <= 90))
            d[searchPattern[i]] = i;
      } else {
         if ((searchPattern[i] >= 97) && (searchPattern[i] <= 122))
            d[searchPattern[i]-64] = i;
      }
   }
   i = n - m-1;
   do {
      j = -1;
      k = i;
      do {
         k++;
         j++;
      } while ((j < m) && CharEqual(searchPattern[j], actualLine[k], cs));
      i = i - d[actualLine[i+1]-32];
   } while ((j < m) && (i >= 0));

   if (j >= m)
      return i + m + d[searchPattern[1] - 32];

  return -1;
}

//______________________________________________________________________________
Bool_t TGText::CharEqual(char first, char second, Bool_t cs)
{
   // Check if two characters are equal. If cs is true be case sensitive.

   if (first == second)
      return kTRUE;

   if (!cs) {
      if ((first >= 65) && (first <= 90))
         if (first+32 == second)
            return kTRUE;

      if ((first >= 97) && (first <= 122))
         if (first-32 == second)
            return kTRUE;
   }

   return kFALSE;
}

//______________________________________________________________________________
Bool_t TGText::Replace(TGLongPosition pos, const char *oldText, const char *newText,
                       Bool_t direction, Bool_t caseSensitive)
{
   // Replace oldText by newText. Returns false if nothing replaced.

   TGLongPosition delEnd;
   if (!SetCurrentRow(pos.fY))
      return kFALSE;
   if (direction) {
      if (!DownSearchBM(fCurrent->fString+pos.fX, oldText, fCurrent->fLength-pos.fX, caseSensitive))
         return kFALSE;
   } else {
      if (!UpSearchBM(fCurrent->fString+pos.fX, oldText, fCurrent->fLength-pos.fX, caseSensitive))
         return kFALSE;
      pos.fX = pos.fX - strlen(oldText) - 1;
   }
   delEnd.fY = pos.fY;
   delEnd.fX = pos.fX + strlen(oldText) - 1;
   DelText(pos, delEnd);
   InsText(pos, newText);
   return kTRUE;
}

//______________________________________________________________________________
void TGText::LongestLine()
{
   // Set fLongestLine.

   Long_t line_count = 0;
   TGTextLine *travel = fFirst;
   fColCount = 0;
   while (travel) {
      if ((Long_t)travel->fLength > fColCount) {
         fColCount = travel->fLength;
         fLongestLine = line_count;
      }
      travel = travel->fNext;
      line_count++;
   }
}


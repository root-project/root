// @(#)root/gui:$Id$
// Author: Fons Rademakers   05/01/98

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


/** \class  TGString
    \ingroup guiwidgets

TGString wraps a TString and adds some graphics routines like
drawing, size of string on screen depending on font, etc.

\class TGHotString
\ingroup guiwidgets

TGHotString is a string with a "hot" character underlined.

*/


#include "TGString.h"
#include "TVirtualX.h"
#include "ctype.h"


ClassImp(TGString);
ClassImp(TGHotString);

////////////////////////////////////////////////////////////////////////////////
/// constructor

TGString::TGString(const TGString *s) : TString(s->Data())
{
}

////////////////////////////////////////////////////////////////////////////////
/// Draw string.

void TGString::Draw(Drawable_t id, GContext_t gc, Int_t x, Int_t y)
{
   gVirtualX->DrawString(id, gc, x, y, Data(), Length());
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a string in a column with width w. If string is longer than
/// w wrap it to next line.

void TGString::DrawWrapped(Drawable_t id, GContext_t gc,
                           Int_t x, Int_t y, UInt_t w, FontStruct_t font)
{
   const char *p     = Data();
   const char *prev  = p;
   const char *chunk = p;
   int tw, th, len = Length();

   tw = gVirtualX->TextWidth(font, p, len);
   if (tw <= (int)w) {
      gVirtualX->DrawString(id, gc, x, y, p, len);
      return;
   }

   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(font, max_ascent, max_descent);
   th = max_ascent + max_descent + 1;

   while(1) {
      p = strchr(p, ' ');
      if (p == 0) {
         if (chunk) gVirtualX->DrawString(id, gc, x, y, chunk, strlen(chunk));
         break;
      }
      tw = gVirtualX->TextWidth(font, chunk, p-chunk);
      if (tw > (int)w) {
         if (prev == chunk)
            prev = ++p;
         else
            p = prev;
         gVirtualX->DrawString(id, gc, x, y, chunk, prev-chunk-1);
         chunk = prev;
         y += th;
      } else {
         prev = ++p;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Get number of lines of width w the string would take using a certain font.

Int_t TGString::GetLines(FontStruct_t font, UInt_t w)
{
   const char *p     = Data();
   const char *prev  = p;
   const char *chunk = p;
   int tw, nlines, len = Length();

   nlines = 1;

   tw = gVirtualX->TextWidth(font, p, len);
   if (tw <= (int)w) return nlines;

   while(1) {
      p = strchr(p, ' ');
      if (p == 0) break;
      tw = gVirtualX->TextWidth(font, chunk, p-chunk);
      if (tw > (int)w) {
         if (prev == chunk)
            chunk = prev = ++p;
         else
            p = chunk = prev;
         ++nlines;
      } else {
         prev = ++p;
      }
   }
   return nlines;
}


////////////////////////////////////////////////////////////////////////////////
/// Create a hot string.

TGHotString::TGHotString(const char *s) : TGString()
{
   fLastGC = 0;
   fOff1 = fOff2 = 0;

   fHotChar = 0;
   fHotPos  = 0;    // No hotkey defaults the offset to zero

   if (!s) return;

   char *dup = StrDup(s);
   char *p;

   for (p = dup; *p; p++) {
      if (*p == '&') {
         if (p[1] == '&') { // escaped & ?
            // copy the string down over it
            for (char *tmp = p; *tmp; tmp++)
               tmp[0] = tmp[1];
            continue; // and skip to the key char
         }
         // hot key marker - calculate the offset value
         fHotPos  = (p - dup) + 1;
         fHotChar = tolower(p[1]);
         for (; *p; p++) p[0] = p[1];  // copy down
         break;                        // allow only one hotkey per item
      }
   }
   Append(dup);
   delete [] dup;
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a hot string and underline the hot character.

void TGHotString::Draw(Drawable_t id, GContext_t gc, Int_t x, Int_t y)
{
   gVirtualX->DrawString(id, gc, x, y, Data(), Length());

   DrawHotChar(id, gc, x, y);
}

////////////////////////////////////////////////////////////////////////////////
/// Draw a hot string in a column with width w. If string is longer than
/// w wrap it to next line.

void TGHotString::DrawWrapped(Drawable_t id, GContext_t gc,
                              Int_t x, Int_t y, UInt_t w, FontStruct_t font)
{
   const char *p     = Data();
   const char *prev  = p;
   const char *chunk = p;
   int tw, th, len = Length();

   tw = gVirtualX->TextWidth(font, p, len);
   if (tw <= (int)w) {
      gVirtualX->DrawString(id, gc, x, y, p, len);
      DrawHotChar(id, gc, x, y);
      return;
   }

   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(font, max_ascent, max_descent);
   th = max_ascent + max_descent + 1;

   int pcnt = 0;
   while(1) {
      p = strchr(p, ' ');
      if (p == 0) {
         if (chunk) {
            gVirtualX->DrawString(id, gc, x, y, chunk, strlen(chunk));
            if (fHotPos > pcnt && fHotPos <= pcnt+(int)strlen(chunk))
               DrawHotChar(id, gc, x, y);
         }
         break;
      }
      tw = gVirtualX->TextWidth(font, chunk, p-chunk);
      if (tw > (int)w) {
         if (prev == chunk)
            prev = ++p;
         else
            p = prev;
         gVirtualX->DrawString(id, gc, x, y, chunk, prev-chunk-1);
         if (fHotPos > pcnt && fHotPos <= pcnt+prev-chunk-1)
            DrawHotChar(id, gc, x, y);
         pcnt = prev-chunk-1;
         chunk = prev;
         y += th;
      } else {
         prev = ++p;
      }
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Draw the underline under the hot character.

void TGHotString::DrawHotChar(Drawable_t id, GContext_t gc, Int_t x, Int_t y)
{
   if (fHotPos > 0) {
      if (fLastGC != gc) {
         GCValues_t   gcval;
         FontStruct_t font;
         font = gVirtualX->GetGCFont(gc);
         if (font) {
            fOff1   = gVirtualX->TextWidth(font, Data(), fHotPos-1); //+1;
            fOff2   = gVirtualX->TextWidth(font, Data(), fHotPos) - 1;
         }
         else {
            gcval.fMask = kGCFont;
            gVirtualX->GetGCValues(gc, gcval);
            font = gVirtualX->GetFontStruct(gcval.fFont);

            fOff1   = gVirtualX->TextWidth(font, Data(), fHotPos-1); //+1;
            fOff2   = gVirtualX->TextWidth(font, Data(), fHotPos) - 1;

            gVirtualX->FreeFontStruct(font);
         }
         fLastGC = gc;
      }
      gVirtualX->DrawLine(id, gc, x+fOff1, y+1, x+fOff2, y+1);
   }
}


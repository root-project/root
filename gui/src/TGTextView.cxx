// @(#)root/gui:$Name$:$Id$
// Author: Fons Rademakers   23/02/98

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
// A TGTextView displays a file or a text buffer in a frame with a      //
// vertical scrollbar. Internally it uses a TGTextFrame which displays  //
// the text.                                                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGTextView.h"
#include "TGScrollBar.h"


ClassImp(TGTextFrame)
ClassImp(TGTextView)



enum {
   kMaxLines    = 5000,     // initial max number of lines (dynamic)
   kMaxLineSize = 1024,     // max line length
   kMargin      = 5         // margin between border and starting of text
};


//______________________________________________________________________________
TGTextFrame::TGTextFrame(TGWindow *parent, UInt_t w, UInt_t h,
                         UInt_t options, ULong_t back) :
   TGFrame(parent, w, h, options, back)
{
   // Create a text frame.

   // set in TGClient via fgDefaultFontStruct and select font via .rootrc
   fFont = fgDefaultFontStruct;

   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(fFont, max_ascent, max_descent);
   fTHeight = max_ascent + max_descent;

   SetBackgroundColor(back);

   GCValues_t gcv;
   gcv.fMask = kGCFont | kGCForeground | kGCBackground;
   gcv.fForeground = fgBlackPixel;
   gcv.fBackground = back;
   gcv.fFont       = gVirtualX->GetFontHandle(fFont);

   fGC = gVirtualX->CreateGC(fId, &gcv);

   fChars = new char* [kMaxLines];
   fLnlen = new Int_t [kMaxLines];
   fMaxLines = kMaxLines;
   fNlines = fTop = 0;
   memset(fChars, 0, fMaxLines*sizeof(char*));
}

//______________________________________________________________________________
TGTextFrame::~TGTextFrame()
{
   // Delete a text frame object.

   Clear();

   gVirtualX->DeleteGC(fGC);

   delete [] fChars;
   delete [] fLnlen;
}

//______________________________________________________________________________
void TGTextFrame::Expand(Int_t newSize)
{
   // Expand or shrink lines container to newSize.

   if (newSize < 0) {
      Error("Expand", "newSize < 0");
      return;
   }
   if (newSize == fMaxLines)
      return;

   fChars = (char **) TStorage::ReAlloc(fChars, newSize * sizeof(char*),
                                        fMaxLines * sizeof(char*));
   fLnlen = (Int_t *) TStorage::ReAlloc(fLnlen, newSize * sizeof(Int_t),
                                        fMaxLines * sizeof(Int_t));
   fMaxLines = newSize;
}

//______________________________________________________________________________
void TGTextFrame::Clear(Option_t *)
{
   // Clear text frame.

   for (int i = 0; i < fNlines; ++i)
      delete [] fChars[i];

   // go back to default allocation
   Expand(kMaxLines);

   fNlines = fTop = 0;
   gVirtualX->ClearWindow(fId);
}

//______________________________________________________________________________
Bool_t TGTextFrame::LoadFile(const char *filename)
{
   // Load file into text frame.

   FILE *fp;
   int  i, cnt;
   char buf[kMaxLineSize], c, *src;
   char line[kMaxLineSize], *dst;

   if ((fp = fopen(filename, "r")) == 0)
      return kFALSE;

   if (fNlines > 0) Clear();

   // Read each line of the file into the buffer

   i = 0;
   while ((fgets(buf, kMaxLineSize, fp) != 0)) {
      // Expand tabs
      src = buf;
      dst = line;
      cnt = 0;
      while ((c = *src++)) {
         if (c == 0x0D || c == 0x0A)
            break;
         else if (c == 0x09)
            do
               *dst++ = ' ';
            while (((dst-line) & 0x7) && (cnt++ < kMaxLineSize-1));
         else
            *dst++ = c;
         if (cnt++ >= kMaxLineSize-1) break;
      }
      *dst = '\0';
      if (i >= fMaxLines)
         Expand(2*fMaxLines);
      fChars[i] = new char[strlen(line) + 1];
      strcpy(fChars[i], line);
      fLnlen[i] = strlen(fChars[i]);
      ++i;
   }

   fclose(fp);

   // Remember the number of lines, and initialize the current line
   // number to be 0.

   fNlines = i;
   fTop = 0;

   DrawRegion(0, 0, fWidth, fHeight);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGTextFrame::LoadBuffer(const char *txtbuf)
{
   // Load 0 terminated txtbuf buffer into text frame.

   int  i, cnt, last;
   char buf[kMaxLineSize], c, *src;
   char line[kMaxLineSize], *dst, *s;
   const char *tbuf = txtbuf;

   if (!tbuf || !strlen(tbuf))
      return kFALSE;

   if (fNlines > 0) Clear();

   // Read each line of the txtbuf into the buffer

   i = 0;
   last = 0;
next:
      if ((s = (char*)strchr(tbuf, '\n'))) {
         if (s-tbuf+1 >= kMaxLineSize-1) {
            strncpy(buf, tbuf, kMaxLineSize-2);
            buf[kMaxLineSize-2] = '\n';
            buf[kMaxLineSize-1] = 0;
         } else {
            strncpy(buf, tbuf, s-tbuf+1);
            buf[s-tbuf+1] = 0;
         }
         tbuf = s+1;
      } else {
         if (strlen(tbuf) >= kMaxLineSize) {
            strncpy(buf, tbuf, kMaxLineSize-1);
            buf[kMaxLineSize-1] = 0;
         } else
            strcpy(buf, tbuf);
         last = 1;
      }

      // Expand tabs
      src = buf;
      dst = line;
      cnt = 0;
      while ((c = *src++)) {
         if (c == 0x0D || c == 0x0A)
            break;
         else if (c == 0x09)
            do
               *dst++ = ' ';
            while (((dst-line) & 0x7) && (cnt++ < kMaxLineSize-1));
         else
            *dst++ = c;
         if (cnt++ >= kMaxLineSize-1) break;
      }
      *dst = '\0';
      if (i >= fMaxLines)
         Expand(2*fMaxLines);
      fChars[i] = new char[strlen(line) + 1];
      strcpy(fChars[i], line);
      fLnlen[i] = strlen(fChars[i]);
      ++i;

   if (!last) goto next;

   // Remember the number of lines, and initialize the current line
   // number to be 0.

   fNlines = i;
   fTop = 0;

   DrawRegion(0, 0, fWidth, fHeight);

   return kTRUE;
}

//______________________________________________________________________________
void TGTextFrame::DrawRegion(Int_t x, Int_t y, UInt_t w, UInt_t h)
{
   // Draw lines in exposed region.

   int yloc = 0, index = fTop;
   Rectangle_t rect;

   rect.fX = x;
   rect.fY = y;
   rect.fWidth  = w;
   rect.fHeight = h;

   // Set the clip mask of the GC

   gVirtualX->SetClipRectangles(fGC, 0, 0, &rect, 1);

   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(fFont, max_ascent, max_descent);

   // Loop through each line until the bottom of the window is reached,
   // or we run out of lines. Redraw any lines that intersect the exposed
   // region.

   while (index < fNlines && yloc < (Int_t)fHeight) {
      yloc += fTHeight;
      if ((yloc - fTHeight <= rect.fY + rect.fHeight) && (yloc >= rect.fY))
         gVirtualX->DrawString(fId, fGC, kMargin, yloc - max_descent,
                          fChars[index], fLnlen[index]);
      ++index;
   }

   // Set the GC clip mask back to None

   GCValues_t gcv;
   gcv.fMask = kGCClipMask;
   gcv.fClipMask = kNone;
   gVirtualX->ChangeGC(fGC, &gcv);
}

//______________________________________________________________________________
void TGTextFrame::SetTopLine(Int_t new_top)
{
   // Set new top line.

   if (fTop == new_top) return;
   ScrollWindow(new_top);
}

//______________________________________________________________________________
void TGTextFrame::ScrollWindow(Int_t new_top)
{
   // Scrollwindow to make new_top the first line being displayed.

   Point_t points[4];
   int xsrc, ysrc, xdest, ydest;

   // These points are the same for both cases, so set them here.

   points[0].fX = points[3].fX = 0;
   points[1].fX = points[2].fX = fWidth;
   xsrc = xdest = 0;

   if (new_top < fTop) {
      // scroll down...
      ysrc = 0;
      // convert new_top row position to pixels
      ydest = (fTop - new_top) * fTHeight;
      // limit the destination to the window height
      if (ydest > (Int_t)fHeight) ydest = fHeight;
      // Fill in the points array with the bounding box of the area that
      // needs to be redrawn - that is, the area that is not copied.
      points[1].fY = points[0].fY = 0;
      points[3].fY = points[2].fY = ydest + fTHeight; // -1;
   } else {
      // scroll up...
      ydest = 0;
      // convert new_top row position to pixels
      ysrc = (new_top - fTop) * fTHeight;
      // limit the source to the window height
      if (ysrc > (Int_t)fHeight) ysrc = fHeight;
      // Fill in the points array with the bounding box of the area that
      // needs to be redrawn - that is, the area that is not copied.
      points[1].fY = points[0].fY = fHeight - ysrc; // +1;
      points[3].fY = points[2].fY = fHeight;
   }

   // Set the top line of the text buffer
   fTop = new_top;
   // Copy the scrolled region to its new position
   gVirtualX->CopyArea(fId, fId, fGC, xsrc, ysrc, fWidth, fHeight, xdest, ydest);
   // Clear the remaining area of any old text
   gVirtualX->ClearArea(fId, points[0].fX, points[0].fY,
                   0, points[2].fY - points[0].fY);

   DrawRegion(points[0].fX, points[0].fY,
              points[2].fX - points[0].fX, points[2].fY - points[0].fY);
}



//______________________________________________________________________________
TGTextView::TGTextView(TGWindow *parent, UInt_t w, UInt_t h,
                       UInt_t options, ULong_t back) :
   TGCompositeFrame(parent, w, h, options, back)
{
   // Create text view widget.

   SetLayoutManager(new TGHorizontalLayout(this));

   ULong_t background = TGFrame::fgWhitePixel;
   SetBackgroundColor(background);

   fTextCanvas = new TGTextFrame(this, 10, 10, kChildFrame, background);
   fVsb = new TGVScrollBar(this, 10, 10, kChildFrame);

   AddFrame(fTextCanvas, 0);
   AddFrame(fVsb, 0);
}

//______________________________________________________________________________
TGTextView::~TGTextView()
{
   // Delete text view widget.

   delete fTextCanvas;
   delete fVsb;
}

//______________________________________________________________________________
Bool_t TGTextView::LoadFile(const char *fname)
{
   // Load file in text view widget.

   Bool_t retc = fTextCanvas->LoadFile(fname);

   if (retc) {
      fVsb->SetPosition(0);
      Layout();
   }
   return retc;
}

//______________________________________________________________________________
Bool_t TGTextView::LoadBuffer(const char *txtbuf)
{
   // Load 0 terminated txtbuf buffer in text view widget.

   Bool_t retc = fTextCanvas->LoadBuffer(txtbuf);

   if (retc) {
      fVsb->SetPosition(0);
      Layout();
   }
   return retc;
}

//______________________________________________________________________________
void TGTextView::Clear(Option_t *)
{
   // Clear text view widget.

   fTextCanvas->Clear();
   Layout();
}

//______________________________________________________________________________
Bool_t TGTextView::ProcessMessage(Long_t msg, Long_t parm1, Long_t)
{
   // Handle scrollbar messages.

   switch (GET_MSG(msg)) {
      case kC_VSCROLL:
         switch (GET_SUBMSG(msg)) {
            case kSB_SLIDERTRACK:
            case kSB_SLIDERPOS:
               fTextCanvas->SetTopLine((Int_t)parm1);
               break;
          }
          break;
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGTextView::Layout()
{
   // Layout TGTextFrame and vertical scrollbar in the text view widget.

   Int_t   lines, vlines;
   UInt_t  tcw, tch;

   tch = fHeight - (fBorderWidth << 1);
   tcw = fWidth - (fBorderWidth << 1);
   fTextCanvas->SetHeight(tch);
   lines = fTextCanvas->GetLines();
   vlines = fTextCanvas->GetVisibleLines();

   if (lines <= vlines) {
      fVsb->UnmapWindow();
    } else {
      tcw -= fVsb->GetDefaultWidth();
      fVsb->MoveResize(fBorderWidth + (Int_t)tcw, fBorderWidth,
                       fVsb->GetDefaultWidth(), tch);
      fVsb->MapWindow();
      fVsb->SetRange(lines, vlines);
   }

   fTextCanvas->MoveResize(fBorderWidth, fBorderWidth, tcw, tch);
}

//______________________________________________________________________________
void TGTextView::DrawBorder()
{
   // Draw border of text view widget.

   switch (fOptions & (kSunkenFrame | kRaisedFrame | kDoubleBorder)) {
      case kSunkenFrame | kDoubleBorder:
         gVirtualX->DrawLine(fId, fgShadowGC, 0, 0, fWidth-2, 0);
         gVirtualX->DrawLine(fId, fgShadowGC, 0, 0, 0, fHeight-2);
         gVirtualX->DrawLine(fId, fgBlackGC, 1, 1, fWidth-3, 1);
         gVirtualX->DrawLine(fId, fgBlackGC, 1, 1, 1, fHeight-3);

         gVirtualX->DrawLine(fId, fgHilightGC, 0, fHeight-1, fWidth-1, fHeight-1);
         gVirtualX->DrawLine(fId, fgHilightGC, fWidth-1, fHeight-1, fWidth-1, 0);
         gVirtualX->DrawLine(fId, fgBckgndGC,  1, fHeight-2, fWidth-2, fHeight-2);
         gVirtualX->DrawLine(fId, fgBckgndGC,  fWidth-2, 1, fWidth-2, fHeight-2);
         break;

      default:
         TGCompositeFrame::DrawBorder();
         break;
   }
}

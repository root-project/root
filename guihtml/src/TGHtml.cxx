// $Id$
// Author:  Valeriy Onuchin   03/05/2007

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun, Fons Rademakers and Reiner Rohlfs *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**************************************************************************

    HTML widget for xclass. Based on tkhtml 1.28
    Copyright (C) 1997-2000 D. Richard Hipp <drh@acm.org>
    Copyright (C) 2002-2003 Hector Peraza.

    This library is free software; you can redistribute it and/or
    modify it under the terms of the GNU Library General Public
    License as published by the Free Software Foundation; either
    version 2 of the License, or (at your option) any later version.

    This library is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    Library General Public License for more details.

    You should have received a copy of the GNU Library General Public
    License along with this library; if not, write to the Free
    Software Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

**************************************************************************/

#include <ctype.h>
#include <stdlib.h>
#include <string.h>

#include "TGHtml.h"
#include "THashTable.h"
#include "TObjString.h"
#include "TGIdleHandler.h"
#include "TImage.h"
#include "TGScrollBar.h"


ClassImp(TGHtml)

int HtmlTraceMask = 0; //HtmlTrace_Table1 | HtmlTrace_Table4;
int HtmlDepth = 0;

#define DEF_FRAME_BG_COLOR        "#c0c0c0"
#define DEF_FRAME_CURSOR          ""
#define DEF_BUTTON_FG             "black"
#define DEF_BUTTON_HIGHLIGHT_BG   "#d9d9d9"
#define DEF_BUTTON_HIGHLIGHT      "black"


//______________________________________________________________________________
TGHtml::TGHtml(const TGWindow *p, int w, int h, int id) : TGView(p, w, h, id)
{
   //

   int i;

   _exiting = 0;
   pFirst = 0;
   pLast = 0;
   nToken = 0;
   lastSized = 0;
   nextPlaced = 0;
   firstBlock = 0;
   lastBlock = 0;
   firstInput = 0;
   lastInput = 0;
   nInput = 0;
   nForm = 0;
   varId = 0;  // do we need this??
   inputIdx = 0;
   radioIdx = 0;
   selBegin.p = 0;
   selEnd.p = 0;
   pSelStartBlock = 0;
   pSelEndBlock = 0;
   insOnTime = DEF_HTML_INSERT_ON_TIME;
   insOffTime = DEF_HTML_INSERT_OFF_TIME;
   insStatus = 0;
   insTimer = 0;
   ins.p = 0;
   pInsBlock = 0;
   insIndex = 0;
   zText = 0;
   nText = 0;
   nAlloc = 0;
   nComplete = 0;
   iCol = 0;
   iPlaintext = 0;
   pScript = 0;
   fIdle = 0;
   styleStack = 0;
   paraAlignment = ALIGN_None;
   rowAlignment = ALIGN_None;
   anchorFlags = 0;
   inDt = 0;
   inTr = 0;
   inTd = 0;
   anchorStart = 0;
   formStart = 0;
   formElemStart = 0;
   formElemLast = 0;
   loEndPtr = 0;
   loFormStart = 0;
   innerList = 0;
   ResetLayoutContext();
   highlightWidth = 0;
   highlightBgColorPtr = 0;
   highlightColorPtr = 0;
   for (i = 0; i < N_FONT; ++i) aFont[i] = 0;
   memset(fontValid, 0, sizeof(fontValid));
   for (i = 0; i < N_COLOR; ++i) {
      apColor[i] = 0;
      iDark[i] = 0;
      iLight[i] = 0;
   }
   fgColor = AllocColor("black");
   bgColor = AllocColor("white"); //AllocColor("#c0c0c0");
   newLinkColor = AllocColor(DEF_HTML_UNVISITED);
   oldLinkColor = AllocColor(DEF_HTML_VISITED);
   selectionColor = AllocColor(DEF_HTML_SELECTION_COLOR);

   apColor[COLOR_Normal] = fgColor;
   apColor[COLOR_Visited] = oldLinkColor;
   apColor[COLOR_Unvisited] = newLinkColor;
   apColor[COLOR_Selection] = selectionColor;
   apColor[COLOR_Background] = bgColor;

   bgImage = 0;

   SetBackgroundColor(apColor[COLOR_Background]->fPixel);
   SetBackgroundPixmap(0);  // force usage of solid color

   colorUsed = 0;

   for (i = 0; i < N_CACHE_GC; ++i) aGcCache[i].index = 0;
   GcNextToFree = 0;
   imageList = 0;
   zBaseHref = 0;
   innerList = 0;
   formPadding = 5;
   overrideFonts = 0;
   overrideColors = 0;
   HasScript = 0;
   HasFrames = 0;
   AddEndTags = 0;
   TableBorderMin = 0;
   varind = 0;
   idind = 0;
   inParse = 0;
   zGoto = 0;
   exts = 0;
   underlineLinks = kTRUE;
   exportSelection = DEF_HTML_EXPORT_SEL;
   tableRelief = HTML_RELIEF_RAISED;
   ruleRelief = HTML_RELIEF_SUNKEN;
   rulePadding = 5;
   zBase = 0;
   zBaseHref = 0;
   cursor = kPointer;
   maxX = 0;
   maxY = 0;

   margins = TGInsets(HTML_INDENT/4, HTML_INDENT/4,
                    HTML_INDENT/4, HTML_INDENT/4);

   flags = RESIZE_ELEMENTS | RELAYOUT;

   dirtyLeft = LARGE_NUMBER;
   dirtyRight = 0;
   dirtyTop = LARGE_NUMBER;
   dirtyBottom = 0;


   fVsb->SetAccelerated();
   fHsb->SetAccelerated();

   _lastUri = 0;

   AddInput(kExposureMask | kFocusChangeMask);
   AddInput(kButtonPressMask | kButtonReleaseMask | kPointerMotionMask);

   fUidTable = new THashTable(100);
}

//______________________________________________________________________________
TGHtml::~TGHtml()
{
   //

   int i;

   _exiting = 1;
   _Clear();
   for (i = 0; i < N_FONT; i++) {
      if (aFont[i] != 0) fClient->FreeFont(aFont[i]);
   }
   if (insTimer) delete insTimer;
   if (fIdle) delete fIdle;

  // TODO: should also free colors!
}

//______________________________________________________________________________
void TGHtml::UpdateBackgroundStart()
{
   //

   //GCValues_t gcv;
   //unsigned int mask = GCTileStipXOrigin | GCTileStipYOrigin;
//
   //gcv.ts_x_origin = -fVvisible.fX;
   //gcv.ts_y_origin = -fVisibleStart.fY;
   //XChangeGC(GetDisplay(), _backGC, mask, &gcv);
}

//______________________________________________________________________________
void TGHtml::FreeColor(ColorStruct_t *color)
{
   //
  
   gVirtualX->FreeColor(gClient->GetDefaultColormap(), color->fPixel);
   delete color;
}

//______________________________________________________________________________
ColorStruct_t *TGHtml::AllocColor(const char *name)
{
   //

   ColorStruct_t *color = new ColorStruct_t;

   color->fPixel = 0;
   if (gVirtualX->ParseColor(fClient->GetDefaultColormap(), name, *color)) {
      if (!gVirtualX->AllocColor(fClient->GetDefaultColormap(), *color)) {
         // force allocation of pixel 0
         gVirtualX->QueryColor(fClient->GetDefaultColormap(), *color);
         gVirtualX->AllocColor(fClient->GetDefaultColormap(), *color);
      }
   }

   return color;
}

//______________________________________________________________________________
ColorStruct_t *TGHtml::AllocColorByValue(ColorStruct_t *color)
{
   //

   ColorStruct_t *c = new ColorStruct_t;
   *c = *color;

   if (!gVirtualX->AllocColor(gClient->GetDefaultColormap(), *c)) {
      // force allocation of pixel 0
      c->fPixel = 0;
      gVirtualX->QueryColor(gClient->GetDefaultColormap(), *c);
      gVirtualX->AllocColor(gClient->GetDefaultColormap(), *c);
   }

   return c;
}

//______________________________________________________________________________
void TGHtml::Clear(Option_t *)
{
   // Erase all HTML from this widget and clear the screen. This is
   // typically done before loading a new document.

   _Clear();
   TGView::Clear();
   flags |= REDRAW_TEXT | VSCROLL | HSCROLL;
   ScheduleRedraw();
}

//______________________________________________________________________________
int TGHtml::ParseText(char *text, const char *index)
{
   // Appends (or insert at the specified position) the given HTML text to the
   // end of any HTML text that may have been inserted by prior calls to this
   // command.  Then it runs the tokenizer, parser and layout engine as far as
   // possible with the text that is available. The display is updated
   // appropriately.

   SHtmlIndex iStart;
   TGHtmlElement *savePtr;

   iStart.p = 0;
   iStart.i = 0;

   loEndPtr = pLast;

   if (index) {
      int rc = GetIndex(index, &iStart.p, &iStart.i);
      if (rc != 0) return kFALSE;  // malformed index
      if (iStart.p) {
         savePtr = iStart.p->pNext;
         pLast = iStart.p;
         iStart.p->pNext = 0;
      }
   }

   TokenizerAppend(text);

   if (loEndPtr) {
      formStart = loFormStart;
      if (iStart.p && savePtr) {
         AddStyle(loEndPtr);
         pLast->pNext = savePtr;
         savePtr->pPrev = pLast;
         pLast = loEndPtr;
         flags |= REDRAW_TEXT | RELAYOUT;
         ScheduleRedraw();
      } else if (loEndPtr->pNext) {
         AddStyle(loEndPtr->pNext);
      }
   } else if (pFirst) {
      paraAlignment = ALIGN_None;
      rowAlignment = ALIGN_None; 
      anchorFlags = 0;
      inDt = 0;
      anchorStart = 0;
      formStart = 0;  
      innerList = 0;  
      nInput = 0;
      AddStyle(pFirst);
   }
#if 1
   loEndPtr = pLast;
   loFormStart = formStart;
#endif

   flags |= EXTEND_LAYOUT;
   ScheduleRedraw();

   return kTRUE;
}

//______________________________________________________________________________
void TGHtml::SetTableRelief(int relief)
{
   //

   if (tableRelief != relief) {
      tableRelief = relief;
      flags |= RELAYOUT;
      RedrawEverything();
   }
}

//______________________________________________________________________________
void TGHtml::SetRuleRelief(int relief)
{
   //

   if (ruleRelief != relief) {
      ruleRelief = relief;
      flags |= RELAYOUT;
      RedrawEverything();
   }
}

//______________________________________________________________________________
void TGHtml::UnderlineLinks(int onoff)
{
   //

   if (underlineLinks != onoff) {
      underlineLinks = onoff;
//    flags |= RESIZE_ELEMENTS | RELAYOUT;
//    AddStyle(pFirst);

      TGHtmlElement *p;
      SHtmlStyle style = GetCurrentStyle();
      for (p = pFirst; p; p = p->pNext) {
         if (p->type == Html_A) {
            if (anchorStart) {
               style = PopStyleStack(Html_EndA);
               anchorStart = 0;
               anchorFlags = 0;
            }
            char *z = p->MarkupArg("href", 0);
            if (z) {
               style.color = GetLinkColor(z);
               if (underlineLinks) style.flags |= STY_Underline;
               anchorFlags |= STY_Anchor;
               PushStyleStack(Html_EndA, style);
               anchorStart = (TGHtmlAnchor *) p;
            }
         } else if (p->type == Html_EndA) {
            if (anchorStart) {
               ((TGHtmlRef *)p)->pOther = anchorStart;
               style = PopStyleStack(Html_EndA);
               anchorStart = 0;
               anchorFlags = 0;
            }
         }
         p->style.flags &= ~STY_Underline;
         p->style.flags |= (style.flags & STY_Underline);
      }

      RedrawEverything();
   }
}

//______________________________________________________________________________
void TGHtml::SetBaseUri(const char *uri)
{
   //

   if (zBase) delete[] zBase;
   zBase = 0;
   if (uri) zBase = StrDup(uri);
}

//______________________________________________________________________________
int TGHtml::GotoAnchor(const char *name)
{
   //

   char *z;
   TGHtmlElement *p;

   for (p = pFirst; p; p = p->pNext) {
      if (p->type == Html_A) {
         z = p->MarkupArg("name", 0);
         if (z && strcmp(z, name) == 0) {
            ScrollToPosition(TGLongPosition(fVisible.fX, ((TGHtmlAnchor *)p)->y));
            return kTRUE;
         }
      }
   }

   return kFALSE;
}

//______________________________________________________________________________
const char *TGHtml::GetUid(const char *string)
{
   // Given a string, this procedure returns a unique identifier for the
   // string.
   //
   // This procedure returns a pointer to a new char string corresponding to
   // the "string" argument. The new string has a value identical to string
   // (strcmp will return 0), but it's guaranteed that any other calls to this
   // procedure with a string equal to "string" will return exactly the same
   // result (i.e. can compare pointer *values* directly, without having to
   // call strcmp on what they point to).

   //int dummy;

   TObjString *obj = 0;
   obj = (TObjString*)fUidTable->FindObject(string);

   if (!obj) {
      obj = new TObjString(string);
      fUidTable->Add(obj);
   }

   return (const char *)obj->GetName();
}

//______________________________________________________________________________
void TGHtml::ComputeVirtualSize()
{
   //

   fVirtualSize = TGDimension(maxX, maxY);
}

//______________________________________________________________________________
void TGHtml::ClearGcCache()
{
   // Clear the cache of GCs

   int i;

   for (i = 0; i < N_CACHE_GC; i++) {
      if (aGcCache[i].index) {
         gVirtualX->DeleteGC(aGcCache[i].gc);
         aGcCache[i].index = 0;
      }
   }
   GcNextToFree = 0;
}

//______________________________________________________________________________
void TGHtml::ResetLayoutContext()
{
   // Reset the main layout context in the main widget.  This happens
   // before we redo the layout, or just before deleting the widget.

   layoutContext.Reset();
}

//______________________________________________________________________________
void TGHtml::Redraw()
{
   // This routine is invoked in order to redraw all or part of the HTML
   // widget. This might happen because the display has changed, or in
   // response to an expose event. In all cases, though, this routine
   // is called by an idle handler.

   Pixmap_t pixmap;           // The buffer on which to render HTML
   int x, y, w, h;          // Virtual canvas coordinates of area to draw
   int hw;                  // highlight thickness
   int clipwinH, clipwinW;  // Width and height of the clipping window
   TGHtmlBlock *pBlock;      // For looping over blocks to be drawn
   int redoSelection = 0;   // kTRUE to recompute the selection
  
   // Don't do anything if we are in the middle of a parse.

   if (inParse) {
      flags &= ~REDRAW_PENDING;
      return;
   }

   // Recompute the layout, if necessary or requested.
   //
   // Calling LayoutDoc() is tricky because LayoutDoc() may invoke one
   // or more user-overriden methods, and these methods could, in theory,
   // do nasty things. So we have to take precautions:
   //
   // *  Do not remove the REDRAW_PENDING flag until after LayoutDoc()
   //    has been called, to prevent a recursive call to Redraw().

   if ((flags & RESIZE_ELEMENTS) != 0 && (flags & STYLER_RUNNING) == 0) {
      TGHtmlImage *pImage;
      for (pImage = imageList; pImage; pImage = pImage->pNext) {
         pImage->pList = 0;
      }
      lastSized = 0;
      flags &= ~RESIZE_ELEMENTS;
      flags |= RELAYOUT;
   }

   // We used to make a distinction between RELAYOUT and EXTEND_LAYOUT.
   // RELAYOUT would be used when the widget was resized, but the
   // less compute-intensive EXTEND_LAYOUT would be used when new
   // text was appended.
   //
   // Unfortunately, EXTEND_LAYOUT has some problem that arise when
   // tables are used.  The quick fix is to make an EXTEND_LAYOUT do
   // a complete RELAYOUT.  Someday, we need to fix EXTEND_LAYOUT so
   // that it works right...

   if ((flags & (RELAYOUT | EXTEND_LAYOUT)) != 0 
      && (flags & STYLER_RUNNING) == 0) {
      nextPlaced = 0;
      //nInput = 0;
      varId = 0;
      maxX = 0;
      maxY = 0;
      ResetLayoutContext();
      firstBlock = 0;
      lastBlock = 0;
      redoSelection = 1;
      flags &= ~RELAYOUT;
      flags |= HSCROLL | VSCROLL | REDRAW_TEXT | EXTEND_LAYOUT;
   }

   if ((flags & EXTEND_LAYOUT) && pFirst != 0) {
      LayoutDoc();
      flags &= ~EXTEND_LAYOUT;
      FormBlocks();
      MapControls();
      if (redoSelection && selBegin.p && selEnd.p) {
         UpdateSelection(1);
         UpdateInsert();
      }
   }
   flags &= ~REDRAW_PENDING;

   // No need to do any actual drawing if we aren't mapped

////  if (!IsMapped()) return;

   // Update the scrollbars.

   if ((flags & (HSCROLL | VSCROLL)) != 0) {
      ComputeVirtualSize();
      flags &= ~(HSCROLL | VSCROLL);

      if (flags & REDRAW_PENDING) return;
   }

   // Redraw the focus highlight, if requested
   hw = highlightWidth;
   if (flags & REDRAW_FOCUS) {
      if (hw > 0) {
#if 0
      unsigned long color;

      if (flags & GOT_FOCUS) {
        color = highlightColorPtr;
      } else {
        color = highlightBgColorPtr;
      }
      _DrawFocusHighlight(color);
#endif
      }
      flags &= ~REDRAW_FOCUS;
   }

   // If the styler is in a callback, abort further processing.
   // TODO: check this!

   if (flags & STYLER_RUNNING) {
      goto earlyOut;
   }

   MapControls();

   // Compute the virtual canvas coordinates corresponding to the
   // dirty region of the clipping window.

   clipwinW = fCanvas->GetWidth();
   clipwinH = fCanvas->GetHeight();
   if (flags & REDRAW_TEXT) {
      w = clipwinW;
      h = clipwinH;
      x = fVisible.fX;
      y = fVisible.fY;
      dirtyLeft = 0;
      dirtyTop = 0;
      flags &= ~REDRAW_TEXT;
   } else {
      if (dirtyLeft < 0) dirtyLeft = 0;
      if (dirtyRight > clipwinW) dirtyRight = clipwinW;
      if (dirtyTop < 0) dirtyTop = 0;
      if (dirtyBottom > clipwinH) dirtyBottom = clipwinH;
      w = dirtyRight - dirtyLeft;
      h = dirtyBottom - dirtyTop;
      x = fVisible.fX + dirtyLeft;
      y = fVisible.fY + dirtyTop;
   }

   // Skip the rest of the drawing process if the area to be refreshed is
   // less than zero
   if (w > 0 && h > 0) {
      GContext_t gcBg;
      TGRectangle xrec;
      // printf("Redraw %dx%d at %d,%d\n", w, h, x, y);

      // Allocate and clear a pixmap upon which to draw
      gcBg = GetGC(COLOR_Background, FONT_Any);
      pixmap = gVirtualX->CreatePixmap(fCanvas->GetId(), w, h);
      xrec.fX = 0;
      xrec.fY = 0;
      xrec.fW = w;
      xrec.fH = h;
#if 0

//old--    XFillRectangles(GetDisplay(), pixmap, gcBg, &xrec, 1);
//new--    if (bgImage)
//           BGDraw(fVisible.fX, fVisible.fY, w, h, bgImage);
#else

      fWhiteGC.SetTileStipXOrigin(-fVisible.fX - dirtyLeft);
      fWhiteGC.SetTileStipYOrigin(-fVisible.fY - dirtyTop);

      gVirtualX->FillRectangle(pixmap, fWhiteGC.GetGC(), 0, 0, w, h);
      UpdateBackgroundStart();  // back to original
#endif
                       
      // Render all visible HTML onto the pixmap
      for (pBlock = firstBlock; pBlock; pBlock = pBlock->bNext) {
         if (pBlock->top <= y+h && pBlock->bottom >= y-10 &&
            pBlock->left <= x+w && pBlock->right >= x-10) {
            BlockDraw(pBlock, pixmap, x, y, w, h, pixmap);
         }
      }
     
      // Finally, copy the pixmap onto the window and delete the pixmap
      gVirtualX->CopyArea(pixmap, fCanvas->GetId(),
                          gcBg, 0, 0, w, h, dirtyLeft, dirtyTop);
      gVirtualX->Update(kFALSE);

      gVirtualX->DeletePixmap(pixmap);
//    XFlush(GetDisplay());
   }

   // Redraw images, if requested
   if (flags & REDRAW_IMAGES) {
      TGHtmlImage *pImage;
      TGHtmlImageMarkup *pElem;
      int top, bottom, left, right;     // Coordinates of the clipping window
      int imageTop;                     // Top edge of image

      top = fVisible.fY;
      bottom = top + fCanvas->GetHeight();
      left = fVisible.fX;
      right = left + fCanvas->GetWidth();
      for (pImage = imageList; pImage; pImage = pImage->pNext) {
         for (pElem = pImage->pList; pElem; pElem = pElem->iNext) {
            if (pElem->redrawNeeded == 0) continue;
            imageTop = pElem->y - pElem->ascent;
            if (imageTop > bottom || imageTop + pElem->h < top
               || pElem->x > right || pElem->x + pElem->w < left) continue; 

            DrawImage(pElem, fCanvas->GetId(), left, top, right, bottom);
         }
      }
      flags &= ~(REDRAW_IMAGES | ANIMATE_IMAGES);
   }

   // Set the dirty region to the empty set.
earlyOut:
   dirtyTop = LARGE_NUMBER;
   dirtyLeft = LARGE_NUMBER;
   dirtyBottom = 0;
   dirtyRight = 0;

   return;
}

//______________________________________________________________________________
void TGHtml::ScheduleRedraw()
{
   // Make sure that a call to the Redraw() routine has been queued.

   if ((flags & REDRAW_PENDING) == 0 /*&& IsMapped()*/) {
      if (!fIdle) fIdle = new TGIdleHandler(this);
      flags |= REDRAW_PENDING;
   }
}

//______________________________________________________________________________
Bool_t TGHtml::HandleIdleEvent(TGIdleHandler *idle) 
{
   //

   if (idle != fIdle) return kFALSE;
   Redraw();
   delete fIdle;
   fIdle = NULL;
   return kTRUE;
}

//______________________________________________________________________________
void TGHtml::RedrawArea(int left, int top, int right, int bottom)
{
   // If any part of the screen needs to be redrawn, then call this routine
   // with the values of a box (in window coordinates) that needs to be 
   // redrawn. This routine will schedule an idle handler to do the redraw.
   //
   // The box coordinates are relative to the clipping window (fCanvas).

   if (bottom < 0) return;
   if (top > (int)fCanvas->GetHeight()) return;
   if (right < 0) return;
   if (left > (int)fCanvas->GetWidth()) return;
   if (dirtyTop > top) dirtyTop = top;
   if (dirtyLeft > left) dirtyLeft = left;
   if (dirtyBottom < bottom) dirtyBottom = bottom;
   if (dirtyRight < right) dirtyRight = right;
   ScheduleRedraw();
}

//______________________________________________________________________________
void TGHtml::DrawRegion(Int_t x, Int_t y, UInt_t w, UInt_t h) 
{
   //

   TGView::DrawRegion(x, y, w, h);

#if 0
  RedrawArea(x, y, x + w + 1, y + h + 1);
#else
   int left = x;
   int top = y;
   int right = x + w + 1;
   int bottom = y + h + 1;
   if (bottom < 0) return;
   if (top > (int) fCanvas->GetHeight()) return;
   if (right < 0) return;
   if (left > (int)fCanvas->GetWidth()) return;
   if (dirtyTop > top) dirtyTop = top;
   if (dirtyLeft > left) dirtyLeft = left;
   if (dirtyBottom < bottom) dirtyBottom = bottom;
   if (dirtyRight < right) dirtyRight = right;

   flags |= REDRAW_PENDING;
   Redraw();
#endif
   return;
}

//______________________________________________________________________________
Bool_t TGHtml::ItemLayout()
{
   //
#if 0
  flags |= RELAYOUT | VSCROLL | HSCROLL;
  Redraw(); //RedrawEverything();
#else
   nextPlaced = 0;
   //nInput = 0;
   varId = 0;
   maxX = 0;
   maxY = 0;
   ResetLayoutContext();
   firstBlock = 0;
   lastBlock = 0;
   if (pFirst != 0) {
      LayoutDoc();
      FormBlocks();
      MapControls();
      if (selBegin.p && selEnd.p) {
         UpdateSelection(1);
         UpdateInsert();
      }
   }
   ComputeVirtualSize();
   ScheduleRedraw();
#endif
   return kTRUE;
}

//______________________________________________________________________________
void TGHtml::RedrawBlock(TGHtmlBlock *p)
{
   // Redraw the TGHtmlBlock given.

   if (p) {
      RedrawArea(p->left - fVisible.fX, p->top - fVisible.fY,
                 p->right - fVisible.fX + 1, p->bottom - fVisible.fY);
   }
}

//______________________________________________________________________________
void TGHtml::RedrawEverything()
{
   // Call this routine to force the entire widget to be redrawn.

   flags |= REDRAW_FOCUS | REDRAW_TEXT;
   ScheduleRedraw();
}

//______________________________________________________________________________
void TGHtml::RedrawText(int y)
{
   // Call this routine to cause all of the rendered HTML at the
   // virtual canvas coordinate of Y and beyond to be redrawn.

   int clipHeight;     // Height of the clipping window
  
   clipHeight = fCanvas->GetHeight();
   y -= fVisible.fY;
   if (y < clipHeight) {
      RedrawArea(0, y, LARGE_NUMBER, clipHeight);
   }
}

//______________________________________________________________________________
void TGHtml::_Clear()
{
   // Erase all data from the HTML widget. Bring it back to an empty screen.

   int i;
   TGHtmlElement *p, *pNext;

   margins = TGInsets(HTML_INDENT/4, HTML_INDENT/4,
                      HTML_INDENT/4, HTML_INDENT/4);

   DeleteControls();
   for (p = pFirst; p; p = pNext) {
      pNext = p->pNext;
      delete p;
   }
   pFirst = 0;
   pLast = 0;
   nToken = 0;
   if (zText) delete[] zText;
   zText = 0;
   nText = 0;
   nAlloc = 0;
   nComplete = 0;
   iPlaintext = 0;

   for (i = 0; i < N_COLOR; ++i) {
      if (apColor[i] != 0) FreeColor(apColor[i]);
      apColor[i] = 0;
      iDark[i] = 0;
      iLight[i] = 0;
   }

   if (!_exiting) {
      fgColor = AllocColor("black");
      bgColor = AllocColor("white"); //AllocColor("#c0c0c0");
      newLinkColor = AllocColor(DEF_HTML_UNVISITED);
      oldLinkColor = AllocColor(DEF_HTML_VISITED);
      selectionColor = AllocColor(DEF_HTML_SELECTION_COLOR);

      apColor[COLOR_Normal] = fgColor;
      apColor[COLOR_Visited] = oldLinkColor;
      apColor[COLOR_Unvisited] = newLinkColor;
      apColor[COLOR_Selection] = selectionColor;
      apColor[COLOR_Background] = bgColor;

      SetBackgroundColor(apColor[COLOR_Background]->fPixel);
      SetBackgroundPixmap(0);  // use solid color
   }

   colorUsed = 0;
   while (imageList) {
      TGHtmlImage *p = imageList;
      imageList = p->pNext;
      delete p;
   }

   if (bgImage) delete bgImage;
   bgImage = 0;

   while (styleStack) {
      SHtmlStyleStack *p = styleStack;
      styleStack = p->pNext;
      delete p;
   }
   ClearGcCache();
   ResetLayoutContext();
//  if (zBase) delete[] zBase;
//  zBase = 0;

   if (zBaseHref) delete [] zBaseHref;
   zBaseHref = 0;
   lastSized = 0;
   nextPlaced = 0;
   firstBlock = 0;
   lastBlock = 0;
   nInput = 0;
   nForm = 0;
   varId = 0;
   paraAlignment = ALIGN_None;
   rowAlignment = ALIGN_None;
   anchorFlags = 0;
   inDt = 0;
   anchorStart = 0;
   formStart = 0;
   innerList = 0;
   maxX = 0;
   maxY = 0;
#if 0  // in OXView::Clear()
  fVisible = TGPosition(0, 0);
  _virtualSize = TGDimension(0, 0);
  ScrollTTGPosition(fVisible);
#endif
   pInsBlock = 0;
   ins.p = 0;
   selBegin.p = 0;
   selEnd.p = 0;
   pSelStartBlock = 0;
   pSelEndBlock = 0;
   HasScript = 0;
   HasFrames = 0;
   _lastUri = 0;
}

//______________________________________________________________________________
Bool_t TGHtml::HandleTimer(TTimer *t)
{
   if (t == insTimer) {
      if (insTimer) delete insTimer;
      insTimer = NULL;
      FlashCursor();
      return kTRUE;
   } else {
      TGHtmlImage *pImage;
      for (pImage = imageList; pImage; pImage = pImage->pNext) {
         if (pImage->timer == t) {
            AnimateImage(pImage);
            return kTRUE;
         }
      }
   }
   return kFALSE;
}

//______________________________________________________________________________
void TGHtml::FlashCursor()
{
   // Flash the insertion cursor.

   if (pInsBlock == 0 || insOnTime <= 0 || insOffTime <= 0) return;
   RedrawBlock(pInsBlock);
   if ((flags & GOT_FOCUS) == 0) {
      insStatus = 0;
   } else if (insStatus) {
      insTimer = new TTimer(this, insOffTime);
      insStatus = 0;
   } else {
      insTimer = new TTimer(this, insOnTime);
      insStatus = 1;
   }
}

//______________________________________________________________________________
GContext_t TGHtml::GetGC(int color, int font)
{
   // Return a GC from the cache.  As many as N_CACHE_GCs are kept valid
   // at any one time.  They are replaced using an LRU algorithm.
   //
   // A value of FONT_Any (-1) for the font means "don't care".

   int i, j;
   GcCache *p = aGcCache;
   GCValues_t gcValues;
   TGFont *xfont;

   // Check for an existing GC.

   if (color < 0 || color >= N_COLOR) color = 0;
   if (font < FONT_Any || font >= N_FONT) font = FONT_Default;

   for (i = 0; i < N_CACHE_GC; i++, p++) {
      if (p->index == 0) continue;
      if ((font < 0 || p->font == font) && p->color == color) {
         if (p->index > 1) {
            for (j = 0; j < N_CACHE_GC; j++) {
               if (aGcCache[j].index && aGcCache[j].index < p->index ) {
                  aGcCache[j].index++;
               }
            }
            p->index = 1;
         }
         return aGcCache[i].gc;
      }
   }

   // No GC matches. Find a place to allocate a new GC.

   p = aGcCache;
   for (i = 0; i < N_CACHE_GC; i++, p++) {
      if (p->index == 0 || p->index == N_CACHE_GC) break;
   }
   if (i >= N_CACHE_GC) {  // No slot, so free one (round-robin)
      p = aGcCache;
      for (i = 0; i < N_CACHE_GC && i < GcNextToFree; ++i, ++p) {}
         GcNextToFree = (GcNextToFree + 1) % N_CACHE_GC;
         gVirtualX->DeleteGC(p->gc);
   }
   gcValues.fForeground = apColor[color]->fPixel;
   gcValues.fGraphicsExposures = kTRUE;
   gcValues.fMask = kGCForeground | kGCGraphicsExposures;

   if (font < 0) font = FONT_Default;
   xfont = GetFont(font);

   if (xfont) {
      gcValues.fFont = xfont->GetFontHandle();
      gcValues.fMask |= kGCFont;
   }

   p->gc = gVirtualX->CreateGC(fId, &gcValues);

   if (p->index == 0) p->index = N_CACHE_GC + 1;
   for (j = 0; j < N_CACHE_GC; j++) {
      if (aGcCache[j].index && aGcCache[j].index < p->index) {
         aGcCache[j].index++;
      }
   }
   p->index = 1;
   p->font = font;
   p->color = color;

   return p->gc;
}

//______________________________________________________________________________
GContext_t TGHtml::GetAnyGC()
{
   // Retrieve any valid GC. The font and color don't matter since the
   // GC will only be used for copying.

   int i;
   GcCache *p = aGcCache;

   for (i = 0; i < N_CACHE_GC; i++, p++) {
      if (p->index) return p->gc;
   }

   return GetGC(COLOR_Normal, FONT_Default);
}

//______________________________________________________________________________
Bool_t TGHtml::HandleFocusChange(Event_t *event)
{
   //

   if (event->fType == kFocusIn) {
      flags |= GOT_FOCUS | REDRAW_FOCUS;
      ScheduleRedraw();
      UpdateInsert();
   } else {  // FocusOut
      flags &= ~GOT_FOCUS;
      flags |= REDRAW_FOCUS;
      ScheduleRedraw();
  }

  return kTRUE;
}

//______________________________________________________________________________
Bool_t TGHtml::HandleButton(Event_t *event)
{
   //

   if ((event->fType == kButtonPress) && (event->fCode == kButton1)) {
      int x = event->fX + fVisible.fX;
      int y = event->fY + fVisible.fY;
      char *uri = GetHref(x, y);

#if 0  // insertion cursor test
    char ix[20];
    sprintf(ix, "begin");
    SetInsert(ix);
#endif

      if (uri) {
         uri = ResolveUri(uri);
         if (uri) {
            MouseDown(uri);
            //!!delete[] uri;
         }
      }
   } else {
      return TGView::HandleButton(event);
  }
  return kTRUE;
}

//______________________________________________________________________________
Bool_t TGHtml::HandleMotion(Event_t *event)
{
   // handle mouse motion events

   int x = event->fX + fVisible.fX;
   int y = event->fY + fVisible.fY;
   char *uri = GetHref(x, y);

   if (uri) {
      gVirtualX->SetCursor(fId, gVirtualX->CreateCursor(kHand));
   } else {
      gVirtualX->SetCursor(fId, gVirtualX->CreateCursor(kPointer));
   }

   if (uri != _lastUri) {
      _lastUri = uri;
      if (uri) uri = ResolveUri(uri);
      MouseOver(uri);
      //!!if (uri) delete [] uri;
   }

   return kTRUE;
}

//______________________________________________________________________________
TGFont *TGHtml::GetFont(int iFont)
{
   // The rendering and layout routines should call this routine in order to
   // get a font structure. The iFont parameter specifies which of the N_FONT
   // fonts should be obtained. The font is allocated if necessary.

   TGFont *toFree = 0;

   if (iFont < 0) iFont = 0;
   if (iFont >= N_FONT) { iFont = N_FONT - 1; CANT_HAPPEN; }

   // If the font has previously been allocated, but the "fontValid" bitmap
   // shows it is no longer valid, then mark it for freeing later. We use
   // a policy of allocate-before-free because xclass' font cache operates
   // much more efficiently that way.

   if (!FontIsValid(iFont) && aFont[iFont] != 0) {
      toFree = aFont[iFont];
      aFont[iFont] = 0;
   }

   // If we need to allocate a font, first construct the font name then
   // allocate it.

   if (aFont[iFont] == 0) {
      char name[200];         // Name of the font
      char *familyStr = "";
      int iFamily;
      int iSize;
      int size;

      iFamily = FontFamily(iFont) >> 3;
      iSize = FontSize(iFont) + 1;

      switch (iFamily) {
//#define TIMES
#ifdef TIMES
         case 0:  familyStr = "times -%d";                 break;
         case 1:  familyStr = "times -%d bold";            break;
         case 2:  familyStr = "times -%d italic";          break;
         case 3:  familyStr = "times -%d bold italic";     break;
         case 4:  familyStr = "courier -%d";               break;
         case 5:  familyStr = "courier -%d bold";          break;
         case 6:  familyStr = "courier -%d italic";        break;
         case 7:  familyStr = "courier -%d bold italic";   break;
         default: familyStr = "times -16";                 CANT_HAPPEN;
#else
         case 0:  familyStr = "helvetica -%d";             break;
         case 1:  familyStr = "helvetica -%d bold";        break;
         case 2:  familyStr = "helvetica -%d italic";      break;
         case 3:  familyStr = "helvetica -%d bold italic"; break;
         case 4:  familyStr = "courier -%d";               break;
         case 5:  familyStr = "courier -%d bold";          break;
         case 6:  familyStr = "courier -%d italic";        break;
         case 7:  familyStr = "courier -%d bold italic";   break;
         default: familyStr = "helvetica -14";             CANT_HAPPEN;
#endif
      }
#if 0
    switch (iSize) {
      case 1:  size = 6+finc/*8*/;   break;
      case 2:  size = 10+finc/*10*/;  break;
      case 3:  size = 12+finc/*12*/;  break;
      case 4:  size = 14+finc/*14*/;  break;
      case 5:  size = 20+finc/*16*/;  break;
      case 6:  size = 24+finc/*18*/;  break;
      case 7:  size = 30+finc/*24*/;  break;
      default: size = 14+finc/*14*/;  CANT_HAPPEN;
    }
#else
      switch (iSize) {
         case 1:  size = 8;   break;
         case 2:  size = 10;  break;
         case 3:  size = 12;  break;
         case 4:  size = 14;  break;
         case 5:  size = 16;  break;
         case 6:  size = 18;  break;
         case 7:  size = 24;  break;
         default: size = 14;  CANT_HAPPEN;
      }
#endif
#ifdef TIMES
      if (iFamily < 4) size += 2;
#endif

      sprintf(name, familyStr, size);

      // Get the named font
      aFont[iFont] = fClient->GetFont(name);\

      if (aFont[iFont] == 0) {
         fprintf(stderr, "TGHtml: could not get font \"%s\", trying fixed\n",
                      name);
         aFont[iFont] = fClient->GetFont("fixed");
      }
      if (aFont[iFont]==0 ){
         fprintf(stderr, "TGHtml: could not get font \"fixed\", trying "
                      "\"helvetica -12\"\n");
         aFont[iFont] = fClient->GetFont("helvetica -12");
      }
      FontSetValid(iFont);
   }

   // Free the expired font, if any.

   if (toFree) fClient->FreeFont(toFree);

   return aFont[iFont];
}

//______________________________________________________________________________
int TGHtml::InArea(TGHtmlMapArea *p, int left, int top, int x, int y)
{
   // Only support rect and circles for now

   int *ip = p->coords;
   if (!ip) return 0;

   if (p->mType == HTML_MAP_RECT) {
      return ((left + ip[0]) <= x && (left + ip[2]) >= x &&
               (top  + ip[1]) <= y && (top  + ip[3]) >= y);
   } else if (p->mType == HTML_MAP_CIRCLE) {
      int dx = left + ip[0] - x;
      int dy = top + ip[1] - y;
      return (dx * dx + dy * dy <= ip[2] * ip[2]);
   }
   return 0;
}

//______________________________________________________________________________
TGHtmlElement *TGHtml::GetMap(char *name)
{
   //

   TGHtmlElement *p = pFirst;
   char *z, *zb;

   while (p) {
      if (p->type == Html_MAP) {
         z = p->MarkupArg("name", 0);
         zb = p->MarkupArg("shape", 0);
         if (zb && *zb != 'r') return 0;
         if (z && !strcmp(z, name)) return p;
      }
      p = p->pNext;
  }
  return 0;
}

//______________________________________________________________________________
float TGHtml::colorDistance(ColorStruct_t *pA, ColorStruct_t *pB)
{

   // Compute the squared distance between two colors

   float x, y, z;

   x = 0.30 * (pA->fRed - pB->fRed);
   y = 0.61 * (pA->fGreen - pB->fGreen);
   z = 0.11 * (pA->fBlue - pB->fBlue);

   return x*x + y*y + z*z;
}

//______________________________________________________________________________
int TGHtml::GetColorByName(char *zColor)
{
   // This routine returns an index between 0 and N_COLOR-1 which indicates
   // which ColorStruct_t structure in the apColor[] array should be used to describe
   // the color specified by the given name.

   ColorStruct_t *pNew;
   int iColor;
   const char *name;  // unique!
   int i, n;
   char zAltColor[16];

   // Netscape accepts color names that are just HEX values, without
   // the # up front.  This isn't valid HTML, but we support it for
   // compatibility.

   n = strlen(zColor);
   if (n == 6 || n == 3 || n == 9 || n == 12) {
      for (i = 0; i < n; i++) {
         if (!isxdigit(zColor[i])) break;
      }
      if (i == n) {
         sprintf(zAltColor, "#%s", zColor);
      } else {
         strcpy(zAltColor, zColor);
      }
      name = GetUid(zAltColor);
   } else {
      name = GetUid(zColor);
   }

   pNew = AllocColor(name);
   if (pNew == 0) {
      return 0;      // Color 0 is always the default
   }

   iColor = GetColorByValue(pNew);
   FreeColor(pNew);

   return iColor;
}


// Macros used in the computation of appropriate shadow colors.

#define MAX_COLOR    65535
#define MAX(A,B)     ((A)<(B)?(B):(A))
#define MIN(A,B)     ((A)<(B)?(A):(B))

//______________________________________________________________________________
int TGHtml::isDarkColor(ColorStruct_t *p)
{
   // Check to see if the given color is too dark to be easily distinguished
   // from black.

   float x, y, z;

   x = 0.50 * p->fRed;
   y = 1.00 * p->fGreen;
   z = 0.28 * p->fBlue;
   return (x*x + y*y + z*z) < (0.05 * MAX_COLOR * MAX_COLOR);
}

//______________________________________________________________________________
int TGHtml::GetDarkShadowColor(int iBgColor)
{
   // Given that the background color is iBgColor, figure out an
   // appropriate color for the dark part of a 3D shadow.

   if (iDark[iBgColor] == 0) {
      ColorStruct_t *pRef, val;
      pRef = apColor[iBgColor];
      if (isDarkColor(pRef)) {
         int t1, t2;
         t1 = (int) MIN(MAX_COLOR, pRef->fRed * 1.2);
         t2 = (pRef->fRed * 3 + MAX_COLOR) / 4;
         val.fRed = MAX(t1, t2);
         t1 = (int) MIN(MAX_COLOR, pRef->fGreen * 1.2);
         t2 = (pRef->fGreen * 3 + MAX_COLOR) / 4;
         val.fGreen = MAX(t1, t2);
         t1 = (int) MIN(MAX_COLOR, pRef->fBlue * 1.2);
         t2 = (pRef->fBlue * 3 + MAX_COLOR) / 4;
         val.fBlue = MAX(t1, t2);
      } else {
         val.fRed = (unsigned short) (pRef->fRed * 0.6);
         val.fGreen = (unsigned short) (pRef->fGreen * 0.6);
         val.fBlue = (unsigned short) (pRef->fBlue * 0.6);
      }
      iDark[iBgColor] = GetColorByValue(&val) + 1;
   }

   return iDark[iBgColor] - 1;
}

//______________________________________________________________________________
int TGHtml::isLightColor(ColorStruct_t *p)
{
   // Check to see if the given color is too light to be easily distinguished
   // from white.

   return p->fGreen >= 0.85 * MAX_COLOR;
}

//______________________________________________________________________________
int TGHtml::GetLightShadowColor(int iBgColor)
{
   // Given that the background color is iBgColor, figure out an
   // appropriate color for the bright part of the 3D shadow.

   if (iLight[iBgColor] == 0) {
      ColorStruct_t *pRef, val;
      pRef = apColor[iBgColor];
      if (isLightColor(pRef)) {
         val.fRed = (unsigned short) (pRef->fRed * 0.9);
         val.fGreen = (unsigned short) (pRef->fGreen * 0.9);
         val.fBlue = (unsigned short) (pRef->fBlue * 0.9);
      } else {
         int t1, t2;
         t1 = (int) MIN(MAX_COLOR, pRef->fGreen * 1.4);
         t2 = (pRef->fGreen + MAX_COLOR) / 2;
         val.fGreen = MAX(t1, t2);
         t1 = (int) MIN(MAX_COLOR, pRef->fRed * 1.4);
         t2 = (pRef->fRed + MAX_COLOR) / 2;
         val.fRed = MAX(t1, t2);
         t1 = (int) MIN(MAX_COLOR, pRef->fBlue * 1.4);
         t2 = (pRef->fBlue + MAX_COLOR) / 2;
         val.fBlue = MAX(t1, t2);
      }
      iLight[iBgColor] = GetColorByValue(&val) + 1;
   }

   return iLight[iBgColor] - 1;
}

//______________________________________________________________________________
int TGHtml::GetColorByValue(ColorStruct_t *pRef)
{
   // Find a color integer for the color whose color components
   // are given by pRef.

   int i;
   float dist;
   float closestDist;
   int closest;
   int r, g, b;
# define COLOR_MASK  0xf800

   // Search for an exact match
   r = pRef->fRed & COLOR_MASK;
   g = pRef->fGreen & COLOR_MASK;
   b = pRef->fBlue & COLOR_MASK;
   for (i = 0; i < N_COLOR; i++) {
      ColorStruct_t *p = apColor[i];
      if (p && 
         ((p->fRed & COLOR_MASK) == r) && 
         ((p->fGreen & COLOR_MASK) == g) &&
         ((p->fBlue & COLOR_MASK) == b)) {
         colorUsed |= (1<<i);
         return i;
      }
   }

   // No exact matches. Look for a completely unused slot
   for (i = N_PREDEFINED_COLOR; i < N_COLOR; i++) {
      if (apColor[i] == 0) {
         apColor[i] = AllocColorByValue(pRef);
         colorUsed |= (1<<i);
         return i;
      }
   }

   // No empty slots. Look for a slot that contains a color that
   // isn't currently in use.
   for (i = N_PREDEFINED_COLOR; i < N_COLOR; i++) {
      if (((colorUsed >> i) & 1) == 0) {
         FreeColor(apColor[i]);
         apColor[i] = AllocColorByValue(pRef);
         colorUsed |= (1<<i);
         return i;
      }
   }

   // Ok, find the existing color that is closest to the color requested
   // and use it.
   closest = 0;
   closestDist = colorDistance(pRef, apColor[0]);
   for (i = 1; i < N_COLOR; i++) {
      dist = colorDistance(pRef, apColor[i]);
      if (dist < closestDist) {
         closestDist = dist;
         closest = i;
      }
   }

   return closest;
}

//______________________________________________________________________________
char *TGHtml::GetHref(int x, int y, char **target)
{
   // This routine searchs for a hyperlink beneath the coordinates x,y
   // and returns a pointer to the HREF for that hyperlink. The text
   // is held in one of the markup argv[] fields of the <a> markup.

   TGHtmlBlock *pBlock;
   TGHtmlElement *pElem;

   for (pBlock = firstBlock; pBlock; pBlock = pBlock->bNext) {
      if (pBlock->top > y || pBlock->bottom < y ||
          pBlock->left > x || pBlock->right < x) continue;
      pElem = pBlock->pNext;
      if (pElem->type == Html_IMG) {
         TGHtmlImageMarkup *image = (TGHtmlImageMarkup *) pElem;
         if (image->pMap) {
            pElem = image->pMap->pNext;
            while (pElem && pElem->type != Html_EndMAP) {
               if (pElem->type == Html_AREA) {
                  if (InArea((TGHtmlMapArea *) pElem, pBlock->left, pBlock->top, x, y)) {
                     if (target) *target = pElem->MarkupArg("target", 0);
                     return pElem->MarkupArg("href", 0);
                  }
               }
               pElem = pElem->pNext;
            }
            continue;
         }
      }
      if ((pElem->style.flags & STY_Anchor) == 0) continue;
      switch (pElem->type) {
         case Html_Text:
         case Html_Space:
         case Html_IMG:
            while (pElem && pElem->type != Html_A) pElem = pElem->pPrev;
            if (pElem == 0 || pElem->type != Html_A) break;
            if (target) *target = pElem->MarkupArg("target", 0);
            return pElem->MarkupArg("href", 0);

            default:
               break;
      }
  }

  return 0;
}

//______________________________________________________________________________
int TGHtml::ElementCoords(TGHtmlElement *p, int /*i*/, int pct, int *coords)
{
   // Return coordinates of item

   TGHtmlBlock *pBlock;

   while (p && p->type != Html_Block) p = p->pPrev;
   if (!p) return 1;

   pBlock = (TGHtmlBlock *) p;
   if (pct) {
      TGHtmlElement *pEnd = pLast;
      TGHtmlBlock *pb2;
      while (pEnd && pEnd->type != Html_Block) pEnd = pEnd->pPrev;
      pb2 = (TGHtmlBlock *) pEnd;
#define HGCo(dir) pb2->dir ? pBlock->dir * 100 / pb2->dir : 0
      coords[0] = HGCo(left);
      coords[1] = HGCo(top);
      coords[3] = HGCo(right);
      coords[4] = HGCo(bottom);
   } else {
      coords[0] = pBlock->left;
      coords[1] = pBlock->top;
      coords[2] = pBlock->right;
      coords[3] = pBlock->bottom;
   }
   return 0;
}

//______________________________________________________________________________
TGHtmlElement *TGHtml::AttrElem(char *name, char *value)
{
   TGHtmlElement *p;
   char *z;

   for (p = pFirst; p; p = p->pNext) {
      if (p->type != Html_A) continue;
      z = p->MarkupArg(name, 0);
      if (z && (strcmp(z, value) == 0)) return p;
   }
   return 0;
}

//______________________________________________________________________________
void TGHtml::UpdateSelection(int forceUpdate)
{
   // Given the selection end-points in selBegin and selEnd, recompute
   // pSelBeginBlock and pSelEndBlock, then call UpdateSelectionDisplay()
   // to update the display.
   //
   // This routine should be called whenever the selection changes or
   // whenever the set of TGHtmlBlock structures change.

   TGHtmlBlock *pBlock;
   int index;
   int needUpdate = forceUpdate;
   int temp;

   if (selEnd.p == 0) selBegin.p = 0;

   IndexToBlockIndex(selBegin, &pBlock, &index);
   if (needUpdate || pBlock != pSelStartBlock) {
      needUpdate = 1;
      RedrawBlock(pSelStartBlock);
      pSelStartBlock = pBlock;
      selStartIndex = index;
   } else if (index != selStartIndex) {
      RedrawBlock(pBlock);
      selStartIndex = index;
   }

   if (selBegin.p == 0) selEnd.p = 0;

   IndexToBlockIndex(selEnd, &pBlock, &index);
   if (needUpdate || pBlock != pSelEndBlock) {
      needUpdate = 1;
      RedrawBlock(pSelEndBlock);
      pSelEndBlock = pBlock;
      selEndIndex = index;
   } else if (index != selEndIndex) {
      RedrawBlock(pBlock);
      selEndIndex = index;
   }

  if (pSelStartBlock && pSelStartBlock == pSelEndBlock &&
      selStartIndex > selEndIndex) {
    temp = selStartIndex;
    selStartIndex = selEndIndex;
    selEndIndex = temp;
  }

  if (needUpdate) {
    flags |= ANIMATE_IMAGES;
    UpdateSelectionDisplay();
  }
}

//______________________________________________________________________________
void TGHtml::UpdateSelectionDisplay()
{
   // The pSelStartBlock and pSelEndBlock values have been changed.
   // This routine's job is to loop over all TGHtmlBlocks and either 
   // set or clear the HTML_Selected bits in the .flags field
   // as appropriate.  For every TGHtmlBlock where the bit changes,
   // mark that block for redrawing.

   int selected = 0;
   SHtmlIndex tempIndex;
   TGHtmlBlock *pTempBlock;
   int temp;
   TGHtmlBlock *p;

   for (p = firstBlock; p; p = p->bNext) {
      if (p == pSelStartBlock) {
         selected = 1;
         RedrawBlock(p);
      } else if (!selected && p == pSelEndBlock) {
         selected = 1;
         tempIndex = selBegin;
         selBegin = selEnd;
         selEnd = tempIndex;
         pTempBlock = pSelStartBlock;
         pSelStartBlock = pSelEndBlock;
         pSelEndBlock = pTempBlock;
         temp = selStartIndex;
         selStartIndex = selEndIndex;
         selEndIndex = temp;
         RedrawBlock(p);
      }
      if (p->flags & HTML_Selected) {
         if (!selected) {
            p->flags &= ~HTML_Selected;
            RedrawBlock(p);
         }
      } else {
         if (selected) {
            p->flags |= HTML_Selected;
            RedrawBlock(p);
         }
      }
      if (p == pSelEndBlock) {
         selected = 0;
         RedrawBlock(p);
      }
   }
}

//______________________________________________________________________________
void TGHtml::LostSelection()
{
   //

   if (exportSelection) {
      // clear selection
      pSelStartBlock = 0;
      pSelEndBlock = 0;
      selBegin.p = 0;
      selEnd.p = 0;
      UpdateSelectionDisplay();
   }
}

//______________________________________________________________________________
int TGHtml::SelectionSet(const char *startIx, const char *endIx)
{
   //

   SHtmlIndex sBegin, sEnd;
   int bi, ei;

   if (GetIndex(startIx, &sBegin.p, &sBegin.i)) {
      // malformed start index
      return kFALSE;
   }

   if (GetIndex(endIx, &sEnd.p, &sEnd.i)) {
      // malformed end index
      return kFALSE;
   }

   bi = TokenNumber(sBegin.p);
   ei = TokenNumber(sEnd.p);

   if (!(sBegin.p && sEnd.p)) return kTRUE;

   if (bi < ei || (bi == ei && sBegin.i <= sEnd.i)) {
      selBegin = sBegin;
      selEnd = sEnd;
   } else {
      selBegin = sEnd;
      selEnd = sBegin;
   }

   UpdateSelection(0);
   if (exportSelection) {
      // TODO:
      // get selection ownership ... fId, XA_PRIMARY
      // selection lost handler must directly call LostSelection()
   }

   return kTRUE;
}

//______________________________________________________________________________
void TGHtml::UpdateInsert()
{
   // Recompute the position of the insertion cursor based on the
   // position in ins.

   IndexToBlockIndex(ins, &pInsBlock, &insIndex);
   RedrawBlock(pInsBlock);
   if (insTimer == 0) {
      insStatus = 0;
      FlashCursor();
   }
}

//______________________________________________________________________________
int TGHtml::SetInsert(const char *insIx)
{
   //

   SHtmlIndex i;

   if (!insIx) {
      RedrawBlock(pInsBlock);
      insStatus = 0;
      pInsBlock = 0;
      ins.p = 0;
   } else {
      if (GetIndex(insIx, &i.p, &i.i)) {
         // malformed index
         return kFALSE;
      }
      RedrawBlock(pInsBlock);
      ins = i;
      UpdateInsert();
   }

   return kTRUE;
}

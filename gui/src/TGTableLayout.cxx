// @(#)root/gui:$Name:  $:$Id: TGTableLayout.cxx,v 1.3 2001/05/07 18:44:16 rdm Exp $
// Author: Brett Viren   04/15/2001

/*************************************************************************
 * Copyright (C) 2001, Brett Viren                                       *
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
/**************************************************************************

    The Layout algorithm was largely translated from GTK's gtktable.c
    source.  That source is also distributed under LGPL.

**************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTableLayout                                                        //
//                                                                      //
// A LayoutManager which places child frames in a table. This uses      //
// TGTableLayoutHints (not TGLayoutHints). See TGTableLayoutHints       //
// for how to use these. This manager works like TGMatrixLayout with    //
// the addition that:                                                   //
//  - Child frames can span more than one column/row.                   //
//  - Child frames can resize with the frame.                           //
//  - Column and row sizes are not fixed nor (optionally) homogeneous.  //
//  - The number of columns and rows must be fully specified.           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#include "TGTableLayout.h"
#include "TGFrame.h"
#include "TList.h"
#include "TMath.h"
#include "Rtypes.h"


ClassImp(TGTableLayout)
ClassImp(TGTableLayoutHints)

//______________________________________________________________________________
TGTableLayout::TGTableLayout(TGCompositeFrame *main, UInt_t nrows, UInt_t ncols,
                             Bool_t homogeneous, Int_t sep, Int_t hints)
{
   // TGTableLayout constructor.
   // Note:
   // - Number of rows first, number of Columns second
   // - homogeneous == true means all table cells are the same size,
   //   set by the widest and the highest child frame.
   // - s gives the amount of separation in pixels between cells
   // - h are the hints, see TGTableLayoutHints.

   fMain    = main;
   fList    = fMain->GetList();
   fSep     = sep;
   fHints   = hints;
   fNrows   = nrows;
   fNcols   = ncols;
   fRow     = 0;
   fCol     = 0;
   fHomogeneous = homogeneous;
}

//______________________________________________________________________________
TGTableLayout::~TGTableLayout()
{
   // TGTableLayout constructor.

   if (fRow) delete [] fRow;
   if (fCol) delete [] fCol;
}

//______________________________________________________________________________
void TGTableLayout::FindRowColSizes()
{
   // Find the sizes of rows and columns needed to statisfy
   // children's layout policies.

   // This is equiv to GTK's requisition stage

   FindRowColSizesInit();
   FindRowColSizesSinglyAttached();
   FindRowColSizesHomogeneous();
   FindRowColSizesMultiplyAttached();
   FindRowColSizesHomogeneous();
}

//______________________________________________________________________________
void TGTableLayout::FindRowColSizesInit()
{
   // Initialize values needed to determine the size of rows and columns.

   if (fRow) delete [] fRow;
   if (fCol) delete [] fCol;
   fRow = new TableData[fNrows];
   fCol = new TableData[fNcols];

   // Find max of each row and column

   UInt_t i;
   for (i = 0; i < fNrows; ++i) fRow[i].fDefSize = 0;
   for (i = 0; i < fNcols; ++i) fCol[i].fDefSize = 0;
}

//______________________________________________________________________________
void TGTableLayout::FindRowColSizesSinglyAttached()
{
   // Determine the size of rows/cols needed for singly attached children.

   TIter next(fList);
   TGFrameElement *ptr;

   while ((ptr = (TGFrameElement *) next())) {
#ifdef R__RTTI
      TGTableLayoutHints *layout =
            dynamic_cast<TGTableLayoutHints*>(ptr->fLayout);
        if (!layout) {
#else
      TGTableLayoutHints *layout = (TGTableLayoutHints*) ptr->fLayout;
      if (layout->IsA() != TGTableLayoutHints::Class()) {
#endif
         Error("FindRowColSizesSinglyAttached", "didn't get TGTableLayoutHints from %s, layout = 0x%lx",
               ptr->fFrame->GetName(), ptr->fLayout);
         return;
      }
      UInt_t col = layout->GetAttachLeft();
      if (col == (layout->GetAttachRight() - 1))
         fCol[col].fDefSize = TMath::Max(fCol[col].fDefSize,
                                         ptr->fFrame->GetDefaultWidth() +
                                         layout->GetPadLeft() +
                                         layout->GetPadRight());

      UInt_t row = layout->GetAttachTop();
      if (row == (layout->GetAttachBottom() - 1))
         fRow[row].fDefSize = TMath::Max(fRow[row].fDefSize,
                                         ptr->fFrame->GetDefaultHeight() +
                                         layout->GetPadTop() +
                                         layout->GetPadBottom());
   }
}

//______________________________________________________________________________
void TGTableLayout::FindRowColSizesHomogeneous()
{
   // If the table is homogeneous make sure all col/rows are same
   // size as biggest col/row.

   if (!fHomogeneous) return;

   UInt_t max_width = 0, max_height = 0, col, row;

   // find max
   for (col = 0; col < fNcols; ++col)
      max_width = TMath::Max(max_width,fCol[col].fDefSize);

   for (row = 0; row < fNrows; ++row)
      max_height = TMath::Max(max_height,fRow[row].fDefSize);

   // set max
   for (col = 0; col < fNcols; ++col) fCol[col].fDefSize = max_width;
   for (row = 0; row < fNrows; ++row) fRow[row].fDefSize = max_height;
}

//______________________________________________________________________________
void TGTableLayout::FindRowColSizesMultiplyAttached()
{
   // Checks any children which span multiple col/rows.

   TIter next(fList);
   TGFrameElement *ptr;

   while ((ptr = (TGFrameElement *) next())) {
#ifdef R__RTTI
      TGTableLayoutHints *layout =
            dynamic_cast<TGTableLayoutHints*>(ptr->fLayout);
      if (!layout) {
#else
      TGTableLayoutHints *layout = (TGTableLayoutHints*) ptr->fLayout;
      if (layout->IsA() != TGTableLayoutHints::Class()) {
#endif
         Error("FindRowColSizesMultiplyAttached", "didn't get TGTableLayoutHints");
         return;
      }
      UInt_t left   = layout->GetAttachLeft();
      UInt_t right  = layout->GetAttachRight();
      if (left != right-1) {  // child spans multi-columns
         UInt_t width = 0, col;
         for (col = left; col < right; ++col) width += fCol[col].fDefSize;

         // If more space needed, divide space evenly among the columns
         UInt_t child_width = ptr->fFrame->GetDefaultWidth() +
                              layout->GetPadLeft() + layout->GetPadRight();

         if (width < child_width) {
            width = child_width - width;
            for (col = left; col < right; ++col) {
               UInt_t extra = width / (right - col);
               fCol[col].fDefSize += extra;
               width -= extra;
            }
         }
      }
      UInt_t top    = layout->GetAttachTop();
      UInt_t bottom = layout->GetAttachBottom();
      if (top != bottom-1) {  // child spans multi-rows
         UInt_t height = 0, row;
         for (row = top; row < bottom; ++row) height += fRow[row].fDefSize;

         // If more space needed, divide space evenly among the rows
         UInt_t child_height = ptr->fFrame->GetDefaultHeight() +
                               layout->GetPadTop() + layout->GetPadBottom();

         if (height < child_height) {
            height = child_height - height;
            for (row = top; row < bottom; ++row) {
               UInt_t extra = height / (bottom - row);
               fRow[row].fDefSize += extra;
               height -= extra;
            }
         }
      }
   }
}

//______________________________________________________________________________
void TGTableLayout::SetRowColResize(UInt_t real_size, UInt_t nthings,
                                    TableData *thing, Bool_t homogeneous)
{
   // If main frame is bigger or smaller than all children,
   // expand/shrink to fill. This is symmetric under row<-->col
   // switching so it is abstracted out to a normal function to save typing.

   if (homogeneous) {
      UInt_t ind, nexpand = 0, cur_size = 0;

      for (ind = 0; ind < nthings; ++ind)
         cur_size += thing[ind].fDefSize;

      if (cur_size < real_size) {
         for (ind = 0; ind < nthings; ++ind)
            if (thing[ind].fExpand) { ++ nexpand; break; }
         if (nexpand > 0) {
            UInt_t size = real_size;
            for (ind = 0; ind < nthings; ++ ind) {
               UInt_t extra = size / (nthings - ind);
               thing[ind].fRealSize = TMath::Max(1U, extra);
               size -= extra;
            }
         }
      }
   } else {
      UInt_t ind, nshrink=0, nexpand=0, size=0;
      for (ind = 0; ind < nthings; ++ind) {
         size += thing[ind].fDefSize;
         if (thing[ind].fExpand) ++ nexpand;
         if (thing[ind].fShrink) ++ nshrink;
      }

      // Did main frame expand?
      if ((size < real_size) && (nexpand >= 1)) {
         size = real_size - size;
         for (ind = 0; ind < nthings; ++ind) {
            if (thing[ind].fExpand) {
               UInt_t extra = size / nexpand;
               thing[ind].fRealSize += extra;
               size -= extra;
               --nexpand;
            }
         }
      }

      // Did main frame shrink?
      if (size > real_size) {
         UInt_t total_nshrink = nshrink;
         UInt_t extra = size - real_size;
         while (total_nshrink > 0 && extra > 0) {
            nshrink = total_nshrink;
            for (ind = 0; ind < nthings; ++ind)
               if (thing[ind].fShrink) {
                  UInt_t size = thing[ind].fRealSize;
                  thing[ind].fRealSize = TMath::Max(1U,thing[ind].fRealSize - extra / nshrink);
                  extra -= size - thing[ind].fRealSize;
                  --nshrink;
                  if (thing[ind].fRealSize < 2) {
                     total_nshrink -= 1;
                     thing[ind].fShrink = kFALSE;
                  }
               }
         }
      }
   } // not homogeneous
}

//______________________________________________________________________________
void TGTableLayout::SetRowColSizes()
{
   // This gets the new sizes needed to fit the table to the parent
   // frame. To be called after FindRowColSizes.

   SetRowColSizesInit();
   UInt_t border_width = fMain->GetBorderWidth();

   SetRowColResize(fMain->GetWidth() - (fNcols-1)*fSep - 2*border_width,
                   fNcols, fCol, fHomogeneous);
   SetRowColResize(fMain->GetHeight() - (fNrows-1)*fSep - 2*border_width,
                   fNrows, fRow, fHomogeneous);
}

//______________________________________________________________________________
void TGTableLayout::SetRowColSizesInit()
{
   // Initialize rows/cols. By default they do not expand and they
   // do shrink. What the children want determine what the rows/cols do.

   UInt_t col;
   for (col = 0; col < fNcols; ++col) {
      fCol[col].fRealSize = fCol[col].fDefSize;
      fCol[col].fNeedExpand = kFALSE;
      fCol[col].fNeedShrink = kTRUE;
      fCol[col].fExpand = kFALSE;
      fCol[col].fShrink = kTRUE;
      fCol[col].fEmpty = kTRUE;
   }
   UInt_t row;
   for (row = 0; row < fNrows; ++row) {
      fRow[row].fRealSize = fRow[row].fDefSize;
      fRow[row].fNeedExpand = kFALSE;
      fRow[row].fNeedShrink = kTRUE;
      fRow[row].fExpand = kFALSE;
      fRow[row].fShrink = kTRUE;
      fRow[row].fEmpty = kTRUE;
   }

   // Check single row/col children for expand/shrink-ability
   TIter next(fList);
   TGFrameElement *ptr;
   while ((ptr = (TGFrameElement*) next())) {
#ifdef R__RTTI
      TGTableLayoutHints *layout =
            dynamic_cast<TGTableLayoutHints*>(ptr->fLayout);
      if (!layout) {
#else
      TGTableLayoutHints *layout = (TGTableLayoutHints*) ptr->fLayout;
      if (layout->IsA() != TGTableLayoutHints::Class()) {
#endif
         Error("SetRowColSizesInit", "didn't get TGTableLayoutHints");
         return;
      }
      ULong_t hints = layout->GetLayoutHints();

      // columns
      if (layout->GetAttachLeft() == layout->GetAttachRight()-1) {
         if (hints & kLHintsExpandX)
            fCol[layout->GetAttachLeft()].fExpand = kTRUE;
         if (!(hints & kLHintsShrinkX))
            fCol[layout->GetAttachLeft()].fShrink = kFALSE;
         fCol[layout->GetAttachLeft()].fEmpty = kFALSE;
      }
      // rows
      if (layout->GetAttachTop() == layout->GetAttachBottom()-1) {
         if (hints & kLHintsExpandY)
            fRow[layout->GetAttachTop()].fExpand = kTRUE;
         if (!(hints & kLHintsShrinkY))
            fRow[layout->GetAttachTop()].fShrink = kFALSE;
         fRow[layout->GetAttachTop()].fEmpty = kFALSE;
      }
   }

   // Do same for children of spanning multiple col/rows
   next.Reset();
   while ((ptr = (TGFrameElement*) next())) {
#ifdef R__RTTI
      TGTableLayoutHints *layout =
            dynamic_cast<TGTableLayoutHints*>(ptr->fLayout);
      if (!layout) {
#else
      TGTableLayoutHints *layout = (TGTableLayoutHints*) ptr->fLayout;
      if (layout->IsA() != TGTableLayoutHints::Class()) {
#endif
         Error("SetRowColSizesInit", "didn't get TGTableLayoutHints");
         return;
      }
      ULong_t hints = layout->GetLayoutHints();

      // columns
      UInt_t left = layout->GetAttachLeft();
      UInt_t right = layout->GetAttachRight();
      if (left != right - 1) {
         for (col = left; col < right; ++col) fCol[col].fEmpty = kFALSE;
         Bool_t has_expand=kFALSE, has_shrink=kTRUE;
         if (hints & kLHintsExpandX) {
            for (col = left; col < right; ++col)
               if (fCol[col].fExpand) { has_expand = kTRUE; break; }
            if (!has_expand)
               for (col = left; col < right; ++col)
                  fCol[col].fNeedExpand = kTRUE;
         }
         if (!(hints & kLHintsShrinkX)) {
            for (col = left; col < right; ++col)
               if (!fCol[col].fShrink) { has_shrink = kFALSE; break;}
            if (has_shrink)
               for (col = left; col < right; ++col)
                  fCol[col].fNeedShrink = kFALSE;
         }
      }

      // rows
      UInt_t top = layout->GetAttachTop();
      UInt_t bottom = layout->GetAttachBottom();
      if (top != bottom - 1) {
         for (row = top; row < bottom; ++row) fRow[row].fEmpty = kFALSE;
         Bool_t has_expand=kFALSE, has_shrink=kTRUE;
         if (hints & kLHintsExpandY) {
            for (row = top; row < bottom; ++row)
               if (fRow[row].fExpand) { has_expand = kTRUE; break; }
            if (!has_expand)
               for (row = top; row < bottom; ++row)
                  fRow[row].fNeedExpand = kTRUE;
         }
         if (!(hints & kLHintsShrinkY)) {
            for (row = top; row < bottom; ++row)
               if (!fRow[row].fShrink) { has_shrink = kFALSE; break;}
            if (has_shrink)
               for (row = top; row < bottom; ++row)
                  fRow[row].fNeedShrink = kFALSE;
         }
      }
   }

   // Set expand/shrink flags
   for (col = 0; col < fNcols; ++col) {
      if (fCol[col].fEmpty) {
         fCol[col].fExpand = kFALSE;
         fCol[col].fShrink = kFALSE;
      } else {
         if (fCol[col].fNeedExpand) fCol[col].fExpand = kTRUE;
         if (!fCol[col].fNeedShrink) fCol[col].fShrink = kFALSE;
      }
   }
   for (row = 0; row < fNrows; ++row) {
      if (fRow[row].fEmpty) {
         fRow[row].fExpand = kFALSE;
         fRow[row].fShrink = kFALSE;
      } else {
         if (fRow[row].fNeedExpand) fRow[row].fExpand = kTRUE;
         if (!fRow[row].fNeedShrink) fRow[row].fShrink = kFALSE;
      }
   }
}

//______________________________________________________________________________
void TGTableLayout::CheckSanity()
{
   // Sanity check various values.

   TIter next(fList);
   TGFrameElement *ptr;
   UInt_t nerrors = 0;
   while ((ptr = (TGFrameElement*) next())) {
#ifdef R__RTTI
      TGTableLayoutHints *layout =
            dynamic_cast<TGTableLayoutHints*>(ptr->fLayout);
      if (!layout) {
#else
      TGTableLayoutHints *layout = (TGTableLayoutHints*) ptr->fLayout;
      if (layout->IsA() != TGTableLayoutHints::Class()) {
#endif
         Error("CheckSanity", "didn't get TGTableLayoutHints");
         return;
      }

      UInt_t right  = layout->GetAttachRight();
      UInt_t left   = layout->GetAttachLeft();
      UInt_t top    = layout->GetAttachTop();
      UInt_t bottom = layout->GetAttachBottom();

      if (left == right) {
         ++nerrors;
         Error("CheckSanity", "AttachLeft == AttachRight");
      }
      if (left > right) {
         ++nerrors;
         Error("CheckSanity", "AttachLeft > AttachRight");
      }
      if (left > fNcols-1) {
         ++nerrors;
         Error("CheckSanity", "AttachLeft illegal value: %u", left);
      }
      if (right < 1 || right > fNcols) {
         ++nerrors;
         Error("CheckSanity", "AttachRight illegal value: %u", right);
      }

      if (top == bottom) {
         ++nerrors;
         Error("CheckSanity", "AttachTop == AttachBottom");
      }
      if (top > bottom) {
         ++nerrors;
         Error("CheckSanity", "AttachTop > AttachBottom");
      }
      if (top > fNrows-1) {
         ++nerrors;
         Error("CheckSanity", "AttachTop illegal value: %u", top);
      }
      if (bottom < 1 || bottom > fNrows) {
         ++nerrors;
         Error("CheckSanity", "AttachBottom illegal value: %u", bottom);
      }

   }
   if (nerrors) {
      Error("CheckSanity", "errors in %u x %u table", fNcols, fNrows);
   }
}

//______________________________________________________________________________
void TGTableLayout::Layout()
{
    // Make a table layout of all frames in the list.

   CheckSanity();

   FindRowColSizes();

   SetRowColSizes();

   // Do the layout
   TIter next(fList);
   TGFrameElement *ptr;
   UInt_t border_width = fMain->GetBorderWidth();
   while ((ptr = (TGFrameElement*) next())) {
      TGTableLayoutHints *layout = (TGTableLayoutHints*)ptr->fLayout;
      ULong_t hints = layout->GetLayoutHints();
      TGDimension size = ptr->fFrame->GetDefaultSize();

      UInt_t right  = layout->GetAttachRight();
      UInt_t left   = layout->GetAttachLeft();
      UInt_t top    = layout->GetAttachTop();
      UInt_t bottom = layout->GetAttachBottom();

      // Find location and size of cell in which to fit the child frame.
      UInt_t col, cell_x = border_width + left*fSep;
      for (col = 0; col < left; ++col) cell_x += fCol[col].fRealSize;

      UInt_t row, cell_y = border_width + top*fSep;
      for (row = 0; row < top; ++row) cell_y += fRow[row].fRealSize;

      UInt_t cell_width = (right-left-1)*fSep;
      for (col=left; col < right; ++col)
         cell_width += fCol[col].fRealSize;

      UInt_t cell_height = (bottom-top-1)*fSep;
      for (row=top; row < bottom; ++row)
         cell_height += fRow[row].fRealSize;

      UInt_t pad_left   = layout->GetPadLeft();
      UInt_t pad_right  = layout->GetPadRight();
      UInt_t pad_bottom = layout->GetPadBottom();
      UInt_t pad_top    = layout->GetPadTop();

      // find size of child frame
      UInt_t ww,hh;
      if (hints & kLHintsFillX)
         ww = cell_width  - pad_left - pad_right;
      else
         ww = size.fWidth;
      if (hints & kLHintsFillY)
         hh = cell_height - pad_top - pad_bottom;
      else
         hh = size.fHeight;

      // Find location of child frame
      UInt_t xx;
      if (hints & kLHintsFillX) // Fill beats right/center/left hints
         xx = cell_x + pad_left;
      else if (hints & kLHintsRight)
         xx = cell_x + cell_width - pad_right - ww;
      else if (hints & kLHintsCenterX)
         xx = cell_x + cell_width/2 - ww/2; // padding?
      else                    // defaults to kLHintsLeft
         xx = cell_x + pad_left;

      UInt_t yy;
      if (hints & kLHintsFillY) // Fill beats top/center/bottom hings
         yy = cell_y + pad_top;
      else if (hints & kLHintsBottom)
         yy = cell_y + cell_height - pad_bottom - hh;
      else if (hints & kLHintsCenterY)
         yy = cell_y + cell_height/2 - hh/2; // padding?
      else                    // defaults to kLHintsTop
         yy = cell_y + pad_top;

      ptr->fFrame->MoveResize(xx,yy,ww,hh);
      ptr->fFrame->Layout();

   }
}

//______________________________________________________________________________
TGDimension TGTableLayout::GetDefaultSize() const
{
   // Return default dimension of the table layout.

   Int_t border_width = fMain->GetBorderWidth();

   TGDimension size(2*border_width + (fNcols-1)*fSep,
                    2*border_width + (fNrows-1)*fSep);

   UInt_t col, row;
   if (fCol)
      for (col = 0; col < fNcols; ++col) size.fWidth += fCol[col].fDefSize;
   if (fRow)
      for (row = 0; row < fNrows; ++row) size.fHeight += fRow[row].fDefSize;

   return size;
}

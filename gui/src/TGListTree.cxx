// @(#)root/gui:$Name:  $:$Id: TGListTree.cxx,v 1.4 2000/09/05 16:13:36 rdm Exp $
// Author: Fons Rademakers   25/02/98

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
// TGListTree and TGListTreeItem                                        //
//                                                                      //
// A list tree is a widget that can contain a number of items           //
// arranged in a tree structure. The items are represented by small     //
// folder icons that can be either open or closed.                      //
//                                                                      //
// The TGListTree is user callable. The TGListTreeItem is a service     //
// class of the list tree.                                              //
//                                                                      //
// A list tree can generate the following events:                       //
// kC_LISTTREE, kCT_ITEMCLICK, which button, location (y<<16|x).        //
// kC_LISTTREE, kCT_ITEMDBLCLICK, which button, location (y<<16|x).     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <stdlib.h>

#include "TGListTree.h"
#include "TGPicture.h"
#include "TGCanvas.h"
#include "TGToolTip.h"


ClassImp(TGListTreeItem)
ClassImp(TGListTree)


//--- Some utility functions ---------------------------------------------------
static Int_t FontHeight(FontStruct_t f)
{
   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(f, max_ascent, max_descent);
   return max_ascent + max_descent;
}

static Int_t FontAscent(FontStruct_t f)
{
   int max_ascent, max_descent;
   gVirtualX->GetFontProperties(f, max_ascent, max_descent);
   return max_ascent;
}

static Int_t FontTextWidth(FontStruct_t f, const char *c)
{
   return gVirtualX->TextWidth(f, c, strlen(c));
}


//______________________________________________________________________________
TGListTreeItem::TGListTreeItem(TGClient *client, const char *name,
                               const TGPicture *opened,
                               const TGPicture *closed)
{
   // Create list tree item.

   fText = name;

   fOpenPic   = opened;
   fClosedPic = closed;
   fPicWidth  = TMath::Max(fOpenPic->GetWidth(), fClosedPic->GetWidth());

   fOpen = fActive = kFALSE;

   fParent =
   fFirstchild =
   fPrevsibling =
   fNextsibling = 0;

   fUserData = 0;

   fClient = client;
}

//______________________________________________________________________________
TGListTreeItem::~TGListTreeItem()
{
   // Delete list tree item.

   fClient->FreePicture(fOpenPic);
   fClient->FreePicture(fClosedPic);
}

//______________________________________________________________________________
void TGListTreeItem::Rename(const char *new_name)
{
   // Rename a list tree item.

   fText = new_name;
}

//___________________________________________________________________________
void TGListTreeItem::SetPictures(const TGPicture* opened, const TGPicture* closed)
{
   // Change list tree item icons.

   fClient->FreePicture(fOpenPic);
   fClient->FreePicture(fClosedPic);
   fOpenPic   = opened;
   fClosedPic = closed;
}
//--------------------------------------------------------------------

//______________________________________________________________________________
TGListTree::TGListTree(TGWindow *p, UInt_t w, UInt_t h, UInt_t options,
                       ULong_t back) :
   TGFrame(p, w, h, options, back)
{
   // Create a list tree widget.

   GCValues_t gcv;

   fMsgWindow = p;
   fTip       = 0;
   fTipItem   = 0;
   fAutoTips  = kFALSE;

   fFont = fgDefaultFontStruct;

   if (!fClient->GetColorByName("#808080", fGrayPixel))
      fClient->GetColorByName("black", fGrayPixel);

   gcv.fLineStyle = kLineSolid;
   gcv.fLineWidth = 0;
   gcv.fFillStyle = kFillSolid;
   gcv.fFont = gVirtualX->GetFontHandle(fFont);
   gcv.fBackground = fgWhitePixel;
   gcv.fForeground = fgBlackPixel;

   gcv.fMask = kGCLineStyle  | kGCLineWidth  | kGCFillStyle |
               kGCForeground | kGCBackground | kGCFont;
   fDrawGC = gVirtualX->CreateGC(fId, &gcv);

   gcv.fLineStyle = kLineOnOffDash;
   gcv.fForeground = fGrayPixel;
   fLineGC = gVirtualX->CreateGC(fId, &gcv);
   gVirtualX->SetDashes(fLineGC, 0, "\x1\x1", 2);

   gcv.fBackground = fgDefaultSelectedBackground;
   gcv.fForeground = fgWhitePixel;
   gcv.fLineStyle = kLineSolid;
   gcv.fMask = kGCLineStyle  | kGCLineWidth  | kGCFillStyle |
               kGCForeground | kGCBackground | kGCFont;
   fHighlightGC = gVirtualX->CreateGC(fId, &gcv);

   fFirst = fSelected = 0;
   fDefw = fDefh = 1;

   fHspacing = 2;
   fVspacing = 2;  // 0;
   fIndent   = 3;  // 0;
   fMargin   = 2;

   gVirtualX->GrabButton(fId, kAnyButton, kAnyModifier,
                    kButtonPressMask | kButtonReleaseMask,
                    kNone, kNone);

   AddInput(kPointerMotionMask | kEnterWindowMask | kLeaveWindowMask);
}

//______________________________________________________________________________
TGListTree::~TGListTree()
{
   // Delete list tree widget.

   TGListTreeItem *item, *sibling;

   delete fTip;

   gVirtualX->DeleteGC(fDrawGC);
   gVirtualX->DeleteGC(fLineGC);
   gVirtualX->DeleteGC(fHighlightGC);
   item = fFirst;
   while (item) {
      if (item->fFirstchild) PDeleteChildren(item->fFirstchild);
      sibling = item->fNextsibling;
      delete item;
      item = sibling;
   }
}


//---- highlighting utilities

//______________________________________________________________________________
void TGListTree::HighlightItem(TGListTreeItem *item, Bool_t state, Bool_t draw)
{
   // Highlight tree item.

   if (item) {
      if ((item == fSelected) && !state) {
         fSelected = 0;
         if (draw) DrawItemName(item);
      } else if (state != item->fActive) {
         item->fActive = state;
         if (draw) DrawItemName(item);
      }
   }
}

//______________________________________________________________________________
void TGListTree::HighlightChildren(TGListTreeItem *item, Bool_t state, Bool_t draw)
{
   // Higlight item children.

   while (item) {
      HighlightItem(item, state, draw);
      if (item->fFirstchild)
         HighlightChildren(item->fFirstchild, state, (item->fOpen) ? draw : kFALSE);
      item = item->fNextsibling;
   }
}

//______________________________________________________________________________
void TGListTree::UnselectAll(Bool_t draw)
{
   // Unselect all items.

   HighlightChildren(fFirst, kFALSE, draw);
}

//______________________________________________________________________________
Bool_t TGListTree::HandleButton(Event_t *event)
{
   // Handle button events in the list tree.

   TGListTreeItem *item;

   if (fTip) fTip->Hide();

   if (event->fType == kButtonPress) {
      if ((item = FindItem(event->fY)) != 0) {
         if (fSelected) fSelected->fActive = kFALSE;
         fLastY = event->fY;
         UnselectAll(kTRUE);
         fSelected = item;
         //item->fActive = kTRUE; // this is done below w/redraw
         HighlightItem(item, kTRUE, kTRUE);
         SendMessage(fMsgWindow, MK_MSG(kC_LISTTREE, kCT_ITEMCLICK),
                     event->fCode, (event->fYRoot << 16) | event->fXRoot);
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGListTree::HandleDoubleClick(Event_t *event)
{
   // Handle double click event in the list tree (only for kButton1).

   TGListTreeItem *item;

   if (event->fCode == kButton1 && (item = FindItem(event->fY)) != 0) {
      item->fOpen = !item->fOpen;
      if (item != fSelected) { // huh?!
         if (fSelected) fSelected->fActive = kFALSE;
         UnselectAll(kTRUE);
         fSelected = item;
         //item->fActive = kTRUE; // this is done below w/redraw
         HighlightItem(item, kTRUE, kTRUE);
      }
      //fClient->NeedRedraw(this); //DoRedraw();
      SendMessage(fMsgWindow, MK_MSG(kC_LISTTREE, kCT_ITEMDBLCLICK),
                  event->fCode, (event->fYRoot << 16) | event->fXRoot);
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGListTree::HandleExpose(Event_t *event)
{
   // Handle expose event in the list tree.

   Draw(event->fY, (Int_t)event->fHeight);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGListTree::HandleCrossing(Event_t *event)
{
   // Handle mouse crossing event.

   if (fTip) {
      if (event->fType == kLeaveNotify) {
         fTip->Hide();
         fTipItem = 0;
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGListTree::HandleMotion(Event_t *event)
{
   // Handle mouse motion event. Only used to set tool tip.

   TGListTreeItem *item;

   if ((item = FindItem(event->fY)) != 0) {
      if (fTipItem == item) return kTRUE;

      if (fTip)
         fTip->Hide();

      if (item->fTipText.Length() > 0) {

         UInt_t width = FontTextWidth(fFont, item->fText.Data());
         SetToolTipText(item->fTipText.Data(), item->fXtext+width,
                        item->fY+item->fHeight-4, 1000);

      } else if (fAutoTips && item->GetUserData()) {
         // must derive from TObject (in principle user can put pointer
         // to anything in user data field). Add check.
         TObject *obj = (TObject *)item->GetUserData();
         if (obj->InheritsFrom(TObject::Class())) {
            UInt_t width = FontTextWidth(fFont, item->fText.Data());
            SetToolTipText(obj->GetTitle(), item->fXtext+width,
                           item->fY+item->fHeight-4, 1000);
         }
      }
      fTipItem = item;
   }
   return kTRUE;
}

//---- drawing functions

//______________________________________________________________________________
void TGListTree::DoRedraw()
{
   // Redraw list tree.

   gVirtualX->ClearWindow(fId);
   Draw(0, (Int_t)fHeight);
}

//______________________________________________________________________________
void TGListTree::Draw(Int_t yevent, Int_t hevent)
{
   // Draw list tree widget.

   TGListTreeItem *item;
   Int_t  x, y, xbranch;
   UInt_t width, height, old_width, old_height;

   // Overestimate the expose region to be sure to draw an item that gets
   // cut by the region
   fExposeTop = yevent - FontHeight(fFont);
   fExposeBottom = yevent + hevent + FontHeight(fFont);
   old_width  = fDefw;
   old_height = fDefh;
   fDefw = fDefh = 1;

   x = fMargin;
   y = fMargin;
   item = fFirst;
   while (item) {
      xbranch = -1;
      DrawItem(item, x, y, &xbranch, &width, &height);

      width += x + fHspacing + fMargin;

      if (width > fDefw) fDefw = width;

      y += height + fVspacing;
      if (item->fFirstchild && item->fOpen)
         y = DrawChildren(item->fFirstchild, x, y, xbranch);

      item = item->fNextsibling;
   }

   fDefh = y + fMargin;

   if ((old_width != fDefw) || (old_height != fDefh))
      ((TGCanvas *)GetParent()->GetParent())->Layout();
}

//______________________________________________________________________________
Int_t TGListTree::DrawChildren(TGListTreeItem *item, Int_t x, Int_t y, Int_t xroot)
{
   // Draw children of item in list tree.

   UInt_t width, height;
   Int_t  xbranch;

   x += fIndent + (Int_t)item->fPicWidth;
   while (item) {
      xbranch = xroot;
      DrawItem(item, x, y, &xbranch, &width, &height);

      width += x + fHspacing + fMargin;

      if (width > fDefw) fDefw = width;

      y += height + fVspacing;
      if ((item->fFirstchild) && (item->fOpen))
         y = DrawChildren(item->fFirstchild, x, y, xbranch);

      item = item->fNextsibling;
   }
   return y;
}

//______________________________________________________________________________
void TGListTree::DrawItem(TGListTreeItem *item, Int_t x, Int_t y, Int_t *xroot,
                          UInt_t *retwidth, UInt_t *retheight)
{
   // Draw list tree item.

   Int_t  xpic, ypic, xbranch, ybranch, xtext, ytext, yline, xc;
   UInt_t height;
   const TGPicture *pic;

   // Select the pixmap to use, if any
   if (item->fOpen)
      pic = item->fOpenPic;
   else
      pic = item->fClosedPic;

   // Compute the height of this line
   height = FontHeight(fFont);
   xpic = x;
   xtext = x + fHspacing + (Int_t)item->fPicWidth;
   if (pic) {
      if (pic->GetHeight() > height) {
         ytext = y + (Int_t)((pic->GetHeight() - height) >> 1);
         height = pic->GetHeight();
         ypic = y;
      } else {
         ytext = y;
         ypic = y + (Int_t)((height - pic->GetHeight()) >> 1);
      }
      xbranch = xpic + (Int_t)(item->fPicWidth >> 1);
      ybranch = ypic + (Int_t)pic->GetHeight();
      yline = ypic + (Int_t)(pic->GetHeight() >> 1);
   } else {
      ypic = ytext = y;
      xbranch = xpic + (Int_t)(item->fPicWidth >> 1);
      yline = ybranch = ypic + (Int_t)(height >> 1);
      yline = ypic + (Int_t)(height >> 1);
   }

   // height must be even, otherwise our dashed line wont appear properly
   ++height; height &= ~1;

   // Save the basic graphics info for use by other functions
   item->fY      = y;
   item->fXtext  = xtext;
   item->fYtext  = ytext;
   item->fHeight = height;

   if ((y+(Int_t)height >= fExposeTop) && (y <= fExposeBottom)) {
      if (*xroot >= 0) {
         xc = *xroot;

         if (item->fNextsibling)
            gVirtualX->DrawLine(fId, fLineGC, xc, y, xc, y+height);
         else
            gVirtualX->DrawLine(fId, fLineGC, xc, y, xc, yline);

         TGListTreeItem *p = item->fParent;
         while (p) {
            xc -= (fIndent + (Int_t)item->fPicWidth);
            if (p->fNextsibling)
               gVirtualX->DrawLine(fId, fLineGC, xc, y, xc, y+height);
            p = p->fParent;
         }
         gVirtualX->DrawLine(fId, fLineGC, *xroot, yline, xpic/*xbranch*/, yline);
         DrawNode(item, *xroot, yline);
      }
      if (item->fOpen && item->fFirstchild)
         gVirtualX->DrawLine(fId, fLineGC, xbranch, ybranch/*yline*/,
                        xbranch, y+height);

      // if (pic)
      //    pic->Draw(fId, fDrawGC, xpic, ypic);
      if (item->fActive || item == fSelected)
         item->fOpenPic->Draw(fId, fDrawGC, xpic, ypic);
      else
         item->fClosedPic->Draw(fId, fDrawGC, xpic, ypic);

      DrawItemName(item);
   }

   *xroot = xbranch;
   *retwidth = FontTextWidth(fFont, item->fText.Data()) + item->fPicWidth;
   *retheight = height;
}

//______________________________________________________________________________
void TGListTree::DrawItemName(TGListTreeItem *item)
{
   // Draw name of list tree item.

   UInt_t width;

   width = FontTextWidth(fFont, item->fText.Data());
   if (item->fActive || item == fSelected) {
      gVirtualX->SetForeground(fDrawGC, fgDefaultSelectedBackground);
      gVirtualX->FillRectangle(fId, fDrawGC,
                          item->fXtext, item->fYtext, width, FontHeight(fFont));
      gVirtualX->SetForeground(fDrawGC, fgBlackPixel);
      gVirtualX->DrawString(fId, fHighlightGC,
                       item->fXtext, item->fYtext + FontAscent(fFont),
                       item->fText.Data(), item->fText.Length());
   } else {
      gVirtualX->FillRectangle(fId, fHighlightGC,
                          item->fXtext, item->fYtext, width, FontHeight(fFont));
      gVirtualX->DrawString(fId, fDrawGC,
                       item->fXtext, item->fYtext + FontAscent(fFont),
                       item->fText.Data(), item->fText.Length());
   }
}

//______________________________________________________________________________
void TGListTree::DrawNode(TGListTreeItem *item, Int_t x, Int_t y)
{
   // Draw node (little + in box).

   if (item->fFirstchild) {
      gVirtualX->DrawLine(fId, fHighlightGC, x, y-2, x, y+2);
      gVirtualX->SetForeground(fHighlightGC, fgBlackPixel);
      gVirtualX->DrawLine(fId, fHighlightGC, x-2, y, x+2, y);
      if (!item->fOpen)
         gVirtualX->DrawLine(fId, fHighlightGC, x, y-2, x, y+2);
      gVirtualX->SetForeground(fHighlightGC, fGrayPixel);
      gVirtualX->DrawLine(fId, fHighlightGC, x-4, y-4, x+4, y-4);
      gVirtualX->DrawLine(fId, fHighlightGC, x+4, y-4, x+4, y+4);
      gVirtualX->DrawLine(fId, fHighlightGC, x-4, y+4, x+4, y+4);
      gVirtualX->DrawLine(fId, fHighlightGC, x-4, y-4, x-4, y+4);
      gVirtualX->SetForeground(fHighlightGC, fgWhitePixel);
  }
}

//______________________________________________________________________________
void TGListTree::SetToolTipText(const char *text, Int_t x, Int_t y, Long_t delayms)
{
   // Set tool tip text associated with this item. The delay is in
   // milliseconds (minimum 250). To remove tool tip call method with
   // delayms = 0. To change delayms you first have to call this method
   // with delayms=0.

   if (delayms == 0) {
      delete fTip;
      fTip = 0;
      return;
   }

   if (text && strlen(text)) {
      if (!fTip)
         fTip = new TGToolTip(fClient->GetRoot(), this, text, delayms);
      else
         fTip->SetText(text);
      fTip->SetPosition(x, y);
      fTip->Reset();
   }
}

//______________________________________________________________________________
void TGListTree::RemoveReference(TGListTreeItem *item)
{
   // This function removes the specified item from the linked list.
   // It does not do anything with the data contained in the item, though.

   // if there exists a previous sibling, just skip over item to be dereferenced
   if (item->fPrevsibling) {
      item->fPrevsibling->fNextsibling = item->fNextsibling;
      if (item->fNextsibling)
         item->fNextsibling->fPrevsibling = item->fPrevsibling;
   } else {
      // if not, then the deleted item is the first item in some branch
      if (item->fParent)
         item->fParent->fFirstchild = item->fNextsibling;
      else
         fFirst = item->fNextsibling;
      if (item->fNextsibling)
         item->fNextsibling->fPrevsibling = 0;
   }
}

//______________________________________________________________________________
void TGListTree::PDeleteChildren(TGListTreeItem *item)
{
   // Delete children of item from list.

   TGListTreeItem *sibling;

   while (item) {
      if (item->fFirstchild) {
         PDeleteChildren(item->fFirstchild);
         item->fFirstchild = 0;
      }
      sibling = item->fNextsibling;
      delete item;
      item = sibling;
   }
}

//______________________________________________________________________________
void TGListTree::InsertChild(TGListTreeItem *parent, TGListTreeItem *item)
{
   // Insert child in list.

   TGListTreeItem *i;

   item->fParent = parent;
   item->fNextsibling = item->fPrevsibling = 0;

   if (parent) {

      if (parent->fFirstchild) {
         i = parent->fFirstchild;
         while (i->fNextsibling) i = i->fNextsibling;
         i->fNextsibling = item;
         item->fPrevsibling = i;
      } else {
         parent->fFirstchild = item;
      }

   } else {  // if parent == 0, this is a top level entry

      if (fFirst) {
         i = fFirst;
         while (i->fNextsibling) i = i->fNextsibling;
         i->fNextsibling = item;
         item->fPrevsibling = i;
      } else {
         fFirst = item;
      }

   }
}

//______________________________________________________________________________
void TGListTree::InsertChildren(TGListTreeItem *parent, TGListTreeItem *item)
{
   // Insert a list of ALREADY LINKED children into another list

   TGListTreeItem *next, *newnext;

   //while (item) {
   //   next = item->fNextsibling;
   //   InsertChild(parent, item);
   //   item = next;
   //}
   //return;

   // Save the reference for the next item in the new list
   next = item->fNextsibling;

   // Insert the first item in the new list into the existing list
   InsertChild(parent, item);

   // The first item is inserted, with its prev and next siblings updated
   // to fit into the existing list. So, save the existing list reference
   newnext = item->fNextsibling;

   // Now, mark the first item's next sibling to point back to the new list
   item->fNextsibling = next;

   // Mark the parents of the new list to the new parent. The order of the
   // rest of the new list should be OK, and the second item should still
   // point to the first, even though the first was reparented.
   while (item->fNextsibling) {
      item->fParent = parent;
      item = item->fNextsibling;
   }

   // Fit the end of the new list back into the existing list
   item->fNextsibling = newnext;
   if (newnext)
      newnext->fPrevsibling = item;
}

//______________________________________________________________________________
Int_t TGListTree::SearchChildren(TGListTreeItem *item, Int_t y, Int_t findy,
                                 TGListTreeItem **finditem)
{
   // Search child item.

   UInt_t height;
   const TGPicture *pic;

   while (item) {
      // Select the pixmap to use, if any
      if (item->fOpen)
         pic = item->fOpenPic;
      else
         pic = item->fClosedPic;

      // Compute the height of this line
      height = FontHeight(fFont);
      if (pic && (pic->GetHeight() > height))
         height = pic->GetHeight();

      if ((findy >= y) && (findy <= y + (Int_t)height)) {
         *finditem = item;
         return -1;
      }

      y += (Int_t)height + fVspacing;
      if ((item->fFirstchild) && (item->fOpen)) {
         y = SearchChildren(item->fFirstchild, y, findy, finditem);
         if (*finditem) return -1;
      }

      item = item->fNextsibling;
   }

   return y;
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::FindItem(Int_t findy)
{
   // Find item at postion findy.

   Int_t  y;
   UInt_t height;
   TGListTreeItem *item, *finditem;
   const TGPicture *pic;

   y = fMargin;
   item = fFirst;
   finditem = 0;
   while (item && !finditem) {
      // Select the pixmap to use, if any
      if (item->fOpen)
         pic = item->fOpenPic;
      else
         pic = item->fClosedPic;

      // Compute the height of this line
      height = FontHeight(fFont);
      if (pic && (pic->GetHeight() > height))
         height = pic->GetHeight();

      if ((findy >= y) && (findy <= y + (Int_t)height))
         return item;

      y += (Int_t)height + fVspacing;
      if ((item->fFirstchild) && (item->fOpen)) {
         y = SearchChildren(item->fFirstchild, y, findy, &finditem);
         //if (finditem) return finditem;
      }
      item = item->fNextsibling;
   }

   return finditem;
}

//----- Public Functions

//______________________________________________________________________________
TGListTreeItem *TGListTree::AddItem(TGListTreeItem *parent, const char *string,
                                    const TGPicture *open, const TGPicture *closed)
{
   // Add item to list tree. Returns new item.

   TGListTreeItem *item;

   if (!open)   open   = fClient->GetPicture("ofolder_t.xpm");
   if (!closed) closed = fClient->GetPicture("folder_t.xpm");

   item = new TGListTreeItem(fClient, string, open, closed);
   InsertChild(parent, item);

   //fClient->NeedRedraw(this);

   return item;
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::AddItem(TGListTreeItem *parent, const char *string,
                                    void *userData, const TGPicture *open,
                                    const TGPicture *closed)
{
   // Add item to list tree. If item with same userData already exists
   // don't add it. Returns new item.

   TGListTreeItem *item = FindChildByData(parent, userData);
   if (!item) {
      item = AddItem(parent, string, open, closed);
      if (item) item->SetUserData(userData);
   }
   return item;
}

//______________________________________________________________________________
void TGListTree::RenameItem(TGListTreeItem *item, const char *string)
{
   // Rename item in list tree.

   if (item)
      item->Rename(string);

   //fClient->NeedRedraw(this);
}

//______________________________________________________________________________
Int_t TGListTree::DeleteItem(TGListTreeItem *item)
{
   // Delete item from list tree.

   if (item->fFirstchild)
      PDeleteChildren(item->fFirstchild);

   item->fFirstchild = 0;

   RemoveReference(item);

   if (fSelected == item)
      fSelected = 0;

   delete item;

   //fClient->NeedRedraw(this);

   return 1;
}

//______________________________________________________________________________
void TGListTree::OpenItem(TGListTreeItem *item)
{
   // Open item in list tree (i.e. show child items).

   if (item)
      item->fOpen = kTRUE;

   //fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGListTree::CloseItem(TGListTreeItem *item)
{
   // Close item in list tree (i.e. hide child items).

   if (item)
      item->fOpen = kFALSE;

   //fClient->NeedRedraw(this);
}

//______________________________________________________________________________
Int_t TGListTree::RecursiveDeleteItem(TGListTreeItem *item, void *ptr)
{
   // Delete item with fUserData == ptr. Search tree downwards starting
   // at item.

   if (item && ptr) {
      if (item->fUserData == ptr)
         DeleteItem(item);
      else {
         if (item->fOpen && item->fFirstchild)
            RecursiveDeleteItem(item->fFirstchild,  ptr);
         RecursiveDeleteItem(item->fNextsibling, ptr);
      }
   }
   return 1;
}

//______________________________________________________________________________
void TGListTree::SetToolTipItem(TGListTreeItem *item, const char *string)
{
   // Set tooltip text for this item. By default an item for which the
   // userData is a pointer to an TObject the TObject::GetTitle() will
   // be used to get the tip text.

   if (item)
      item->fTipText = string;
}

//______________________________________________________________________________
Int_t TGListTree::DeleteChildren(TGListTreeItem *item)
{
   // Delete children of item from list.

   if (item->fFirstchild)
      PDeleteChildren(item->fFirstchild);

   item->fFirstchild = 0;

   //fClient->NeedRedraw(this);

   return 1;
}

//______________________________________________________________________________
Int_t TGListTree::Reparent(TGListTreeItem *item, TGListTreeItem *newparent)
{
   // Make newparent the new parent of item.

   // Remove the item from its old location.
   RemoveReference(item);

   // The item is now unattached. Reparent it.
   InsertChild(newparent, item);

   //fClient->NeedRedraw(this);

   return 1;
}

//______________________________________________________________________________
Int_t TGListTree::ReparentChildren(TGListTreeItem *item,
                                 TGListTreeItem *newparent)
{
   // Make newparent the new parent of the children of item.

   TGListTreeItem *first;

   if (item->fFirstchild) {
      first = item->fFirstchild;
      item->fFirstchild = 0;

      InsertChildren(newparent, first);

      //fClient->NeedRedraw(this);
      return 1;
   }
   return 0;
}

//______________________________________________________________________________
static Int_t Compare(const void *item1, const void *item2)
{
   return strcmp((*((TGListTreeItem **) item1))->GetText(),
                 (*((TGListTreeItem **) item2))->GetText());
}

//______________________________________________________________________________
Int_t TGListTree::Sort(TGListTreeItem *item)
{
   // Sort items starting with item.

   TGListTreeItem *first, *parent, **list;
   size_t i, count;

   // Get first child in list;
   while (item->fPrevsibling) item = item->fPrevsibling;

   first = item;
   parent = first->fParent;

   // Count the children
   count = 1;
   while (item->fNextsibling) item = item->fNextsibling, count++;
   if (count <= 1) return 1;

   list = new TGListTreeItem* [count];
   list[0] = first;
   count = 1;
   while (first->fNextsibling) {
      list[count] = first->fNextsibling;
      count++;
      first = first->fNextsibling;
   }

   ::qsort(list, count, sizeof(TGListTreeItem*), &::Compare);

   list[0]->fPrevsibling = 0;
   for (i = 0; i < count; i++) {
      if (i < count - 1)
         list[i]->fNextsibling = list[i + 1];
      if (i > 0)
         list[i]->fPrevsibling = list[i - 1];
   }
   list[count - 1]->fNextsibling = 0;
   if (parent)
      parent->fFirstchild = list[0];
   else
      fFirst = list[0];

   delete [] list;

   //fClient->NeedRedraw(this);

   return 1;
}

//______________________________________________________________________________
Int_t TGListTree::SortSiblings(TGListTreeItem *item)
{
   // Sort siblings of item.

   return Sort(item);
}

//______________________________________________________________________________
Int_t TGListTree::SortChildren(TGListTreeItem *item)
{
   // Sort children of item.

   TGListTreeItem *first;

   if (item) {
      first = item->fFirstchild;
      if (first)
         SortSiblings(first);
   } else {
      if (fFirst) {
         first = fFirst->fFirstchild;
         if (first)
           SortSiblings(first);
      }
   }
   return 1;
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::FindSiblingByName(TGListTreeItem *item, const char *name)
{
   // Find sibling of item by name.

   // Get first child in list
   if (item) {
      while (item->fPrevsibling)
         item = item->fPrevsibling;

      while (item) {
         if (item->fText == name)
            return item;
         item = item->fNextsibling;
      }
      return item;
   }
   return 0;
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::FindSiblingByData(TGListTreeItem *item, void *userData)
{
   // Find sibling of item by userData.

   // Get first child in list
   if (item) {
      while (item->fPrevsibling)
         item = item->fPrevsibling;

      while (item) {
         if (item->fUserData == userData)
            return item;
         item = item->fNextsibling;
      }
      return item;
   }
   return 0;
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::FindChildByName(TGListTreeItem *item, const char *name)
{
   // Find child of item by name.

   // Get first child in list
   if (item && item->fFirstchild) {
      item = item->fFirstchild;
   } else if (!item && fFirst) {
      item = fFirst;
   } else {
      item = 0;
   }

   while (item) {
      if (item->fText == name)
         return item;
      item = item->fNextsibling;
   }
   return 0;
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::FindChildByData(TGListTreeItem *item, void *userData)
{
   // Find child of item by userData.

   // Get first child in list
   if (item && item->fFirstchild) {
      item = item->fFirstchild;
   } else if (!item && fFirst) {
      item = fFirst;
   } else {
      item = 0;
   }

   while (item) {
      if (item->fUserData == userData)
         return item;
      item = item->fNextsibling;
   }
   return 0;
}

//______________________________________________________________________________
TGListTreeItem *TGListTree::FindItemByPathname(const char *path)
{
   // Find item by pathname. Pathname is in the form of /xx/yy/zz. If zz
   // in path /xx/yy is found it returns item, 0 otherwise.

   if (!path || !*path) return 0;

   const char *p = path, *s;
   char dirname[256];
   TGListTreeItem *item = 0;

   while (1) {
      while (*p && *p == '/') p++;
      if (!*p) break;

      s = strchr(p, '/');
      if (!s) {
         strcpy(dirname, p);
      } else {
         strncpy(dirname, p, s-p);
         dirname[s-p] = 0;
      }
      item = FindChildByName(item, dirname);
      if (!item || !s) return item;
      p = ++s;
   }
   return 0;
}

//______________________________________________________________________________
void TGListTree::HighlightItem(TGListTreeItem *item)
{
   // Highlight item.

   UnselectAll(kFALSE);
   HighlightItem(item, kTRUE, kFALSE);
   //fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGListTree::ClearHighlighted()
{
   // Un highlight items.

   UnselectAll(kFALSE);
   //fClient->NeedRedraw(this);
}

//______________________________________________________________________________
void TGListTree::GetPathnameFromItem(TGListTreeItem *item, char *path, Int_t depth)
{
   // Get pathname from item. Use depth to limit path name to last
   // depth levels. By default depth is not limited.

   char tmppath[1024];

   *path = '\0';
   while (item) {
      sprintf(tmppath, "/%s%s", item->fText.Data(), path);
      strcpy(path, tmppath);
      item = item->fParent;
      if (--depth == 0 && item) {
         sprintf(tmppath, "...%s", path);
         strcpy(path, tmppath);
         return;
      }
   }
}

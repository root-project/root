// @(#)root/gui:$Name:  $:$Id: TGListTree.h,v 1.3 2000/09/05 10:56:50 rdm Exp $
// Author: Fons Rademakers   25/02/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGListTree
#define ROOT_TGListTree


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

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TGWidget
#include "TGWidget.h"
#endif

class TGPicture;
class TGToolTip;
class TGCanvas;


class TGListTreeItem {

friend class TGListTree;

private:
   TGClient        *fClient;       // pointer to TGClient
   TGListTreeItem  *fParent;       // pointer to parent
   TGListTreeItem  *fFirstchild;   // pointer to first child item
   TGListTreeItem  *fPrevsibling;  // pointer to previous sibling
   TGListTreeItem  *fNextsibling;  // pointer to next sibling
   Bool_t           fOpen;         // true if item is open
   Bool_t           fActive;       // true if item is active
   TString          fText;         // item text
   TString          fTipText;      // tooltip text
   Int_t            fY;            // y position of item
   Int_t            fXtext;        // x position of item text
   Int_t            fYtext;        // y position of item text
   UInt_t           fHeight;       // item height
   UInt_t           fPicWidth;     // width of item icon
   const TGPicture *fOpenPic;      // icon for open state
   const TGPicture *fClosedPic;    // icon for closed state
   void            *fUserData;     // pointer to user data structure

public:
   TGListTreeItem(TGClient *fClient, const char *name,
                  const TGPicture *opened, const TGPicture *closed);
   virtual ~TGListTreeItem();

   void Rename(const char *new_name);

   TGListTreeItem *GetParent() const { return fParent; }
   TGListTreeItem *GetFirstChild() const { return fFirstchild; }
   TGListTreeItem *GetPrevSibling() const { return fPrevsibling; }
   TGListTreeItem *GetNextSibling() const { return fNextsibling; }
   Bool_t          IsActive() const { return fActive; }
   Bool_t          IsOpen() const { return fOpen; }
   const char     *GetText() const { return fText.Data(); }
   const char     *GetTipText() const { return fTipText.Data(); }
   void            SetUserData(void *userData) { fUserData = userData; }
   void           *GetUserData() const { return fUserData; }
   void            SetPictures(const TGPicture *opened, const TGPicture *closed);

   ClassDef(TGListTreeItem,0)  //Item that goes into a TGListTree container
};


class TGListTree : public TGFrame {

friend class TGClient;

protected:
   TGListTreeItem  *fFirst;          // pointer to first item in list
   TGListTreeItem  *fSelected;       // pointer to selected item in list
   Int_t            fHspacing;       // horizontal spacing between items
   Int_t            fVspacing;       // vertical spacing between items
   Int_t            fIndent;         // number of pixels indentation
   Int_t            fMargin;         // number of pixels margin from left side
   Int_t            fLastY;          // last used y position
   ULong_t          fGrayPixel;      // gray draw color
   GContext_t       fDrawGC;         // icon drawing context
   GContext_t       fLineGC;         // dashed line drawing context
   GContext_t       fHighlightGC;    // highlighted icon drawing context
   FontStruct_t     fFont;           // font used to draw item text
   UInt_t           fDefw;           // default list width
   UInt_t           fDefh;           // default list height
   Int_t            fExposeTop;      // top y postion of visible region
   Int_t            fExposeBottom;   // bottom y position of visible region
   const TGWindow  *fMsgWindow;      // pointer to window handling list messages
   TGToolTip       *fTip;            // tooltip shown when moving over list items
   TGListTreeItem  *fTipItem;        // item for which tooltip is set
   TGCanvas        *fCanvas;         // canvas which contains the tree
   Bool_t           fAutoTips;       // assume item->fUserData is TObject and use GetTitle() for tip text

   static FontStruct_t   fgDefaultFontStruct;

   virtual void DoRedraw();

   void  Draw(Int_t yevent, Int_t hevent);
   void  Draw(Option_t * ="") { MayNotUse("Draw(Option_t*)"); }
   Int_t DrawChildren(TGListTreeItem *item, Int_t x, Int_t y, Int_t xroot);
   void  DrawItem(TGListTreeItem *item, Int_t x, Int_t y, Int_t *xroot,
                  UInt_t *retwidth, UInt_t *retheight);
   void  DrawItemName(TGListTreeItem *item);
   void  DrawNode(TGListTreeItem *item, Int_t x, Int_t y);
   void  SetToolTipText(const char *text, Int_t x, Int_t y, Long_t delayms);

   void  HighlightItem(TGListTreeItem *item, Bool_t state, Bool_t draw);
   void  HighlightChildren(TGListTreeItem *item, Bool_t state, Bool_t draw);
   void  UnselectAll(Bool_t draw);

   void  RemoveReference(TGListTreeItem *item);
   void  PDeleteChildren(TGListTreeItem *item);
   void  InsertChild(TGListTreeItem *parent, TGListTreeItem *item);
   void  InsertChildren(TGListTreeItem *parent, TGListTreeItem *item);
   Int_t SearchChildren(TGListTreeItem *item, Int_t y, Int_t findy,
                        TGListTreeItem **finditem);
   TGListTreeItem *FindItem(Int_t findy);

public:
   TGListTree(TGWindow *p, UInt_t w, UInt_t h,
              UInt_t options, ULong_t back = fgWhitePixel);
   virtual ~TGListTree();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleDoubleClick(Event_t *event);
   virtual Bool_t HandleExpose(Event_t *event);
   virtual Bool_t HandleCrossing(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);

   virtual void Associate(const TGWindow *w) { fMsgWindow = w; }
   virtual void SetCanvas(TGCanvas *canvas) { fCanvas = canvas; }

   virtual TGDimension GetDefaultSize() const
            { return TGDimension(fDefw, fDefh); }

   TGListTreeItem *AddItem(TGListTreeItem *parent, const char *string,
                           const TGPicture *open = 0,
                           const TGPicture *closed = 0);
   TGListTreeItem *AddItem(TGListTreeItem *parent, const char *string,
                           void *userData, const TGPicture *open = 0,
                           const TGPicture *closed = 0);
   void  RenameItem(TGListTreeItem *item, const char *string);
   Int_t DeleteItem(TGListTreeItem *item);
   void  OpenItem(TGListTreeItem *item);
   void  CloseItem(TGListTreeItem *item);
   Int_t RecursiveDeleteItem(TGListTreeItem *item, void *userData);
   Int_t DeleteChildren(TGListTreeItem *item);
   Int_t Reparent(TGListTreeItem *item, TGListTreeItem *newparent);
   Int_t ReparentChildren(TGListTreeItem *item, TGListTreeItem *newparent);
   void  SetToolTipItem(TGListTreeItem *item, const char *string);
   void  SetAutoTips(Bool_t on = kTRUE) { fAutoTips = on; }

   Int_t Sort(TGListTreeItem *item);
   Int_t SortSiblings(TGListTreeItem *item);
   Int_t SortChildren(TGListTreeItem *item);
   void  HighlightItem(TGListTreeItem *item);
   void  ClearHighlighted();
   void  GetPathnameFromItem(TGListTreeItem *item, char *path, Int_t depth = 0);

   TGListTreeItem *GetFirstItem() const { return fFirst; }
   TGListTreeItem *GetSelected() const { return fSelected; }
   TGListTreeItem *FindSiblingByName(TGListTreeItem *item, const char *name);
   TGListTreeItem *FindSiblingByData(TGListTreeItem *item, void *userData);
   TGListTreeItem *FindChildByName(TGListTreeItem *item, const char *name);
   TGListTreeItem *FindChildByData(TGListTreeItem *item, void *userData);
   TGListTreeItem *FindItemByPathname(const char *path);

   ClassDef(TGListTree,0)  //Show items in a tree structured list
};

#endif

// @(#)root/gui:$Name:  $:$Id: TGListTree.h,v 1.29 2006/07/03 16:10:45 brun Exp $
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

#ifndef ROOT_TGCanvas
#include "TGCanvas.h"
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
   Bool_t           fCheckBox;     // true if checkbox is visible
   Bool_t           fChecked;      // true if item is checked
   TString          fText;         // item text
   TString          fTipText;      // tooltip text
   Int_t            fY;            // y position of item
   Int_t            fXtext;        // x position of item text
   Int_t            fYtext;        // y position of item text
   UInt_t           fHeight;       // item height
   UInt_t           fPicWidth;     // width of item icon
   const TGPicture *fOpenPic;      // icon for open state
   const TGPicture *fClosedPic;    // icon for closed state
   const TGPicture *fCheckedPic;   // icon for checked item
   const TGPicture *fUncheckedPic; // icon for unchecked item
   void            *fUserData;     // pointer to user data structure

   Bool_t           fHasColor;     // true if item has assigned color
   Color_t          fColor;        // item's color

   TGListTreeItem(const TGListTreeItem&);             // not implemented
   TGListTreeItem& operator=(const TGListTreeItem&);  // not implemented

public:
   TGListTreeItem(TGClient *fClient = gClient, const char *name = 0,
                  const TGPicture *opened = 0, const TGPicture *closed = 0,
                  Bool_t checkbox = kFALSE);
   virtual ~TGListTreeItem();

   void Rename(const char *new_name);

   TGListTreeItem *GetParent() const { return fParent; }
   TGListTreeItem *GetFirstChild() const { return fFirstchild; }
   TGListTreeItem *GetPrevSibling() const { return fPrevsibling; }
   TGListTreeItem *GetNextSibling() const { return fNextsibling; }
   Bool_t          IsActive() const { return fActive; }
   Bool_t          IsOpen() const { return fOpen; }
   const char     *GetText() const { return fText.Data(); }
   void            SetTipText(const char *tip) { fTipText = tip; }
   const char     *GetTipText() const { return fTipText.Data(); }
   void            SetUserData(void *userData) { fUserData = userData; }
   void           *GetUserData() const { return fUserData; }
   void            SetPictures(const TGPicture *opened, const TGPicture *closed);
   void            SetCheckBoxPictures(const TGPicture *checked, const TGPicture *unchecked);

   void            SetCheckBox(Bool_t on = kTRUE);
   Bool_t          HasCheckBox() const { return fCheckBox; }
   void            CheckItem(Bool_t checked = kTRUE) { fChecked = checked; }
   void            Toggle() { fChecked = !fChecked; }
   Bool_t          IsChecked() const { return fChecked; }

   Color_t         GetColor() const { return fColor; }
   void            SetColor(Color_t color) { fHasColor = true;fColor = color; }
   void            ClearColor() { fHasColor = false; }

   void            SavePrimitive(ostream &out, Option_t *option, Int_t n);

   ClassDef(TGListTreeItem,0)  //Item that goes into a TGListTree container
};


class TGListTree : public TGContainer {

public:
   //---- color markup mode of tree items
   enum EColorMarkupMode {
      kDefault        = 0,
      kColorUnderline = BIT(0),
      kColorBox       = BIT(1)
   };

protected:
   TGListTreeItem  *fFirst;          // pointer to first item in list
   TGListTreeItem  *fSelected;       // pointer to selected item in list
   Int_t            fHspacing;       // horizontal spacing between items
   Int_t            fVspacing;       // vertical spacing between items
   Int_t            fIndent;         // number of pixels indentation
   Int_t            fMargin;         // number of pixels margin from left side
   Int_t            fLastY;          // last used y position
   Pixel_t          fGrayPixel;      // gray draw color
   GContext_t       fDrawGC;         // icon drawing context
   GContext_t       fLineGC;         // dashed line drawing context
   GContext_t       fHighlightGC;    // highlighted icon drawing context
   FontStruct_t     fFont;           // font used to draw item text
   UInt_t           fDefw;           // default list width
   UInt_t           fDefh;           // default list height
   Int_t            fExposeTop;      // top y postion of visible region
   Int_t            fExposeBottom;   // bottom y position of visible region
   TGToolTip       *fTip;            // tooltip shown when moving over list items
   TGListTreeItem  *fTipItem;        // item for which tooltip is set
   Bool_t           fAutoTips;       // assume item->fUserData is TObject and use GetTitle() for tip text
   Bool_t           fDisableOpen;    // disable branch opening on double-clicks

   EColorMarkupMode fColorMode;      // if/how to render item's main color
   GContext_t       fColorGC;        // drawing context for main item color

   static Pixel_t        fgGrayPixel;
   static const TGFont  *fgDefaultFont;
   static TGGC          *fgDrawGC;
   static TGGC          *fgLineGC;
   static TGGC          *fgHighlightGC;
   static TGGC          *fgColorGC;

   static Pixel_t       GetGrayPixel();
   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDrawGC();
   static const TGGC   &GetLineGC();
   static const TGGC   &GetHighlightGC();
   static const TGGC   &GetColorGC();

   virtual void DoRedraw();
   void  Draw(Int_t yevent, Int_t hevent);
   void  Draw(Option_t * ="") { MayNotUse("Draw(Option_t*)"); }
   Int_t DrawChildren(TGListTreeItem *item, Int_t x, Int_t y, Int_t xroot);
   void  DrawItem(TGListTreeItem *item, Int_t x, Int_t y, Int_t *xroot,
                  UInt_t *retwidth, UInt_t *retheight);
   void  DrawItemName(TGListTreeItem *item);
   void  DrawNode(TGListTreeItem *item, Int_t x, Int_t y);
   void  UpdateChecked(TGListTreeItem *item, Bool_t redraw = kFALSE);

   void  SaveChildren(ostream &out, TGListTreeItem *item, Int_t &n);
   void  RemoveReference(TGListTreeItem *item);
   void  PDeleteChildren(TGListTreeItem *item);
   void  InsertChild(TGListTreeItem *parent, TGListTreeItem *item);
   void  InsertChildren(TGListTreeItem *parent, TGListTreeItem *item);
   Int_t SearchChildren(TGListTreeItem *item, Int_t y, Int_t findy,
                        TGListTreeItem **finditem);
   TGListTreeItem *FindItem(Int_t findy);
   void *FindItem(const TString& name,
                  Bool_t direction = kTRUE,
                  Bool_t caseSensitive = kTRUE,
                  Bool_t beginWith = kFALSE)
      { return TGContainer::FindItem(name, direction, caseSensitive, beginWith); }

   virtual void Layout() {}

   void OnMouseOver(TGFrame*) { }
   void CurrentChanged(Int_t /*x*/, Int_t /*y*/) { }
   void CurrentChanged(TGFrame *) { }
   void ReturnPressed(TGFrame*) { }
   void Clicked(TGFrame *, Int_t /*btn*/) { }
   void Clicked(TGFrame *, Int_t /*btn*/, Int_t /*x*/, Int_t /*y*/) { }
   void DoubleClicked(TGFrame *, Int_t /*btn*/) { }
   void DoubleClicked(TGFrame *, Int_t /*btn*/, Int_t /*x*/, Int_t /*y*/) { }
   void KeyPressed(TGFrame *, UInt_t /*keysym*/, UInt_t /*mask*/) { }

private:
   TGListTree(const TGListTree&);               // not implemented
   TGListTree& operator=(const TGListTree&);    // not implemented

public:
   TGListTree(TGWindow *p = 0, UInt_t w = 1, UInt_t h = 1,
              UInt_t options = 0, Pixel_t back = GetWhitePixel());
   TGListTree(TGCanvas *p, UInt_t options, Pixel_t back = GetWhitePixel());

   virtual ~TGListTree();

   virtual Bool_t HandleButton(Event_t *event);
   virtual Bool_t HandleDoubleClick(Event_t *event);
   virtual Bool_t HandleExpose(Event_t *event);
   virtual Bool_t HandleCrossing(Event_t *event);
   virtual Bool_t HandleMotion(Event_t *event);
   virtual Bool_t HandleKey(Event_t *event);

   virtual void SetCanvas(TGCanvas *canvas) { fCanvas = canvas; }
   virtual void DrawRegion(Int_t x, Int_t y, UInt_t w, UInt_t h);

   virtual TGDimension GetDefaultSize() const
            { return TGDimension(fDefw, fDefh); }

   TGListTreeItem *AddItem(TGListTreeItem *parent, const char *string,
                           const TGPicture *open = 0,
                           const TGPicture *closed = 0,
                           Bool_t checkbox = kFALSE);
   TGListTreeItem *AddItem(TGListTreeItem *parent, const char *string,
                           void *userData, const TGPicture *open = 0,
                           const TGPicture *closed = 0,
                           Bool_t checkbox = kFALSE);
   void  RenameItem(TGListTreeItem *item, const char *string);
   Int_t DeleteItem(TGListTreeItem *item);
   void  OpenItem(TGListTreeItem *item);
   void  CloseItem(TGListTreeItem *item);
   void  CheckItem(TGListTreeItem *item, Bool_t check = kTRUE);
   void  SetCheckBox(TGListTreeItem *item, Bool_t on = kTRUE);
   void  ToggleItem(TGListTreeItem *item);
   Int_t RecursiveDeleteItem(TGListTreeItem *item, void *userData);

   Int_t DeleteChildren(TGListTreeItem *item);
   Int_t Reparent(TGListTreeItem *item, TGListTreeItem *newparent);
   Int_t ReparentChildren(TGListTreeItem *item, TGListTreeItem *newparent);
   void  SetToolTipItem(TGListTreeItem *item, const char *string);
   void  SetAutoTips(Bool_t on = kTRUE) { fAutoTips = on; }
   void  SetSelected(TGListTreeItem *item) { fSelected = item; }
   void  AdjustPosition(TGListTreeItem *item);
   void  AdjustPosition() { TGContainer::AdjustPosition(); }

   // overwrite TGContainer's methods
   void Home(Bool_t select = kFALSE);
   void End(Bool_t select = kFALSE);
   void PageUp(Bool_t select = kFALSE);
   void PageDown(Bool_t select = kFALSE);
   void LineUp(Bool_t select = kFALSE);
   void LineDown(Bool_t select = kFALSE);
   void Search(Bool_t close = kTRUE);

   Int_t Sort(TGListTreeItem *item);
   Int_t SortSiblings(TGListTreeItem *item);
   Int_t SortChildren(TGListTreeItem *item);
   void  HighlightItem(TGListTreeItem *item);
   void  ClearHighlighted();
   void  GetPathnameFromItem(TGListTreeItem *item, char *path, Int_t depth = 0);
   void  UnselectAll(Bool_t draw);
   void  SetToolTipText(const char *text, Int_t x, Int_t y, Long_t delayms);
   void  HighlightItem(TGListTreeItem *item, Bool_t state, Bool_t draw);
   void  HighlightChildren(TGListTreeItem *item, Bool_t state, Bool_t draw);
   void  DisableOpen(Bool_t disable = kTRUE) { fDisableOpen = disable;}

   TGListTreeItem *GetFirstItem() const { return fFirst; }
   TGListTreeItem *GetSelected() const { return fSelected; }
   TGListTreeItem *FindSiblingByName(TGListTreeItem *item, const char *name);
   TGListTreeItem *FindSiblingByData(TGListTreeItem *item, void *userData);
   TGListTreeItem *FindChildByName(TGListTreeItem *item, const char *name);
   TGListTreeItem *FindChildByData(TGListTreeItem *item, void *userData);
   TGListTreeItem *FindItemByPathname(const char *path);
   TGListTreeItem *FindItemByObj(TGListTreeItem *item, void *ptr);

   void  AddItem(const char *string) { AddItem(fSelected, string); } //*MENU*
   void  AddRoot(const char *string) { AddItem(0, string); } //*MENU*
   Int_t DeleteSelected() { return (fSelected ? DeleteItem(fSelected) : 0); } //*MENU*
   void  RenameSelected(const char *string) { if (fSelected) RenameItem(fSelected, string); } //*MENU*

   virtual void OnMouseOver(TGListTreeItem *entry);  //*SIGNAL*
   virtual void KeyPressed(TGListTreeItem *entry, UInt_t keysym, UInt_t mask);  //*SIGNAL*
   virtual void ReturnPressed(TGListTreeItem *entry);  //*SIGNAL*
   virtual void Clicked(TGListTreeItem *entry, Int_t btn);  //*SIGNAL*
   virtual void Clicked(TGListTreeItem *entry, Int_t btn, Int_t x, Int_t y);  //*SIGNAL*
   virtual void DoubleClicked(TGListTreeItem *entry, Int_t btn);  //*SIGNAL*
   virtual void DoubleClicked(TGListTreeItem *entry, Int_t btn, Int_t x, Int_t y);  //*SIGNAL*
   virtual void Checked(TObject *obj, Bool_t check);  //*SIGNAL*

   EColorMarkupMode GetColorMode() const { return fColorMode; }
   void SetColorMode(EColorMarkupMode colorMode) { fColorMode = colorMode; }

   virtual void SavePrimitive(ostream &out, Option_t *option = "");

   ClassDef(TGListTree,0)  //Show items in a tree structured list
};

#endif

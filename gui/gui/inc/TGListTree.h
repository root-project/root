// @(#)root/gui:$Id$
// Author: Fons Rademakers   25/02/98

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGListTree
#define ROOT_TGListTree


#include "TGCanvas.h"
#include "TGWidget.h"
#include "TGDNDManager.h"

class TGPicture;
class TGToolTip;
class TGCanvas;
class TList;
class TBufferFile;

class TGListTreeItem
{
   friend class TGListTree;

private:
   TGListTreeItem(const TGListTreeItem&) = delete;
   TGListTreeItem& operator=(const TGListTreeItem&) = delete;

protected:
   TGClient        *fClient;       ///< pointer to TGClient
   TGListTreeItem  *fParent;       ///< pointer to parent
   TGListTreeItem  *fFirstchild;   ///< pointer to first child item
   TGListTreeItem  *fLastchild;    ///< pointer to last child item
   TGListTreeItem  *fPrevsibling;  ///< pointer to previous sibling
   TGListTreeItem  *fNextsibling;  ///< pointer to next sibling

   Bool_t           fOpen;         ///< true if item is open

   Int_t            fDNDState;     ///< EDNDFlags

   ///@{
   ///@name State managed by TGListTree during drawing.
   Int_t            fY;            // y position of item
   Int_t            fXtext;        // x position of item text
   Int_t            fYtext;        // y position of item text
   UInt_t           fHeight;       // item height
   ///@}

public:
   TGListTreeItem(TGClient *client = gClient);
   virtual ~TGListTreeItem() {}

   TGListTreeItem *GetParent()      const { return fParent; }
   TGListTreeItem *GetFirstChild()  const { return fFirstchild; }
   TGListTreeItem *GetLastChild()   const { return fLastchild;  }
   TGListTreeItem *GetPrevSibling() const { return fPrevsibling; }
   TGListTreeItem *GetNextSibling() const { return fNextsibling; }

   virtual Bool_t          IsOpen()    const { return fOpen; }
   virtual void            SetOpen(Bool_t o) { fOpen = o; }

   virtual Bool_t          IsActive() const = 0;
   virtual Pixel_t         GetActiveColor() const = 0;
   virtual void            SetActive(Bool_t) {}

   void                    Rename(const char* new_name) { SetText(new_name); }
   virtual const char     *GetText() const = 0;
   virtual Int_t           GetTextLength() const = 0;
   virtual const char     *GetTipText() const = 0;
   virtual Int_t           GetTipTextLength() const = 0;
   virtual void            SetText(const char *) {}
   virtual void            SetTipText(const char *) {}

   virtual void            SetUserData(void *, Bool_t=kFALSE) {}
   virtual void           *GetUserData() const = 0;

   virtual const TGPicture*GetPicture() const = 0;
   virtual void            SetPictures(const TGPicture*, const TGPicture*) {}
   virtual const TGPicture*GetCheckBoxPicture() const = 0;
   virtual void            SetCheckBoxPictures(const TGPicture*, const TGPicture*) {}
   virtual UInt_t          GetPicWidth() const;

   virtual void            SetCheckBox(Bool_t=kTRUE) {}
   virtual Bool_t          HasCheckBox() const = 0;
   virtual void            CheckItem(Bool_t=kTRUE) = 0;
   virtual void            Toggle() { SetCheckBox( ! IsChecked()); }
   virtual Bool_t          IsChecked() const = 0;

   // Propagation of checked-state form children to parents.
   virtual void            CheckAllChildren(Bool_t=kTRUE) {}
   virtual void            CheckChildren(TGListTreeItem*, Bool_t) {}
   virtual Bool_t          HasCheckedChild(Bool_t=kFALSE)   { return kTRUE; } // !!!!
   virtual Bool_t          HasUnCheckedChild(Bool_t=kFALSE) { return kTRUE; } // !!!!
   virtual void            UpdateState() {}

   // Item coloration (underline + minibox)
   virtual Bool_t          HasColor() const = 0;
   virtual Color_t         GetColor() const = 0;
   virtual void            SetColor(Color_t) {}
   virtual void            ClearColor() {}

   // Drag and drop.
   void            SetDNDSource(Bool_t onoff)
                   { if (onoff) fDNDState |= kIsDNDSource; else fDNDState &= ~kIsDNDSource; }
   void            SetDNDTarget(Bool_t onoff)
                   { if (onoff) fDNDState |= kIsDNDTarget; else fDNDState &= ~kIsDNDTarget; }
   Bool_t          IsDNDSource() const { return fDNDState & kIsDNDSource; }
   Bool_t          IsDNDTarget() const { return fDNDState & kIsDNDTarget; }

   // Allow handling by the items themselves ... NOT USED in TGListTree yet !!!!
   virtual Bool_t  HandlesDragAndDrop() const { return kFALSE; }
   virtual void    HandleDrag() {}
   virtual void    HandleDrop() {}

   virtual void    SavePrimitive(std::ostream&, Option_t*, Int_t) {}

   ClassDef(TGListTreeItem,0)  // Abstract base-class for items that go into a TGListTree container.
};


class TGListTreeItemStd : public TGListTreeItem
{
private:
   Bool_t           fActive;       ///< true if item is active
   Bool_t           fCheckBox;     ///< true if checkbox is visible
   Bool_t           fChecked;      ///< true if item is checked
   Bool_t           fOwnsData;     ///< true if user data has to be deleted
   TString          fText;         ///< item text
   TString          fTipText;      ///< tooltip text
   const TGPicture *fOpenPic;      ///< icon for open state
   const TGPicture *fClosedPic;    ///< icon for closed state
   const TGPicture *fCheckedPic;   ///< icon for checked item
   const TGPicture *fUncheckedPic; ///< icon for unchecked item
   void            *fUserData;     ///< pointer to user data structure

   Bool_t           fHasColor;     ///< true if item has assigned color
   Color_t          fColor;        ///< item's color

   TGListTreeItemStd(const TGListTreeItemStd&) = delete;
   TGListTreeItemStd& operator=(const TGListTreeItemStd&) = delete;

public:
   TGListTreeItemStd(TGClient *fClient = gClient, const char *name = nullptr,
                     const TGPicture *opened = nullptr, const TGPicture *closed = nullptr,
                     Bool_t checkbox = kFALSE);
   virtual ~TGListTreeItemStd();

   Pixel_t         GetActiveColor() const override;
   Bool_t          IsActive() const override { return fActive; }
   void            SetActive(Bool_t a) override { fActive = a; }

   const char     *GetText() const override { return fText.Data(); }
   Int_t           GetTextLength() const override { return fText.Length(); }
   const char     *GetTipText() const override { return fTipText.Data(); }
   Int_t           GetTipTextLength() const override { return fTipText.Length(); }
   void            SetText(const char *text) override { fText = text; }
   void            SetTipText(const char *tip) override { fTipText = tip; }

   void            SetUserData(void *userData, Bool_t own = kFALSE) override { fUserData = userData; fOwnsData=own; }
   void           *GetUserData() const override { return fUserData; }

   const TGPicture *GetPicture() const override { return fOpen ? fOpenPic : fClosedPic; }
   const TGPicture *GetCheckBoxPicture() const override { return fCheckBox ? (fChecked ? fCheckedPic : fUncheckedPic) : nullptr; }
   void            SetPictures(const TGPicture *opened, const TGPicture *closed) override;
   void            SetCheckBoxPictures(const TGPicture *checked, const TGPicture *unchecked) override;

   void            SetCheckBox(Bool_t on = kTRUE) override;
   Bool_t          HasCheckBox() const override { return fCheckBox; }
   void            CheckItem(Bool_t checked = kTRUE) override { fChecked = checked; }
   void            Toggle() override { fChecked = !fChecked; }
   Bool_t          IsChecked() const override { return fChecked; }

   void            CheckAllChildren(Bool_t state = kTRUE) override;
   void            CheckChildren(TGListTreeItem *item, Bool_t state) override;
   Bool_t          HasCheckedChild(Bool_t first=kFALSE) override;
   Bool_t          HasUnCheckedChild(Bool_t first=kFALSE) override;
   void            UpdateState() override;

   Bool_t          HasColor() const override { return fHasColor; }
   Color_t         GetColor() const override { return fColor; }
   void            SetColor(Color_t color) override { fHasColor = true;fColor = color; }
   void            ClearColor() override { fHasColor = false; }

   void            SavePrimitive(std::ostream &out, Option_t *option, Int_t n) override;

   ClassDefOverride(TGListTreeItemStd,0)  //Item that goes into a TGListTree container
};


class TGListTree : public TGContainer {

public:
   //---- color markup mode of tree items
   enum EColorMarkupMode {
      kDefault        = 0,
      kColorUnderline = BIT(0),
      kColorBox       = BIT(1)
   };
   enum ECheckMode {
      kSimple    = BIT(2),
      kRecursive = BIT(3)
   };

protected:
   TGListTreeItem  *fFirst;          ///< pointer to first item in list
   TGListTreeItem  *fLast;           ///< pointer to last item in list
   TGListTreeItem  *fSelected;       ///< pointer to selected item in list
   TGListTreeItem  *fCurrent;        ///< pointer to current item in list
   TGListTreeItem  *fBelowMouse;     ///< pointer to item below mouses cursor
   Int_t            fHspacing;       ///< horizontal spacing between items
   Int_t            fVspacing;       ///< vertical spacing between items
   Int_t            fIndent;         ///< number of pixels indentation
   Int_t            fMargin;         ///< number of pixels margin from left side
   Pixel_t          fGrayPixel;      ///< gray draw color
   GContext_t       fActiveGC;       ///< activated (selected) drawing context
   GContext_t       fDrawGC;         ///< icon drawing context
   GContext_t       fLineGC;         ///< dashed line drawing context
   GContext_t       fHighlightGC;    ///< highlighted icon drawing context
   FontStruct_t     fFont;           ///< font used to draw item text
   UInt_t           fDefw;           ///< default list width
   UInt_t           fDefh;           ///< default list height
   Int_t            fExposeTop;      ///< top y postion of visible region
   Int_t            fExposeBottom;   ///< bottom y position of visible region
   TGToolTip       *fTip;            ///< tooltip shown when moving over list items
   TGListTreeItem  *fTipItem;        ///< item for which tooltip is set
   TBufferFile     *fBuf;            ///< buffer used for Drag and Drop
   TDNDData         fDNDData;        ///< Drag and Drop data
   Atom_t          *fDNDTypeList;    ///< handles DND types
   TGListTreeItem  *fDropItem;       ///< item on which DND is over
   Bool_t           fAutoTips;       ///< assume item->fUserData is TObject and use GetTitle() for tip text
   Bool_t           fAutoCheckBoxPic;///< change check box picture if parent and children have diffrent state
   Bool_t           fDisableOpen;    ///< disable branch opening on double-clicks
   Bool_t           fUserControlled; ///< let user decides what is the behaviour on events
   Bool_t           fEventHandled;   ///< flag used from user code to bypass standard event handling
   UInt_t           fLastEventState; ///< modifier state of the last keyboard event

   EColorMarkupMode fColorMode;      ///< if/how to render item's main color
   ECheckMode       fCheckMode;      ///< how to propagate check properties through the tree
   GContext_t       fColorGC;        ///< drawing context for main item color

   static Pixel_t          fgGrayPixel;
   static const TGFont    *fgDefaultFont;
   static TGGC            *fgActiveGC;
   static TGGC            *fgDrawGC;
   static TGGC            *fgLineGC;
   static TGGC            *fgHighlightGC;
   static TGGC            *fgColorGC;
   static const TGPicture *fgOpenPic;       ///< icon for open item
   static const TGPicture *fgClosedPic;     ///< icon for closed item
   static const TGPicture *fgCheckedPic;    ///< icon for checked item
   static const TGPicture *fgUncheckedPic;  ///< icon for unchecked item

   static Pixel_t       GetGrayPixel();
   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetActiveGC();
   static const TGGC   &GetDrawGC();
   static const TGGC   &GetLineGC();
   static const TGGC   &GetHighlightGC();
   static const TGGC   &GetColorGC();

   void  Draw(Handle_t id, Int_t yevent, Int_t hevent);
   void  Draw(Option_t * ="") override { MayNotUse("Draw(Option_t*)"); }
   Int_t DrawChildren(Handle_t id, TGListTreeItem *item, Int_t x, Int_t y, Int_t xroot);
   void  DrawItem(Handle_t id, TGListTreeItem *item, Int_t x, Int_t y, Int_t *xroot,
                  UInt_t *retwidth, UInt_t *retheight);
   void  DrawItemName(Handle_t id, TGListTreeItem *item);
   void  DrawNode(Handle_t id, TGListTreeItem *item, Int_t x, Int_t y);
   virtual void UpdateChecked(TGListTreeItem *item, Bool_t redraw = kFALSE);

   void  SaveChildren(std::ostream &out, TGListTreeItem *item, Int_t &n);
   void  RemoveReference(TGListTreeItem *item);
   void  PDeleteItem(TGListTreeItem *item);
   void  PDeleteChildren(TGListTreeItem *item);
   void  InsertChild(TGListTreeItem *parent, TGListTreeItem *item);
   void  InsertChildren(TGListTreeItem *parent, TGListTreeItem *item);
   Int_t SearchChildren(TGListTreeItem *item, Int_t y, Int_t findy,
                        TGListTreeItem **finditem);
   TGListTreeItem *FindItem(Int_t findy);
   void *FindItem(const TString& name,
                  Bool_t direction = kTRUE,
                  Bool_t caseSensitive = kTRUE,
                  Bool_t beginWith = kFALSE) override
      { return TGContainer::FindItem(name, direction, caseSensitive, beginWith); }

   void Layout() override {}

   void OnMouseOver(TGFrame*) override { }
   void CurrentChanged(Int_t /*x*/, Int_t /*y*/) override { }
   void CurrentChanged(TGFrame *) override { }
   void ReturnPressed(TGFrame*) override { }
   void Clicked(TGFrame *, Int_t /*btn*/) override { }
   void Clicked(TGFrame *, Int_t /*btn*/, Int_t /*x*/, Int_t /*y*/) override { }
   void DoubleClicked(TGFrame *, Int_t /*btn*/) override { }
   void DoubleClicked(TGFrame *, Int_t /*btn*/, Int_t /*x*/, Int_t /*y*/) override { }
   void KeyPressed(TGFrame *, UInt_t /*keysym*/, UInt_t /*mask*/) override { }

private:
   TGListTree(const TGListTree&) = delete;
   TGListTree& operator=(const TGListTree&) = delete;

public:
   TGListTree(TGWindow *p = nullptr, UInt_t w = 1, UInt_t h = 1,
              UInt_t options = 0, Pixel_t back = GetWhitePixel());
   TGListTree(TGCanvas *p, UInt_t options, Pixel_t back = GetWhitePixel());

   virtual ~TGListTree();

   Bool_t HandleButton(Event_t *event) override;
   Bool_t HandleDoubleClick(Event_t *event) override;
   Bool_t HandleCrossing(Event_t *event) override;
   Bool_t HandleMotion(Event_t *event) override;
   Bool_t HandleKey(Event_t *event) override;

   virtual void SetCanvas(TGCanvas *canvas) { fCanvas = canvas; }
   void DrawRegion(Int_t x, Int_t y, UInt_t w, UInt_t h) override;

   virtual void DrawOutline(Handle_t id, TGListTreeItem *item, Pixel_t col=0xbbbbbb,
                            Bool_t clear=kFALSE);
   virtual void DrawActive(Handle_t id, TGListTreeItem *item);

   TGDimension GetDefaultSize() const override
      { return TGDimension(fDefw, fDefh); }

   void            AddItem(TGListTreeItem *parent, TGListTreeItem *item);
   TGListTreeItem *AddItem(TGListTreeItem *parent, const char *string,
                           const TGPicture *open = nullptr,
                           const TGPicture *closed = nullptr,
                           Bool_t checkbox = kFALSE);
   TGListTreeItem *AddItem(TGListTreeItem *parent, const char *string,
                           void *userData, const TGPicture *open = nullptr,
                           const TGPicture *closed = nullptr,
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
   void  SetAutoCheckBoxPic(Bool_t on) { fAutoCheckBoxPic = on; }
   void  SetSelected(TGListTreeItem *item) { fSelected = item; }
   void  AdjustPosition(TGListTreeItem *item);
   void  AdjustPosition() override { TGContainer::AdjustPosition(); }

   // overwrite TGContainer's methods
   void Home(Bool_t select = kFALSE) override;
   void End(Bool_t select = kFALSE) override;
   void PageUp(Bool_t select = kFALSE) override;
   void PageDown(Bool_t select = kFALSE) override;
   void LineUp(Bool_t select = kFALSE) override;
   void LineDown(Bool_t select = kFALSE) override;
   void Search(Bool_t close = kTRUE) override;

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
   void  GetChecked(TList *checked);
   void  GetCheckedChildren(TList *checked, TGListTreeItem *item);
   void  CheckAllChildren(TGListTreeItem *item, Bool_t state);

   TGListTreeItem *GetFirstItem()  const { return fFirst; }
   TGListTreeItem *GetSelected()   const { return fSelected; }
   TGListTreeItem *GetCurrent()    const { return fCurrent; }
   TGListTreeItem *GetBelowMouse() const { return fBelowMouse; }
   TGListTreeItem *FindSiblingByName(TGListTreeItem *item, const char *name);
   TGListTreeItem *FindSiblingByData(TGListTreeItem *item, void *userData);
   TGListTreeItem *FindChildByName(TGListTreeItem *item, const char *name);
   TGListTreeItem *FindChildByData(TGListTreeItem *item, void *userData);
   TGListTreeItem *FindItemByPathname(const char *path);
   TGListTreeItem *FindItemByObj(TGListTreeItem *item, void *ptr);

   void  AddItem(const char *string) { AddItem(fSelected, string); } //*MENU*
   void  AddRoot(const char *string) { AddItem(nullptr, string); } //*MENU*
   Int_t DeleteSelected() { return fSelected ? DeleteItem(fSelected) : 0; } //*MENU*
   void  RenameSelected(const char *string) { if (fSelected) RenameItem(fSelected, string); } //*MENU*

   virtual void MouseOver(TGListTreeItem *entry);  //*SIGNAL*
   virtual void MouseOver(TGListTreeItem *entry, UInt_t mask);  //*SIGNAL*
   virtual void KeyPressed(TGListTreeItem *entry, UInt_t keysym, UInt_t mask);  //*SIGNAL*
   virtual void ReturnPressed(TGListTreeItem *entry);  //*SIGNAL*
   virtual void Clicked(TGListTreeItem *entry, Int_t btn);  //*SIGNAL*
   virtual void Clicked(TGListTreeItem *entry, Int_t btn, Int_t x, Int_t y);  //*SIGNAL*
   virtual void Clicked(TGListTreeItem *entry, Int_t btn, UInt_t mask, Int_t x, Int_t y);  //*SIGNAL*
   virtual void DoubleClicked(TGListTreeItem *entry, Int_t btn);  //*SIGNAL*
   virtual void DoubleClicked(TGListTreeItem *entry, Int_t btn, Int_t x, Int_t y);  //*SIGNAL*
   virtual void Checked(TObject *obj, Bool_t check);  //*SIGNAL*
   virtual void DataDropped(TGListTreeItem *item, TDNDData *data);  //*SIGNAL*

   // Utility functions
   Int_t        FontHeight();
   Int_t        FontAscent();
   Int_t        TextWidth(const char *c);

   static const TGPicture *GetOpenPic();
   static const TGPicture *GetClosedPic();
   static const TGPicture *GetCheckedPic();
   static const TGPicture *GetUncheckedPic();

   // User control
   void         SetUserControl(Bool_t ctrl=kTRUE) { fUserControlled = ctrl; }
   Bool_t       HasUserControl() const { return fUserControlled; }
   void         SetEventHandled(Bool_t eh=kTRUE) { fEventHandled = eh; }
   Bool_t       IsEventHandled() const { return fEventHandled; }

   Bool_t   HandleDNDDrop(TDNDData *data) override;
   Atom_t   HandleDNDPosition(Int_t x, Int_t y, Atom_t action,
                              Int_t xroot, Int_t yroot) override;
   Atom_t   HandleDNDEnter(Atom_t * typelist) override;
   Bool_t   HandleDNDLeave() override;

   TDNDData *GetDNDData(Atom_t) override { return &fDNDData; }

   EColorMarkupMode GetColorMode() const { return fColorMode; }
   void SetColorMode(EColorMarkupMode colorMode) { fColorMode = colorMode; }

   ECheckMode GetCheckMode() const { return fCheckMode; }
   void SetCheckMode(ECheckMode mode) { fCheckMode = mode; }

   void SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGListTree,0)  //Show items in a tree structured list
};

#endif

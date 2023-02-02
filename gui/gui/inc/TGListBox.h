// @(#)root/gui:$Id$
// Author: Fons Rademakers   12/01/98

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGListBox
#define ROOT_TGListBox


#include "TGFrame.h"
#include "TGCanvas.h"
#include "TGScrollBar.h"

class TGListBox;
class TList;


class TGLBEntry : public TGFrame {

protected:
   Int_t      fEntryId;          ///< message id of listbox entry
   Pixel_t    fBkcolor;          ///< entry background color
   Bool_t     fActive;           ///< true if entry is active

   void DoRedraw() override { }

public:
   TGLBEntry(const TGWindow *p = nullptr, Int_t id = -1, UInt_t options = kHorizontalFrame,
             Pixel_t back = GetWhitePixel());

   void Activate(Bool_t a) override;
   virtual void Toggle();
   virtual void Update(TGLBEntry *) { }  // this is needed on TGComboBoxes :(
   Int_t  EntryId() const { return fEntryId; }
   Bool_t IsActive() const override { return fActive;  }
   void SetBackgroundColor(Pixel_t col) override { TGFrame::SetBackgroundColor(col); fBkcolor = col; }

   ClassDefOverride(TGLBEntry,0)  // Basic listbox entry
};


class TGTextLBEntry : public TGLBEntry {

protected:
   TGString     *fText;           ///< entry text string
   UInt_t        fTWidth;         ///< text width
   UInt_t        fTHeight;        ///< text height
   Bool_t        fTextChanged;    ///< true if text has been changed
   GContext_t    fNormGC;         ///< text drawing graphics context
   FontStruct_t  fFontStruct;     ///< font used to draw string

   void DoRedraw() override;

   static const TGFont *fgDefaultFont;
   static TGGC         *fgDefaultGC;

private:
   TGTextLBEntry(const TGTextLBEntry &) = delete;
   TGTextLBEntry &operator=(const TGTextLBEntry &) = delete;

public:
   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

   TGTextLBEntry(const TGWindow *p = nullptr, TGString *s = nullptr, Int_t id = -1,
                 GContext_t norm = GetDefaultGC()(),
                 FontStruct_t font = GetDefaultFontStruct(),
                 UInt_t options = kHorizontalFrame,
                 Pixel_t back = GetWhitePixel());
   virtual ~TGTextLBEntry();

   TGDimension GetDefaultSize() const override { return TGDimension(fTWidth, fTHeight+1); }
   const TGString *GetText() const { return fText; }
   void SetText(TGString *new_text);
   const char *GetTitle() const override { return fText->Data(); }
   virtual void  SetTitle(const char *text) { *fText = text; }

   void  DrawCopy(Handle_t id, Int_t x, Int_t y) override;
   void  Update(TGLBEntry *e) override
           { SetText(new TGString(((TGTextLBEntry *)e)->GetText())); }

   GContext_t     GetNormGC() const { return fNormGC; }
   FontStruct_t   GetFontStruct() const { return fFontStruct; }

   void SavePrimitive(std::ostream &out, Option_t * = "") override;

   ClassDefOverride(TGTextLBEntry,0)  // Text listbox entry
};


class TGLineLBEntry : public TGTextLBEntry {

private:
   TGLineLBEntry(const TGLineLBEntry&) = delete;
   TGLineLBEntry operator=(const TGLineLBEntry&) = delete;

protected:
   UInt_t      fLineWidth;       ///< line width
   Style_t     fLineStyle;       ///< line style
   UInt_t      fLineLength;      ///< line length
   TGGC       *fLineGC;          ///< line graphics context

   void DoRedraw() override;

public:
   TGLineLBEntry(const TGWindow *p = nullptr, Int_t id = -1, const char *str = nullptr,
                     UInt_t w = 0, Style_t s = 0,
                     UInt_t options = kHorizontalFrame,
                     Pixel_t back = GetWhitePixel());
   virtual ~TGLineLBEntry();

   TGDimension   GetDefaultSize() const override
                  { return TGDimension(fTWidth, fTHeight+1); }
   virtual Int_t GetLineWidth() const { return fLineWidth; }
   virtual void  SetLineWidth(Int_t width);
   Style_t       GetLineStyle() const { return fLineStyle; }
   virtual void  SetLineStyle(Style_t style);
   TGGC         *GetLineGC() const { return fLineGC; }
   void          Update(TGLBEntry *e) override;
   void          DrawCopy(Handle_t id, Int_t x, Int_t y) override;

   ClassDefOverride(TGLineLBEntry, 0)  // Line width listbox entry
};


class TGIconLBEntry : public TGTextLBEntry {

private:
   TGIconLBEntry(const TGIconLBEntry&) = delete;
   TGIconLBEntry operator=(const TGIconLBEntry&) = delete;

protected:
   const TGPicture *fPicture;    // icon

   void DoRedraw() override;

public:
   TGIconLBEntry(const TGWindow *p = nullptr, Int_t id = -1, const char *str = nullptr,
                 const TGPicture *pic = nullptr,
                 UInt_t w = 0, Style_t s = 0,
                 UInt_t options = kHorizontalFrame,
                 Pixel_t back = GetWhitePixel());
   virtual ~TGIconLBEntry();

   TGDimension   GetDefaultSize() const override
                  { return TGDimension(fTWidth, fTHeight+1); }
   const TGPicture *GetPicture() const { return fPicture; }
   virtual void  SetPicture(const TGPicture *pic = nullptr);

   void  Update(TGLBEntry *e) override;
   void  DrawCopy(Handle_t id, Int_t x, Int_t y) override;

   ClassDefOverride(TGIconLBEntry, 0)  // Icon + text listbox entry
};


class TGLBContainer : public TGContainer {

friend class TGListBox;

private:
   TGLBContainer(const TGLBContainer&) = delete;
   TGLBContainer operator=(const TGLBContainer&) = delete;

protected:
   TGLBEntry      *fLastActive;    ///< last active listbox entry in single selection listbox
   TGListBox      *fListBox;       ///< list box which contains this container
   Bool_t          fMultiSelect;   ///< true if multi selection is switched on
   Int_t           fChangeStatus;  ///< defines the changes (select or unselect) while the mouse
                                   ///< moves over a multi selectable list box

   void OnAutoScroll() override;
   void DoRedraw() override;

public:
   TGLBContainer(const TGWindow *p = nullptr, UInt_t w = 1, UInt_t h = 1,
                 UInt_t options = kSunkenFrame,
                 Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGLBContainer();

   virtual void AddEntry(TGLBEntry *lbe, TGLayoutHints *lhints);
   virtual void AddEntrySort(TGLBEntry *lbe, TGLayoutHints *lhints);
   virtual void InsertEntry(TGLBEntry *lbe, TGLayoutHints *lhints, Int_t afterID);
   virtual void RemoveEntry(Int_t id);
   virtual void RemoveEntries(Int_t from_ID, Int_t to_ID);
   void         RemoveAll() override;

   void               ActivateItem(TGFrameElement *el) override;
   void               Associate(const TGWindow *w) override { fMsgWindow = w; }
   virtual void       SetListBox(TGListBox *lb) { fListBox = lb; }
   TGListBox         *GetListBox() const { return fListBox; }
   Bool_t             HandleButton(Event_t *event) override;
   Bool_t             HandleDoubleClick(Event_t *event) override;
   Bool_t             HandleMotion(Event_t *event) override;
   virtual Int_t      GetSelected() const;
   virtual Bool_t     GetSelection(Int_t id);
   virtual Int_t      GetPos(Int_t id);
   TGLBEntry         *GetSelectedEntry() const { return fLastActive; }
   virtual void       GetSelectedEntries(TList *selected);
   virtual TGLBEntry *Select(Int_t id, Bool_t sel);
   virtual TGLBEntry *Select(Int_t id);

   TGVScrollBar  *GetVScrollbar() const override;
   void           SetVsbPosition(Int_t newPos) override;
   void           Layout() override;
   UInt_t         GetDefaultWidth() const override { return fWidth; }

   virtual void   SetMultipleSelections(Bool_t multi);
   virtual Bool_t GetMultipleSelections() const { return fMultiSelect; }

   ClassDefOverride(TGLBContainer,0)  // Listbox container
};


class TGListBox : public TGCompositeFrame, public TGWidget {

private:
   TGListBox(const TGListBox&) = delete;
   TGListBox operator=(const TGListBox&) = delete;

protected:
   UInt_t           fItemVsize;       ///< maximum height of single entry
   Bool_t           fIntegralHeight;  ///< true if height should be multiple of fItemVsize
   TGLBContainer   *fLbc;             ///< listbox container
   TGViewPort      *fVport;           ///< listbox viewport (see TGCanvas.h)
   TGVScrollBar    *fVScrollbar;      ///< vertical scrollbar

   void SetContainer(TGFrame *f) { fVport->SetContainer(f); }

   virtual void InitListBox();

public:
   TGListBox(const TGWindow *p = nullptr, Int_t id = -1,
             UInt_t options = kSunkenFrame | kDoubleBorder,
             Pixel_t back = GetWhitePixel());
   virtual ~TGListBox();

   virtual void AddEntry(TGString *s, Int_t id);
   virtual void AddEntry(const char *s, Int_t id);
   virtual void AddEntry(TGLBEntry *lbe, TGLayoutHints *lhints);
   virtual void AddEntrySort(TGString *s, Int_t id);
   virtual void AddEntrySort(const char *s, Int_t id);
   virtual void AddEntrySort(TGLBEntry *lbe, TGLayoutHints *lhints);
   virtual void InsertEntry(TGString *s, Int_t id, Int_t afterID);
   virtual void InsertEntry(const char *s , Int_t id, Int_t afterID);
   virtual void InsertEntry(TGLBEntry *lbe, TGLayoutHints *lhints, Int_t afterID);
   virtual void NewEntry(const char *s = "Entry");             //*MENU*
   virtual void RemoveEntry(Int_t id = -1);                    //*MENU*
   void         RemoveAll() override;                          //*MENU*
   virtual void RemoveEntries(Int_t from_ID, Int_t to_ID);
   void         ChangeBackground(Pixel_t back) override;
   virtual void SetTopEntry(Int_t id = -1);
   virtual void SetMultipleSelections(Bool_t multi = kTRUE)
                  { fLbc->SetMultipleSelections(multi); }      //*TOGGLE* *GETTER=GetMultipleSelections
   virtual Bool_t GetMultipleSelections() const
                  { return fLbc->GetMultipleSelections(); }
   virtual Int_t  GetNumberOfEntries() const
                  { return fLbc->GetList()->GetSize(); }
   virtual TGLBEntry    *GetEntry(Int_t id) const;
   virtual TGLBEntry    *FindEntry(const char *s) const;
   virtual TGFrame      *GetContainer() const { return fVport->GetContainer(); }
   virtual TGViewPort   *GetViewPort() const { return fVport; }
   virtual TGScrollBar  *GetScrollBar() const { return fVScrollbar; }
   virtual TGVScrollBar *GetVScrollbar() const { return fVScrollbar; }

   void         DrawBorder() override;
   void         Resize(UInt_t w, UInt_t h) override;
   void         Resize(TGDimension size) override { Resize(size.fWidth, size.fHeight); }
   void         MoveResize(Int_t x, Int_t y, UInt_t w, UInt_t h) override;
   void         Layout() override;
   void         SetLayoutManager(TGLayoutManager *) override {}
   virtual void SortByName(Bool_t ascend = kTRUE);   //*MENU*icon=bld_sortup.png*
   virtual void IntegralHeight(Bool_t mode) { fIntegralHeight = mode; }
   TGDimension  GetDefaultSize() const override;

   Bool_t       ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2) override;

   virtual TGLBEntry *Select(Int_t id, Bool_t sel = kTRUE)
                                       { return fLbc->Select(id, sel); }
   virtual Int_t GetSelected() const;
   virtual Bool_t GetSelection(Int_t id) { return fLbc->GetSelection(id); }
   virtual TGLBEntry *GetSelectedEntry() const { return fLbc->GetSelectedEntry(); }
   virtual void GetSelectedEntries(TList *selected);
   UInt_t  GetItemVsize() const { return fItemVsize; }

   void SavePrimitive(std::ostream &out, Option_t *option = "") override;

   virtual void Selected(Int_t widgetId, Int_t id);   //*SIGNAL*
   virtual void Selected(Int_t id) { Emit("Selected(Int_t)", id); } //*SIGNAL*
   virtual void Selected(const char *txt) { Emit("Selected(char*)", txt); } //*SIGNAL
   virtual void DoubleClicked(Int_t widgetId, Int_t id);   //*SIGNAL*
   virtual void DoubleClicked(Int_t id) { Emit("DoubleClicked(Int_t)", id); } //*SIGNAL*
   virtual void DoubleClicked(const char *txt) { Emit("DoubleClicked(char*)", txt); } //*SIGNAL
   virtual void SelectionChanged() { Emit("SelectionChanged()"); } //*SIGNAL*

   ClassDefOverride(TGListBox,0)  // Listbox widget
};

#endif

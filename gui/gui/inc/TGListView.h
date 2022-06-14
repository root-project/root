// @(#)root/gui:$Id$
// Author: Fons Rademakers   17/01/98

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGListView
#define ROOT_TGListView


#include "TGCanvas.h"
#include "TGWidget.h"
#include "TGSplitter.h"


enum EListViewMode {
   kLVLargeIcons,
   kLVSmallIcons,
   kLVList,
   kLVDetails
};


class TGSelectedPicture;
class TGTextButton;
class TGListView;
class TGLVContainer;
class TGHeaderFrame;


class TGLVEntry : public TGFrame {

private:
   TGLVEntry(const TGLVEntry&) = delete;
   TGLVEntry& operator=(const TGLVEntry&) = delete;

protected:
   TGString           *fItemName;    ///< name of item
   TGString          **fSubnames;    ///< sub names of item (details)
   Int_t              *fCpos;        ///< position of sub names
   Int_t              *fJmode;       ///< alignment for sub names
   Int_t              *fCtw;         ///< width of sub names
   UInt_t              fTWidth;      ///< width of name
   UInt_t              fTHeight;     ///< height of name
   Bool_t              fActive;      ///< true if item is active
   Bool_t              fChecked;     ///< true if item is checked
   EListViewMode       fViewMode;    ///< list view viewing mode
   const TGPicture    *fBigPic;      ///< big icon
   const TGPicture    *fSmallPic;    ///< small icon
   const TGPicture    *fCurrent;     ///< current icon
   const TGPicture    *fCheckMark;   ///< checkmark
   TGSelectedPicture  *fSelPic;      ///< selected icon
   GContext_t          fNormGC;      ///< drawing graphics context
   FontStruct_t        fFontStruct;  ///< text font
   void               *fUserData;    ///< pointer to user data structure

   static const TGFont *fgDefaultFont;
   static TGGC         *fgDefaultGC;

   void DoRedraw() override;

   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

public:
   TGLVEntry(const TGWindow *p = nullptr,
             const TGPicture *bigpic = nullptr, const TGPicture *smallpic = nullptr,
             TGString *name = nullptr, TGString **subnames = nullptr,
             EListViewMode ViewMode = kLVDetails,
             UInt_t options = kChildFrame,
             Pixel_t back = GetWhitePixel());

   TGLVEntry(const TGLVContainer *p,
             const TString& name, const TString& cname, TGString **subnames = nullptr,
             UInt_t options = kChildFrame, Pixel_t back = GetWhitePixel());

   virtual ~TGLVEntry();

   virtual void SetViewMode(EListViewMode viewMode);

   void                Activate(Bool_t a) override;
   Bool_t              IsActive() const override { return fActive; }
   TGString           *GetItemName() const { return fItemName; }
   const char         *GetTitle() const override { return fItemName->GetString(); }
   virtual void        SetTitle(const char *text) { *fItemName = text; }
   void                SetItemName(const char *name) { *fItemName = name; }
   const TGPicture    *GetPicture() const { return fCurrent; }
   EListViewMode       GetViewMode() const { return fViewMode; }
   void                SetUserData(void *userData) { fUserData = userData; }
   void               *GetUserData() const { return fUserData; }
   virtual TGString  **GetSubnames() const { return fSubnames; }
   virtual TGString   *GetSubname(Int_t idx) const { if (fSubnames) return fSubnames[idx]; else return nullptr; }
   virtual void        SetSubnames(const char* n1="",const char* n2="",const char* n3="",
                                   const char* n4="",const char* n5="",const char* n6="",
                                   const char* n7="",const char* n8="",const char* n9="",
                                   const char* n10="",const char* n11="",const char* n12="");
   virtual void        SetPictures(const TGPicture *bigpic = nullptr, const TGPicture *smallpic = nullptr);
   virtual void        SetColumns(Int_t *cpos, Int_t *jmode) { fCpos = cpos; fJmode = jmode; }
   virtual void        SetCheckedEntry(Bool_t check = kTRUE) { fChecked = check; }

   TGDimension         GetDefaultSize() const override;
   virtual Int_t       GetSubnameWidth(Int_t idx) const { return fCtw[idx]; }

   void                DrawCopy(Handle_t id, Int_t x, Int_t y) override;

   ClassDefOverride(TGLVEntry,0)  // Item that goes into a TGListView container
};


class TGListView : public TGCanvas {

private:
   TGListView(const TGListView&) = delete;
   TGListView& operator=(const TGListView&) = delete;

protected:
   Int_t                 fNColumns;      ///< number of columns
   Int_t                *fColumns;       ///< column width
   Int_t                *fJmode;         ///< column text alignment
   EListViewMode         fViewMode;      ///< view mode if list view widget
   TGDimension           fMaxSize;       ///< maximum item size
   TGTextButton        **fColHeader;     ///< column headers for in detailed mode
   TString              *fColNames;      ///< column titles for in detailed mode
   TGVFileSplitter     **fSplitHeader;   ///< column splitters
   GContext_t            fNormGC;        ///< drawing graphics context
   FontStruct_t          fFontStruct;    ///< text font
   TGHeaderFrame        *fHeader;        ///< frame used as container for column headers
   Bool_t                fJustChanged;   ///< Indicate whether the view mode was just changed to Detail
   UInt_t                fMinColumnSize; ///< Minimun column size

   static const TGFont  *fgDefaultFont;
   static TGGC          *fgDefaultGC;

   static FontStruct_t   GetDefaultFontStruct();
   static const TGGC    &GetDefaultGC();

public:
   TGListView(const TGWindow *p, UInt_t w, UInt_t h,
              UInt_t options = kSunkenFrame | kDoubleBorder,
              Pixel_t back = GetDefaultFrameBackground());
   virtual ~TGListView();

   virtual void   ResizeColumns();
   void           Layout() override;
   virtual void   LayoutHeader(TGFrame *head);
   Bool_t         ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2) override;
   virtual void   ScrollHeader(Int_t pos);
   void           SetContainer(TGFrame *f) override;
   virtual void   AdjustHeaders() { fJustChanged = kTRUE; LayoutHeader(0); }
   virtual void   SetHeaders(Int_t ncolumns);
   virtual void   SetHeader(const char *s, Int_t hmode, Int_t cmode, Int_t idx);
   virtual void   SetDefaultHeaders();
   virtual void   SetViewMode(EListViewMode viewMode);
   TGTextButton** GetHeaderButtons() { return fColHeader; }
   UInt_t         GetNumColumns() { return fNColumns; }
   EListViewMode  GetViewMode() const { return fViewMode; }
   virtual const char *GetHeader(Int_t idx) const;
   void           SavePrimitive(std::ostream &out, Option_t *option = "") override;
   virtual void   SetIncrements(Int_t hInc, Int_t vInc);
   virtual void   SetDefaultColumnWidth(TGVFileSplitter* splitter);
   TGDimension    GetMaxItemSize() const { return fMaxSize; }

   virtual void SelectionChanged() { Emit("SelectionChanged()"); }  //*SIGNAL*
   virtual void Clicked(TGLVEntry *entry, Int_t btn);  //*SIGNAL*
   virtual void Clicked(TGLVEntry *entry, Int_t btn, Int_t x, Int_t y);  //*SIGNAL*
   virtual void DoubleClicked(TGLVEntry *entry, Int_t btn);  //*SIGNAL*
   virtual void DoubleClicked(TGLVEntry *entry, Int_t btn, Int_t x, Int_t y);  //*SIGNAL*

   ClassDefOverride(TGListView,0)  // List view widget (iconbox, small icons or tabular view)
};


class TGLVContainer : public TGContainer {

private:
   TGLVContainer(const TGLVContainer&) = delete;
   TGLVContainer& operator=(const TGLVContainer&) = delete;

protected:
   TGLayoutHints     *fItemLayout;    ///< item layout hints
   EListViewMode      fViewMode;      ///< list view viewing mode
   Int_t             *fCpos;          ///< position of sub names
   Int_t             *fJmode;         ///< alignment of sub names
   Bool_t             fMultiSelect;   ///< true = multiple file selection
   TGListView        *fListView;      ///< listview which contains this container
   TGLVEntry         *fLastActive;    ///< last active item

   void ActivateItem(TGFrameElement* el) override;
   void DeActivateItem(TGFrameElement* el) override;

public:
   TGLVContainer(const TGWindow *p, UInt_t w, UInt_t h,
                 UInt_t options = kSunkenFrame,
                 Pixel_t back = GetDefaultFrameBackground());
   TGLVContainer(TGCanvas *p, UInt_t options = kSunkenFrame,
                 Pixel_t back = GetDefaultFrameBackground());

   virtual ~TGLVContainer();

   TGListView  *GetListView() const { return fListView; }

   virtual void AddItem(TGLVEntry *item)
                  { AddFrame(item, fItemLayout); item->SetColumns(fCpos, fJmode); fTotal++; }
   virtual void SelectEntry(TGLVEntry *item);

   virtual void  SetListView(TGListView *lv) { fListView = lv; }
   virtual void  RemoveItemWithData(void *userData);
   virtual void  SetViewMode(EListViewMode viewMode);
   EListViewMode GetViewMode() const { return fViewMode; }
   virtual void  SetColumns(Int_t *cpos, Int_t *jmode);

   TGDimension GetPageDimension() const override;
   virtual TGDimension GetMaxItemSize() const;
   virtual Int_t GetMaxSubnameWidth(Int_t idx) const;
   virtual void  SetColHeaders(const char* n1="",const char* n2="",const char* n3="",
                               const char* n4="",const char* n5="",const char* n6="",
                               const char* n7="",const char* n8="",const char* n9="",
                               const char* n10="",const char* n11="",const char* n12="");
   void LineUp(Bool_t select = kFALSE) override;
   void LineDown(Bool_t select = kFALSE) override;
   void LineLeft(Bool_t select = kFALSE) override;
   void LineRight(Bool_t select = kFALSE) override;

   Bool_t HandleButton(Event_t* event) override;
   TList *GetSelectedItems();
   TList *GetSelectedEntries();
   Bool_t GetMultipleSelection() const { return fMultiSelect; };
   void   SetMultipleSelection(Bool_t multi = kTRUE) { fMultiSelect = multi; };
   void   SetHeaders(Int_t ncolumns) { fListView->SetHeaders(ncolumns); }
   void   SetHeader(const char *s, Int_t hmode, Int_t cmode, Int_t idx)
                              { fListView->SetHeader(s,hmode,cmode,idx); }
   void   SetDefaultHeaders() { fListView->SetDefaultHeaders(); }
   const char *GetHeader(Int_t idx) const { return fListView->GetHeader(idx); }
   void   SavePrimitive(std::ostream &out, Option_t *option = "") override;

   ClassDefOverride(TGLVContainer,0)  // Listview container
};

#endif

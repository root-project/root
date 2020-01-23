// @(#)root/gui:$Id$
// Author: Fons Rademakers   17/01/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGListView
#define ROOT_TGListView


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGListView, TGLVContainer and TGLVEntry                              //
//                                                                      //
// A list view is a widget that can contain a number of items           //
// arranged in a grid or list. The items can be represented either      //
// by a string or by an icon.                                           //
//                                                                      //
// The TGListView is user callable. The other classes are service       //
// classes of the list view.                                            //
//                                                                      //
// A list view can generate the following events:                       //
// kC_CONTAINER, kCT_SELCHANGED, total items, selected items.           //
// kC_CONTAINER, kCT_ITEMCLICK, which button, location (y<<16|x).       //
// kC_CONTAINER, kCT_ITEMDBLCLICK, which button, location (y<<16|x).    //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

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
   TGLVEntry(const TGLVEntry&); // Not implemented
   TGLVEntry& operator=(const TGLVEntry&); // Not implemented

protected:
   TGString           *fItemName;    // name of item
   TGString          **fSubnames;    // sub names of item (details)
   Int_t              *fCpos;        // position of sub names
   Int_t              *fJmode;       // alignment for sub names
   Int_t              *fCtw;         // width of sub names
   UInt_t              fTWidth;      // width of name
   UInt_t              fTHeight;     // height of name
   Bool_t              fActive;      // true if item is active
   Bool_t              fChecked;     // true if item is checked
   EListViewMode       fViewMode;    // list view viewing mode
   const TGPicture    *fBigPic;      // big icon
   const TGPicture    *fSmallPic;    // small icon
   const TGPicture    *fCurrent;     // current icon
   const TGPicture    *fCheckMark;   // checkmark
   TGSelectedPicture  *fSelPic;      // selected icon
   GContext_t          fNormGC;      // drawing graphics context
   FontStruct_t        fFontStruct;  // text font
   void               *fUserData;    // pointer to user data structure

   static const TGFont *fgDefaultFont;
   static TGGC         *fgDefaultGC;

   virtual void DoRedraw();

   static FontStruct_t  GetDefaultFontStruct();
   static const TGGC   &GetDefaultGC();

public:
   TGLVEntry(const TGWindow *p = 0,
             const TGPicture *bigpic = 0, const TGPicture *smallpic = 0,
             TGString *name = 0, TGString **subnames = 0,
             EListViewMode ViewMode = kLVDetails,
             UInt_t options = kChildFrame,
             Pixel_t back = GetWhitePixel());

   TGLVEntry(const TGLVContainer *p,
             const TString& name, const TString& cname, TGString **subnames = 0,
             UInt_t options = kChildFrame, Pixel_t back = GetWhitePixel());

   virtual ~TGLVEntry();

   virtual void SetViewMode(EListViewMode viewMode);

   virtual void        Activate(Bool_t a);
   Bool_t              IsActive() const { return fActive; }
   TGString           *GetItemName() const { return fItemName; }
   virtual const char *GetTitle() const { return fItemName->GetString(); }
   virtual void        SetTitle(const char *text) { *fItemName = text; }
   void                SetItemName(const char *name) { *fItemName = name; }
   const TGPicture    *GetPicture() const { return fCurrent; }
   EListViewMode       GetViewMode() const { return fViewMode; }
   void                SetUserData(void *userData) { fUserData = userData; }
   void               *GetUserData() const { return fUserData; }
   virtual TGString  **GetSubnames() const { return fSubnames; }
   virtual TGString   *GetSubname(Int_t idx) const { if (fSubnames) return fSubnames[idx]; else return 0; }
   virtual void        SetSubnames(const char* n1="",const char* n2="",const char* n3="",
                                   const char* n4="",const char* n5="",const char* n6="",
                                   const char* n7="",const char* n8="",const char* n9="",
                                   const char* n10="",const char* n11="",const char* n12="");
   virtual void        SetPictures(const TGPicture *bigpic = 0, const TGPicture *smallpic = 0);
   virtual void        SetColumns(Int_t *cpos, Int_t *jmode) { fCpos = cpos; fJmode = jmode; }
   virtual void        SetCheckedEntry(Bool_t check = kTRUE) { fChecked = check; }

   virtual TGDimension GetDefaultSize() const;
   virtual Int_t       GetSubnameWidth(Int_t idx) const { return fCtw[idx]; }

   virtual void DrawCopy(Handle_t id, Int_t x, Int_t y);

   ClassDef(TGLVEntry,0)  // Item that goes into a TGListView container
};


class TGListView : public TGCanvas {

private:
   TGListView(const TGListView&); // Not implemented
   TGListView& operator=(const TGListView&); // Not implemented

protected:
   Int_t                 fNColumns;      // number of columns
   Int_t                *fColumns;       // column width
   Int_t                *fJmode;         // column text alignment
   EListViewMode         fViewMode;      // view mode if list view widget
   TGDimension           fMaxSize;       // maximum item size
   TGTextButton        **fColHeader;     // column headers for in detailed mode
   TString              *fColNames;      // column titles for in detailed mode
   TGVFileSplitter     **fSplitHeader;   // column splitters
   GContext_t            fNormGC;        // drawing graphics context
   FontStruct_t          fFontStruct;    // text font
   TGHeaderFrame        *fHeader;        // frame used as container for column headers
   Bool_t                fJustChanged;   // Indicate whether the view mode was just changed to Detail
   UInt_t                fMinColumnSize; // Minimun column size

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
   virtual void   Layout();
   virtual void   LayoutHeader(TGFrame *head);
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   virtual void   ScrollHeader(Int_t pos);
   virtual void   SetContainer(TGFrame *f);
   virtual void   AdjustHeaders() { fJustChanged = kTRUE; LayoutHeader(0); }
   virtual void   SetHeaders(Int_t ncolumns);
   virtual void   SetHeader(const char *s, Int_t hmode, Int_t cmode, Int_t idx);
   virtual void   SetDefaultHeaders();
   virtual void   SetViewMode(EListViewMode viewMode);
   TGTextButton** GetHeaderButtons() { return fColHeader; }
   UInt_t         GetNumColumns() { return fNColumns; }
   EListViewMode  GetViewMode() const { return fViewMode; }
   virtual const char *GetHeader(Int_t idx) const;
   virtual void   SavePrimitive(std::ostream &out, Option_t *option = "");
   virtual void   SetIncrements(Int_t hInc, Int_t vInc);
   virtual void   SetDefaultColumnWidth(TGVFileSplitter* splitter);
   TGDimension    GetMaxItemSize() const { return fMaxSize; }

   virtual void SelectionChanged() { Emit("SelectionChanged()"); }  //*SIGNAL*
   virtual void Clicked(TGLVEntry *entry, Int_t btn);  //*SIGNAL*
   virtual void Clicked(TGLVEntry *entry, Int_t btn, Int_t x, Int_t y);  //*SIGNAL*
   virtual void DoubleClicked(TGLVEntry *entry, Int_t btn);  //*SIGNAL*
   virtual void DoubleClicked(TGLVEntry *entry, Int_t btn, Int_t x, Int_t y);  //*SIGNAL*

   ClassDef(TGListView,0)  // List view widget (iconbox, small icons or tabular view)
};


class TGLVContainer : public TGContainer {

private:
   TGLVContainer(const TGLVContainer&); // Not implemented
   TGLVContainer& operator=(const TGLVContainer&); // Not implemented

protected:
   TGLayoutHints     *fItemLayout;    // item layout hints
   EListViewMode      fViewMode;      // list view viewing mode
   Int_t             *fCpos;          // position of sub names
   Int_t             *fJmode;         // alignment of sub names
   Bool_t             fMultiSelect;   // true = multiple file selection
   TGListView        *fListView;      // listview which contains this container
   TGLVEntry         *fLastActive;    // last active item

   virtual void ActivateItem(TGFrameElement* el);
   virtual void DeActivateItem(TGFrameElement* el);

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

   virtual TGDimension GetPageDimension() const;
   virtual TGDimension GetMaxItemSize() const;
   virtual Int_t GetMaxSubnameWidth(Int_t idx) const;
   virtual void  SetColHeaders(const char* n1="",const char* n2="",const char* n3="",
                               const char* n4="",const char* n5="",const char* n6="",
                               const char* n7="",const char* n8="",const char* n9="",
                               const char* n10="",const char* n11="",const char* n12="");
   virtual void LineUp(Bool_t select = kFALSE);
   virtual void LineDown(Bool_t select = kFALSE);
   virtual void LineLeft(Bool_t select = kFALSE);
   virtual void LineRight(Bool_t select = kFALSE);

   virtual Bool_t HandleButton(Event_t* event);
   TList *GetSelectedItems();
   TList *GetSelectedEntries();
   Bool_t GetMultipleSelection() const { return fMultiSelect; };
   void   SetMultipleSelection(Bool_t multi = kTRUE) { fMultiSelect = multi; };
   void   SetHeaders(Int_t ncolumns) { fListView->SetHeaders(ncolumns); }
   void   SetHeader(const char *s, Int_t hmode, Int_t cmode, Int_t idx)
                              { fListView->SetHeader(s,hmode,cmode,idx); }
   void   SetDefaultHeaders() { fListView->SetDefaultHeaders(); }
   const char *GetHeader(Int_t idx) const { return fListView->GetHeader(idx); }
   virtual void   SavePrimitive(std::ostream &out, Option_t *option = "");

   ClassDef(TGLVContainer,0)  // Listview container
};

#endif

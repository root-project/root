// @(#)root/gui:$Name:  $:$Id: TGListView.h,v 1.10 2002/06/12 17:56:25 rdm Exp $
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

#ifndef ROOT_TGCanvas
#include "TGCanvas.h"
#endif
#ifndef ROOT_TGWidget
#include "TGWidget.h"
#endif


enum EListViewMode {
   kLVLargeIcons,
   kLVSmallIcons,
   kLVList,
   kLVDetails
};


class TGSelectedPicture;
class TGTextButton;
class TGListView;


class TGLVEntry : public TGFrame {

friend class TGClient;

protected:
   TGString           *fName;        // name of item
   TGString          **fSubnames;    // sub names of item (details)
   Int_t              *fCpos;        // position of sub names
   Int_t              *fJmode;       // alignment for sub names
   Int_t              *fCtw;         // width of sub names
   UInt_t              fTWidth;      // width of name
   UInt_t              fTHeight;     // height of name
   Bool_t              fActive;      // true if item is active
   EListViewMode       fViewMode;    // list view viewing mode
   const TGPicture    *fBigPic;      // big icon
   const TGPicture    *fSmallPic;    // small icon
   const TGPicture    *fCurrent;     // current icon
   TGSelectedPicture  *fSelPic;      // selected icon
   GContext_t          fNormGC;      // drawing graphics context
   FontStruct_t        fFontStruct;  // text font
   void               *fUserData;    // pointer to user data structure

   static FontStruct_t   fgDefaultFontStruct;
   static TGGC           fgDefaultGC;
   static ULong_t        fgSelPixel;

   virtual void DoRedraw();

public:
   TGLVEntry(const TGWindow *p,
             const TGPicture *bigpic, const TGPicture *smallpic,
             TGString *name, TGString **subnames, EListViewMode ViewMode,
             UInt_t options = kChildFrame,
             ULong_t back = GetWhitePixel());
   virtual ~TGLVEntry();

   virtual void SetViewMode(EListViewMode ViewMode);

   virtual void     Activate(Bool_t a);
   Bool_t           IsActive() const { return fActive; }
   TGString        *GetItemName() const { return fName; }
   void             SetItemName(const char *name) { *fName = name; }
   const TGPicture *GetPicture() const { return fCurrent; }
   void             SetUserData(void *userData) { fUserData = userData; }
   void            *GetUserData() const { return fUserData; }

   void SetColumns(Int_t *cpos, Int_t *jmode) { fCpos = cpos; fJmode = jmode; }

   virtual TGDimension GetDefaultSize() const;
   virtual Int_t       GetSubnameWidth(Int_t idx) const { return fCtw[idx]; }

    virtual void DrawCopy(Handle_t id,Int_t x, Int_t y);

  ClassDef(TGLVEntry,0)  // Item that goes into a TGListView container
};



class TGLVContainer : public TGContainer {

friend class TGClient;

protected:
   TGLayoutHints     *fItemLayout;    // item layout hints
   EListViewMode      fViewMode;      // list view viewing mode
   Int_t             *fCpos;          // position of sub names
   Int_t             *fJmode;         // alignment of sub names
   TGListView        *fListView;      // listview which contains this container
   TGLVEntry         *fLastActive;    // last active item

   virtual void ActivateItem(TGFrameElement* el);

public:
   TGLVContainer(const TGWindow *p, UInt_t w, UInt_t h,
                 UInt_t options = kSunkenFrame,
                 ULong_t back = GetDefaultFrameBackground());
   virtual ~TGLVContainer();

   TGListView  *GetListView() const { return fListView; }

   virtual void AddItem(TGLVEntry *item)
              { AddFrame(item, fItemLayout); item->SetColumns(fCpos, fJmode); }

   virtual void SetListView(TGListView *lv) { fListView = lv; }
   virtual void RemoveItemWithData(void *userData);
   virtual void SetViewMode(EListViewMode ViewMode);
   Int_t        GetViewMode() const { return fViewMode; }
   virtual void SetColumns(Int_t *cpos, Int_t *jmode);

   virtual TGDimension GetMaxItemSize() const;
   virtual Int_t       GetMaxSubnameWidth(Int_t idx) const;

   ClassDef(TGLVContainer,0)  // Listview container
};



class TGListView : public TGCanvas {

friend class TGClient;

protected:
   Int_t           fNColumns;     // number of columns
   Int_t          *fColumns;      // column width
   Int_t          *fJmode;        // column text alignment
   EListViewMode   fViewMode;     // view mode if list view widget
   TGDimension     fMaxSize;      // maximum item size
   TGTextButton  **fColHeader;    // column headers for in detailed mode

   static TGGC          fgDefaultGC;
   static FontStruct_t  fgDefaultFontStruct;

public:
   TGListView(const TGWindow *p, UInt_t w, UInt_t h,
              UInt_t options = kSunkenFrame | kDoubleBorder,
              ULong_t back = GetDefaultFrameBackground());
   virtual ~TGListView();

   virtual void   Layout();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   virtual void   SetContainer(TGFrame *f);
   virtual void   SetHeaders(Int_t ncolumns);
   virtual void   SetHeader(const char *s, Int_t hmode, Int_t cmode, Int_t idx);
   virtual void   SetDefaultHeaders();
   virtual void   SetViewMode(EListViewMode ViewMode);
   virtual const char *GetHeader(Int_t idx) const;

   virtual void SelectionChanged() { Emit("SelectionChanged()"); }  //*SIGNAL*
   virtual void Clicked(TGLVEntry *entry, Int_t btn);  //*SIGNAL*
   virtual void Clicked(TGLVEntry *entry, Int_t btn, Int_t x, Int_t y);  //*SIGNAL*
   virtual void DoubleClicked(TGLVEntry *entry, Int_t btn);  //*SIGNAL*
   virtual void DoubleClicked(TGLVEntry *entry, Int_t btn, Int_t x, Int_t y);  //*SIGNAL*

   ClassDef(TGListView,0)  // List view widget (iconbox, small icons or tabular view)
};

#endif

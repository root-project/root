// @(#)root/gui:$Name:  $:$Id: TRootBrowser.h,v 1.9 2003/10/08 09:50:47 brun Exp $
// Author: Fons Rademakers   27/02/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TRootBrowser
#define ROOT_TRootBrowser

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootBrowser                                                         //
//                                                                      //
// This class creates a ROOT object browser (looking like Windows       //
// Explorer). The widgets used are the new native ROOT GUI widgets.     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TBrowserImp
#include "TBrowserImp.h"
#endif
#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGMenuBar;
class TGPopupMenu;
class TGLayoutHints;
class TGStatusBar;
class TGHorizontal3DLine;
class TGToolBar;
class TGButton;
class TGFSComboBox;
class TGLabel;
class TGListView;
class TRootIconBox;
class TGCanvas;
class TGListTree;
class TGListTreeItem;
class TGFileItem;
class TList;
class TGFileContainer;

class TRootBrowser : public TGMainFrame, public TBrowserImp {

friend class TRootIconBox;

private:
   TGMenuBar           *fMenuBar;
   TGToolBar           *fToolBar;
   TGHorizontal3DLine  *fToolBarSep;
   TGVerticalFrame     *fV1;
   TGVerticalFrame     *fV2;
   TGLabel             *fLbl1;
   TGLabel             *fLbl2;
   TGHorizontalFrame   *fHf;
   TGCompositeFrame    *fTreeHdr;
   TGCompositeFrame    *fListHdr;

   TGLayoutHints       *fMenuBarLayout;
   TGLayoutHints       *fMenuBarItemLayout;
   TGLayoutHints       *fMenuBarHelpLayout;
   TGLayoutHints       *fComboLayout;
   TGLayoutHints       *fBarLayout;

   TList               *fWidgets;
   Cursor_t             fWaitCursor;        // busy cursor

   void  CreateBrowser(const char *name);
   void  ListTreeHighlight(TGListTreeItem *item);
   void  IconBoxAction(TObject *obj);
   void  Chdir(TGListTreeItem *item);
   void  DisplayDirectory();
   void  DisplayTotal(Int_t total, Int_t selected);
   void  SetViewMode(Int_t new_mode, Bool_t force = kFALSE);
   void  SetSortMode(Int_t new_mode);
   void  ToUpSystemDirectory();

protected:
   TGPopupMenu         *fFileMenu;
   TGPopupMenu         *fViewMenu;
   TGPopupMenu         *fOptionMenu;
   TGPopupMenu         *fHelpMenu;
   TGPopupMenu         *fSortMenu;
   TGListView          *fListView;
   TRootIconBox        *fIconBox;
   TGCanvas            *fTreeView;
   TGListTree          *fLt;
   TGButton            *fToolBarButton[7];  // same size as gToolBarData[]
   TGFSComboBox        *fFSComboBox;
   TGStatusBar         *fStatusBar;
   TGListTreeItem      *fListLevel;         // current TGListTree level
   Bool_t               fTreeLock;          // true when we want to lock TGListTree
   Int_t                fViewMode;          // current IconBox view mode
   Int_t                fSortMode;          // current IconBox sort mode

public:
   TRootBrowser(TBrowser *b, const char *title, UInt_t width, UInt_t height);
   TRootBrowser(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual ~TRootBrowser();

   virtual void Add(TObject *obj, const char *name = 0);
   virtual void AddToBox(TObject *obj, const char *name);
   virtual void AddToTree(TObject *obj, const char *name);
   virtual void BrowseObj(TObject *obj);            //*SIGNAL*
   virtual void ExecuteDefaultAction(TObject *obj); //*SIGNAL*
   virtual void DoubleClicked(TObject *obj);        //*SIGNAL*
   virtual void Iconify() { }
   virtual void RecursiveRemove(TObject *obj);
   virtual void Refresh(Bool_t force = kFALSE);
   virtual void ResizeBrowser() { }
   virtual void ShowToolBar(Bool_t show = kTRUE);
   virtual void ShowStatusBar(Bool_t show = kTRUE);
   virtual void Show() { MapRaised(); }
   virtual void SetDefaults(const char *iconStyle = 0, const char *sortBy = 0);
   TGListTree      *GetListTree()  const { return fLt; }
   TGFileContainer *GetIconBox()   const;
   TGStatusBar     *GetStatusBar() const { return fStatusBar; }
   TGMenuBar       *GetMenuBar()   const { return  fMenuBar; }
   TGToolBar       *GetToolBar()   const { return fToolBar; }

   // overridden from TGMainFrame
   void     CloseWindow();
   Bool_t   ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   void     ReallyDelete();

   ClassDef(TRootBrowser,0)  //ROOT native GUI version of browser
};

#endif

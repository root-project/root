// @(#)root/gui:$Name$:$Id$
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


class TRootBrowser : public TGMainFrame, public TBrowserImp {

private:
   TGHorizontal3DLine  *fToolBarSep;
   TGToolBar           *fToolBar;
   TGButton            *fToolBarButton[7];  // same size as gToolBarData[]
   TGFSComboBox        *fFSComboBox;
   TGStatusBar         *fStatusBar;
   TGVerticalFrame     *fV1;
   TGVerticalFrame     *fV2;
   TGLabel             *fLbl1;
   TGLabel             *fLbl2;
   TGHorizontalFrame   *fHf;
   TGCompositeFrame    *fTreeHdr;
   TGCompositeFrame    *fListHdr;
   TGListView          *fListView;
   TRootIconBox        *fIconBox;

   TGCanvas            *fTreeView;
   TGListTree          *fLt;

   TGLayoutHints       *fMenuBarLayout;
   TGLayoutHints       *fMenuBarItemLayout;
   TGLayoutHints       *fMenuBarHelpLayout;
   TGLayoutHints       *fComboLayout;
   TGLayoutHints       *fBarLayout;

   TGMenuBar           *fMenuBar;
   TGPopupMenu         *fFileMenu;
   TGPopupMenu         *fViewMenu;
   TGPopupMenu         *fOptionMenu;
   TGPopupMenu         *fHelpMenu;
   TGPopupMenu         *fSortMenu;

   TList               *fWidgets;

   char                 fCurrentDir[1024];
   Cursor_t             fWaitCursor;        // busy cursor
   TGListTreeItem      *fListLevel;         // current TGListTree level
   Bool_t               fTreeLock;          // true when we want to lock TGListTree
   Int_t                fViewMode;          // current IconBox view mode
   Int_t                fSortMode;          // current IconBox sort mode

   void  CreateBrowser(const char *name);
   void  ListTreeHighlight(TGListTreeItem *item);
   void  IconBoxAction(TObject *obj);
   void  Chdir(TGListTreeItem *item);
   void  DisplayDirectory();
   void  DisplayTotal(Int_t total, Int_t selected);
   void  SetDefaults();
   void  SetViewMode(Int_t new_mode, Bool_t force = kFALSE);
   void  SetSortMode(Int_t new_mode);

public:
   TRootBrowser(TBrowser *b, const char *title, UInt_t width, UInt_t height);
   TRootBrowser(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual ~TRootBrowser();

   void     Add(TObject *obj, const char *name = 0);
   void     AddToBox(TObject *obj, const char *name);
   void     AddToTree(TObject *obj, const char *name);
   void     BrowseObj(TObject *obj);
   void     ExecuteDefaultAction(TObject *obj);
   void     Iconify() { }
   void     RecursiveRemove(TObject *obj);
   void     Refresh(Bool_t force = kFALSE);
   void     ResizeBrowser() { }
   void     ShowToolBar(Bool_t show = kTRUE);
   void     ShowStatusBar(Bool_t show = kTRUE);
   void     Show() { MapRaised(); }

   // overridden from TGMainFrame
   void     CloseWindow();
   Bool_t   ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(TRootBrowser,0)  //ROOT native GUI version of browser
};

#endif

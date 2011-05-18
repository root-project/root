// @(#)root/gui:$Id$
// Author: Fons Rademakers   27/02/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TRootBrowserLite
#define ROOT_TRootBrowserLite

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TRootBrowserLite                                                     //
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
class TGComboBox;
class TGTextEdit;

class TRootBrowserLite : public TGMainFrame, public TBrowserImp {

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
   TGComboBox          *fDrawOption;         // drawing option entry
   TGLayoutHints       *fExpandLayout;       //
   Bool_t               fBrowseTextFile;     //
   TString              fTextFileName;

   TList               *fWidgets;
   TList               *fHistory;            // history of browsing
   TObject             *fHistoryCursor;      // current hsitory position
   const TGPicture     *fIconPic;            // icon picture

   void  CreateBrowser(const char *name);
   void  ListTreeHighlight(TGListTreeItem *item);
   void  DeleteListTreeItem(TGListTreeItem *item);
   void  HighlightListLevel();
   void  AddToHistory(TGListTreeItem *item);
   void  IconBoxAction(TObject *obj);
   void  Chdir(TGListTreeItem *item);
   void  DisplayDirectory();
   void  DisplayTotal(Int_t total, Int_t selected);
   void  SetViewMode(Int_t new_mode, Bool_t force = kFALSE);
   void  ToSystemDirectory(const char *dirname);
   void  UpdateDrawOption();
   void  Search();
   void  BrowseTextFile(const char *file);
   void  HideTextEdit();
   void  ShowMacroButtons(Bool_t show = kTRUE);

   Bool_t  HistoryBackward();
   Bool_t  HistoryForward();
   void    ClearHistory();

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
   TGTextEdit          *fTextEdit;          // contents of browsed text file

public:
   TRootBrowserLite(TBrowser *b = 0, const char *title = "ROOT Browser", UInt_t width = 800, UInt_t height = 500);
   TRootBrowserLite(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height);
   virtual ~TRootBrowserLite();

   virtual void Add(TObject *obj, const char *name = 0, Int_t check = -1);
   virtual void AddToBox(TObject *obj, const char *name);
   virtual void AddToTree(TObject *obj, const char *name, Int_t check = -1);

   virtual void AddCheckBox(TObject *obj, Bool_t check = kFALSE);
   virtual void CheckObjectItem(TObject *obj, Bool_t check = kFALSE);
   virtual void RemoveCheckBox(TObject *obj);

   virtual void BrowseObj(TObject *obj);             //*SIGNAL*
   virtual void ExecuteDefaultAction(TObject *obj);  //*SIGNAL*
   virtual void DoubleClicked(TObject *obj);         //*SIGNAL*
   virtual void Checked(TObject *obj, Bool_t check); //*SIGNAL*
   virtual void CloseTabs() { }
   virtual void Iconify() { }
   virtual void RecursiveRemove(TObject *obj);
   virtual void Refresh(Bool_t force = kFALSE);
   virtual void ResizeBrowser() { }
   virtual void ShowToolBar(Bool_t show = kTRUE);
   virtual void ShowStatusBar(Bool_t show = kTRUE);
   virtual void Show() { MapRaised(); }
   virtual void SetDefaults(const char *iconStyle = 0, const char *sortBy = 0);
   virtual Bool_t HandleKey(Event_t *event);
   virtual void SetStatusText(const char *txt, Int_t col);

   TGListTree      *GetListTree()  const { return fLt; }
   TGFileContainer *GetIconBox()   const;
   TGStatusBar     *GetStatusBar() const { return fStatusBar; }
   TGMenuBar       *GetMenuBar()   const { return  fMenuBar; }
   TGToolBar       *GetToolBar()   const { return fToolBar; }
   void             SetDrawOption(Option_t *option="");
   Option_t        *GetDrawOption() const;
   void             SetSortMode(Int_t new_mode);
   TGMainFrame     *GetMainFrame() const { return (TGMainFrame *)this; }

   // overridden from TGMainFrame
   void     CloseWindow();
   Bool_t   ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   void     ReallyDelete();

   // auxilary (a la privae) methods
   void     ExecMacro();
   void     InterruptMacro();

   static TBrowserImp *NewBrowser(TBrowser *b = 0, const char *title = "ROOT Browser", UInt_t width = 800, UInt_t height = 500, Option_t *opt="");
   static TBrowserImp *NewBrowser(TBrowser *b, const char *title, Int_t x, Int_t y, UInt_t width, UInt_t height, Option_t *opt="");

   ClassDef(TRootBrowserLite,0)  //ROOT native GUI version of browser
};

#endif

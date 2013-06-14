// @(#)root/treeviewer:$Id$
//Author : Andrei Gheata   16/08/00

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeViewer
#define ROOT_TTreeViewer

////////////////////////////////////////////////////
//                                                //
// TTreeViewer - A GUI oriented tree viewer       //
//                                                //
////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TTreeViewer;
class TTVLVContainer;
class TTVLVEntry;
class TTVSession;
class TGSelectBox;
class TTree;
class TBranch;
class TContextMenu;
class TList;
class TGPicture;
class TTimer;
class TGLayoutHints;
class TGMenuBar;
class TGPopupMenu;
class TGToolBar;
class TGLabel;
class TGCheckButton;
class TGComboBox;
class TGTextButton;
class TGTextEntry;
class TGDoubleVSlider;
class TGPictureButton;
class TGStatusBar;
class TGCanvas;
class TGListTree;
class TGListTreeItem;
class TGListView;
class TGHProgressBar;
class TGButton;


class TTreeViewer : public TGMainFrame {

friend class TGClient;
friend class TGButton;

public:
   //---- item types used as user data
   enum EListItemType {
      kLTNoType            = 0,
      kLTPackType          = BIT(0),
      kLTTreeType          = BIT(1),
      kLTBranchType        = BIT(2),
      kLTLeafType          = BIT(3),
      kLTActionType        = BIT(4),
      kLTDragType          = BIT(5),
      kLTExpressionType    = BIT(6),
      kLTCutType           = BIT(7)
   };

private:
   TTree                *fTree;                 // selected tree
   TTVSession           *fSession;              // current tree-viewer session
   const char           *fFilename;             // name of the file containing the tree
   const char           *fSourceFile;           // name of the C++ source file - default treeviewer.C
   TString              fLastOption;            // last graphic option
   TTree                *fMappedTree;           // listed tree
   TBranch              *fMappedBranch;         // listed branch
   Int_t                fDimension;             // histogram dimension
   Bool_t               fVarDraw;               // true if an item is double-clicked
   Bool_t               fScanMode;              // flag activated when Scan Box is double-clicked
   TContextMenu         *fContextMenu;          // context menu for tree viewer
   TGSelectBox          *fDialogBox;            // expression editor
   TList                *fTreeList;             // list of mapped trees
   Int_t                fTreeIndex;             // index of current tree in list
   const TGPicture      *fPicX, *fPicY, *fPicZ; // pictures for X, Y and Z expressions
   const TGPicture      *fPicDraw, *fPicStop;   // pictures for Draw/Stop buttons
   const TGPicture      *fPicRefr;              // pictures for Refresh buttons //ia
   Cursor_t             fDefaultCursor;         // default cursor
   Cursor_t             fWatchCursor;           // watch cursor
   TTimer               *fTimer;                // tree viewer timer
   Bool_t               fCounting;              // true if timer is counting
   Bool_t               fStopMapping;           // true if branch don't need remapping
   Bool_t               fEnableCut;             // true if cuts are enabled
   Int_t                fNexpressions;          // number of expression widgets
// menu bar, menu bar entries and layouts
   TGLayoutHints        *fMenuBarLayout;
   TGLayoutHints        *fMenuBarItemLayout;
   TGLayoutHints        *fMenuBarHelpLayout;
   TGMenuBar            *fMenuBar;
   TGPopupMenu          *fFileMenu;
   TGPopupMenu          *fEditMenu;
   TGPopupMenu          *fRunMenu;
   TGPopupMenu          *fOptionsMenu;
   TGPopupMenu          *fOptionsGen;
   TGPopupMenu          *fOptions1D;
   TGPopupMenu          *fOptions2D;
   TGPopupMenu          *fHelpMenu;
// toolbar and hints
   TGToolBar            *fToolBar;
   TGLayoutHints        *fBarLayout;
// widgets on the toolbar
   TGLabel              *fBarLbl1;      // label of command text entry
   TGLabel              *fBarLbl2;      // label of option text entry
   TGLabel              *fBarLbl3;      // label of histogram name text entry
   TGCheckButton        *fBarH;         // checked for drawing current histogram with different graphic option
   TGCheckButton        *fBarScan;      // checked for tree scan
   TGCheckButton        *fBarRec;       // command recording toggle
   TGTextEntry          *fBarCommand;   // user command entry
   TGTextEntry          *fBarOption;    // histogram drawing option entry
   TGTextEntry          *fBarHist;      // histogram name entry
// frames
   TGHorizontalFrame    *fHf;           // main horizontal frame
   TGDoubleVSlider      *fSlider;       // vertical slider to select processed tree entries;
   TGVerticalFrame      *fV1;           // list tree mother
   TGVerticalFrame      *fV2;           // list view mother
   TGCompositeFrame     *fTreeHdr;      // header for list tree
   TGCompositeFrame     *fListHdr;      // header for list view
   TGLabel              *fLbl1;         // label for list tree
   TGLabel              *fLbl2;         // label for list view
   TGHorizontalFrame    *fBFrame;       // button frame
   TGHorizontalFrame    *fHpb;          // progress bar frame
   TGHProgressBar       *fProgressBar;  // progress bar
   TGLabel              *fBLbl4;        // label for input list entry
   TGLabel              *fBLbl5;        // label for output list entry
   TGTextEntry          *fBarListIn;    // tree input event list name entry
   TGTextEntry          *fBarListOut;   // tree output event list name entry
   TGPictureButton      *fDRAW;         // DRAW button
   TGTextButton         *fSPIDER;       // SPIDER button
   TGPictureButton      *fSTOP;         // interrupt current command (not yet)
   TGPictureButton      *fREFR;         // REFRESH button  //ia
   TGStatusBar          *fStatusBar;    // status bar
   TGComboBox           *fCombo;        // combo box with session records
   TGPictureButton      *fBGFirst;
   TGPictureButton      *fBGPrevious;
   TGPictureButton      *fBGRecord;
   TGPictureButton      *fBGNext;
   TGPictureButton      *fBGLast;
   TGTextButton         *fReset;        // clear expression's entries
// ListTree
   TGCanvas             *fTreeView;     // ListTree canvas container
   TGListTree           *fLt;           // ListTree with file and tree items
// ListView
   TGListView           *fListView;     // ListView with branches and leaves
   TTVLVContainer       *fLVContainer;  // container for listview

   TList                *fWidgets;      // list of widgets to be deleted

private:
// private methods
   void          BuildInterface();
   const char   *Cut();
   Int_t         Dimension();
   const char   *EmptyBrackets(const char* name);
   const char   *Ex();
   const char   *Ey();
   const char   *Ez();
   const char   *En(Int_t n);
   void          MapBranch(TBranch *branch, const char *prefix="", TGListTreeItem *parent = 0, Bool_t listIt = kTRUE);
   void          MapOptions(Long_t parm1);
   void          MapTree(TTree *tree, TGListTreeItem *parent = 0, Bool_t listIt = kTRUE);
   void          SetFile();
   const char   *ScanList();
   void          SetParentTree(TGListTreeItem *item);
   void          DoError(int level, const char *location, const char *fmt, va_list va) const;

public:
   TTreeViewer(const char* treeName = 0);
   TTreeViewer(const TTree *tree);
   virtual       ~TTreeViewer();
// public methods
   void          AppendTree(TTree *tree);
   void          ActivateButtons(Bool_t first, Bool_t previous,
                                 Bool_t next , Bool_t last);
   virtual void  CloseWindow();
   virtual void  Delete(Option_t *) { }                          // *MENU*
   void          DoRefresh();
   void          EditExpression();
   void          Empty();
   void          EmptyAll();                                     // *MENU*
   void          ExecuteCommand(const char* command, Bool_t fast = kFALSE); // *MENU*
   void          ExecuteDraw();
   void          ExecuteSpider();
   TTVLVEntry   *ExpressionItem(Int_t index);
   TList        *ExpressionList();
   const char   *GetGrOpt();
   TTree        *GetTree() {return fTree;}
   Bool_t        HandleTimer(TTimer *timer);
   Bool_t        IsCutEnabled() {return fEnableCut;}
   Bool_t        IsScanRedirected();
   Int_t         MakeSelector(const char* selector = 0);         // *MENU*
   void          Message(const char* msg);
   void          NewExpression();                                // *MENU*
   void          PrintEntries();
   Long64_t      Process(const char* filename, Option_t *option="", Long64_t nentries=1000000000, Long64_t firstentry=0); // *MENU*
   Bool_t        ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   void          RemoveItem();
   void          RemoveLastRecord();                             // *MENU*
   void          SaveSource(const char* filename="", Option_t *option="");            // *MENU*
   void          SetHistogramTitle(const char *title);
   void          SetCutMode(Bool_t enabled = kTRUE) {fEnableCut = enabled;}
   void          SetCurrentRecord(Long64_t entry);
   void          SetGrOpt(const char *option);
   void          SetNexpressions(Int_t expr);
   void          SetRecordName(const char *name);                // *MENU*
   void          SetScanFileName(const char *name="");           // *MENU*
   void          SetScanMode(Bool_t mode=kTRUE) {fScanMode = mode;}
   void          SetScanRedirect(Bool_t mode);
   void          SetSession(TTVSession *session);
   void          SetUserCode(const char *code, Bool_t autoexec=kTRUE); // *MENU*
   void          SetTree(TTree* tree);
   void          SetTreeName(const char* treeName);              // *MENU*
   Bool_t        SwitchTree(Int_t index);
   void          UpdateCombo();
   void          UpdateRecord(const char *name="new name");      // *MENU*

   ClassDef(TTreeViewer,0)  // A GUI oriented tree viewer
};

#endif

//Author : Andrei Gheata   16/08/00
#ifndef ROOT_TTreeView
#define ROOT_TTreeView

//////////////////////////////////////////////////
//                                              //
// TTreeView - A GUI oriented tree viewer       //
//                                              //
//////////////////////////////////////////////////

//#include <stdlib.h>

#include <TSystem.h>
#include <TApplication.h>
#include <TVirtualX.h>
#include <TGClient.h>
#include <TKey.h>
#include <TFile.h>
#include <TGMenu.h>
#include <TGFrame.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TTree.h>
#include <TObjArray.h>
#include <TGButton.h>
#include <TGTextEntry.h>
#include <TGComboBox.h>
#include <TGLabel.h>
#include <TGListView.h>
#include <TGListTree.h>
#include <TGMimeTypes.h>
#include <TGSplitter.h>
#include <TGDoubleSlider.h>
#include <TGToolBar.h>
#include <TGStatusBar.h>
#include <Getline.h>
#include <TTimer.h>
#include <TG3DLine.h>
#include "HelpTextTV.h"

#include "TGTreeLVC.h"

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


class TGTreeLVC;
class TGSelectBox;

class TTreeView:public TGMainFrame
{
   friend class TGClient;

private:
// data members
   TTree                *fTree;                 // selected tree
   TTree                *fMappedTree;           // listed tree
   TBranch              *fMappedBranch;         // listed branch
   Int_t                fDimension;             // histogram dimension
   Bool_t               fVarDraw;               // true if an item is double-clicked
   TContextMenu         *fContextMenu;          // context menu for tree viewer
   TGSelectBox          *fDialogBox;            // expression editor
   TList                *fTreeList;             // list of mapped trees
   Int_t                fTreeIndex;             // index of current tree in list
   const TGPicture      *fPicX, *fPicY, *fPicZ; // pictures for X, Y and Z expressions
   const TGPicture      *fPicDraw, *fPicStop;   // pictures for Draw/Stop buttons
   Cursor_t             fDefaultCursor;         // default cursor
   Cursor_t             fWatchCursor;           // watch cursor
   TTimer               *fTimer;                // tree viewer timer
   Bool_t               fCounting;              // true if timer is counting
   Bool_t               fStopMapping;           // true if branch don't need remapping
   Bool_t               fEnableCut;             // true if cuts are enabled
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
   TGCheckButton	*fBarH;         // checked for drawing current histogram with different graphic option
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
   TGLabel              *fBLbl4;        // label for input list entry
   TGLabel              *fBLbl5;        // label for output list entry
   TGTextEntry          *fBarListIn;    // tree input event list name entry
   TGTextEntry          *fBarListOut;   // tree output event list name entry
   TGPictureButton      *fbDRAW;        // DRAW button
   TGPictureButton      *fbSTOP;        // interrupt current command (not yet)
   TGStatusBar          *fStatusBar;    // status bar
   TGTextButton         *fReset;        // clear expression's entries 
// ListTree
   TGCanvas             *fTreeView;     // ListTree canvas container
   TGListTree           *fLt;           // ListTree with file and tree items
// ListView
   TGListView           *fListView;     // ListView with branches and leaves
   TGTreeLVC            *fLVContainer;  // container for listview
	
   TList                *fWidgets;      // list of widgets to be deleted
private:
// private methods
   void         BuildInterface();
   const char*  Cut();	
   Int_t        Dimension();	
   const char*  Ex();
   const char*  Ey();
   const char*  Ez();
   void         MapBranch(TBranch *branch, TGListTreeItem *parent = 0, Bool_t listIt = kTRUE);
   void         MapOptions(Long_t parm1);
   void         MapTree(TTree *tree, TGListTreeItem *parent = 0, Bool_t listIt = kTRUE);
   void         Message(const char* msg);
   const char*  ScanList();
   void         SetParentTree(TGListTreeItem *item);
   void         Warning(const char* msg);
public:
   TTreeView(const char* treeName = 0);
   virtual      ~TTreeView();
// public methods
   virtual void CloseWindow();
   virtual void	Delete(Option_t *option = "") {}                // *MENU*
   void         EditExpression();                               // *MENU*
   void         Empty();                                        // *MENU*
   void	        EmptyAll();                                     // *MENU*
   void         ExecuteCommand(const char* command, Bool_t fast = kFALSE); // *MENU*
   void         ExecuteDraw();
   TTree*       GetTree() {return fTree;};
   Bool_t       HandleTimer(TTimer *timer);
   Int_t        MakeSelector(const char* selector = 0);         // *MENU*	
   void         PrintEntries();
   Int_t        Process(const char* filename, Option_t *option="", Int_t nentries=1000000000, Int_t firstentry=0); // *MENU*
   Bool_t       ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);
   void         RemoveItem();					// *MENU*
   void         SetTree(TTree* tree); 
   void         SetTreeName(const char* treeName);              // *MENU*
   Bool_t       SwitchTree(Int_t index);

  ClassDef(TTreeView,0)         // A GUI oriented tree viewer
};

#endif

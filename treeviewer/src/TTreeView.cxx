#include "TTreeView.h"
#include <iostream.h>
#include <TROOT.h>
#include <TGX11.h>
#include <TGMsgBox.h>
#include <TTreePlayer.h>
#include <TContextMenu.h>
#include <TInterpreter.h>
#include <TLeaf.h>
#include <TRootHelpDialog.h>


// drawing options
static const char* optgen[16] =
{	
   "","AXIS","HIST","SAME","CYL","POL","SPH","PSR","LEGO","LEGO1","LEGO2",
   "SURF","SURF1","SURF2","SURF3","SURF4"
};
static const char* opt1D[12] =
{
   "","AH","B","C","E","E1","E2","E3","E4","L","P","*H"
};
static const char* opt2D[14] =
{
   "","ARR","BOX","COL","COL2","CONT","CONT0","CONT1","CONT2","CONT3",
   "FB","BB","SCAT","PROF"
};

// Menu command id's
enum ERootTreeViewCommands {
   kFileCanvas,
   kFileBrowse,
   kFileLoadLibrary = 3,
   kFileSaveSettings,
   kFileSaveMacro,
   kFilePrint,
   kFileClose,
   kFileQuit,
   
   kEditExpression,
   kEditCut,
   kEditMacro,
   kEditEvent,
   
   kRunCommand,
   kRunMacro,

   kOptionsGeneral = 20,   
   kOptions1D = 50,
   kOptions2D = 70,
   
   kHelpAbout = 100,
   kHelpStart,
   kHelpLayout,
   kHelpBrowse,
   kHelpDragging,
   kHelpEditing,
   kHelpCommands,
   kHelpContext,
   kHelpDrawing,
   kHelpMacros,
   
   kBarCommand,
   kBarOption,
   kBarCut,
   kAxis
};

// button Id's
enum EButtonIdentifiers {
   kDRAW,
   kRESET,
   kSTOP,
   kCLOSE,
   kSLIDER
};

ClassImp(TTreeView)

//______________________________________________________________________________//*-*
//*-*   TTreeView is GUI version of TTreeViewer, designed to handle ROOT trees and
//*-*   to take advantage of TTree class features in a graphical manner. 
//
//	It uses ROOT native GUI widgets and has capability to work with several trees 
//	in the same session. It provides the following functionality :
//  - browsing all root files in the working directory and mapping trees inside;
//  - once a tree is mapped, the user can browse branches and work with the
//  corresponding sub-branches if there is no need for the whole tree;
//  - fast drawing of branches by double-click;
//  - easy edit the expressions to be drawn on X, Y and Z axis and/or selection;
//  - dragging expressions to one axis and aliasing of expression names;
//  - handle input/output event lists;
//  - usage of predefined compatible drawing options;
//  - possibility of executing user commands and macros and echoing of the current
//  command;
//  - possibility of interrupting the current command or the event loop (not yet);
//  - possibility of selecting the tree entries to be processed (not yet);
//  - take advantage of TTree class features via context menu;
//
//   The layout has the following items :
//
//  - a menu bar with entries : File, Edit, Run, Options and Help;
//  - a toolbar in the upper part where you can issue user commands, change
//  the drawing option and the histogram name, two check buttons Hist and Rec
//  which toggles histogram drawing mode and command recording respectively;
//  - a button bar in the lower part with : buttons DRAW/STOP that issue histogram
//  drawing and stop the current command respectively, two text widgets where 
//  input and output event lists can be specified, a message box and a RESET
//  button on the right that clear edited expression content (see Editing...)
//  - a tree-type list on the main left panel where you can browse the root files
//  from the working directory and load the trees inside by double clicking.
//  When the first tree is loaded, a new item called "TreeList" will pop-up on
//  the list menu and will have the selected tree inside with all branches mapped
//  Mapped trees are provided with context menus, activated by right-clicking;
//  - a view-type list on the main right panel. The first column contain X, Y and
//  Z expression items, an optional cut and ten optional editable expressions.
//  The other items in this list are activated when a mapped item from the
//  "TreeList" is left-clicked (tree or branch) and will describe the conyent
//  of the tree (branch). Expressions and leaf-type items can be dragged or
//  deleted. A right click on the list-box or item activates a general context
//  menu.
//
//   Browsing root files from the working directory :
//
// Just double-click on the directory item on the left and you will see all
// root files from this directory. Do it once more on the files with a leading +
// and you will see the trees inside. If you want one or more of those to
// be loaded, double-click on them and they will be mapped in a new item called
// "TreeList".
//
//   Browsing trees :
//
// Left-clicking on trees from the TreeList will expand their content on the list
// from the right side. Double-clicking them will also open their content on the
// left, where you can click on branches to expand them on the right.
//
//   Dragging items :
//
//   Items that can be dragged from the list in the right : expressions and 
// leaves. Dragging an item and dropping to another will copy the content of first
// to the last (leaf->expression, expression->expression). Items far to the right
// side of the list can be easily dragged to the left (where expressions are
// placed) by dragging them to the left at least 10 pixels.
//
//   Editing expressions
//
//   All editable expressions from the right panel has two components : a
// true name (that will be used when TTree::Draw() commands are issued) and an
// alias (used for labeling axes - not yet). The visible name is the alias if
// there is one and the true name otherwise.
//   The expression editor can be activated by right clicking on an
// expression item via the command EditExpression from the context menu.
// An alternative is to use the Edit-Expression menu after the desired expression
// is selected. The editor will pop-up in the left part, but it can be moved.
// The editor usage is the following :
//  - you can write C expressions made of leaf names by hand or you can insert
//  any item from the right panel by clicking on it (recommandable);
//  - you should write the item alias by hand since it not ony make the expression

//   User commands can be issued directly from the textbox labeled "Command"
// from the upper-left toolbar by typing and pressing Enter at the end.
//   An other way is from the right panel context menu : ExecuteCommand.
// All commands can be interrupted at any time by pressing the STOP button
// from the bottom-left (not yet)
// You can toggle recording of the current command in the history file by
// checking the Rec button from the top-right
//
//   Context menus
//
//   You can activate context menus by right-clicking on items or inside the
// box from the right.
// Context menus for mapped items from the left tree-type list :
//  The items from the left that are provided with context menus are tree and
// branch items. You can directly activate the *MENU* marked methods of TTree
// from this menu.
// Context menu for the right panel :
//  A general context manu of class TTreeView is acivated if the user
// right-clicks the right panel. Commands are :
//  - EmptyAll        : empty the content of all expressions;
//  - Empty           : empty the content of the clicked expression;
//  - EditExpression  : pops-up the expression editor;
//  - ExecuteCommand  : execute a user command;
//  - MakeSelector    : equivalent of TTree::MakeSelector();
//  - Process         : equivalent of TTree::Process();
//  - RemoveExpression: removes clicked item from the list;
//
//   Starting the viewer
//
//   The quickest way to start the tree viewer is to start a ROOT session in 
// your working directory where you have the root files containing trees.
// You will need first to load the library for TTreeView and optionally other
// libraries for user defined classes (you can do this later in the session) :
//    root [0] gSystem->Load(\"TTreeView\");
//    root [1] new TTreeView;
// or, to load the tree Mytree from the file Myfile :
//    root [1] TFile file(\"Myfile\");
//    root [2] new TTreeView(\"Mytree\");
// This will work if uou have the path to the library TTreeView defined in your
// .rootrc file.
//
//Begin_Html
/*
<img src="treeview.gif">
*/
//End_Html
//
		
//______________________________________________________________________________
TTreeView::TTreeView(const char* treeName)
          :TGMainFrame(gClient->GetRoot(),10,10,kVerticalFrame)
{
//*-*-*-*-*-*-*-*-*-*-*-*TTreeView default constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                    ================================
   fTree = 0;
   BuildInterface();
   SetTreeName(treeName);
}

//______________________________________________________________________________
void TTreeView::SetTreeName(const char* treeName)
{
//*-*-*-*-*-*-*-*-*-*-*-*Allow geting the tree from the context menu*-*-*-*-*-*-*-*-*-*-*
//*-*                    ==========================================
   TTree *tree = (TTree *) gROOT->FindObject(treeName);
   if (!tree) return;
   if (fTreeList) {
      if (fTreeList->FindObject(treeName)) return;   
   }
   if (fTree != tree) {
      fTree = tree;
      // load the tree via the interpreter 
      char command[100];
      command[0] = 0;
      // define a global "tree" variable for the same tree
      sprintf(command, "TTree *tree = (TTree *) gROOT->FindObject(\"%s\");", treeName);
      ExecuteCommand(command);
   } 
   //--- add the list of trees
   if (!fTreeList) {
      fTreeList = new TList();
      ExecuteCommand("TList *list = new TList;"); 
   }      
   //--- add the tree to the list if it is noy already in
   fTreeList->Add(fTree);
   ExecuteCommand("list->Add(tree);");
   //--- map this tree
   TGListTreeItem *base = 0;
   TGListTreeItem *parent = fLt->FindChildByName(base, "TreeList");
   if (!parent) parent = fLt->AddItem(base, "TreeList", new ULong_t(kLTNoType));
   ULong_t *itemType = new ULong_t((fTreeIndex << 8) | kLTTreeType);
   fTreeIndex++;
   TGListTreeItem *lTreeItem = fLt->AddItem(parent, treeName, itemType,
               gClient->GetPicture("tree_t.xpm"), gClient->GetPicture("tree_t.xpm"));
   MapTree(fTree, lTreeItem, kFALSE);
   fLt->OpenItem(parent);
   fLt->HighlightItem(lTreeItem);
   fClient->NeedRedraw(fLt);
   
   //--- map slider and list view
   SwitchTree(fTreeIndex-1);
   fLVContainer->RemoveNonStatic();
   MapTree(fTree);
   fListView->Layout();   
}



//______________________________________________________________________________
void TTreeView::BuildInterface()
{
//*-*-*-*-*-*-*-*-*Create all viewer widgets*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================
   //--- timer
   fCounting = kFALSE;
   fEnableCut = kTRUE;
   fTimer = new TTimer(this, 50, kTRUE);
   //--- cursors
   fDefaultCursor = gVirtualX->CreateCursor(kPointer);
   fWatchCursor = gVirtualX->CreateCursor(kWatch);
   //--- colours   
   ULong_t color;
   gClient->GetColorByName("blue",color);
   //--- pictures for X, Y and Z expression items
   fPicX = gClient->GetPicture("x_pic.xpm");
   fPicY = gClient->GetPicture("y_pic.xpm");
   fPicZ = gClient->GetPicture("z_pic.xpm");

   //--- general context menu
   fContextMenu = new TContextMenu("TreeViewer context menu","");
   fMappedTree = 0;
   fMappedBranch = 0;
   fDialogBox = 0;
   fDimension = 0;
   fVarDraw = kFALSE;
   fStopMapping = kFALSE;
   //--- lists : trees and widgets to be removed
   fTreeList = 0;
   fTreeIndex = 0;
   fWidgets = new TList();
   //--- create menus --------------------------------------------------------
   //--- File menu
   fFileMenu = new TGPopupMenu(fClient->GetRoot());
   fFileMenu->AddEntry("&New canvas",      kFileCanvas);
   fFileMenu->AddEntry("&Browse",          kFileBrowse);
   fFileMenu->AddEntry("&Load Library...", kFileLoadLibrary);
   fFileMenu->AddEntry("&Save Settings",   kFileSaveSettings);
   fFileMenu->AddEntry("Save &Macro",      kFileSaveMacro);
   fFileMenu->AddSeparator();
   fFileMenu->AddEntry("&Print",           kFilePrint);
   fFileMenu->AddEntry("&Close",           kFileClose);
   fFileMenu->AddSeparator();
   fFileMenu->AddEntry("&Quit ROOT",       kFileQuit);   
   
   fFileMenu->DisableEntry(kFileBrowse);
//   fFileMenu->DisableEntry(kFileLoadLibrary);
   fFileMenu->DisableEntry(kFileSaveSettings);
   fFileMenu->DisableEntry(kFileSaveMacro);
   fFileMenu->DisableEntry(kFilePrint);
   //--- Edit menu
   fEditMenu = new TGPopupMenu(gClient->GetRoot());
   fEditMenu->AddEntry("&Expression...",   kEditExpression);
   fEditMenu->AddEntry("&Cut...",          kEditCut);
   fEditMenu->AddEntry("&Macro...",        kEditMacro);
   fEditMenu->AddEntry("E&Vent...",        kEditEvent);
   
   fEditMenu->DisableEntry(kEditMacro);
   fEditMenu->DisableEntry(kEditEvent);
   //---Run menu
   fRunMenu = new TGPopupMenu(gClient->GetRoot());
   fRunMenu->AddEntry("&Macro...",         kRunMacro);
   fRunMenu->DisableEntry(kRunMacro);
   //--- Options menu
   //--- General options
   fOptionsGen = new TGPopupMenu(gClient->GetRoot());
   fOptionsGen->AddEntry("Default",        kOptionsGeneral);
   fOptionsGen->AddSeparator();
   fOptionsGen->AddEntry("Axis only",      kOptionsGeneral+1);  // "AXIS"
   fOptionsGen->AddEntry("Contour only",   kOptionsGeneral+2);  // "HIST"
   fOptionsGen->AddEntry("Superimpose",    kOptionsGeneral+3);  //"SAME"	
   fOptionsGen->AddEntry("Cylindrical",    kOptionsGeneral+4);  //"CYL"
   fOptionsGen->AddEntry("Polar",          kOptionsGeneral+5);  //"POL"
   fOptionsGen->AddEntry("Spherical",      kOptionsGeneral+6);  //"SPH"
   fOptionsGen->AddEntry("PsRap/Phi",      kOptionsGeneral+7);  //"PSR"
   fOptionsGen->AddEntry("Lego HLR",       kOptionsGeneral+8);  //"LEGO"
   fOptionsGen->AddEntry("Lego HSR",       kOptionsGeneral+9);  //"LEGO1"
   fOptionsGen->AddEntry("Lego Color",     kOptionsGeneral+10); //"LEGO2"
   fOptionsGen->AddEntry("Surface HLR",    kOptionsGeneral+11); //"SURF"
   fOptionsGen->AddEntry("Surface HSR",    kOptionsGeneral+12); //"SURF1"
   fOptionsGen->AddEntry("Surface Col",    kOptionsGeneral+13); //"SURF2"
   fOptionsGen->AddEntry("Surf+Cont",      kOptionsGeneral+14); //"SURF3"
   fOptionsGen->AddEntry("Gouraud",        kOptionsGeneral+15); //"SURF4"
   fOptionsGen->Associate(this);
   //--- 1D options
   fOptions1D = new TGPopupMenu(gClient->GetRoot());
   fOptions1D->AddEntry("Default",         kOptions1D);
   fOptions1D->AddSeparator();
   fOptions1D->AddEntry("No labels/ticks", kOptions1D+1);       // "AH"
   fOptions1D->AddEntry("Bar chart",       kOptions1D+2);       // "B"
   fOptions1D->AddEntry("Smooth curve",    kOptions1D+3);       // "C"	
   fOptions1D->AddEntry("Errors",          kOptions1D+4);       // "E"
   fOptions1D->AddEntry("Errors 1",        kOptions1D+5);       // "E1"
   fOptions1D->AddEntry("Errors 2",        kOptions1D+6);       // "E2"
   fOptions1D->AddEntry("Errors 3",        kOptions1D+7);       // "E3"
   fOptions1D->AddEntry("Errors 4",        kOptions1D+8);       // "E4"
   fOptions1D->AddEntry("Line",            kOptions1D+9);       // "L"
   fOptions1D->AddEntry("Markers",         kOptions1D+10);      // "P"
   fOptions1D->AddEntry("Stars",           kOptions1D+11);      // "*H"
   fOptions1D->Associate(this);
   //--- 2D options
   fOptions2D = new TGPopupMenu(gClient->GetRoot());
   fOptions2D->AddEntry("Default",         kOptions2D);
   fOptions2D->AddSeparator();
   fOptions2D->AddEntry("Arrows",          kOptions2D+1);       // "ARR"
   fOptions2D->AddEntry("Box/Surf",        kOptions2D+2);       // "BOX"
   fOptions2D->AddEntry("Box/Color",       kOptions2D+3);       // "COL"
   fOptions2D->AddEntry("Box/ColMap",      kOptions2D+4);       // "COLZ"
   fOptions2D->AddEntry("Contour",         kOptions2D+5);       // "CONT"	
   fOptions2D->AddEntry("Contour 0",       kOptions2D+6);       // "CONT0"
   fOptions2D->AddEntry("Contour 1",       kOptions2D+7);       // "CONT1"
   fOptions2D->AddEntry("Contour 2",       kOptions2D+8);       // "CONT2"
   fOptions2D->AddEntry("Contour 3",       kOptions2D+9);       // "CONT3"
   fOptions2D->AddEntry("No front-box",    kOptions2D+10);      // "FB"
   fOptions2D->AddEntry("No back-box",     kOptions2D+11);      // "BB"
   fOptions2D->AddEntry("Scatter",         kOptions2D+12);      // "SCAT"
   fOptions2D->AddEntry("Profile",         kOptions2D+13);      // "SCAT"
   fOptions2D->Associate(this);

   fOptionsMenu = new TGPopupMenu(gClient->GetRoot());
   fOptionsMenu->AddPopup("&General Options...", fOptionsGen);
   fOptionsMenu->AddPopup("&1D Options",         fOptions1D);
   fOptionsMenu->AddPopup("&2D Options",         fOptions2D);
   //--- Help menu
   fHelpMenu = new TGPopupMenu(gClient->GetRoot());
   fHelpMenu->AddEntry("&About...",              kHelpAbout);
   fHelpMenu->AddSeparator();
   fHelpMenu->AddEntry("&Starting...",           kHelpStart);
   fHelpMenu->AddEntry("&Layout...",             kHelpLayout);
   fHelpMenu->AddEntry("&Browsing...",           kHelpBrowse);
   fHelpMenu->AddEntry("&Dragging...",           kHelpDragging);
   fHelpMenu->AddEntry("&Editing expressions...",kHelpEditing);
   fHelpMenu->AddEntry("&User commands...",      kHelpCommands);
   fHelpMenu->AddEntry("&Context menus...",      kHelpContext);
   fHelpMenu->AddEntry("D&rawing...",            kHelpDrawing);
   fHelpMenu->AddEntry("&Macros...",             kHelpMacros);

   fFileMenu->Associate(this);
   fEditMenu->Associate(this);
   fRunMenu->Associate(this);
   fOptionsMenu->Associate(this);
   fHelpMenu->Associate(this);
   
   //--- menubar layout hints
   fMenuBarLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0,0,1,1);
   fMenuBarItemLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0);
   fMenuBarHelpLayout = new TGLayoutHints(kLHintsTop | kLHintsRight);
   //--- create menubar and add popup menus
   fMenuBar = new TGMenuBar(this, 1, 1, kHorizontalFrame);
   
   fMenuBar->AddPopup("&File", fFileMenu, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Edit", fEditMenu, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Run",  fRunMenu,  fMenuBarItemLayout);
   fMenuBar->AddPopup("&Options", fOptionsMenu,	fMenuBarItemLayout);
   fMenuBar->AddPopup("&Help", fHelpMenu, fMenuBarHelpLayout);
   
   AddFrame(fMenuBar, fMenuBarLayout);
   //--- toolbar ----------------------------------------------------------------
   fToolBar = new TGToolBar(this, 10, 10, kHorizontalFrame);
   fBarLayout = new TGLayoutHints(kLHintsTop | kLHintsExpandX);

   TGLayoutHints *lo;
   lo = new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 4,4,0,0);
   fWidgets->Add(lo);
   //--- label for Command text entry
   fBarLbl1 = new TGLabel(fToolBar,"Command");
   fToolBar->AddFrame(fBarLbl1,lo);
   //--- command text entry
   fBarCommand = new TGTextEntry(fToolBar, new TGTextBuffer(100),kBarCommand);
   fBarCommand->SetWidth(120);
   fBarCommand->Associate(this);
   fToolBar->AddFrame(fBarCommand, lo);
   //--- first vertical separator
   TGVertical3DLine *vSeparator = new TGVertical3DLine(fToolBar);
   lo = new TGLayoutHints(kLHintsLeft | kLHintsExpandY, 4,4,0,0);
   fWidgets->Add(lo);
   fWidgets->Add(vSeparator);
   fToolBar->AddFrame(vSeparator, lo);

   lo = new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 4,4,0,0);
   fWidgets->Add(lo);
   //--- label for Option text entry
   fBarLbl2 = new TGLabel(fToolBar,"Option");
   fToolBar->AddFrame(fBarLbl2, lo);
   //--- drawing option text entry
   fBarOption = new TGTextEntry(fToolBar, new TGTextBuffer(100),kBarOption);
   fBarOption->SetWidth(100);
   fBarOption->Associate(this);
   fToolBar->AddFrame(fBarOption, lo);
   //--- second vertical separator
   vSeparator = new TGVertical3DLine(fToolBar);
   lo = new TGLayoutHints(kLHintsLeft | kLHintsExpandY, 4,4,0,0);
   fWidgets->Add(lo);
   fWidgets->Add(vSeparator);
   fToolBar->AddFrame(vSeparator, lo);

   lo = new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 4,4,0,0);
   fWidgets->Add(lo);
   //--- label for Histogram text entry
   fBarLbl3 = new TGLabel(fToolBar,"Histogram");
   fToolBar->AddFrame(fBarLbl3, lo);
   //--- histogram name text entry
   fBarHist = new TGTextEntry(fToolBar, new TGTextBuffer(100));
   fBarHist->SetWidth(50);
   fBarHist->SetText("htemp");
   fToolBar->AddFrame(fBarHist, lo);
   //--- Hist check button
   fBarH = new TGCheckButton(fToolBar, "Hist");
   fBarH->SetToolTipText("Checked : redraw only current histogram");
   fBarH->SetState(kButtonUp);
   fToolBar->AddFrame(fBarH, lo);
   //--- Scan check button
   fBarScan = new TGCheckButton(fToolBar, "Scan");
   fBarScan->SetState(kButtonUp);
   fBarScan->SetToolTipText("Check to scan branches colected in the scan box ");
   fToolBar->AddFrame(fBarScan, lo);
   //--- Rec check button
   fBarRec = new TGCheckButton(fToolBar, "Rec");
   fBarRec->SetState(kButtonDown);
   fBarRec->SetToolTipText("Check to record commands in history file and be verbose");
   fToolBar->AddFrame(fBarRec, lo);
   //--- 1'st horizontal tool bar separator ----------------------------------------
   TGHorizontal3DLine *toolBarSep = new TGHorizontal3DLine(this);
   fWidgets->Add(toolBarSep);
   AddFrame(toolBarSep, fBarLayout);
   AddFrame(fToolBar, fBarLayout);
   //--- 2'nd horizontal tool bar separator ----------------------------------------
   toolBarSep = new TGHorizontal3DLine(this);
   fWidgets->Add(toolBarSep);
   AddFrame(toolBarSep, fBarLayout);   
   //--- Horizontal mother frame
   fHf = new TGHorizontalFrame(this, 10, 10);
   //--- Vertical frames
   fSlider = new TGDoubleVSlider(fHf, 10, kDoubleScaleNo, kSLIDER);
//   fSlider->SetBackgroundColor(color);
   fSlider->Associate(this);
   fV1 = new TGVerticalFrame(fHf, 10, 10, kFixedWidth);
   fV2 = new TGVerticalFrame(fHf, 10, 10);
   //--- Headers and labels
   fTreeHdr = new TGCompositeFrame(fV1, 10, 10, kSunkenFrame);
   fListHdr = new TGCompositeFrame(fV2, 10, 10, kSunkenFrame);
   fLbl1 = new TGLabel(fTreeHdr, "Current folder");
   fLbl2 = new TGLabel(fListHdr, "Current tree :                 ");
   
   lo = new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 3, 0, 0, 0);
   fWidgets->Add(lo);
   fTreeHdr->AddFrame(fLbl1, lo);
   fListHdr->AddFrame(fLbl2, lo);

   lo = new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0, 0, 1, 2);
   fWidgets->Add(lo);
   fV1->AddFrame(fTreeHdr, lo);
   fV2->AddFrame(fListHdr, lo);
   
   fV1->Resize(fTreeHdr->GetDefaultWidth()+100, fV1->GetDefaultHeight());
   lo = new TGLayoutHints(kLHintsLeft | kLHintsExpandY);
   fWidgets->Add(lo);
   fHf->AddFrame(fSlider, lo);
   fHf->AddFrame(fV1, lo);
   //--- vertical splitter
   TGVSplitter *splitter = new TGVSplitter(fHf);
   splitter->SetFrame(fV1,kTRUE);
   lo = new TGLayoutHints(kLHintsLeft | kLHintsExpandY);
   fWidgets->Add(splitter);
   fWidgets->Add(lo);
   fHf->AddFrame(splitter,lo);
	
   lo = new TGLayoutHints(kLHintsRight | kLHintsExpandX | kLHintsExpandY);
   fWidgets->Add(lo);
   fHf->AddFrame(fV2,lo);
   //--- tree view canvas on the left -------------------------------------------
   fTreeView = new TGCanvas(fV1, 10, 10, kSunkenFrame | kDoubleBorder);
   //--- container frame
   fLt = new TGListTree(fTreeView->GetViewPort(), 10, 10, kHorizontalFrame,
                        fgWhitePixel);
   fLt->Associate(this);
   fTreeView->SetContainer(fLt);
	
   lo = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY);
   fWidgets->Add(lo);
   fV1->AddFrame(fTreeView, lo);
   //-- listview for the content of the tree/branch -----------------------------
   fListView = new TGListView(fV2,400,300);
   //--- container frame
   fLVContainer = new TGTreeLVC(fListView->GetViewPort(),400,300);
   fLVContainer->Associate(this);
   fLVContainer->SetListView(fListView);
   fLVContainer->SetBackgroundColor(fgWhitePixel);
   fListView->GetViewPort()->SetBackgroundColor(fgWhitePixel);
   fListView->SetContainer(fLVContainer);
   fListView->SetViewMode(kLVList);

   fV2->AddFrame(fListView,lo);
   AddFrame(fHf, lo);	
   //--- bottom button frame ----------------------------------------------------
   fBFrame = new TGHorizontalFrame(this,10,10);
   fPicDraw = gClient->GetPicture("draw_t.xpm");
//   fPicStop = gClient->GetPicture("mb_stop_s.xpm");
   fPicStop = gClient->GetPicture("stop_t.xpm");
   //--- DRAW button
   fbDRAW  = new TGPictureButton(fBFrame,fPicDraw,kDRAW);
   fbDRAW->SetToolTipText("Draw current selection");
   fbDRAW->Associate(this);
   //--- STOP button (breaks current operation)
   fbSTOP  = new TGPictureButton(fBFrame,fPicStop,kSTOP);
   fbSTOP->SetToolTipText("Abort current operation");
   fbSTOP->Associate(this);
   lo = new TGLayoutHints(kLHintsTop | kLHintsLeft, 2,2,2,2);
   fWidgets->Add(lo);

   fBFrame->AddFrame(fbDRAW, lo);
   fBFrame->AddFrame(fbSTOP, lo);
   //--- label for IList text entry
   fBLbl4 = new TGLabel(fBFrame,"IList");
   fBFrame->AddFrame(fBLbl4, lo);
   //--- IList text entry
   fBarListIn =  new TGTextEntry(fBFrame, new TGTextBuffer(100));
   fBarListIn->SetWidth(50);
   fBFrame->AddFrame(fBarListIn, lo);
   //--- label for OList text entry
   fBLbl5 = new TGLabel(fBFrame,"OList");
   fBFrame->AddFrame(fBLbl5, lo);
   //--- OList text entry
   fBarListOut =  new TGTextEntry(fBFrame, new TGTextBuffer(100));
   fBarListOut->SetWidth(50);
   fBFrame->AddFrame(fBarListOut, lo);
   //--- Status bar
   fStatusBar = new TGStatusBar(fBFrame, 10, 10);
   fStatusBar->SetWidth(200);
   lo = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 2,2,2,2);
   fWidgets->Add(lo);
   fBFrame->AddFrame(fStatusBar, lo);
   //--- RESET button
   TGTextButton* fReset = new TGTextButton(fBFrame,"RESET",kRESET);
   fReset->SetToolTipText("Reset variable's fields and drawing options");
   fReset->Associate(this);
   lo = new TGLayoutHints(kLHintsTop | kLHintsRight, 2,2,2,2);
   fWidgets->Add(lo);

   fBFrame->AddFrame(fReset,lo);
   lo = new TGLayoutHints(kLHintsExpandX,2,2,2,0);
   fWidgets->Add(lo);
   AddFrame(fBFrame,lo);
	
   // map the window
   SetWindowName("TreeViewer");
   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();
   
   // put default items in the listview on the right
   const TGPicture *pic, *spic;
   
   fLVContainer->RemoveAll();
   TGLVTreeEntry* entry;
   Char_t symbol;
   entry = new TGLVTreeEntry(fLVContainer,fPicX,fPicX,new TGString(),0,kLVSmallIcons);
   symbol = 'X';
   entry->SetUserData(new ULong_t((symbol << 8) | kLTExpressionType | kLTTreeType));
   //--- X item
   fLVContainer->AddThisItem(entry);
   entry->MapWindow();

   entry = new TGLVTreeEntry(fLVContainer,fPicY,fPicY,new TGString(),0,kLVSmallIcons);
   symbol = 'Y';
   entry->SetUserData(new ULong_t((symbol << 8) | kLTExpressionType | kLTTreeType));
   //--- Y item
   fLVContainer->AddThisItem(entry);
   entry->MapWindow();

   entry = new TGLVTreeEntry(fLVContainer,fPicZ,fPicZ,new TGString(),0,kLVSmallIcons);
   symbol = 'Z';
   entry->SetUserData(new ULong_t((symbol << 8) | kLTExpressionType | kLTTreeType));
   //--- Z item
   fLVContainer->AddThisItem(entry);
   entry->MapWindow();

   pic = gClient->GetPicture("cut_t.xpm");
   spic = gClient->GetPicture("cut_t.xpm");
   entry = new TGLVTreeEntry(fLVContainer,pic,spic,new TGString(),0,kLVSmallIcons);
   entry->SetUserData(new ULong_t(kLTExpressionType | kLTCutType));
   //--- Cut item (scissors icon)
   fLVContainer->AddThisItem(entry);
   entry->MapWindow();

   pic = gClient->GetPicture("pack_t.xpm");
   spic = gClient->GetPicture("pack-empty_t.xpm");
   entry = new TGLVTreeEntry(fLVContainer,pic,spic,new TGString("Scan box"),0,kLVSmallIcons);
   entry->SetUserData(new ULong_t(kLTExpressionType | kLTPackType));
   //--- Cut item (scissors icon)
   fLVContainer->AddThisItem(entry);
   entry->MapWindow();
   entry->SetTrueName("");
   
   pic = gClient->GetPicture("expression_t.xpm");
   spic = gClient->GetPicture("expression_t.xpm");   
   //--- 10 expression items
   for (Int_t i=0; i<10; i++) {
      entry = new TGLVTreeEntry(fLVContainer,pic,spic,new TGString(),0,kLVSmallIcons);
      entry->SetUserData(new ULong_t(kLTExpressionType | kLTDragType));
      fLVContainer->AddThisItem(entry);
      entry->MapWindow();   
   }

   fListView->Layout();
   // map the tree if it was supplied in the constructor

   SetMWMHints(kMWMDecorAll, kMWMFuncAll, kMWMInputModeless);
   if (!fTree) {
      fSlider->SetRange(0,1000000);
      fSlider->SetPosition(0,1000000);
   } else {
      fSlider->SetRange(0,fTree->GetEntries()-1);
      fSlider->SetPosition(0,fTree->GetEntries()-1);
   }
   PrintEntries();
}

//______________________________________________________________________________
TTreeView::~TTreeView() 
{
//*-*-*-*-*-*-*-*-*-*-*TTreeView default destructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============================

   gClient->FreePicture(fPicX);
   gClient->FreePicture(fPicY);   
   gClient->FreePicture(fPicZ);   
   gClient->FreePicture(fPicDraw);   
   gClient->FreePicture(fPicStop);   

   fDialogBox = TGSelectBox::GetInstance();
   if (fDialogBox) delete fDialogBox;

   delete fContextMenu;

   delete fBarLbl1;
   delete fBarLbl2;
   delete fBarLbl3;
   delete fBLbl4;
   delete fBLbl5;
   delete fBarCommand;
   delete fBarOption;
   delete fBarHist;
   delete fBarListIn;
   delete fBarListOut;

   delete fBarH;
   delete fBarScan;
   delete fBarRec;

   delete fToolBar;

   delete fSlider;
   delete fV1;
   delete fV2;
   delete fLbl1;
   delete fLbl2;
   delete fHf;
   delete fTreeHdr;
   delete fListHdr;
   delete fLt;
   delete fTreeView;
   delete fLVContainer; 
   delete fListView;

   delete fbDRAW;
   delete fbSTOP;
   delete fReset;  
   delete fBFrame;

   delete fMenuBar;
   delete fFileMenu;
   delete fEditMenu;

   delete fOptionsGen;
   delete fOptions1D;
   delete fOptions2D;
   delete fOptionsMenu;
   delete fHelpMenu;
   delete fMenuBarLayout;
   delete fMenuBarItemLayout;
   delete fMenuBarHelpLayout;
   delete fBarLayout;
   
   fWidgets->Delete();
   delete fWidgets;
   if (fTreeList) {
      fTreeList->Delete();
      delete fTreeList;
   }
   delete fTimer;
}
//______________________________________________________________________________
const char* TTreeView::Cut()
{
   return fLVContainer->Cut();
}
//______________________________________________________________________________
const char* TTreeView::ScanList()
{
   return fLVContainer->ScanList();
}
//______________________________________________________________________________
void TTreeView::EmptyAll()
{
//*-*-*-*-*-*-*-*-*Clear the content of all items in the list view*-*-*-*-*-*-*
//*-*              ================================================
   fLVContainer->EmptyAll();
}
//______________________________________________________________________________
void TTreeView::Empty()
//*-*-*-*-*-*-*-*-*Empty the content of the selected expression*-*-*-*-*-*-*-*-*-*-*
//*-*              ============================================
{
   void *p = 0;
   TGLVTreeEntry *item = 0;
   if ((item = (TGLVTreeEntry *) fLVContainer->GetNextSelected(&p)) == 0) {
      Warning("No item selected.");
      return;
   }
   ULong_t *itemType = (ULong_t *) item->GetUserData();
   if (!(*itemType & kLTExpressionType)) {
      Warning("Not expression type.");
      return;
   }
   if (*itemType & kLTPackType) {
      item->SetSmallPic(fClient->GetPicture("pack-empty_t.xpm"));
      item->SetTrueName("");
      return;
   }
   item->Empty();
}

//______________________________________________________________________________
Int_t TTreeView::Dimension()
{
//*-*-*-*-*-*-*-*-*Compute dimension of the histogram*-*-*-*-*-*-*-*-*-*-*
//*-*              ==================================
   fDimension = 0;
   if (strlen(Ex())) fDimension++;
   if (strlen(Ey())) fDimension++;
   if (strlen(Ez())) fDimension++;
   return fDimension;
}
//______________________________________________________________________________
void TTreeView::ExecuteDraw()
{
//*-*-*-*-*-*-*-*-*Called when the DRAW button is executed*-*-*-*-*-*-*-*-*-*-*
//*-*              ========================================
   char varexp[100];
   varexp[0] = 0;
   char command[512];
   command[0] = 0;
   // fill in expressions
   if (fVarDraw) {
      void *p = 0;
      TGLVTreeEntry *item;
      if (!(item = (TGLVTreeEntry *) fLVContainer->GetNextSelected(&p))) return;
      sprintf(varexp, item->GetTrueName());
   } else {
      if (strlen(Ez())) sprintf(varexp, Ez());
      if (strlen(Ez()) && (strlen(Ex()) || strlen(Ey()))) strcat(varexp, ":");
      if (strlen(Ey())) strcat(varexp, Ey());
      if (strlen(Ey()) && strlen(Ex())) strcat(varexp, ":");
      if (strlen(Ex())) strcat(varexp, Ex());
   }
   // find ListIn
   fTree->SetEventList(0);
   TEventList *elist = 0;
   if (strlen(fBarListIn->GetText())) {
      elist = (TEventList *) gROOT->FindObject(fBarListIn->GetText());
      if (elist) fTree->SetEventList(elist);
   }
   // find ListOut
   if (strlen(fBarListOut->GetText())) sprintf(varexp, ">>%s", fBarListOut->GetText());
   // find histogram name
   if (strcmp("htemp", fBarHist->GetText())) {
      strcat(varexp, ">>");
      strcat(varexp, fBarHist->GetText());
   }
   // find canvas/pad where to draw
   TPad *pad = (TPad*)gROOT->GetSelectedPad();
   if (pad) {
      pad->cd();
   } else {
      new TCanvas("c1");
   }
   // find graphics option
   const char* gopt = fBarOption->GetText();   
   // just in case a previous interrupt was posted
   gROOT->SetInterrupt(kFALSE);
   // check if only histogram has to be updated
   if (fBarH->GetState() == kButtonDown) {
      // reset 'Hist' mode
      fBarH->SetState(kButtonUp);
      TH1 *hist = fTree->GetHistogram();
      if (hist) {
         hist->Draw(gopt);
         gPad->Update();
         return;
      }
   }
   // check if cut is enabled
   const char *cut = "";
   if (fEnableCut) cut = Cut();
   
   // get entries to be processed   
   Int_t nentries = (Int_t)(fSlider->GetMaxPosition() - 
                            fSlider->GetMinPosition() + 1);
   Int_t firstentry =(Int_t) fSlider->GetMinPosition();

   // check if Scan is checked and if there is something in the box
   if (strlen(ScanList())) sprintf(varexp, ScanList());
   if (fBarScan->GetState() == kButtonDown) {
      sprintf(command, "tree->Scan(\"%s\",\"%s\",\"%s\", %i, %i);", 
              varexp, cut, gopt, nentries, firstentry);
      ExecuteCommand(command, kTRUE);
      return;
   }
   // send draw command
   sprintf(command, "tree->Draw(\"%s\",\"%s\",\"%s\", %i, %i);", 
           varexp, cut, gopt, nentries, firstentry);
   ExecuteCommand(command, kTRUE);
   gPad->Update();
}
//______________________________________________________________________________
const char* TTreeView::Ex()
{
//*-*-*-*-*-*-*-*-*Get the expression to be drawn on X axis*-*-*-*-*-*-*-*-*-*-*
//*-*              ========================================
   return fLVContainer->Ex();
}
//______________________________________________________________________________
const char* TTreeView::Ey()
{
//*-*-*-*-*-*-*-*-*Get the expression to be drawn on Y axis*-*-*-*-*-*-*-*-*-*-*
//*-*              ========================================
   return fLVContainer->Ey();
}
//______________________________________________________________________________
const char* TTreeView::Ez()
{
//*-*-*-*-*-*-*-*-*Get the expression to be drawn on Z axis*-*-*-*-*-*-*-*-*-*-*
//*-*              ========================================
   return fLVContainer->Ez();
}
//______________________________________________________________________________
void TTreeView::EditExpression()
{
//*-*-*-*-*-*-*-*-*Start the expression editor*-*-*-*-*-*-*-*-*-*-*
//*-*              ===========================
   void *p = 0;
   // get the selected item
   TGLVTreeEntry *item = 0;
   if ((item = (TGLVTreeEntry *) fLVContainer->GetNextSelected(&p)) == 0) {
      Warning("No item selected.");
      return;
   }
   // check if it is an expression
   ULong_t *itemType = (ULong_t *) item->GetUserData();
   if (!(*itemType & kLTExpressionType)) {
      Warning("Not expression type.");
      return;
   }
   // check if the editor is already active 
   fDialogBox = TGSelectBox::GetInstance();
   if (!fDialogBox) {
      fDialogBox = new TGSelectBox(fClient->GetRoot(), this, fV1->GetWidth() - 10);
   }
   // copy current item data into editor boxes
   fDialogBox->SetEntry(item);
   fDialogBox->SetWindowName("Expression editor");
   // check if you are editing the cut expression
   if (*itemType & kLTCutType) {
      fDialogBox->SetLabel("Cut");
   } else {
      fDialogBox->SetLabel("Expression");
   } 
}
//______________________________________________________________________________
Int_t TTreeView::MakeSelector(const char* selector)
{
//*-*-*-*-*-*-*-*-*get use of TTree::MakeSelector() via the context menu*-*-*-*-*-*-*-*-*-*-*
//*-*              =====================================================
   if (!fTree) return 0;
   return fTree->MakeSelector(selector);
}
//______________________________________________________________________________
Int_t TTreeView::Process(const char* filename, Option_t *option, Int_t nentries, Int_t firstentry)
{
//*-*-*-*-*-*-*-*-*get use of TTree::Process() via the context menu*-*-*-*-*-*
//*-*              ================================================
   if (!fTree) return 0;
   return fTree->Process(filename, option, nentries, firstentry);
}
//______________________________________________________________________________
void TTreeView::RemoveItem()
{
//*-*-*-*-*-*-*-*-*Remove the selected item from the list*-*-*-*-*-*-*-*-*-*-*
//*-*              ======================================
   void *p = 0;
   TGLVTreeEntry *item = 0;
   // get the selected item
   if ((item = (TGLVTreeEntry *) fLVContainer->GetNextSelected(&p)) == 0) {
      Warning("No item selected.");
      return;
   }
   // check if it is removable
   ULong_t *itemType = (ULong_t *) item->GetUserData();
   if (!(*itemType & kLTDragType)) {
      Warning("Not removable type.");
      return;
   }
   fLVContainer->RemoveItem(item);
   fListView->Layout();
}
//______________________________________________________________________________
Bool_t TTreeView::HandleTimer(TTimer *timer)
{
// This function is called by the fTimer object 
   timer->Reset();
   fCounting = kTRUE;
   // functionality to be added 
   if (gPad) gPad->SetCursor(kWatch);
   cout << "time\n";
   return kFALSE;
}
//______________________________________________________________________________
Bool_t TTreeView::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
//*-*-*-*-*-*-*-*-*Handle menu and other commands generated*-*-*-*-*-*-*-*-*-*-*
//*-*              ========================================

   TRootHelpDialog *hd;

   switch (GET_MSG(msg)) {
      case kC_VSLIDER :
	 // handle slider messages
	 PrintEntries();
      break;
      case kC_TEXTENTRY:
         switch (GET_SUBMSG(msg)) {
         // handle enter posted by the Command text entry
            case kTE_ENTER:
               if ((Int_t)parm1 == kBarCommand) {
                  ExecuteCommand(fBarCommand->GetText());
               }
	       if ((Int_t)parm1 == kBarOption) {
	          fVarDraw = kFALSE;
                  fBarH->SetState(kButtonDown);
		  ExecuteDraw();
		  fBarH->SetState(kButtonUp);
	       }
               break;
            default:
               break;
         }
         break;
      case kC_LISTTREE:
         switch (GET_SUBMSG(msg)) {
         // handle mouse messages in the list-tree (left panel)
            case kCT_ITEMCLICK :
               if (parm1 == kButton1) {
                  TGListTreeItem *ltItem = 0;
                  // get item that sent this
                  if ((ltItem = fLt->GetSelected()) != 0) {
                  // get item type
                     ULong_t *itemType = (ULong_t *)ltItem->GetUserData();
                     if (*itemType & kLTTreeType) {
                     // already mapped tree item clicked 
                        Int_t index = (Int_t)(*itemType >> 8);
                        SwitchTree(index);
                        if (fTree != fMappedTree) {
			// switch also the global "tree" variable
                           fLVContainer->RemoveNonStatic();
                           // map it on the right panel			   
                           MapTree(fTree);
                           fListView->Layout();
                        }
                     }

                     if (*itemType & kLTBranchType) {
                     // branch item clicked
                        SetParentTree(ltItem);
                        if (!fTree) break; // really needed ?
                        TBranch *branch = fTree->GetBranch(ltItem->GetText());
                        if (!branch) break; 
                        // check if it is mapped on the right panel
                        if (branch != fMappedBranch) {
                           fLVContainer->RemoveNonStatic();
                           MapBranch(branch);
			   fStopMapping = kFALSE;
                           fListView->Layout();
                        }
                     }
		   
                     if (*itemType & kLTLeafType) {
                     // leaf item clicked
                        SetParentTree(ltItem);
                        if (!fTree) break;
                        // find parent branch
                        TBranch *branch = fTree->GetBranch(ltItem->GetParent()->GetText());
                        if (!branch) {
			   if (fTree != fMappedTree) {
			      fLVContainer->RemoveNonStatic();
			      MapTree(fTree);
			      fListView->Layout();
			   }
			} else {
                           // check if it is already mapped
                           if (branch != fMappedBranch) {
                              fLVContainer->RemoveNonStatic();
                              MapBranch(branch);
			      fStopMapping = kFALSE;
                              fListView->Layout();
                           }
			}   
                        // select corresponding leaf on the right panel
                        fLVContainer->SelectItem(ltItem->GetText());
                     }
                  }      
               }
	       
               if (parm1 == kButton3) {
               // get item that sent this
                  TGListTreeItem *ltItem = 0;
                  if ((ltItem = fLt->GetSelected()) != 0) {
                  // get item type
                     ULong_t *itemType = (ULong_t *)ltItem->GetUserData();
                     if (*itemType & kLTTreeType) {
                     // already mapped tree item clicked 
                        Int_t index = (Int_t)(*itemType >> 8);
                        SwitchTree(index);
                        if (fTree != fMappedTree) {
                           fLVContainer->RemoveNonStatic();			   
                           MapTree(fTree);
                           fListView->Layout();
                        }
                        // activate context menu for this tree
                        Int_t x = (Int_t)(parm2 &0xffff);
                        Int_t y = (Int_t)((parm2 >> 16) & 0xffff);
                        fContextMenu->Popup(x, y, fTree);				   
                     }
                     if (*itemType & kLTBranchType) {
                     // branch item clicked
                        SetParentTree(ltItem);
                        if (!fTree) break; // really needed ?
                        TBranch *branch = fTree->GetBranch(ltItem->GetText());
                        if (!branch) break; 
                        if (branch != fMappedBranch) {
                           fLVContainer->RemoveNonStatic();
                           MapBranch(branch);
			   fStopMapping = kFALSE;
                           fListView->Layout();
                        }
                        // activate context menu for this branch (no *MENU* methods ):)
                        Int_t x = (Int_t)(parm2 &0xffff);
                        Int_t y = (Int_t)((parm2 >> 16) & 0xffff);
                        fContextMenu->Popup(x, y, branch);				   
                     }
                  }
               }
               break;
	    case kCT_ITEMDBLCLICK :
	       fClient->NeedRedraw(fLt);
	       break;
            default:
               break;
         }
         break;
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)){
            case kCM_BUTTON:
               switch (parm1) {
               // handle button messages
                  case kRESET:
                     EmptyAll();
                     break;
                  case kDRAW:
                     fVarDraw = kFALSE;
//                     gVirtualX->SetCursor(GetId(), fWatchCursor);
                     ExecuteDraw();
                     break;
                  case kSTOP:
                     gROOT->SetInterrupt(kTRUE); // not working :(
                     if (fCounting) {
                        fTimer->TurnOff();
                     }
                     break;
                  case kCLOSE:
                     CloseWindow();
                     break;
                  default:
                     break;
               }
               break;
            case kCM_MENU:
            // hanlde menu messages
               // check if sent by Options menu
               if ((parm1>=kOptionsGeneral) && (parm1<kHelpAbout)) {
                  Dimension();
                  if ((fDimension==0) && (parm1>=kOptions1D)) {
                     Warning("Edit expressions first");
                     break;
                  }
                  if ((fDimension==1) && (parm1>=kOptions2D)) {
                     Warning("You have only one expression active");
                     break;
                  }
                  if ((fDimension==2) && (parm1>=kOptions1D) &&(parm1<kOptions2D)) {
                     Warning("1D drawing options not apply to 2D histograms");
                     break;
                  }
                  // make composed option
                  MapOptions(parm1);
                  break;
               }
               switch (parm1) {
	          case kFileCanvas:
		     gROOT->GetMakeDefCanvas()();
		     break;
                  case kFileBrowse:
                     break;
                  case kFileLoadLibrary:
		     fBarCommand->SetText("gSystem->Load(\"\");");
		     if (1) {
		        Event_t event;
			event.fType = kButtonPress;
			event.fCode = kButton1;
			
			fBarCommand->HandleButton(&event);
		     }
		     fBarCommand->SetCursorPosition(15);
                     break;
                  case kFileSaveSettings:
                     break;
                  case kFileSaveMacro:
                     break;
                  case kFilePrint:
                     break;
                  case kFileClose:
                     CloseWindow();
                     break;
                  case kFileQuit:
                     gApplication->Terminate(0);
                     break;
                  case kEditExpression:
                     EditExpression();
                     break;
                  case kEditCut:
                     EditExpression();
                     break;
                  case kEditMacro:
                     break;
                  case kEditEvent:
                     break;
                  case kRunMacro:
                     break;
                  case kHelpAbout:
                     hd = new TRootHelpDialog(this, "About TTreeView...", 600, 400);
                     hd->SetText(gTVHelpAbout);
                     hd->Resize(hd->GetDefaultSize());
                     hd->Popup();
                     break;
                  case kHelpStart:
                     hd = new TRootHelpDialog(this, "Quick start...", 600, 400);
                     hd->SetText(gTVHelpStart);
                     hd->Popup();
                     break;
                  case kHelpLayout:
                     hd = new TRootHelpDialog(this, "The layout...", 600, 400);
                     hd->SetText(gTVHelpLayout);
                     hd->Popup();
                     break;
                  case kHelpBrowse:
                     hd = new TRootHelpDialog(this, "Browsing...", 600, 400);
                     hd->SetText(gTVHelpBrowse);
                     hd->Popup();
                     break;
                  case kHelpDragging:
                     hd = new TRootHelpDialog(this, "Dragging items...", 600, 400);
                     hd->SetText(gTVHelpDraggingItems);
                     hd->Popup();
                     break;
                  case kHelpEditing:
                     hd = new TRootHelpDialog(this, "Editing expressions...", 600, 400);
                     hd->SetText(gTVHelpEditExpressions);
                     hd->Popup();
                     break;
                  case kHelpCommands:
                     hd = new TRootHelpDialog(this, "Executing user commands...", 600, 400);
                     hd->SetText(gTVHelpUserCommands);
                     hd->Popup();
                     break;
                  case kHelpContext:
                     hd = new TRootHelpDialog(this, "Context menus...", 600, 400);
                     hd->SetText(gTVHelpContext);
                     hd->Popup();
                     break;
                  case kHelpDrawing:
                     hd = new TRootHelpDialog(this, "Drawing histograms...", 600, 400);
                     hd->SetText(gTVHelpDrawing);
                     hd->Popup();
                     break;
                  case kHelpMacros:
                     hd = new TRootHelpDialog(this, "Using macros...", 600, 400);
                     hd->SetText(gTVHelpMacros);
                     hd->Popup();
                     break;
                  default:
                     break;			
               }
               break;
            default:
               break;
         }
	 break;
      case kC_CONTAINER:
         switch (GET_SUBMSG(msg)) {
         // handle messages sent from the listview (right panel)
            case kCT_SELCHANGED:
               break;
            case kCT_ITEMCLICK:
            // handle mouse messages
               switch (parm1) {
                  case kButton1:
                     if (fLVContainer->NumSelected()) {
                     // get item that sent this
                        void *p = 0;
                        TGLVTreeEntry *item;
                        if ((item = (TGLVTreeEntry *) fLVContainer->GetNextSelected(&p)) != 0) {
                           TString trueName = item->GetTrueName();
                           char* msg = new char[256];
                           // get item type
                           ULong_t *itemType = (ULong_t *) item->GetUserData();
                           if (*itemType & kLTTreeType) {
                           // X, Y or Z clicked
                              char symbol = (char)((*itemType) >> 8);
                              sprintf(msg, "%c expression : %s", symbol, item->GetTrueName());
                           } else {
                              if (*itemType & kLTCutType) {
                              // scissors clicked
                                 sprintf(msg, "Cut : %s", item->GetTrueName());
                              } else {
                                 if (*itemType & kLTExpressionType) {
                                 // expression clicked
                                    sprintf(msg, "Expression : %s", item->GetTrueName());
                                 } else {
                                    if (*itemType & kLTBranchType) {
                                       sprintf(msg, "Branch : %s", item->GetTrueName());
                                    } else {
                                       sprintf(msg, "Leaf : %s", item->GetTrueName());
                                    }
                                 }
                              }
                           }
                           // write who is responsable for this
                           Message(msg);
                           delete[] msg;
                           // check if this should be pasted into the expression editor
                           if ((*itemType & kLTBranchType) || (*itemType & kLTCutType)) break;
                           fDialogBox = TGSelectBox::GetInstance();
                           if (!fDialogBox) break;
                           // paste it
                           char first = (char) trueName(0);
			   TString insert("");
                           if (first != '(') insert += "(";
                           insert += item->GetTrueName();
                           if (first != '(') insert += ")";

			   fDialogBox->GrabPointer();
			   fDialogBox->InsertText(insert.Data());
			   // put the cursor at the right position
                        }
                     }
                     break;
                  case kButton2:
                     break;
                  case kButton3:
                  // activate general context menu
                     if (fLVContainer->NumSelected()) {
                        void *p = 0;
                        Int_t x = (Int_t)(parm2 &0xffff);
                        Int_t y = (Int_t)((parm2 >> 16) & 0xffff);
                        TGLVTreeEntry *item;
                        if ((item = (TGLVTreeEntry *) fLVContainer->GetNextSelected(&p)) != 0) {
                           fContextMenu->Popup(x, y, this);
                        }
                     } else {        // empty click
                        Int_t x = (Int_t)(parm2 &0xffff);
                        Int_t y = (Int_t)((parm2 >> 16) & 0xffff);
                        fContextMenu->Popup(x, y, this);
                     }
                     break;
                  default:
                     break;
               }
               break;
            case kCT_ITEMDBLCLICK:
               switch (parm1) {
                  case kButton1:
                     if (fLVContainer->NumSelected()) {
                     // get item that sent this
                        void *p = 0;
                        TGLVTreeEntry *item;
                        if ((item = (TGLVTreeEntry *) fLVContainer->GetNextSelected(&p)) != 0) {
                        // get item type
                           ULong_t *itemType = (ULong_t *) item->GetUserData();
                           if (!(*itemType & kLTCutType) && !(*itemType & kLTBranchType)) {
                              if (strlen(item->GetTrueName())) {
                                 fVarDraw = kTRUE;
                                 // draw on double-click
                                 ExecuteDraw();
				 break;
                              }
                           }
			   if (*itemType & kLTCutType) {
			      fEnableCut = !fEnableCut;
			      if (fEnableCut) {
			         item->SetSmallPic(gClient->GetPicture("cut_t.xpm"));
			      } else {
			         item->SetSmallPic(gClient->GetPicture("cut-disable_t.xpm"));
			      } 
			   }
                        }
                     }
                     break;
                  case kButton2:
                     break;
                  case kButton3:
                     break;
                  default:
                     break;
               }
               break;
            case 4:
//               cout << "Dragging Item" << endl;
            default:
               break;
         }
         break;
      default:
         break;
   }
   return kTRUE;
}
//______________________________________________________________________________
void TTreeView::CloseWindow()
{
// Close the viewer
   gVirtualX->UnmapWindow(GetId());
   delete this;
   cout << "Tree Viewer deleted\n";
}
//______________________________________________________________________________
void TTreeView::ExecuteCommand(const char* command, Bool_t fast)
{
//*-*-*-*-*-*-*-*-*Execute all user commands*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================
// Execute the command, write it to history file and echo it to output
   if (fBarRec->GetState() == kButtonDown) {
   // show the command on the command line
      printf("%s\n", command);
      char comm[256];
      comm[0] = 0;
      sprintf(comm, command);
      // print the command to history file
      Gl_histadd(comm);
   }
   gROOT->SetInterrupt(kFALSE);
   fTimer->TurnOn();
   // execute it
   if (fast) {
      gROOT->ProcessLineFast(command);
   } else {
      gROOT->ProcessLine(command);
   }
//   HandleTimer(fTimer);
   fTimer->TurnOff();
   fCounting = kFALSE;
   // make sure that 'draw on double-click' flag is reset
   fVarDraw = kFALSE;
}
//______________________________________________________________________________
void TTreeView::MapOptions(Long_t parm1)
{
//*-*-*-*-*-*-*-*-*Scan the selected options from option menu*-*-*-*-*-*-*-*-*-*-*
//*-*              ==========================================
   Int_t ind;
   if (parm1 < kOptions1D) {
      if (fOptionsGen->IsEntryChecked(parm1)) {
         fOptionsGen->UnCheckEntry(parm1);
      } else {
         fOptionsGen->CheckEntry(parm1);
         if (parm1 != kOptionsGeneral) fOptionsGen->UnCheckEntry(kOptionsGeneral);
      }
      if (fOptionsGen->IsEntryChecked(kOptionsGeneral)) {	
      // uncheck all in this menu
         for (ind=kOptionsGeneral+1; ind<kOptionsGeneral+16; ind++) {
            fOptionsGen->UnCheckEntry(ind);
         }
      }
   }

   if (parm1 < kOptions2D) {
      if (fOptions1D->IsEntryChecked(parm1)) {
         fOptions1D->UnCheckEntry(parm1);
      } else {
         fOptions1D->CheckEntry(parm1);
         if (parm1 != kOptions1D) fOptions1D->UnCheckEntry(kOptions1D);
      }
      if (fOptions1D->IsEntryChecked(kOptions1D)) {	
      // uncheck all in this menu
         for (ind=kOptions1D+1; ind<kOptions1D+12; ind++) {
            fOptions1D->UnCheckEntry(ind);
         }
      }
   }
   
   if (parm1 >= kOptions2D) {
      if (fOptions2D->IsEntryChecked(parm1)) {
         fOptions2D->UnCheckEntry(parm1);
      } else {
         fOptions2D->CheckEntry(parm1);
         if (parm1 != kOptions2D) fOptions2D->UnCheckEntry(kOptions2D);
      }
      if (fOptions2D->IsEntryChecked(kOptions2D)) {	
      // uncheck all in this menu
         for (ind=kOptions2D+1; ind<kOptions1D+14; ind++) {
            fOptions2D->UnCheckEntry(ind);
         }
      }
   }
   // concatenate options
   fBarOption->SetText("");
   for (ind=kOptionsGeneral; ind<kOptionsGeneral+16; ind++) {
      if (fOptionsGen->IsEntryChecked(ind)) 
         fBarOption->AppendText(optgen[ind-kOptionsGeneral]);	
   }
   if (Dimension() == 1) {
      for (ind=kOptions1D; ind<kOptions1D+12; ind++) {
         if (fOptions1D->IsEntryChecked(ind))
            fBarOption->AppendText(opt1D[ind-kOptions1D]);   
      }
   }
   if (Dimension() == 2) {
      for (ind=kOptions2D; ind<kOptions2D+14; ind++) {
         if (fOptions2D->IsEntryChecked(ind))
            fBarOption->AppendText(opt2D[ind-kOptions2D]);   
      }
   }
}
//______________________________________________________________________________
void TTreeView::MapTree(TTree *tree, TGListTreeItem *parent, Bool_t listIt)
{
//*-*-*-*-*-*-*-*-*Map current tree and expand its content in the lists*-*-*-*-*-*-*-*-*-*-*
//*-*              ====================================================
   if (!tree) return;
   TObjArray *Branches = tree->GetListOfBranches();
   TBranch   *branch;
   // loop on branches
   for (Int_t id=0; id<Branches->GetEntries(); id++) {
      branch = (TBranch *)Branches->At(id);
      TString name = branch->GetName();
      if (name.Contains(".fBits") || name.Contains(".fUniqueID")) continue;
      // now map sub-branches
      MapBranch(branch, parent, listIt);
      fStopMapping = kFALSE;
   }
   // tell who was last mapped
   if (listIt) {
      fMappedTree    = tree;
      fMappedBranch  = 0;
   }
}
//______________________________________________________________________________
void TTreeView::MapBranch(TBranch *branch, TGListTreeItem *parent, Bool_t listIt)
{
//*-*-*-*-*-*-*-*-*Map current branch and expand its content in the list view*-*-*-*-*-*-*-*-*-*-*
//*-*              ==========================================================
   if (!branch) return;
   TString   name = branch->GetName();
   Int_t     ind;
   TGListTreeItem *branchItem = 0;
   ULong_t *itemType;
// map this branch
   if (name.Contains(".fBits") || name.Contains(".fUniqueID")) return;
   if (parent) {
   // make list tree items for each branch according to the type
      const TGPicture *pic, *spic;
      if (branch->GetListOfBranches()->GetEntries()) {
         itemType = new ULong_t(kLTBranchType);
         if (branch->InheritsFrom("TBranchObject")) {
            pic = gClient->GetPicture("branch-ob_t.xpm");
            spic = gClient->GetPicture("branch-ob_t.xpm");
         } else {
            if (branch->InheritsFrom("TBranchClones")) {
               pic = gClient->GetPicture("branch-cl_t.xpm");
               spic = gClient->GetPicture("branch-cl_t.xpm");
            } else {
               pic = gClient->GetPicture("branch_t.xpm");
               spic = gClient->GetPicture("branch_t.xpm");		   
            }
         }
         branchItem = fLt->AddItem(parent, branch->GetName(), itemType,
                      pic, spic);
      } else {
         itemType = new ULong_t(kLTLeafType);
         pic = gClient->GetPicture("leaf_t.xpm");
         spic = gClient->GetPicture("leaf_t.xpm");
         branchItem = fLt->AddItem(parent, branch->GetName(), itemType,
                                   pic, spic);
      }
   }
   // list branch in list view if necessary
   if (listIt) {
      TGString *textEntry;
      const TGPicture *pic, *spic;
      TGLVTreeEntry *entry;
      // make list view items in the right frame
      if (!fStopMapping) {
         fMappedBranch = branch;
         fMappedTree = 0;
         fStopMapping = kTRUE;
      }
      textEntry = new TGString(name.Data()); 
      if (branch->GetListOfBranches()->GetEntries()) {
         if (branch->InheritsFrom("TBranchObject")) {
            pic = gClient->GetPicture("branch-ob_t.xpm");
            spic = gClient->GetPicture("branch-ob_t.xpm");
         } else {
            if (branch->InheritsFrom("TBranchClones")) {
               pic = gClient->GetPicture("branch-cl_t.xpm");
               spic = gClient->GetPicture("branch-cl_t.xpm");
            } else {
               pic = gClient->GetPicture("branch_t.xpm");
               spic = gClient->GetPicture("branch_t.xpm");		   
            }
         }
         entry = new TGLVTreeEntry(fLVContainer,pic,spic,textEntry,0,kLVSmallIcons);
         entry->SetUserData(new UInt_t(kLTBranchType));
         fLVContainer->AddThisItem(entry);
         entry->MapWindow();
      } else {
         pic = (gClient->GetMimeTypeList())->GetIcon("TLeaf",kFALSE);
         if (!pic) pic = gClient->GetPicture("leaf_t.xpm");
         spic = gClient->GetMimeTypeList()->GetIcon("TLeaf",kTRUE);
         if (!spic) spic = gClient->GetPicture("leaf_t.xpm");
         entry = new TGLVTreeEntry(fLVContainer,pic,spic,textEntry,0,kLVSmallIcons);
         entry->SetUserData(new UInt_t(kLTDragType));
         fLVContainer->AddThisItem(entry);
         entry->MapWindow();
      }
   }

   TObjArray *Branches 	= branch->GetListOfBranches(); 
   TBranch   *branchDaughter = 0;
   
   // loop all sub-branches
   for (ind=0; ind<Branches->GetEntries(); ind++) {
      branchDaughter = (TBranch *)Branches->UncheckedAt(ind);
      // map also all sub-branches
      MapBranch(branchDaughter, branchItem, listIt);
   }
}
//______________________________________________________________________________
void TTreeView::SetParentTree(TGListTreeItem *item)
{
//*-*-*-*-*-*-*-*-*Find parent tree of a clicked item*-*-*-*-*-*-*-*-*-*-*
//*-*              ==================================
   if (!item) return;
   ULong_t *itemType = (ULong_t *)item->GetUserData();
   TGListTreeItem *parent = 0;
   Int_t index;
   if (!(*itemType & kLTTreeType)) {
      parent = item->GetParent();
      SetParentTree(parent); 
   } else {
      index = (Int_t)(*itemType >> 8);
      SwitchTree(index);
   }
}
//______________________________________________________________________________
void TTreeView::Message(const char* msg)
{
//*-*-*-*-*-*-*-*-*Send a message on the status bar*-*-*-*-*-*-*-*-*-*-*
//*-*              ================================
   fStatusBar->SetText(msg);
}
//______________________________________________________________________________
void TTreeView::Warning(const char* msg)
{
//*-*-*-*-*-*-*-*-*Pops-up a warning message*-*-*-*-*-*-*-*-*-*-*
//*-*              =========================
   TGMsgBox *mBox = new TGMsgBox(fClient->GetRoot(), this, "", msg, kMBIconExclamation);
   gClient->WaitFor(mBox);
}
//______________________________________________________________________________
void TTreeView::PrintEntries()
{
//*-*-*-*-*-*-*-*-*Print the number of selected entries on status-bar*-*-*-*-*
//*-*              ==================================================
   if (!fTree) return;
   char * msg = new char[100];
   sprintf(msg, "First entry : %i Last entry : %i", 
           (Int_t)fSlider->GetMinPosition(), (Int_t)fSlider->GetMaxPosition());
   Message(msg);
   delete[] msg;
}
//______________________________________________________________________________
Bool_t TTreeView::SwitchTree(Int_t index)
{
//*-*-*-*-*-*-*-*-*Makes current the tree at a given index in the list*-*-*-*-*
//*-*              ===================================================
   TTree *tree = (TTree *) fTreeList->At(index);
   if (!tree) {
      printf("Error : SwitchTree() : No tree at index %i\n", index);
      return kFALSE;
   }
   if ((tree == fTree) && (tree == fMappedTree)) return kFALSE;     // nothing to switch
   char *command = new char[50];
   if (tree != fTree) {
      sprintf(command, "tree = (TTree *) list->At(%i);", index);
      ExecuteCommand(command);
   }

   fTree = tree;
   fSlider->SetRange(0,fTree->GetEntries()-1);
   fSlider->SetPosition(0,fTree->GetEntries()-1);
   sprintf(command, "Current tree : %s", fTree->GetName());
   fLbl2->SetText(new TGString(command));
   delete[] command;
   PrintEntries();
   return kTRUE;
}
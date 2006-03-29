// @(#)root/guibuilder:$Name:  $:$Id: TRootGuiBuilder.cxx,v 1.18 2005/12/08 13:03:57 brun Exp $
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TRootGuiBuilder.h"
#include "TGuiBldDragManager.h"
#include "TGuiBldEditor.h"

#include "TGShutter.h"
#include "TGSplitter.h"
#include "TGLayout.h"
#include "TGResourcePool.h"
#include "TGButton.h"
#include "TROOT.h"
#include "TGDockableFrame.h"
#include "TGMdi.h"
#include "TGStatusBar.h"
#include "TG3DLine.h"
#include "TGLabel.h"
#include "TColor.h"
#include "TGToolBar.h"
#include "TGToolTip.h"
#include "KeySymbols.h"
#include "TGFileDialog.h"
#include "TGMsgBox.h"
#include "TSystem.h"
#include "TApplication.h"
#include "TRootHelpDialog.h"

//////////////////////////////////////////////////////////////////////////
//
// TRootGuiBuilder
//
//
//  ************************************************
//                ROOT GUI Builder principles
//  ************************************************
//
//  With the GUI builder, we try to make the next step from WYSIWYG
//  to embedded editing concept - WYSIWYE ("what you see is what you edit").
//  The ROOT GUI Builder allows modifying real GUI objects.
//  For example, one can edit the existing GUI application $ROOTSYS/tutorials/guitest.C.
//  GUI components can be added to a design area from a widget palette,
//  or can be borrowed from another application.
//  One can drag and and drop TCanvas's menu bar into the application.
//  GUI objects can be resized and dragged, copied and pasted.
//  ROOT GUI Builder allows changing the layout, snap to grid, change object's
//  layout order via the GUI Builder toolbar, or by options in the right-click
//  context menus.
//  A final design can be immediatly tested and used, or saved as a C++ macro.
//  For example, it's possible to rearrange buttons in control bar,
//  add separators etc. and continue to use a new fancy control bar in the
//  application.
//
//  ************************************************
//
//  The following is a short description of the GUI Builder actions and key shortcuts:
//
//   o Press Ctrl-Double-Click to start/stop edit mode
//   o Press Double-Click to activate quick edit action (defined in root.mimes)
//
//                 Selection, grabbing, dropping
//       ************************************************
//    It is possible to select, drag any frame and drop it to any frame
//
//   o Click left mouse button or Ctrl-Click to select an object to edit.
//   o Press right mouse button to activate context menu
//   o Mutiple selection (grabbing):
//      - draw lasso and press Return key
//      - press Shift key and draw lasso
//   o Dropping:
//      - select frame and press Ctrl-Return key
//   o Changing layout order:
//      - select frame and use arrow keys to change layout order
//   o Alignment:
//      - draw lasso and press arrow keys (or Shift-Arrow key) to align frames
//
//                    Key shortcuts
//       ************************************************
//   o Return    - grab selected frames
//   o Ctrl-Return - drop frames
//   o Del       - delete selected frame
//   o Shift-Del - crop action
//   o Ctrl-X    - cut action
//   o Ctrl-C    - copy action
//   o Ctrl-V    - paste action
//   o Ctrl-R    - replace action
//   o Ctrl-L    - compact
//   o Ctrl-B    - enable/disable layout 
//   o Ctrl-H    - switch horizontal-vertical layout
//   o Ctrl-G    - switch on/off grid
//   o Ctrl-S    - save action
//   o Ctrl-O    - open and execute a ROOT macro file.  GUI components created
//                 after macro execution will be emebedded to currently edited
//                 design area.
//   o Ctrl-N    - create new main frame
//
//Begin_Html
/*
<img src="gif/RootGuiBuilder.gif">
*/
//End_Html


const char gHelpBuilder[] = "\
 o Press Ctrl-Double-Click to start/stop edit mode\n\
 o Press Double-Click to activate quick edit action (defined in root.mimes)\n\
\n\
               Selection, grabbing, dropping\n\
     ************************************************\n\
  It is possible to select, drag any frame and drop to any frame\n\
\n\
 o Press left mouse button Click or Ctrl-Click to select an object to edit.\n\
 o Press right mouse button to activate context menu\n\
 o Mutiple selection (grabbing):\n\
      - draw lasso and press Return key\n\
      - press Shift key and draw lasso\n\
 o Dropping:\n\
      - select frame and press Ctrl-Return key\n\
 o Changing layout order:\n\
      - select frame and use arrow keys to change layout order\n\
 o Alignment:\n\
      - draw lasso and press arrow keys (or Shift-Arrow key) to align frames\n\
\n\
                    Key shortcuts\n\
     ************************************************\n\
 o Return    - grab selected frames\n\
 o Ctrl-Return - drop frames\n\
 o Del       - delete selected frame\n\
 o Shift-Del - crop\n\
 o Ctrl-X    - cut\n\
 o Ctrl-C    - copy\n\
 o Ctrl-V    - paste\n\
 o Ctrl-R    - replace\n\
 o Ctrl-L    - compact frame\n\
 o Ctrl-B    - enable/disable layout\n\
 o Ctrl-H    - switch Horizontal-Vertical layout\n\
 o Ctrl-G    - switch ON/OFF grid\n\
 o Ctrl-S    - save\n\
 o Ctrl-O    - open and execute ROOT macro file\n\
 o Ctrl-N    - create new main frame\n\
 o Ctrl-Z    - undo last action (not implemented)\n\
 o Shift-Ctrl-Z - redo (not implemented)\n\
";

const char gHelpAboutBuilder[] = "\
                  ROOT Gui Builder\n\
\n\
************************************************************\n\
* Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.  *\n\
* All rights reserved.                                     *\n\
*                                                          *\n\
* For the licensing terms see $ROOTSYS/LICENSE.            *\n\
* For the list of contributors see $ROOTSYS/README/CREDITS.*\n\
************************************************************\n\
";

//----- Toolbar stuff...

static ToolBarData_t gToolBarData[] = {
   { "bld_edit.xpm",   "Start Edit (Ctrl-Dbl-Click)",   kFALSE, kEditableAct, 0 },
   { "",                 "",               kFALSE, -1, 0 },
   { "bld_new.xpm",   "New (Ctrl-N)",   kFALSE, kNewAct, 0 },
   { "bld_open.xpm",   "Open (Ctrl-O)",   kFALSE, kOpenAct, 0 },
   { "bld_save.xpm",   "Save (Ctrl-S)",   kFALSE, kSaveAct, 0 },
   { "",                 "",               kFALSE, -1, 0 },
//   { "bld_pointer.xpm",   "Selector (Ctrl-Click)",   kTRUE, kSelectAct, 0 },
   //{ "bld_grab.xpm",   "Grab Selected Frames (Return)",   kTRUE, kGrabAct, 0 },
   { "",                 "",               kFALSE, -1, 0 },
   { "bld_layout.xpm",   "Compact (Ctrl-L)",        kFALSE,  kCompactAct, 0 },
   { "bld_break.xpm",   "Disable/Enable layout (Ctrl-B)",        kFALSE,  kBreakLayoutAct, 0 },
   { "bld_hbox.xpm",  "Lay Out Horizontally (Ctrl-H)",    kFALSE,  kLayoutHAct, 0 },
   { "bld_vbox.xpm",   "Lay Out Vertically (Ctrl-H)",    kFALSE,  kLayoutVAct, 0 },
   { "bld_grid.xpm",   "On/Off Grid (Ctrl+G)",     kFALSE,  kGridAct, 0 },
   { "",                 "",               kFALSE, -1, 0 },
   { "bld_AlignTop.xpm",   "Align Top (Up|Shift  Arrow)",        kFALSE,  kUpAct, 0 },
   { "bld_AlignBtm.xpm",   "Align Bottom (Down|Shift Arrow)",        kFALSE,  kDownAct, 0 },
   { "bld_AlignLeft.xpm",   "Align Left (Left|Shift  Arrow)",        kFALSE,  kLeftAct, 0 },
   { "bld_AlignRight.xpm",   "Align Right (Right|Shift  Arrow)",        kFALSE,  kRightAct, 0 },
   { "",                 "",               kFALSE, -1, 0 },
   { "bld_cut.xpm",   "Cut (Ctrl-X)",        kFALSE,  kCutAct, 0 },
   { "bld_copy.xpm",   "Copy (Ctrl-C)",        kFALSE,  kCopyAct, 0 },
   { "bld_paste.xpm",   "Paste (Ctrl-V)",        kFALSE,  kPasteAct, 0 },
   { "bld_replace.xpm",   "Replace (Ctrl-R)",        kFALSE,  kReplaceAct, 0 },
   { "bld_delete.xpm",   "Delete (Del/Backspace)",        kFALSE,  kDeleteAct, 0 },
   { "bld_crop.xpm",   "Crop (Shift-Del)",        kFALSE,  kCropAct, 0 },
//   { "",                 "",               kFALSE, -1, 0 },
//   { "bld_undo.xpm",   "Undo (Ctrl-Z)",        kFALSE,  kUndoAct, 0 },
//   { "bld_redo.xpm",   "Redo (Shift-Ctrl-Z)",        kFALSE,  kRedoAct, 0 },
   { 0,                  0,                kFALSE, 0, 0 }
};


ClassImp(TRootGuiBuilder)


////////////////////////////////////////////////////////////////////////////////
class TRootGuiBuilderContainer : public TGMdiContainer {

public:
   TRootGuiBuilderContainer(const TGMdiMainFrame *p) :
         TGMdiContainer(p, 10, 10, kOwnBackground) {
      const TGPicture *pbg = fClient->GetPicture("bld_bg.xpm");
      if (pbg) SetBackgroundPixmap(pbg->GetPicture());
   }
   virtual ~TRootGuiBuilderContainer() {}
   void SetEditable(Bool_t) {}
   Bool_t HandleEvent(Event_t*) { return kFALSE; }
};

////////////////////////////////////////////////////////////////////////////////
class TRootGuiBldMain : public TGMdiMainFrame {
public:
   TRootGuiBuilder *fBuilder;

public:
   TRootGuiBldMain(TRootGuiBuilder *bld, TGCompositeFrame *cf, TGMdiMenuBar *bar) :
      TGMdiMainFrame(cf, bar, 1, 1), fBuilder(bld)
   {

   }
   virtual ~TRootGuiBldMain() {}
};



////////////////////////////////////////////////////////////////////////////////
TRootGuiBuilder::TRootGuiBuilder(const TGWindow *p) : TGuiBuilder(),
             TGMainFrame(p ? p : gClient->GetDefaultRoot(), 1, 1)
{
   // ctor

   SetCleanup(kDeepCleanup);
   fEditDisabled = kEditDisable;
   gGuiBuilder  = this;
   fActionButton = 0;

   if (gDragManager) {
      fManager = (TGuiBldDragManager *)gDragManager;
   } else {
      gDragManager = fManager = new TGuiBldDragManager();
   }
   fManager->SetBuilder(this);

   fMenuBar = new TGMdiMenuBar(this, 10, 10);
   AddFrame(fMenuBar, new TGLayoutHints(kLHintsTop | kLHintsExpandX));
   InitMenu();

   TGHorizontal3DLine *hl = new TGHorizontal3DLine(this);
   AddFrame(hl, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0,0,2,2));

   fToolDock = new TGDockableFrame(this);
   AddFrame(fToolDock, new TGLayoutHints(kLHintsExpandX, 0, 0, 1, 0));
   fToolDock->SetWindowName("ROOT GuiBuilder ToolBar");

   fToolBar = new TGToolBar(fToolDock, 60, 20, kHorizontalFrame);
   fToolDock->AddFrame(fToolBar, new TGLayoutHints(kLHintsTop | kLHintsExpandX));

   int spacing = 8;

   for (int i = 0; gToolBarData[i].fPixmap; i++) {
      if (strlen(gToolBarData[i].fPixmap) == 0) {
         spacing = 8;
         continue;
      }
      TGPictureButton *pb = (TGPictureButton*)fToolBar->AddButton(this, &gToolBarData[i], spacing);
      spacing = 0;

      if (gToolBarData[i].fId == kEditableAct) {
         fStartButton = pb;
         continue;
      }

      TGToolTip *tip = pb->GetToolTip();
      if (tip) {
         tip->Connect("Reset()", "TRootGuiBuilder", this, "UpdateStatusBar()");
         tip->Connect("Hide()", "TRootGuiBuilder", this, "EraseStatusBar()");
      }

      TString pname = gToolBarData[i].fPixmap;
      pname.ReplaceAll(".", "_d.");

      const TGPicture *dpic = fClient->GetPicture(pname.Data());
      if (dpic) pb->SetDisabledPicture(dpic);

      if ((gToolBarData[i].fId == kUndoAct) || (gToolBarData[i].fId == kRedoAct)) {
         pb->SetState(kButtonDisabled);
      }
   }

   fToolBar->Connect("Clicked(Int_t)", "TGuiBldDragManager", fManager, "HandleAction(Int_t)");

   hl = new TGHorizontal3DLine(this);
   AddFrame(hl, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0,0,2,5));

   TGCompositeFrame *cf = new TGHorizontalFrame(this, 1, 1);
   AddFrame(cf, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   //fShutterDock = new TGDockableFrame(cf);
   //cf->AddFrame(fShutterDock, new TGLayoutHints(kLHintsNormal ));
   //fShutterDock->SetWindowName("Widget Factory");
   //fShutterDock->EnableUndock(kTRUE);
   //fShutterDock->EnableHide(kTRUE);
   //fShutterDock->DockContainer();

   fShutter = new TGShutter(cf, kSunkenFrame);
   cf->AddFrame(fShutter, new TGLayoutHints(kLHintsNormal | kLHintsExpandY));
   fShutter->ChangeOptions(fShutter->GetOptions() | kFixedWidth);

   TGVSplitter *splitter = new TGVSplitter(cf);
   splitter->SetFrame(fShutter, kTRUE);
   cf->AddFrame(splitter, new TGLayoutHints(kLHintsLeft | kLHintsExpandY));

   fMain = new TRootGuiBldMain(this, cf, fMenuBar);
   cf->AddFrame(fMain, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   TGFrame *mdicont = fMain->GetContainer();
   fMain->GetViewPort()->RemoveFrame(mdicont);
   delete mdicont;
   fMain->SetContainer(new TRootGuiBuilderContainer(fMain));

   if (fManager) {
      fEditor = new TGuiBldEditor(cf);
      cf->AddFrame(fEditor, new TGLayoutHints(kLHintsNormal | kLHintsExpandY));
      fManager->SetPropertyEditor(fEditor);
      fEditor->SetEmbedded();

//      ed->ChangeOptions(ed->GetOptions() | kFixedWidth);
//      splitter = new TGVSplitter(cf);
//      splitter->SetFrame(ed, kFALSE);
//      cf->AddFrame(splitter, new TGLayoutHints(kLHintsLeft | kLHintsExpandY));
   }

   AddSection("Projects");
   AddSection("Buttons");
   AddSection("Containers");
   AddSection("Input");
   AddSection("Display");
//   AddSection("Complex Input");
//   AddSection("Extended");

   TGuiBldAction *act = new TGuiBldAction("TGMainFrame", "Main Frame", kGuiBldProj);
   act->fAct = "new TGMainFrame(gClient->GetRoot(), 300, 300)";
   act->fPic = "bld_mainframe.xpm";
   AddAction(act, "Projects");

   // Standard
   act = new TGuiBldAction("TGTextButton", "Text Button", kGuiBldCtor);
   act->fAct = "new TGTextButton()";
   act->fPic = "bld_textbutton.xpm";
   AddAction(act, "Buttons");

   act = new TGuiBldAction("TGCheckButton", "Check Button", kGuiBldCtor);
   act->fAct = "new TGCheckButton()";
   act->fPic = "bld_checkbutton.xpm";
   AddAction(act, "Buttons");

   act = new TGuiBldAction("TGRadioButton", "Radio Button", kGuiBldCtor);
   act->fAct = "new TGRadioButton()";
   act->fPic = "bld_radiobutton.xpm";
   AddAction(act, "Buttons");


   act = new TGuiBldAction("TGPictureButton", "Picture Button", kGuiBldCtor);
   act->fAct = "new TGPictureButton()";
   act->fPic = "bld_image.xpm";
   AddAction(act, "Buttons");

   act = new TGuiBldAction("TGTextEntry", "Text Entry", kGuiBldCtor);
   act->fAct = "new TGTextEntry()";
   act->fPic = "bld_entry.xpm";
   AddAction(act, "Input");

   act = new TGuiBldAction("TGNumberEntry", "Number Entry", kGuiBldCtor);
   act->fAct = "new TGNumberEntry()";
   act->fPic = "bld_numberentry.xpm";
   AddAction(act, "Input");

   act = new TGuiBldAction("TGHSlider", "Horizontal Slider", kGuiBldCtor);
   act->fAct = "new TGHSlider()";
   act->fPic = "bld_hslider.xpm";
   AddAction(act, "Input");

   act = new TGuiBldAction("TGVSlider", "Vertical Slider", kGuiBldCtor);
   act->fAct = "new TGVSlider()";
   act->fPic = "bld_vslider.xpm";
   AddAction(act, "Input");

   act = new TGuiBldAction("TGLabel", "Text Label", kGuiBldCtor);
   act->fAct = "new TGLabel()";
   act->fPic = "bld_label.xpm";
   AddAction(act, "Display");

   act = new TGuiBldAction("TGHorizontal3DLine", "Horizontal Line", kGuiBldCtor);
   act->fAct = "new TGHorizontal3DLine()";
   act->fPic = "bld_hseparator.xpm";
   AddAction(act, "Display");

   act = new TGuiBldAction("TGVertical3DLine", "Vertical Line", kGuiBldCtor);
   act->fAct = "new TGVertical3DLine()";
   act->fPic = "bld_vseparator.xpm";
   AddAction(act, "Display");

   act = new TGuiBldAction("TGStatusBar", "Status Bar", kGuiBldCtor);
   act->fAct = "new TGStatusBar()";
   act->fPic = "bld_statusbar.xpm";
   act->fHints = new TGLayoutHints(kLHintsBottom | kLHintsExpandX);
   AddAction(act, "Display");

   act = new TGuiBldAction("TGHProgressBar", "Progress Bar", kGuiBldCtor);
   act->fAct = "new TGHProgressBar()";
   act->fPic = "bld_hprogressbar.xpm";
   AddAction(act, "Display");

   act = new TGuiBldAction("TRootEmbeddedCanvas", "Embed Canvas", kGuiBldCtor);
   act->fAct = "new TRootEmbeddedCanvas()";
   act->fPic = "bld_embedcanvas.xpm";
   AddAction(act, "Display");

   // Containers
   act = new TGuiBldAction("TGHorizontalFrame", "Horizontal Frame", kGuiBldCtor);
   act->fAct = "new TGHorizontalFrame()";
   act->fPic = "bld_hbox.xpm";
   AddAction(act, "Containers");

   act = new TGuiBldAction("TGVerticalFrame", "Vertical Frame", kGuiBldCtor);
   act->fAct = "new TGVerticalFrame()";
   act->fPic = "bld_vbox.xpm";
   AddAction(act, "Containers");

   act = new TGuiBldAction("TGGroupFrame", "Group Frame", kGuiBldCtor);
   act->fAct = "new TGGroupFrame()";
   act->fPic = "bld_groupframe.xpm";
   AddAction(act, "Containers");

   act = new TGuiBldAction("TGTab", "Tabbed Frame", kGuiBldCtor);
   act->fAct = "new TGTab()";
   act->fPic = "bld_tab.xpm";
   AddAction(act, "Containers");

/*
   act = new TGuiBldAction("TGVSplitter", "Horizontal Panes", kGuiBldFunc);
   act->fAct = "TRootGuiBuilder::VSplitter()";
   act->fPic = "bld_hpaned.xpm";
   AddAction(act, "Containers");

   act = new TGuiBldAction("TGHSplitter", "Vertical Panes", kGuiBldFunc);
   act->fAct = "TRootGuiBuilder::HSplitter()";
   act->fPic = "bld_vpaned.xpm";
   AddAction(act, "Containers");
*/

   fShutter->Resize(140, fShutter->GetHeight());

   fStatusBar = new TGStatusBar(this, 40, 10);
   AddFrame(fStatusBar, new TGLayoutHints(kLHintsBottom | kLHintsExpandX, 0, 0, 3, 0));

   MapSubwindows();
	
   Int_t qq; 
	UInt_t ww; 
	UInt_t hh;
	gVirtualX->GetWindowSize(gVirtualX->GetDefaultRootWindow(), qq, qq, ww, hh);
	Resize(ww - 100, hh - 100);

   SetWindowName("ROOT GuiBuilder");
   SetIconName("ROOT GuiBuilder");
   fIconPic = SetIconPixmap("bld_rgb.xpm");
   SetClassHints("GuiBuilder", "GuiBuilder");

   fSelected = 0;
   Update();

   fMenuFile->Connect("Activated(Int_t)", "TRootGuiBuilder", this,
                      "HandleMenu(Int_t)");
   fMenuWindow->Connect("Activated(Int_t)", "TRootGuiBuilder", this,
                        "HandleMenu(Int_t)");
   fMenuHelp->Connect("Activated(Int_t)", "TRootGuiBuilder", this,
                      "HandleMenu(Int_t)");

   fMain->Connect("FrameClosed(Int_t)", "TRootGuiBuilder", this, "HandleWindowClosed(Int_t)");

   BindKeys();
   UpdateStatusBar("Ready");
   MapRaised();
}

//______________________________________________________________________________
TRootGuiBuilder::~TRootGuiBuilder()
{
   // destructor

   if (fIconPic) gClient->FreePicture(fIconPic);
   delete fMenuFile;
   delete fMenuWindow;
   delete fMenuHelp;
   gGuiBuilder = 0;
}

//______________________________________________________________________________
void TRootGuiBuilder::CloseWindow()
{
   // close GUI builder via "Close" button

   TGWindow *root = (TGWindow*)fClient->GetRoot();
   if (root) root->SetEditable(kFALSE);

   fEditor->Reset();

   if (fMain->GetNumberOfFrames() == 0) {
      fMenuFile->DisableEntry(kGUIBLD_FILE_CLOSE);
      fMenuFile->DisableEntry(kGUIBLD_FILE_STOP);
      fMenuFile->DisableEntry(kGUIBLD_FILE_START);
   } else {
      fMenuFile->DisableEntry(kGUIBLD_FILE_STOP);
      fMenuFile->EnableEntry(kGUIBLD_FILE_START);
      fMenuFile->EnableEntry(kGUIBLD_FILE_CLOSE);
   }

   fMain->CloseAll();
   Hide();
}

//______________________________________________________________________________
void TRootGuiBuilder::AddAction(TGuiBldAction *act, const char *sect)
{
   // add new action to widget palette

   if (!act || !sect) return;

   TGShutterItem *item = fShutter->GetItem(sect);
   TGButton *btn = 0;

   if (!item) return;
   TGCompositeFrame *cont = (TGCompositeFrame *)item->GetContainer();
   cont->SetBackgroundColor(TColor::Number2Pixel(18));

   const TGPicture *pic = fClient->GetPicture(act->fPic);

   TGHorizontalFrame *hf = new TGHorizontalFrame(cont);

   if (pic) {
      btn = new TGPictureButton(hf, pic);
   } else {
      btn = new TGTextButton(hf, act->GetName());
   }

   btn->SetToolTipText(act->GetTitle());
   btn->SetUserData((void*)act);
   btn->Connect("Clicked()", "TRootGuiBuilder", this, "HandleButtons()");

   hf->AddFrame(btn, new TGLayoutHints(kLHintsTop | kLHintsCenterY, 1, 1, 1, 1));

   TGLabel *lb = new TGLabel(hf, act->GetTitle());
   lb->SetBackgroundColor(cont->GetBackground());
   hf->AddFrame(lb, new TGLayoutHints(kLHintsTop | kLHintsCenterY, 1, 1, 1, 1));
   hf->SetBackgroundColor(cont->GetBackground());

   cont->AddFrame(hf, new TGLayoutHints(kLHintsTop, 5, 5, 5, 0));
   cont->MapSubwindows();
   cont->Resize();  // invoke Layout()
}

//______________________________________________________________________________
void TRootGuiBuilder::AddSection(const char *sect)
{
   //

   static int id = 10000;
   TGShutterItem *item = new TGShutterItem(fShutter, new TGHotString(sect), id++);
   fShutter->AddItem(item);
}

//______________________________________________________________________________
void TRootGuiBuilder::HandleButtons()
{
   //

   TGFrame *parent;

   if (fActionButton) {
      parent = (TGFrame*)fActionButton->GetParent();
      parent->ChangeOptions(parent->GetOptions() & ~kSunkenFrame);
      fClient->NeedRedraw(parent, kTRUE);
   }

   if (!fClient->IsEditable()) {
      HandleMenu(kGUIBLD_FILE_START);
   }

   fActionButton = (TGButton *)gTQSender;
   TGuiBldAction *act  = (TGuiBldAction *)fActionButton->GetUserData();
   parent = (TGFrame*)fActionButton->GetParent();
   parent->ChangeOptions(parent->GetOptions() | kSunkenFrame);
   fClient->NeedRedraw(parent, kTRUE);

   if (act) {
      fAction = act;
      if (fAction->fType == kGuiBldProj) ExecuteAction();
   }
}

//______________________________________________________________________________
TGFrame *TRootGuiBuilder::ExecuteAction()
{
   //

   if (!fAction || fAction->fAct.IsNull()) return 0;

   TGFrame *ret = 0;

   if (!fClient->IsEditable()) { 
      TGMdiFrame *current = fMain->GetCurrent();
      if (current) current->SetEditable(kTRUE);
   }

   switch (fAction->fType) {
      case kGuiBldProj:
         NewProject();
         fAction = 0;
         break;
      default:
         ret = (TGFrame *)gROOT->ProcessLineFast(fAction->fAct.Data());
         break;
   }

   Update();

   return ret;
}

//______________________________________________________________________________
void TRootGuiBuilder::InitMenu()
{
   // inititiate Gui Builder menu

   fMenuBar->SetEditDisabled(1);

   fMenuFile = new TGPopupMenu(fClient->GetDefaultRoot());
   fMenuFile->SetEditDisabled(1);
   fMenuFile->AddEntry(new TGHotString("&Edit (Ctrl-dbl-click)"), kGUIBLD_FILE_START);
   fMenuFile->AddEntry(new TGHotString("&Stop (Ctrl-dbl-click)"), kGUIBLD_FILE_STOP);
   fMenuFile->DisableEntry(kGUIBLD_FILE_STOP);
   fMenuFile->DisableEntry(kGUIBLD_FILE_START);
   fMenuFile->AddSeparator();
   fMenuFile->AddEntry(new TGHotString("&New Window"), kGUIBLD_FILE_NEW);
   fMenuFile->AddEntry(new TGHotString("&Close Window"), kGUIBLD_FILE_CLOSE);
   fMenuFile->DisableEntry(kGUIBLD_FILE_CLOSE);
   fMenuFile->AddSeparator();
   fMenuFile->AddEntry(new TGHotString("E&xit"), kGUIBLD_FILE_EXIT);

   fMenuWindow = new TGPopupMenu(fClient->GetDefaultRoot());
   fMenuWindow->SetEditDisabled(1);
   fMenuWindow->AddEntry(new TGHotString("Tile &Horizontally"), kGUIBLD_WINDOW_HOR);
   fMenuWindow->AddEntry(new TGHotString("Tile &Vertically"), kGUIBLD_WINDOW_VERT);
   fMenuWindow->AddEntry(new TGHotString("&Cascade"), kGUIBLD_WINDOW_CASCADE);
   fMenuWindow->AddSeparator();
   //fMenuWindow->AddPopup(new TGHotString("&Windows"), fMain->GetWinListMenu());
   fMenuWindow->AddSeparator();
   fMenuWindow->AddEntry(new TGHotString("&Arrange icons"), kGUIBLD_WINDOW_ARRANGE);
   fMenuWindow->AddSeparator();
   fMenuWindow->AddEntry(new TGHotString("&Opaque resize"), kGUIBLD_WINDOW_OPAQUE);

   fMenuWindow->CheckEntry(kGUIBLD_WINDOW_OPAQUE);

   fMenuHelp = new TGPopupMenu(fClient->GetDefaultRoot());
   fMenuHelp->SetEditDisabled(1);
   fMenuHelp->AddEntry(new TGHotString("&Contents"), kGUIBLD_HELP_CONTENTS);
   fMenuHelp->AddSeparator();
   fMenuHelp->AddEntry(new TGHotString("&About"), kGUIBLD_HELP_ABOUT);
   //fMenuHelp->AddSeparator();
   //fMenuHelp->AddEntry(new TGHotString("&Send Bug Report"), kGUIBLD_HELP_BUG);

   fMenuBar->AddPopup(new TGHotString("&File"), fMenuFile,
                      new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));
   fMenuBar->AddPopup(new TGHotString("&Windows"), fMenuWindow,
                      new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));
   fMenuBar->AddPopup(new TGHotString("&Help"), fMenuHelp,
                      new TGLayoutHints(kLHintsTop | kLHintsRight, 4, 4, 0, 0));
}

//______________________________________________________________________________
void TRootGuiBuilder::ChangeSelected(TGFrame *f)
{
   //

   fSelected = f;
   Update();
}

//______________________________________________________________________________
void TRootGuiBuilder::EnableLassoButtons(Bool_t on)
{
   //

   TGButton *btn = 0;

   btn = fToolBar->GetButton(kUpAct);
   if (btn) {
      btn->SetState(!on ? kButtonDisabled : kButtonUp);
   }

   btn = fToolBar->GetButton(kDownAct);
   if (btn) {
      btn->SetState(!on ? kButtonDisabled : kButtonUp);
   }

   btn = fToolBar->GetButton(kRightAct);
   if (btn) {
      btn->SetState(!on ? kButtonDisabled : kButtonUp);
   }

   btn = fToolBar->GetButton(kLeftAct);
   if (btn) {
      btn->SetState(!on ? kButtonDisabled : kButtonUp);
   }

   btn = fToolBar->GetButton(kGrabAct);
   if (btn) {
      btn->SetState(kButtonUp);
   }
}

//______________________________________________________________________________
void TRootGuiBuilder::EnableSelectedButtons(Bool_t on)
{
   //

   fSelected = fManager->GetSelected();

   if (!fSelected) {
      return;
   }

   TGButton *btn = 0;
   Bool_t comp = kFALSE;
   TGLayoutManager *lm = 0;
   Bool_t hor = kFALSE;
   Bool_t fixed = kFALSE;
   Bool_t enable = on;
   Bool_t compact_disable = kTRUE;

   if (fSelected->InheritsFrom(TGCompositeFrame::Class())) {
      lm = ((TGCompositeFrame*)fSelected)->GetLayoutManager();
      comp = kTRUE;
      hor = lm && lm->InheritsFrom(TGHorizontalLayout::Class());
      fixed = !fManager->CanChangeLayout(fSelected);
      compact_disable = !fManager->CanCompact(fSelected);
   } else {
      enable = kFALSE;
   }

   btn = fToolBar->GetButton(kCompactAct);
   if (btn) btn->SetState(enable && comp && !fixed && !compact_disable ? 
                          kButtonUp : kButtonDisabled);

   btn = fToolBar->GetButton(kLayoutHAct);
   if (btn) {
      btn->SetState(enable && comp && !hor && !fixed ? kButtonUp : kButtonDisabled);
   }

   btn = fToolBar->GetButton(kLayoutVAct);
   if (btn) {
      btn->SetState(enable && comp && hor && !fixed ? kButtonUp : kButtonDisabled);
   }

   btn = fToolBar->GetButton(kBreakLayoutAct);
   if (btn) {
      btn->SetState(enable && comp && !fixed ? kButtonUp : kButtonDisabled);
   }
/*
   btn = fToolBar->GetButton(kGrabAct);
   if (btn) {
      btn->SetState(enable && comp ? kButtonDown : kButtonUp);
      TGToolTip *tt = btn->GetToolTip();
      tt->SetText(btn->IsDown() ? "Drop Frames (Ctrl-Return)" :
                                  "Grab Selected Frames (Return)");
   }
*/
}

//______________________________________________________________________________
void TRootGuiBuilder::EnableEditButtons(Bool_t on)
{
   //

   TGButton *btn = 0;

   btn = fToolBar->GetButton(kReplaceAct);
   if (btn) {
      btn->SetState(!on ? kButtonDisabled : kButtonUp);
   }

   btn = fToolBar->GetButton(kGridAct);
   if (btn) {
      btn->SetState(!on ? kButtonDisabled : kButtonUp);
   }

   btn = fToolBar->GetButton(kCutAct);
   if (btn) {
      btn->SetState(!on ? kButtonDisabled : kButtonUp);
   }

   btn = fToolBar->GetButton(kDropAct);
   if (btn) {
      btn->SetState(!on ? kButtonDisabled : kButtonUp);
   }

   btn = fToolBar->GetButton(kCopyAct);
   if (btn) {
      btn->SetState(!on ? kButtonDisabled : kButtonUp);
   }

   btn = fToolBar->GetButton(kPasteAct);
   if (btn) {
      btn->SetState(!on ? kButtonDisabled : kButtonUp);
   }

   btn = fToolBar->GetButton(kCropAct);
   if (btn) {
      btn->SetState(!on ? kButtonDisabled : kButtonUp);
   }

   btn = fToolBar->GetButton(kDeleteAct);
   if (btn) {
      btn->SetState(!on ? kButtonDisabled : kButtonUp);
   }
}

//______________________________________________________________________________
void TRootGuiBuilder::Update()
{
   //

   if (!fManager) {
      return;
   }

   EnableLassoButtons(fManager->IsLassoDrawn());
   fSelected = fManager->GetSelected();
   EnableSelectedButtons(fSelected);
   EnableEditButtons(fClient->IsEditable());

   if (fActionButton) {
      TGFrame *parent = (TGFrame*)fActionButton->GetParent();
      parent->ChangeOptions(parent->GetOptions() & ~kSunkenFrame);
      fClient->NeedRedraw(parent, kTRUE);
   }

   if (!fClient->IsEditable()) {
      UpdateStatusBar("");
      fMenuFile->EnableEntry(kGUIBLD_FILE_START);
      fMenuFile->DisableEntry(kGUIBLD_FILE_STOP);
      fEditable = 0;
      //fShutter->SetSelectedItem(fShutter->GetItem("Projects"));
   } else {
      fMenuFile->DisableEntry(kGUIBLD_FILE_START);
      fMenuFile->EnableEntry(kGUIBLD_FILE_STOP);
   }

   SwitchToolbarButton();
   fActionButton = 0;
}

//______________________________________________________________________________
Bool_t TRootGuiBuilder::IsSelectMode() const
{
   //

   TGButton *btn = 0;
   btn = fToolBar->GetButton(kSelectAct);

   if (!btn) return kFALSE;

   return btn->IsDown();
}

//______________________________________________________________________________
Bool_t TRootGuiBuilder::IsGrabButtonDown() const
{
   //

   TGButton *btn = fToolBar->GetButton(kGrabAct);

   if (!btn) return kFALSE;

   return btn->IsDown();
}

////////////////////////////////////////////////////////////////////////////////
class TGuiBldSaveFrame : public TGMainFrame {

public:
   TGuiBldSaveFrame(const TGWindow *p, UInt_t w , UInt_t h) : TGMainFrame(p, w, h) {}
   void SetList(TList *li) { fList = li; }
};

static const char *gSaveMacroTypes[] = { "Macro files", "*.C",
                                         "All files",   "*",
                                         0,             0 };

//______________________________________________________________________________
Bool_t TRootGuiBuilder::HandleKey(Event_t *event)
{
   // keys handling

   if (event->fType == kGKeyPress) {
      UInt_t keysym;
      char str[2];
      gVirtualX->LookupString(event, str, sizeof(str), keysym);

      if (event->fState & kKeyControlMask) {
         if (str[0] == 19) {  // ctrl-s
            if (fMain->GetCurrent()) {
               return SaveProject(event);
            } else {
               return kFALSE; //TGMainFrame::HandleKey(event);
            }
         } else if (str[0] == 14) { //ctrl-n
            return NewProject(event);
         } else if (str[0] == 15) { // ctrl-o
            return OpenProject(event);
         }
      }
      fManager->HandleKey(event);
      return TGMainFrame::HandleKey(event);
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootGuiBuilder::NewProject(Event_t *)
{
   // create a new project

   TGWindow *root = (TGWindow*)fClient->GetRoot();

   if (root) root->SetEditable(kFALSE);
   fEditable = new TGMdiFrame(fMain, 500, 400, kOwnBackground);
   fEditable->SetMdiHints(kMdiDefaultHints);
   fEditable->SetWindowName(fEditable->GetName());
   fEditable->SetEditDisabled(0);   // enable editting
   fEditable->MapRaised();
   fEditable->AddInput(kKeyPressMask | kButtonPressMask);
   fEditable->SetEditable(kTRUE);
   fManager->SetEditable(kTRUE);
   fMenuFile->EnableEntry(kGUIBLD_FILE_CLOSE);
   fMenuFile->EnableEntry(kGUIBLD_FILE_STOP);
   fEditable->SetCleanup(kDeepCleanup);
   fEditable->SetLayoutBroken(kTRUE);

   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootGuiBuilder::OpenProject(Event_t *event)
{
   // open new gui builder project

   TGFileInfo fi;
   static TString dir(".");
   static Bool_t overwr = kFALSE;
   const char *fname;

   fi.fFileTypes = gSaveMacroTypes;
   fi.fIniDir    = StrDup(dir);
   fi.fOverwrite = overwr;
   TGWindow *root = (TGWindow*)fClient->GetRoot();
   root->SetEditable(kFALSE);
   SetEditable(kFALSE);

   new TGFileDialog(fClient->GetDefaultRoot(), this, kFDOpen, &fi);

   if (!fi.fFilename) {
      root->SetEditable(kTRUE);
      SetEditable(kTRUE);
      return kFALSE;
   }

   dir = fi.fIniDir;
   overwr = fi.fOverwrite;
   fname = gSystem->UnixPathName(fi.fFilename);

   if (strstr(fname, ".C")) {
      NewProject();
      gROOT->Macro(fname);
   } else {
      Int_t retval;
      new TGMsgBox(fClient->GetDefaultRoot(), this, "Error...",
                   Form("file (%s) must have extension .C", fname),
                   kMBIconExclamation, kMBRetry | kMBCancel, &retval);

      if (retval == kMBRetry) {
         HandleKey(event);
      }
   }
   root->SetEditable(kTRUE);
   SetEditable(kTRUE);
   fMenuFile->EnableEntry(kGUIBLD_FILE_CLOSE);
   fMenuFile->EnableEntry(kGUIBLD_FILE_STOP);
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TRootGuiBuilder::SaveProject(Event_t *event)
{
   //

   TGMdiFrame *savfr = fMain->GetCurrent();
   if (!savfr) return kFALSE;

   TGWindow *root = (TGWindow*)fClient->GetRoot();
   TGFileInfo fi;
   static TString dir(".");
   static Bool_t overwr = kFALSE;
   const char *fname;
   root->SetEditable(kFALSE);

   fi.fFileTypes = gSaveMacroTypes;
   fi.fIniDir    = StrDup(dir);
   fi.fOverwrite = overwr;

   new TGFileDialog(fClient->GetDefaultRoot(), this, kFDSave, &fi);

   if (!fi.fFilename) {
      root->SetEditable(kTRUE);
      SetEditable(kTRUE);
      return kFALSE;
   }

   dir = fi.fIniDir;
   overwr = fi.fOverwrite;
   fname = gSystem->UnixPathName(fi.fFilename);

   if (strstr(fname, ".C")) {
      TGuiBldSaveFrame *main = new TGuiBldSaveFrame(fClient->GetDefaultRoot(),
                                                    savfr->GetWidth(),
                                                    savfr->GetHeight());
      TList *list = main->GetList();
      TString name = savfr->GetName();
      savfr->SetName(main->GetName());
      main->SetList(savfr->GetList());

      main->SetLayoutBroken(savfr->GetDefaultWidth() != savfr->GetWidth());
      main->SaveSource(fname, "");

      main->SetList(list);
      savfr->SetName(name.Data());
      delete main;
   } else {
      Int_t retval;
      new TGMsgBox(fClient->GetDefaultRoot(), this, "Error...",
                   Form("file (%s) must have extension .C", fname),
                   kMBIconExclamation, kMBRetry | kMBCancel, &retval);
      if (retval == kMBRetry) {
         HandleKey(event);
      }
   }
   //root->SetEditable(kTRUE);

   return kTRUE;
}

//______________________________________________________________________________
TGMdiFrame *TRootGuiBuilder::FindEditableMdiFrame(const TGWindow *win)
{
   //

   const TGWindow *parent = win;

   while (parent && (parent != fClient->GetDefaultRoot())) {
      if (parent->InheritsFrom(TGMdiFrame::Class())) {
         fEditable = (TGMdiFrame*)parent;
         return fEditable;
      }
      parent = parent->GetParent();
   }
   return 0;
}

//______________________________________________________________________________
void TRootGuiBuilder::SwitchToolbarButton()
{
   //

   static const TGPicture *start = fClient->GetPicture("bld_edit.xpm");
   static const TGPicture *stop = fClient->GetPicture("bld_stop.xpm");

   if (fClient->IsEditable()) {
      fStartButton->SetPicture(stop);
      fToolBar->SetId(fStartButton, kEndEditAct);
      fStartButton->SetToolTipText("Stop Edit (Ctrl-Dbl-Click)");
   } else {
      fStartButton->SetPicture(start);
      fToolBar->SetId(fStartButton, kEditableAct);
      fStartButton->SetToolTipText("Start Edit (Ctrl-Dbl-Click)");
   }

   fClient->NeedRedraw(fStartButton, kTRUE);
}

//______________________________________________________________________________
void TRootGuiBuilder::HandleMenu(Int_t id)
{
   // Handle menu items.

   TGWindow *root = (TGWindow*)fClient->GetRoot();
   TRootHelpDialog *hd;

   switch (id) {
      case kGUIBLD_FILE_START:
         fEditable = fMain->GetCurrent();
         if (fEditable) {
            fEditable->SetEditable(kTRUE);
         }
         UpdateStatusBar("Start edit");
         fMenuFile->EnableEntry(kGUIBLD_FILE_STOP);
         fMenuFile->DisableEntry(kGUIBLD_FILE_START);
         SwitchToolbarButton();
         break;

      case kGUIBLD_FILE_STOP:
         fEditable = FindEditableMdiFrame(root);
         if (fEditable) {
            root->SetEditable(kFALSE);

            UpdateStatusBar("Stop edit");
            fMenuFile->EnableEntry(kGUIBLD_FILE_START);
            fMenuFile->DisableEntry(kGUIBLD_FILE_STOP);
            fEditable = 0;
            SwitchToolbarButton();
         }
         break;

      case kGUIBLD_FILE_NEW:
         NewProject();
         break;

      case kGUIBLD_FILE_CLOSE:
         fEditable = FindEditableMdiFrame(root);
         if (fEditable && (fEditable == fMain->GetCurrent())) {
            root->SetEditable(kFALSE);
         }
         fEditor->Reset();
         UpdateStatusBar("");
         fMain->Close(fMain->GetCurrent());

         if (fMain->GetNumberOfFrames() <= 1) {
            fMenuFile->DisableEntry(kGUIBLD_FILE_CLOSE);
            fMenuFile->DisableEntry(kGUIBLD_FILE_STOP);
            fMenuFile->DisableEntry(kGUIBLD_FILE_START);
            break;
         }

         if (fClient->IsEditable()) {
            fMenuFile->DisableEntry(kGUIBLD_FILE_START);
            fMenuFile->EnableEntry(kGUIBLD_FILE_STOP);
         } else {
            fMenuFile->EnableEntry(kGUIBLD_FILE_START);
            fMenuFile->DisableEntry(kGUIBLD_FILE_STOP);
         }

         break;

      case kGUIBLD_FILE_EXIT:
         CloseWindow();
         break;

      case kGUIBLD_WINDOW_HOR:
         fMain->TileHorizontal();
         break;

      case kGUIBLD_WINDOW_VERT:
         fMain->TileVertical();
         break;

      case kGUIBLD_WINDOW_CASCADE:
         fMain->Cascade();
         break;

      case kGUIBLD_WINDOW_ARRANGE:
         fMain->ArrangeMinimized();
         break;

      case kGUIBLD_WINDOW_OPAQUE:
         if (fMenuWindow->IsEntryChecked(kGUIBLD_WINDOW_OPAQUE)) {
            fMenuWindow->UnCheckEntry(kGUIBLD_WINDOW_OPAQUE);
            fMain->SetResizeMode(kMdiNonOpaque);
         } else {
            fMenuWindow->CheckEntry(kGUIBLD_WINDOW_OPAQUE);
            fMain->SetResizeMode(kMdiOpaque);
         }
         break;
      case  kGUIBLD_HELP_CONTENTS:
         root->SetEditable(kFALSE);
         hd = new TRootHelpDialog(this, "Help on Gui Builder...", 600, 400);
         hd->SetText(gHelpBuilder);
         hd->SetEditDisabled();
         hd->Popup();
         root->SetEditable(kTRUE);
         break;

      case  kGUIBLD_HELP_ABOUT:
         root->SetEditable(kFALSE);
         hd = new TRootHelpDialog(this, "About Gui Builder...", 520, 160);
         hd->SetEditDisabled();
         hd->SetText(gHelpAboutBuilder);
         hd->Popup();
         root->SetEditable(kTRUE);
         break;

      default:
         fMain->SetCurrent(id);
         break;
   }
}

//______________________________________________________________________________
void TRootGuiBuilder::HandleWindowClosed(Int_t )
{
   //

   fEditable = 0;

   if (fClient->IsEditable()) {
      fManager->SetEditable(kFALSE);
      fMenuFile->DisableEntry(kGUIBLD_FILE_START);
      fMenuFile->EnableEntry(kGUIBLD_FILE_STOP);
   } else {
      fMenuFile->EnableEntry(kGUIBLD_FILE_START);
      fMenuFile->DisableEntry(kGUIBLD_FILE_STOP);
   }
   fEditor->Reset();
   UpdateStatusBar("");

   if (fMain->GetNumberOfFrames() == 0) {
      fMenuFile->DisableEntry(kGUIBLD_FILE_CLOSE);
      fMenuFile->DisableEntry(kGUIBLD_FILE_STOP);
      fMenuFile->DisableEntry(kGUIBLD_FILE_START);
      return;
   }
}

//______________________________________________________________________________
void TRootGuiBuilder::UpdateStatusBar(const char *txt)
{
   //

   if (!fStatusBar) return;

   const char *text = 0;

   if (!txt) {
      TObject *o = (TObject *)gTQSender;

      if (o && o->InheritsFrom(TGToolTip::Class())) {
         TGToolTip *tip = (TGToolTip*)o;
         text = tip->GetText()->Data();
      }
   } else {
      text = txt;
   }
   fStatusBar->SetText(text);
}

//______________________________________________________________________________
void TRootGuiBuilder::EraseStatusBar()
{
   //

   if (!fStatusBar) return;

   fStatusBar->SetText("");
}

//______________________________________________________________________________
void TRootGuiBuilder::BindKeys()
{
   //

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_a),
                      kKeyControlMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_n),
                      kKeyControlMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_o),
                      kKeyControlMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_Return),
                      kKeyControlMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_Return),
                      0, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_Enter),
                      kKeyControlMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_Enter),
                      0, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_x),
                      kKeyControlMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_c),
                      kKeyControlMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_v),
                      kKeyControlMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_r),
                      kKeyControlMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_z),
                      kKeyControlMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_z),
                      kKeyControlMask | kKeyShiftMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_b),
                      kKeyControlMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_l),
                      kKeyControlMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_g),
                      kKeyControlMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_h),
                      kKeyControlMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_Delete),
                      0, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_Backspace),
                      0, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_Space),
                      0, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_Left),
                      0, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_Right),
                      0, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_Up),
                      0, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_Down),
                      0, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_Left),
                      kKeyShiftMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_Right),
                      kKeyShiftMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_Up),
                      kKeyShiftMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_Down),
                      kKeyShiftMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_Delete),
                      kKeyShiftMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_Backspace),
                      kKeyShiftMask, kTRUE);
}

//______________________________________________________________________________
TGFrame *TRootGuiBuilder::VSplitter()
{
   // creates new TGVSplitter

   TGHorizontalFrame *ret = new TGHorizontalFrame();
   ret->SetCleanup(kDeepCleanup);
   TGVerticalFrame *v1 = new TGVerticalFrame(ret, 40, 10, kSunkenFrame |  kFixedWidth);
   ret->AddFrame(v1, new TGLayoutHints(kLHintsLeft | kLHintsExpandY));

   TGVSplitter *splitter = new TGVSplitter(ret);
   splitter->SetFrame(v1, kTRUE);
   ret->AddFrame(splitter, new TGLayoutHints(kLHintsLeft | kLHintsExpandY));

   TGVerticalFrame *v2 = new TGVerticalFrame(ret, 10, 10, kSunkenFrame);
   v2->ChangeOptions(kSunkenFrame);
   ret->AddFrame(v2, new TGLayoutHints(kLHintsRight | kLHintsExpandX | kLHintsExpandY));

   ret->MapSubwindows();
   ret->SetLayoutBroken(kFALSE);
   return ret;
}

//______________________________________________________________________________
TGFrame *TRootGuiBuilder::HSplitter()
{
   //  creates new TGHSplitter

   TGVerticalFrame *ret = new TGVerticalFrame();
   ret->SetCleanup(kDeepCleanup);
   TGHorizontalFrame *v1 = new TGHorizontalFrame(ret, 10, 40, kSunkenFrame | kFixedHeight);
   ret->AddFrame(v1, new TGLayoutHints(kLHintsTop | kLHintsExpandX));

   TGHSplitter *splitter = new TGHSplitter(ret);
   splitter->SetFrame(v1, kTRUE);
   ret->AddFrame(splitter, new TGLayoutHints(kLHintsTop | kLHintsExpandX));

   TGHorizontalFrame *v2 = new TGHorizontalFrame(ret, 10, 10);
   v2->ChangeOptions(kSunkenFrame);
   ret->AddFrame(v2, new TGLayoutHints(kLHintsBottom | kLHintsExpandX | kLHintsExpandY));

   ret->MapSubwindows();
   ret->SetLayoutBroken(kFALSE);
   return ret;
}

//______________________________________________________________________________
void TRootGuiBuilder::Hide()
{
   // cleanup and hide gui builder

   //fMain->CloseAll();
   UnmapWindow();
}



// @(#)root/guibuilder:$Name:  $:$Id: TGuiBuilder.cxx,v 1.5 2004/09/20 21:00:40 brun Exp $
// Author: Valeriy Onuchin   12/09/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGuiBuilder                                                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGuiBuilder.h"
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
#include "TGToolTip.h"


enum EMenuIds {
   M_FILE_NEW,
   M_FILE_CLOSE,
   M_FILE_EXIT,

   M_WINDOW_HOR,
   M_WINDOW_VERT,
   M_WINDOW_CASCADE,
   M_WINDOW_OPAQUE,
   M_WINDOW_ARRANGE,

   M_HELP_CONTENTS,
   M_HELP_ABOUT
};

const char gHelpBuilder[] = "\
 o Ctrl-Double-Click - Start/Stop edit mode\n\
\n\
               Selection, grabbing, dropping\n\
     ************************************************\n\
 o Use left mouse button Click or Ctrl-Click to select an object to edit.\n\
 o Use right mouse button to activate context menu\n\
 o Mutilple selection (grabbing):\n\
      - draw lasso and press Return key\n\
      - press Shift key and draw lasso\n\
 o Dropping:\n\
      - select frame and press Ctrl-Return key\n\
 o Changing layout order:\n\
      - select frame and use arrow keys to change layout order\n\
 o Alignment:\n\
      - draw lasso and use arrow keys (or Shift-Arrow key) to align frames\n\
\n\
                    Key shortcuts\n\
     ************************************************\n\
 o Ctrl-X - Cut\n\
 o Ctrl-C - Copy\n\
 o Ctrl-V - Paste\n\
 o Ctrl-R - Replace\n\
 o Ctrl-L - Compact Layout\n\
 o Ctrl-B - Break Layout\n\
 o Ctrl-H - Switch Horizontal-Vertical Layout\n\
 o Ctrl-G - Switch ON/OFF Grid\n\
 o Ctrl-S - Save\n\
 o Ctrl-O - Open ROOT macro file\n\
 o Ctrl-N - new TGMainFrame\n\
 o Ctrl-Z - Undo (not implemented)\n\
 o Shift-Ctrl-Z - Redo (not implemented)\n\
";


//----- Toolbar stuff...

static ToolBarData_t gToolBarData[] = {
   { "bld_new.xpm",   "New (Ctrl-N)",   kFALSE, kNewAct, 0 },
   { "bld_open.xpm",   "Open (Ctrl-O)",   kFALSE, kOpenAct, 0 },
   { "bld_save.xpm",   "Save (Ctrl-S)",   kFALSE, kSaveAct, 0 },
   { "",                 "",               kFALSE, -1, 0 },
   { "bld_pointer.xpm",   "Selector (Ctrl-Click)",   kTRUE, kSelectAct, 0 },
   { "bld_grab.xpm",   "Grab Selected Frames (Return)",   kTRUE, kGrabAct, 0 },
   { "",                 "",               kFALSE, -1, 0 },
   { "bld_hbox.xpm",  "Lay Out Horizontally (Ctrl-H)",    kFALSE,  kLayoutHAct, 0 },
   { "bld_vbox.xpm",   "Lay Out Vertically (Ctrl-H)",    kFALSE,  kLayoutVAct, 0 },
   { "bld_grid.xpm",   "Lay Out in a Grid (Ctrl+G)",     kFALSE,  kGridAct, 0 },
   { "bld_layout.xpm",   "Compact Layout (Ctrl-L)",        kFALSE,  kCompactAct, 0 },
   { "bld_break.xpm",   "Break Layout (Ctrl-B)",        kFALSE,  kBreakLayoutAct, 0 },
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
   { "",                 "",               kFALSE, -1, 0 },
   { "bld_undo.xpm",   "Undo (Ctrl-Z)",        kFALSE,  kUndoAct, 0 },
   { "bld_redo.xpm",   "Redo (Shift-Ctrl-Z)",        kFALSE,  kRedoAct, 0 },
   { 0,                  0,                kFALSE, 0, 0 }
};


ClassImp(TGuiBuilder)

////////////////////////////////////////////////////////////////////////////////
class TGuiBuilderContainer : public TGMdiContainer {

public:
   TGuiBuilderContainer(const TGMdiMainFrame *p) : 
         TGMdiContainer(p, 10, 10, kOwnBackground) {
      const TGPicture *pbg = fClient->GetPicture("bld_bg.xpm");
      if (pbg) SetBackgroundPixmap(pbg->GetPicture());
      //SetEditDisabled(kFALSE);
   }
   virtual ~TGuiBuilderContainer() {}
   void SetEditable(Bool_t) {}
   //Bool_t HandleDoubleClick(Event_t *) { printf("qq 1\n"); return kFALSE; }
   //Bool_t HandleEvent(Event_t *) { return kFALSE; }
}; 

////////////////////////////////////////////////////////////////////////////////
TGuiBuilder::TGuiBuilder(const TGWindow *p) : TVirtualGuiBld(),
             TGMainFrame(p ? p : gClient->GetDefaultRoot(), 1, 1)
{
   // ctor

   SetCleanup(kTRUE);
   fEditDisabled = kTRUE;

   if (gDragManager) {
      fManager = (TGuiBldDragManager *)gDragManager;
      fManager->SetBuilder(this);
   }

   fMenuBar = new TGMdiMenuBar(this, 10, 10);
   AddFrame(fMenuBar, new TGLayoutHints(kLHintsTop | kLHintsExpandX));
   InitMenu();

   TGHorizontal3DLine *hl = new TGHorizontal3DLine(this);
   AddFrame(hl, new TGLayoutHints(kLHintsTop | kLHintsExpandX, 0,0,2,2));

   fToolDock = new TGDockableFrame(this);
   AddFrame(fToolDock, new TGLayoutHints(kLHintsExpandX, 0, 0, 1, 0));
   fToolDock->SetWindowName("ROOT GuiBuilder ToolBar");

   fToolBar = new TGToolBar(this, 60, 20, kHorizontalFrame);
   fToolDock->AddFrame(fToolBar, new TGLayoutHints(kLHintsTop | kLHintsExpandX));

   int spacing = 8;

   for (int i = 0; gToolBarData[i].fPixmap; i++) {
      if (strlen(gToolBarData[i].fPixmap) == 0) {
         spacing = 8;
         continue;
      }
      TGPictureButton *pb = (TGPictureButton*)fToolBar->AddButton(this, &gToolBarData[i], spacing);
      TGToolTip *tip = pb->GetToolTip();
      if (tip) {
         tip->Connect("Reset()", "TGuiBuilder", this, "UpdateStatusBar()");
         tip->Connect("Hide()", "TGuiBuilder", this, "EraseStatusBar()");
      }

      TString pname = gToolBarData[i].fPixmap;
      pname.ReplaceAll(".", "_d.");
      const TGPicture *dpic = fClient->GetPicture(pname.Data());
      if (dpic) pb->SetDisabledPicture(dpic);

      if ((gToolBarData[i].fId == kUndoAct) || (gToolBarData[i].fId == kRedoAct) ||
          (gToolBarData[i].fId == kCropAct)) {
         pb->SetState(kButtonDisabled);
      }

      spacing = 0;
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

   fMain = new TGMdiMainFrame(cf, fMenuBar, 1, 1);
   cf->AddFrame(fMain, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   delete fMain->GetContainer();
   fMain->SetContainer(new TGuiBuilderContainer(fMain));

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
   AddSection("Standard");
   AddSection("Containers");
//   AddSection("Extended");

   TGuiBldAction *act = new TGuiBldAction("TGMainFrame", "Main Frame", kGuiBldProj);
   act->fAct = "new TGMainFrame(gClient->GetRoot(), 300, 300)";
   act->fPic = "bld_mainframe.xpm";
   AddAction(act, "Projects");

   // Standard
   act = new TGuiBldAction("TGTextButton", "Text Button", kGuiBldCtor);
   act->fAct = "new TGTextButton()";
   act->fPic = "bld_textbutton.xpm";
   AddAction(act, "Standard");

   act = new TGuiBldAction("TGCheckButton", "Check Button", kGuiBldCtor);
   act->fAct = "new TGCheckButton()";
   act->fPic = "bld_checkbutton.xpm";
   AddAction(act, "Standard");

   act = new TGuiBldAction("TGRadioButton", "Radio Button", kGuiBldCtor);
   act->fAct = "new TGRadioButton()";
   act->fPic = "bld_radiobutton.xpm";
   AddAction(act, "Standard");

   act = new TGuiBldAction("TGTextEntry", "Text Entry", kGuiBldCtor);
   act->fAct = "new TGTextEntry()";
   act->fPic = "bld_entry.xpm";
   AddAction(act, "Standard");

   act = new TGuiBldAction("TGNumberEntry", "Number Entry", kGuiBldCtor);
   act->fAct = "new TGNumberEntry()";
   act->fPic = "bld_numberentry.xpm";
   AddAction(act, "Standard");

   act = new TGuiBldAction("TGLabel", "Text Label", kGuiBldCtor);
   act->fAct = "new TGLabel()";
   act->fPic = "bld_label.xpm";
   AddAction(act, "Standard");

   act = new TGuiBldAction("TGHorizontal3DLine", "Horizontal Line", kGuiBldCtor);
   act->fAct = "new TGHorizontal3DLine()";
   act->fPic = "bld_hseparator.xpm";
   AddAction(act, "Standard");

   act = new TGuiBldAction("TGVertical3DLine", "Vertical Line", kGuiBldCtor);
   act->fAct = "new TGVertical3DLine()";
   act->fPic = "bld_vseparator.xpm";
   AddAction(act, "Standard");

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

   fShutter->Resize(140, fShutter->GetHeight());

   fStatusBar = new TGStatusBar(this, 40, 10);
   AddFrame(fStatusBar, new TGLayoutHints(kLHintsBottom | kLHintsExpandX, 0, 0, 3, 0));

   MapSubwindows();
   Resize(900, 600);

   SetWindowName("ROOT GuiBuilder");
   SetIconName("ROOT GuiBuilder");
   SetIconPixmap("bld_rgb.xpm");
   SetClassHints("GuiBuilder", "GuiBuilder");

   fSelected = 0;
   Update();

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_n),
                      kKeyControlMask, kTRUE);

   gVirtualX->GrabKey(fId, gVirtualX->KeysymToKeycode(kKey_o),
                      kKeyControlMask, kTRUE);

   fMenuFile->Connect("Activated(Int_t)", "TGuiBuilder", this,
                      "HandleMenu(Int_t)");
   fMenuWindow->Connect("Activated(Int_t)", "TGuiBuilder", this,
                        "HandleMenu(Int_t)");
   fMenuHelp->Connect("Activated(Int_t)", "TGuiBuilder", this,
                      "HandleMenu(Int_t)");
 
   fMain->Connect("FrameClosed(Int_t)", "TGuiBuilder", this, "HandleWindowClosed(Int_t)");

   MapRaised();
}

//______________________________________________________________________________
TGuiBuilder::~TGuiBuilder()
{
   // destructor

}

//______________________________________________________________________________
void TGuiBuilder::AddAction(TGuiBldAction *act, const char *sect)
{
   //

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
   btn->Connect("Clicked()", "TGuiBuilder", this, "HandleButtons()");

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
void TGuiBuilder::AddSection(const char *sect)
{
   //

   static int id = 10000;
   TGShutterItem *item = new TGShutterItem(fShutter, new TGHotString(sect), id++);
   fShutter->AddItem(item);
}

//______________________________________________________________________________
void TGuiBuilder::HandleButtons()
{
   //

   TGButton *btn = (TGButton *)gTQSender;
   TGuiBldAction *act  = (TGuiBldAction *)btn->GetUserData();

   if (act) {
      fAction = act;
      if (fAction->fType == kGuiBldProj) ExecuteAction();
   }
}

//______________________________________________________________________________
TGFrame *TGuiBuilder::ExecuteAction()
{
   //

   if (!fAction || fAction->fAct.IsNull()) return 0;

   TGFrame *ret = 0;

   switch (fAction->fType) {
      case kGuiBldProj:
         NewProject();
         break;
      default:
         ret = (TGFrame *)gROOT->ProcessLineFast(fAction->fAct.Data());
         break;
   }

   fAction = 0;
   Update();

   return ret;
}

//______________________________________________________________________________
void TGuiBuilder::InitMenu()
{
   //

   fMenuFile = new TGPopupMenu(fClient->GetDefaultRoot());
   fMenuFile->AddEntry(new TGHotString("&New Window"), M_FILE_NEW);
   fMenuFile->AddEntry(new TGHotString("&Close Window"), M_FILE_CLOSE);
   fMenuFile->AddSeparator();
   fMenuFile->AddEntry(new TGHotString("E&xit"), M_FILE_EXIT);

   fMenuWindow = new TGPopupMenu(fClient->GetDefaultRoot());
   fMenuWindow->AddEntry(new TGHotString("Tile &Horizontally"), M_WINDOW_HOR);
   fMenuWindow->AddEntry(new TGHotString("Tile &Vertically"), M_WINDOW_VERT);
   fMenuWindow->AddEntry(new TGHotString("&Cascade"), M_WINDOW_CASCADE);
   fMenuWindow->AddSeparator();
   //fMenuWindow->AddPopup(new TGHotString("&Windows"), fMain->GetWinListMenu());
   fMenuWindow->AddSeparator();
   fMenuWindow->AddEntry(new TGHotString("&Arrange icons"), M_WINDOW_ARRANGE);
   fMenuWindow->AddSeparator();
   fMenuWindow->AddEntry(new TGHotString("&Opaque resize"), M_WINDOW_OPAQUE);

   fMenuWindow->CheckEntry(M_WINDOW_OPAQUE);

   fMenuHelp = new TGPopupMenu(fClient->GetDefaultRoot());
   fMenuHelp->AddEntry(new TGHotString("&Contents"), M_HELP_CONTENTS);
   fMenuHelp->AddSeparator();
   fMenuHelp->AddEntry(new TGHotString("&About"), M_HELP_ABOUT);

   fMenuBar->AddPopup(new TGHotString("&File"), fMenuFile, 
                      new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));
   fMenuBar->AddPopup(new TGHotString("&Windows"), fMenuWindow,
                      new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0));
   fMenuBar->AddPopup(new TGHotString("&Help"), fMenuHelp, 
                      new TGLayoutHints(kLHintsTop | kLHintsRight, 4, 4, 0, 0));
}

//______________________________________________________________________________
void TGuiBuilder::ChangeSelected(TGFrame *f)
{
   //

   fSelected = f;
   Update();
}

//______________________________________________________________________________
void TGuiBuilder::EnableLassoButtons(Bool_t on)
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
void TGuiBuilder::EnableSelectedButtons(Bool_t on)
{
   //

   TGButton *btn = 0;
   Bool_t comp = kFALSE;
   TGLayoutManager *lm = 0;
   Bool_t hor = kFALSE;

   if (fSelected && fSelected->InheritsFrom(TGCompositeFrame::Class())) {
      lm = ((TGCompositeFrame*)fSelected)->GetLayoutManager();
      comp = kTRUE;
      hor = lm && lm->InheritsFrom(TGHorizontalLayout::Class());
   }

   btn = fToolBar->GetButton(kCompactAct);
   if (btn) btn->SetState(on && comp ? kButtonUp : kButtonDisabled);

   btn = fToolBar->GetButton(kLayoutHAct);
   if (btn) {
      btn->SetState(on && comp && !hor ? kButtonUp : kButtonDisabled);
   }

   btn = fToolBar->GetButton(kLayoutVAct);
   if (btn) {
      btn->SetState(on && comp && hor ? kButtonUp : kButtonDisabled);
   }

   btn = fToolBar->GetButton(kBreakLayoutAct);
   if (btn) {
      btn->SetState(on && comp ? kButtonUp : kButtonDisabled);
   }

   btn = fToolBar->GetButton(kGrabAct);
   if (btn) {
      btn->SetState(on && comp ? kButtonDown : kButtonUp);
      TGToolTip *tt = btn->GetToolTip();
      tt->SetText(btn->IsDown() ? "Drop Frames (Ctrl-Return)" : 
                                  "Grab Selected Frames (Return)");
   }
}

//______________________________________________________________________________
void TGuiBuilder::EnableEditButtons(Bool_t on)
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

   /*btn = fToolBar->GetButton(kCropAct);
   if (btn) {
      btn->SetState(!on ? kButtonDisabled : kButtonUp);
   }*/

   btn = fToolBar->GetButton(kDeleteAct);
   if (btn) {
      btn->SetState(!on ? kButtonDisabled : kButtonUp);
   }
}

//______________________________________________________________________________
void TGuiBuilder::Update()
{
   //

   EnableLassoButtons(fManager && fManager->IsLassoDrawn());
   EnableSelectedButtons(fManager && (fSelected = fManager->GetSelected()));
   EnableEditButtons(fClient->IsEditable());
}

//______________________________________________________________________________
Bool_t TGuiBuilder::IsSelectMode() const
{
   //

   TGButton *btn = 0;
   btn = fToolBar->GetButton(kSelectAct);

   if (!btn) return kFALSE;

   return btn->IsDown();
}

//______________________________________________________________________________
Bool_t TGuiBuilder::IsGrabButtonDown() const
{
   //

   TGButton *btn = fToolBar->GetButton(kGrabAct);

   if (!btn) return kFALSE;

   return btn->IsDown();
}

class TGuiBldSaveFrame : public TGMainFrame {

public:
   TGuiBldSaveFrame(const TGWindow *p, UInt_t w , UInt_t h) : TGMainFrame(p, w, h) {}
   void SetList(TList *li) { fList = li; }
};

static const char *gSaveMacroTypes[] = { "Macro files", "*.C",
                                         "All files",   "*",
                                         0,             0 };

//______________________________________________________________________________
Bool_t TGuiBuilder::HandleKey(Event_t *event)
{
   //

   fEditable = FindEditableMdiFrame(fClient->GetRoot());

   if ((event->fType == kGKeyPress) && (event->fState & kKeyControlMask)) {
      UInt_t keysym;
      char str[2];
      gVirtualX->LookupString(event, str, sizeof(str), keysym);

      if ((str[0] == 19) && fEditable) {  // ctrl-s
         SaveProject(event);
      } else if (str[0] == 14) { //ctrl-n
         NewProject(event);
      } else if (str[0] == 15) { // ctrl-o
         OpenProject(event);
      }
   }
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGuiBuilder::NewProject(Event_t *)
{
   //

   TGWindow *root = (TGWindow*)fClient->GetRoot();

   root->SetEditable(kFALSE);
   fEditable = new TGMdiFrame(fMain, 300, 300, kOwnBackground);
   fEditable->SetMdiHints(kMdiDefaultHints);
   fEditable->SetWindowName(fEditable->GetName());
   fEditable->SetEditDisabled(kFALSE);
   fEditable->MapRaised();
   fEditable->AddInput(kButtonPressMask);
   fEditable->SetEditable(kTRUE);
   fEditable->AddInput(kKeyPressMask);

   return kTRUE;
}

class TGuiBldFileDialog : public TGFileDialog
{
public:
   TGuiBldFileDialog(const TGWindow *p = 0, const TGWindow *main = 0,
                        EFileDialogMode dlg_type = kFDOpen, TGFileInfo *file_info = 0) :
                        TGFileDialog(p, main, dlg_type, file_info)
   {
      fEditDisabled = kTRUE;
   }
};

//______________________________________________________________________________
Bool_t TGuiBuilder::OpenProject(Event_t *event)
{
   //

   TGFileInfo fi;
   static TString dir(".");
   const char *fname;

   fi.fFileTypes = gSaveMacroTypes;
   fi.fIniDir    = StrDup(dir);
   TGWindow *root = (TGWindow*)fClient->GetRoot();
   root->SetEditable(kFALSE);

   new TGuiBldFileDialog(fClient->GetDefaultRoot(), this, kFDSave, &fi);

   if (!fi.fFilename) {
      root->SetEditable(kTRUE);
      return kFALSE;
  }

   dir = fi.fIniDir;
   fname = gSystem->BaseName(gSystem->UnixPathName(fi.fFilename));

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
   return kTRUE;
}

//______________________________________________________________________________
Bool_t TGuiBuilder::SaveProject(Event_t *event)
{
   //

   TGWindow *root = (TGWindow*)fClient->GetRoot();
   fEditable = FindEditableMdiFrame(root);

   if (!fEditable) return kFALSE;

   TGFileInfo fi;
   static TString dir(".");
   const char *fname;
   root->SetEditable(kFALSE);

   fi.fFileTypes = gSaveMacroTypes;
   fi.fIniDir    = StrDup(dir);

   new TGuiBldFileDialog(fClient->GetDefaultRoot(), this, kFDSave, &fi);
  
   if (!fi.fFilename) {
      root->SetEditable(kTRUE);
      return kFALSE;
  }

   dir = fi.fIniDir;
   fname = gSystem->BaseName(gSystem->UnixPathName(fi.fFilename));

   if (strstr(fname, ".C")) {
      TGuiBldSaveFrame *main = new TGuiBldSaveFrame(fClient->GetDefaultRoot(),
                                                    fEditable->GetWidth(),
                                                    fEditable->GetHeight());
      TList *list = main->GetList();
      TString name = fEditable->GetName();
      fEditable->SetName(main->GetName());
      main->SetList(fEditable->GetList());

      main->SetLayoutBroken(fEditable->IsLayoutBroken());
      main->SaveSource(fname, "");

      main->SetList(list);
      fEditable->SetName(name.Data());
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
   root->SetEditable(kTRUE);
   return kTRUE;
}

//______________________________________________________________________________
TGMdiFrame *TGuiBuilder::FindEditableMdiFrame(const TGWindow *win)
{
   //

   const TGWindow *parent = win;

   while (parent && (parent != fClient->GetDefaultRoot())) {
      if (parent->InheritsFrom(TGMdiFrame::Class())) {
         fEditable = (TGMdiFrame*)parent;
         return fEditable;
      }
   }
   return 0;
}

//______________________________________________________________________________
void TGuiBuilder::HandleMenu(Int_t id)
{
   // Handle menu items.

   TGWindow *root = (TGWindow*)fClient->GetRoot();
   TRootHelpDialog *hd;

   switch (id) {
      case M_FILE_NEW:
         NewProject();
         break;

      case M_FILE_CLOSE:
         fEditable = FindEditableMdiFrame(root);
         if (fEditable && (fEditable == fMain->GetCurrent())) {
            root->SetEditable(kFALSE);
         }
         fMain->Close(fMain->GetCurrent());
         break;

      case M_FILE_EXIT:
         CloseWindow();
         break;

      case M_WINDOW_HOR:
         fMain->TileHorizontal();
         break;

      case M_WINDOW_VERT:
         fMain->TileVertical();
         break;

      case M_WINDOW_CASCADE:
         fMain->Cascade();
         break;

      case M_WINDOW_ARRANGE:
         fMain->ArrangeMinimized();
         break;

      case M_WINDOW_OPAQUE:
         if (fMenuWindow->IsEntryChecked(M_WINDOW_OPAQUE)) {
            fMenuWindow->UnCheckEntry(M_WINDOW_OPAQUE);
            fMain->SetResizeMode(kMdiNonOpaque);
         } else {
            fMenuWindow->CheckEntry(M_WINDOW_OPAQUE);
            fMain->SetResizeMode(kMdiOpaque);
         }
         break;
      case  M_HELP_CONTENTS:
         root->SetEditable(kFALSE);
         hd = new TRootHelpDialog(this, "Help on Gui Builder...", 600, 400);
         hd->SetText(gHelpBuilder);
         hd->SetEditDisabled();
         hd->Popup();
         root->SetEditable(kTRUE);
         break;
      case  M_HELP_ABOUT:
         root->SetEditable(kFALSE);
         hd = new TRootHelpDialog(this, "About Gui Builder...", 400, 100);
         hd->SetEditDisabled();
         hd->SetText("         ROOT Gui Builder\n\
\n\
Author: Valeriy Onuchin (Valeri.Onoutchine@cern.ch)");
         hd->Popup();
         root->SetEditable(kTRUE);
         break;
      default:
         fMain->SetCurrent(id);
         break;
   }
}

//______________________________________________________________________________
void TGuiBuilder::HandleWindowClosed(Int_t id)
{
   //

   if (!fClient->IsEditable()) return;

   TGWindow *root = (TGWindow*)fClient->GetRoot();
   fEditable = FindEditableMdiFrame(root);
 
   if (id == (Int_t)fEditable->GetId()) {
      fManager->SetEditable(kFALSE);
      root->SetEditable(kFALSE);
   }
}

//______________________________________________________________________________
void TGuiBuilder::CloseWindow()
{
   //

   gApplication->Terminate(0);
}

//______________________________________________________________________________
void TGuiBuilder::UpdateStatusBar()
{
   //

   if (!fStatusBar) return;

   TObject *o = (TObject *)gTQSender;
   const char *text = 0;

   if (o && o->InheritsFrom(TGToolTip::Class())) {
      TGToolTip *tip = (TGToolTip*)o;
      text = tip->GetText()->Data(); 
   }

   fStatusBar->SetText(text);
}

//______________________________________________________________________________
void TGuiBuilder::EraseStatusBar()
{
   //

   if (!fStatusBar) return;

   fStatusBar->SetText("");
}

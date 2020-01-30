/// \file
/// \ingroup tutorial_gui
/// Test program for ROOT native GUI classes
/// Exactly like $ROOTSYS/test/guitest.cxx but using the new signal and slots communication mechanism.
/// It is now possible to run this entire test program in the interpreter.
/// Do either:
/// ~~~{.cpp}
/// .x guitest.C
/// .x guitest.C++
/// ~~~
///
/// \macro_code
///
/// \authors Ilka Antcheva, Bertrand Bellenot, Fons Rademakers, Valeri Onuchin

#include <stdlib.h>

#include <TROOT.h>
#include <TClass.h>
#include <TApplication.h>
#include <TVirtualX.h>
#include <TVirtualPadEditor.h>
#include <TGResourcePool.h>
#include <TGListBox.h>
#include <TGListTree.h>
#include <TGFSContainer.h>
#include <TGClient.h>
#include <TGFrame.h>
#include <TGIcon.h>
#include <TGLabel.h>
#include <TGButton.h>
#include <TGTextEntry.h>
#include <TGNumberEntry.h>
#include <TGMsgBox.h>
#include <TGMenu.h>
#include <TGCanvas.h>
#include <TGComboBox.h>
#include <TGTab.h>
#include <TGSlider.h>
#include <TGDoubleSlider.h>
#include <TGFileDialog.h>
#include <TGTextEdit.h>
#include <TGShutter.h>
#include <TGProgressBar.h>
#include <TGColorSelect.h>
#include <RQ_OBJECT.h>
#include <TRootEmbeddedCanvas.h>
#include <TCanvas.h>
#include <TColor.h>
#include <TH1.h>
#include <TH2.h>
#include <TRandom.h>
#include <TSystem.h>
#include <TSystemDirectory.h>
#include <TEnv.h>
#include <TFile.h>
#include <TKey.h>
#include <TGDockableFrame.h>
#include <TGFontDialog.h>


enum ETestCommandIdentifiers {
   M_FILE_OPEN,
   M_FILE_SAVE,
   M_FILE_SAVEAS,
   M_FILE_PRINT,
   M_FILE_PRINTSETUP,
   M_FILE_EXIT,

   M_TEST_DLG,
   M_TEST_MSGBOX,
   M_TEST_SLIDER,
   M_TEST_SHUTTER,
   M_TEST_DIRLIST,
   M_TEST_FILELIST,
   M_TEST_PROGRESS,
   M_TEST_NUMBERENTRY,
   M_TEST_FONTDIALOG,
   M_TEST_NEWMENU,

   M_VIEW_ENBL_DOCK,
   M_VIEW_ENBL_HIDE,
   M_VIEW_DOCK,
   M_VIEW_UNDOCK,

   M_HELP_CONTENTS,
   M_HELP_SEARCH,
   M_HELP_ABOUT,

   M_CASCADE_1,
   M_CASCADE_2,
   M_CASCADE_3,

   M_NEW_REMOVEMENU,

   VId1,
   HId1,
   VId2,
   HId2,

   VSId1,
   HSId1,
   VSId2,
   HSId2
};


Int_t mb_button_id[13] = { kMBYes, kMBNo, kMBOk, kMBApply,
                           kMBRetry, kMBIgnore, kMBCancel,
                           kMBClose, kMBYesAll, kMBNoAll,
                           kMBNewer, kMBAppend, kMBDismiss};

EMsgBoxIcon mb_icon[4] = { kMBIconStop, kMBIconQuestion,
                           kMBIconExclamation, kMBIconAsterisk };

const char *filetypes[] = { "All files",     "*",
                            "ROOT files",    "*.root",
                            "ROOT macros",   "*.C",
                            "Text files",    "*.[tT][xX][tT]",
                            0,               0 };

struct shutterData_t {
   const char *pixmap_name;
   const char *tip_text;
   Int_t       id;
   TGButton   *button;
};

shutterData_t histo_data[] = {
   { "h1_s.xpm",        "TH1",      1001,  0 },
   { "h2_s.xpm",        "TH2",      1002,  0 },
   { "h3_s.xpm",        "TH3",      1003,  0 },
   { "profile_s.xpm",   "TProfile", 1004,  0 },
   { 0,                 0,          0,     0 }
};

shutterData_t function_data[] = {
   { "f1_s.xpm",        "TF1",      2001,  0 },
   { "f2_s.xpm",        "TF2",      2002,  0 },
   { 0,                 0,          0,     0 }
};

shutterData_t tree_data[] = {
   { "ntuple_s.xpm",    "TNtuple",  3001,  0 },
   { "tree_s.xpm",      "TTree",    3002,  0 },
   { "chain_s.xpm",     "TChain",   3003,  0 },
   { 0,                 0,          0,     0 }
};


const char *editortxt1 =
"This is the ROOT text edit widget TGTextEdit. It is not intended as\n"
"a full developers editor, but it is relatively complete and can ideally\n"
"be used to edit scripts or to present users editable config files, etc.\n\n"
"The text edit widget supports standard emacs style ctrl-key navigation\n"
"in addition to the arrow keys. By default the widget has under the right\n"
"mouse button a popup menu giving access to several built-in functions.\n\n"
"Cut, copy and paste between different editor windows and any other\n"
"standard text handling application is supported.\n\n"
"Text can be selected with the mouse while holding the left button\n"
"or with the arrow keys while holding the shift key pressed. Use the\n"
"middle mouse button to paste text at the current mouse location."
;
const char *editortxt2 =
"Mice with scroll-ball are properly supported.\n\n"
"This are the currently defined key bindings:\n"
"Left Arrow\n"
"    Move the cursor one character leftwards.\n"
"    Scroll when cursor is out of frame.\n"
"Right Arrow\n"
"    Move the cursor one character rightwards.\n"
"    Scroll when cursor is out of frame.\n"
"Backspace\n"
"    Deletes the character on the left side of the text cursor and moves the\n"
"    cursor one position to the left. If a text has been marked by the user"
;
const char *editortxt3 =
"    (e.g. by clicking and dragging) the cursor will be put at the beginning\n"
"    of the marked text and the marked text will be removed.\n"
"Home\n"
"    Moves the text cursor to the left end of the line. If mark is TRUE text\n"
"    will be marked towards the first position, if not any marked text will\n"
"    be unmarked if the cursor is moved.\n"
"End\n"
"    Moves the text cursor to the right end of the line. If mark is TRUE text\n"
"    will be marked towards the last position, if not any marked text will\n"
"    be unmarked if the cursor is moved.\n"
"Delete"
;
const char *editortxt4 =
"    Deletes the character on the right side of the text cursor. If a text\n"
"    has been marked by the user (e.g. by clicking and dragging) the cursor\n"
"    will be put at the beginning of the marked text and the marked text will\n"
"    be removed.\n"
"Shift - Left Arrow\n"
"    Mark text one character leftwards.\n"
"Shift - Right Arrow\n"
"    Mark text one character rightwards.\n"
"Control-A\n"
"    Select the whole text.\n"
"Control-B\n"
"    Move the cursor one character leftwards."
;
const char *editortxt5 =
"Control-C\n"
"    Copy the marked text to the clipboard.\n"
"Control-D\n"
"    Delete the character to the right of the cursor.\n"
"Control-E\n"
"    Move the cursor to the end of the line.\n"
"Control-F\n"
"    Start Search Dialog.\n"
"Control-H\n"
"    Delete the character to the left of the cursor.\n"
"Control-K\n"
"    Delete marked text if any or delete all\n"
"    characters to the right of the cursor.\n"
"Control-L\n"
"    Start GoTo Line Dialog"
;
const char *editortxt6 =
"Control-U\n"
"    Delete all characters on the line.\n"
"Control-V\n"
"    Paste the clipboard text into line edit.\n"
"Control-X\n"
"    Cut the marked text, copy to clipboard.\n"
"Control-Y\n"
"    Paste the clipboard text into line edit.\n"
"Control-Z\n"
"    Undo action.\n\n"
"All other keys with valid ASCII codes insert themselves into the line.";


class TileFrame;


class TestMainFrame {

RQ_OBJECT("TestMainFrame")

private:
   TGMainFrame        *fMain;
   TGDockableFrame    *fMenuDock;
   TGCompositeFrame   *fStatusFrame;
   TGCanvas           *fCanvasWindow;
   TileFrame          *fContainer;
   TGTextEntry        *fTestText;
   TGButton           *fTestButton;
   TGColorSelect      *fColorSel;

   TGMenuBar          *fMenuBar;
   TGPopupMenu        *fMenuFile, *fMenuTest, *fMenuView, *fMenuHelp;
   TGPopupMenu        *fCascadeMenu, *fCascade1Menu, *fCascade2Menu;
   TGPopupMenu        *fMenuNew1, *fMenuNew2;
   TGLayoutHints      *fMenuBarLayout, *fMenuBarItemLayout, *fMenuBarHelpLayout;

public:
   TestMainFrame(const TGWindow *p, UInt_t w, UInt_t h);
   virtual ~TestMainFrame();

   // slots
   void CloseWindow();
   void DoButton();
   void HandleMenu(Int_t id);
   void HandlePopup() { printf("menu popped up\n"); }
   void HandlePopdown() { printf("menu popped down\n"); }

   void Created() { Emit("Created()"); } //*SIGNAL*
   void Welcome() { printf("TestMainFrame has been created. Welcome!\n"); }
};

class TestDialog {

RQ_OBJECT("TestDialog")

private:
   TGTransientFrame    *fMain;
   TGCompositeFrame    *fFrame1, *fF1, *fF2, *fF3, *fF4, *fF5;
   TGGroupFrame        *fF6, *fF7;
   TGButton            *fOkButton, *fCancelButton, *fStartB, *fStopB;
   TGButton            *fBtn1, *fBtn2, *fChk1, *fChk2, *fRad1, *fRad2;
   TGPictureButton     *fPicBut1;
   TGCheckButton       *fCheck1;
   TGCheckButton       *fCheckMulti;
   TGListBox           *fListBox;
   TGComboBox          *fCombo;
   TGTab               *fTab;
   TGTextEntry         *fTxt1, *fTxt2;
   TGLayoutHints       *fL1, *fL2, *fL3, *fL4;
   TRootEmbeddedCanvas *fEc1, *fEc2;
   Int_t                fFirstEntry;
   Int_t                fLastEntry;
   Bool_t               fFillHistos;
   TH1F                *fHpx;
   TH2F                *fHpxpy;

   void FillHistos();

public:
   TestDialog(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h,
               UInt_t options = kVerticalFrame);
   virtual ~TestDialog();

   // slots
   void DoClose();
   void CloseWindow();
   void DoOK();
   void DoCancel();
   void DoTab(Int_t id);
   void HandleButtons(Int_t id = -1);
   void HandleEmbeddedCanvas(Int_t event, Int_t x, Int_t y, TObject *sel);
};

class TestMsgBox {

RQ_OBJECT("TestMsgBox")

private:
   TGTransientFrame     *fMain;
   TGCompositeFrame     *f1, *f2, *f3, *f4, *f5;
   TGButton             *fTestButton, *fCloseButton;
   TGPictureButton      *fPictButton;
   TGRadioButton        *fR[4];
   TGCheckButton        *fC[13];
   TGGroupFrame         *fG1, *fG2;
   TGLayoutHints        *fL1, *fL2, *fL3, *fL4, *fL5, *fL6, *fL21;
   TGTextEntry          *fTitle, *fMsg;
   TGTextBuffer         *fTbtitle, *fTbmsg;
   TGLabel              *fLtitle, *fLmsg;
   TGGC                  fRedTextGC;

public:
   TestMsgBox(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h,
              UInt_t options = kVerticalFrame);
   virtual ~TestMsgBox();

   // slots
   void TryToClose();
   void CloseWindow();
   void DoClose();
   void DoRadio();
   void DoTest();
};


class TestSliders {

RQ_OBJECT("TestSliders")

private:
   TGTransientFrame  *fMain;
   TGVerticalFrame   *fVframe1, *fVframe2;
   TGLayoutHints     *fBly, *fBfly1;
   TGHSlider         *fHslider1, *fHslider2;
   TGVSlider         *fVslider1;
   TGDoubleVSlider   *fVslider2;
   TGTextEntry       *fTeh1, *fTev1, *fTeh2, *fTev2;
   TGTextBuffer      *fTbh1, *fTbv1, *fTbh2, *fTbv2;

public:
   TestSliders(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h);
   virtual ~TestSliders();

   // slots
   void CloseWindow();
   void DoText(const char *text);
   void DoSlider(Int_t pos = 0);
};


class TestShutter {

RQ_OBJECT("TestShutter")

private:
   TGTransientFrame *fMain;
   TGShutter        *fShutter;
   TGLayoutHints    *fLayout;
   const TGPicture  *fDefaultPic;

public:
   TestShutter(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h);
   ~TestShutter();

   void AddShutterItem(const char *name, shutterData_t *data);

   // slots
   void CloseWindow();
   void HandleButtons();
};


class TestDirList {

RQ_OBJECT("TestDirList")

protected:
   TGTransientFrame *fMain;
   TGListTree       *fContents;
   const TGPicture  *fIcon;
   TString DirName(TGListTreeItem* item);

public:
   TestDirList(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h);
   virtual ~TestDirList();

   // slots
   void OnDoubleClick(TGListTreeItem* item, Int_t btn);
   void CloseWindow();
};


class TestFileList {

RQ_OBJECT("TestFileList")

protected:
   TGTransientFrame *fMain;
   TGFileContainer  *fContents;
   TGPopupMenu      *fMenu;

   void DisplayFile(const TString &fname);
   void DisplayDirectory(const TString &fname);
   void DisplayObject(const TString& fname,const TString& name);

public:
   TestFileList(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h);
   virtual ~TestFileList();

   // slots
   void OnDoubleClick(TGLVEntry*,Int_t);
   void DoMenu(Int_t);
   void CloseWindow();
};

class TestProgress {

private:
   TGTransientFrame  *fMain;
   TGHorizontalFrame *fHframe1;
   TGVerticalFrame   *fVframe1;
   TGLayoutHints     *fHint1, *fHint2, *fHint3, *fHint4, *fHint5;
   TGHProgressBar    *fHProg1, *fHProg2, *fHProg3;
   TGVProgressBar    *fVProg1, *fVProg2;
   TGTextButton      *fGO;
   Bool_t             fClose;

public:
   TestProgress(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h);
   virtual ~TestProgress();

   // slots
   void CloseWindow();
   void DoClose();
   void DoGo();
};


class EntryTestDlg {

private:
   TGTransientFrame     *fMain;
   TGVerticalFrame      *fF1;
   TGVerticalFrame      *fF2;
   TGHorizontalFrame    *fF[13];
   TGLayoutHints        *fL1;
   TGLayoutHints        *fL2;
   TGLayoutHints        *fL3;
   TGLabel              *fLabel[13];
   TGNumberEntry        *fNumericEntries[13];
   TGCheckButton        *fLowerLimit;
   TGCheckButton        *fUpperLimit;
   TGNumberEntry        *fLimits[2];
   TGCheckButton        *fPositive;
   TGCheckButton        *fNonNegative;
   TGButton             *fSetButton;
   TGButton             *fExitButton;

//   static const char *const numlabel[13];
//   static const Double_t numinit[13];

public:
   EntryTestDlg(const TGWindow *p, const TGWindow *main);
   virtual ~EntryTestDlg();

   // slots
   void CloseWindow();
   void SetLimits();
   void DoOK();
};


class Editor {

private:
   TGTransientFrame *fMain;   // main frame of this widget
   TGTextEdit       *fEdit;   // text edit widget
   TGTextButton     *fOK;     // OK button
   TGLayoutHints    *fL1;     // layout of TGTextEdit
   TGLayoutHints    *fL2;     // layout of OK button

public:
   Editor(const TGWindow *main, UInt_t w, UInt_t h);
   virtual ~Editor();

   void   LoadFile(const char *file);
   void   LoadBuffer(const char *buffer);
   void   AddBuffer(const char *buffer);

   TGTextEdit *GetEditor() const { return fEdit; }

   void   SetTitle();
   void   Popup();

   // slots
   void   CloseWindow();
   void   DoOK();
   void   DoOpen();
   void   DoSave();
   void   DoClose();
};


class TileFrame {

RQ_OBJECT("TileFrame")

private:
   TGCompositeFrame *fFrame;
   TGCanvas         *fCanvas;

public:
   TileFrame(const TGWindow *p);
   virtual ~TileFrame() { delete fFrame; }

   TGFrame *GetFrame() const { return fFrame; }

   void SetCanvas(TGCanvas *canvas) { fCanvas = canvas; }
   void HandleMouseWheel(Event_t *event);
};

TileFrame::TileFrame(const TGWindow *p)
{
   // Create tile view container. Used to show colormap.

   fFrame = new TGCompositeFrame(p, 10, 10, kHorizontalFrame,
                                 TGFrame::GetWhitePixel());
   fFrame->Connect("ProcessedEvent(Event_t*)", "TileFrame", this,
                   "HandleMouseWheel(Event_t*)");
   fCanvas = 0;
   fFrame->SetLayoutManager(new TGTileLayout(fFrame, 8));

   gVirtualX->GrabButton(fFrame->GetId(), kAnyButton, kAnyModifier,
                         kButtonPressMask | kButtonReleaseMask |
                         kPointerMotionMask, kNone, kNone);
}

void TileFrame::HandleMouseWheel(Event_t *event)
{
   // Handle mouse wheel to scroll.

   if (event->fType != kButtonPress && event->fType != kButtonRelease)
      return;

   Int_t page = 0;
   if (event->fCode == kButton4 || event->fCode == kButton5) {
      if (!fCanvas) return;
      if (fCanvas->GetContainer()->GetHeight())
         page = Int_t(Float_t(fCanvas->GetViewPort()->GetHeight() *
                              fCanvas->GetViewPort()->GetHeight()) /
                              fCanvas->GetContainer()->GetHeight());
   }

   if (event->fCode == kButton4) {
      //scroll up
      Int_t newpos = fCanvas->GetVsbPosition() - page;
      if (newpos < 0) newpos = 0;
      fCanvas->SetVsbPosition(newpos);
   }
   if (event->fCode == kButton5) {
      // scroll down
      Int_t newpos = fCanvas->GetVsbPosition() + page;
      fCanvas->SetVsbPosition(newpos);
   }
}


TestMainFrame::TestMainFrame(const TGWindow *p, UInt_t w, UInt_t h)
{
   // Create test main frame. A TGMainFrame is a top level window.

   fMain = new TGMainFrame(p, w, h);

   // use hierarchical cleaning
   fMain->SetCleanup(kDeepCleanup);

   fMain->Connect("CloseWindow()", "TestMainFrame", this, "CloseWindow()");

   // Create menubar and popup menus. The hint objects are used to place
   // and group the different menu widgets with respect to eachother.
   fMenuDock = new TGDockableFrame(fMain);
   fMain->AddFrame(fMenuDock, new TGLayoutHints(kLHintsExpandX, 0, 0, 1, 0));
   fMenuDock->SetWindowName("GuiTest Menu");

   fMenuBarLayout = new TGLayoutHints(kLHintsTop | kLHintsExpandX);
   fMenuBarItemLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0);
   fMenuBarHelpLayout = new TGLayoutHints(kLHintsTop | kLHintsRight);

   fMenuFile = new TGPopupMenu(gClient->GetRoot());
   fMenuFile->AddEntry("&Open...", M_FILE_OPEN);
   fMenuFile->AddEntry("&Save", M_FILE_SAVE);
   fMenuFile->AddEntry("S&ave as...", M_FILE_SAVEAS);
   fMenuFile->AddEntry("&Close", -1);
   fMenuFile->AddSeparator();
   fMenuFile->AddEntry("&Print", M_FILE_PRINT);
   fMenuFile->AddEntry("P&rint setup...", M_FILE_PRINTSETUP);
   fMenuFile->AddSeparator();
   fMenuFile->AddEntry("E&xit", M_FILE_EXIT);

   fMenuFile->DisableEntry(M_FILE_SAVEAS);
   fMenuFile->HideEntry(M_FILE_PRINT);

   fCascade2Menu = new TGPopupMenu(gClient->GetRoot());
   fCascade2Menu->AddEntry("ID = 2&3", M_CASCADE_1);
   fCascade2Menu->AddEntry("ID = 2&4", M_CASCADE_2);
   fCascade2Menu->AddEntry("ID = 2&5", M_CASCADE_3);

   fCascade1Menu = new TGPopupMenu(gClient->GetRoot());
   fCascade1Menu->AddEntry("ID = 4&1", 41);
   fCascade1Menu->AddEntry("ID = 4&2", 42);
   fCascade1Menu->AddEntry("ID = 4&3", 43);
   fCascade1Menu->AddSeparator();
   fCascade1Menu->AddPopup("Cascade&d 2", fCascade2Menu);

   fCascadeMenu = new TGPopupMenu(gClient->GetRoot());
   fCascadeMenu->AddEntry("ID = 5&1", 51);
   fCascadeMenu->AddEntry("ID = 5&2", 52);
   fCascadeMenu->AddEntry("ID = 5&3", 53);
   fCascadeMenu->AddSeparator();
   fCascadeMenu->AddPopup("&Cascaded 1", fCascade1Menu);

   fMenuTest = new TGPopupMenu(gClient->GetRoot());
   fMenuTest->AddLabel("Test different features...");
   fMenuTest->AddSeparator();
   fMenuTest->AddEntry("&Dialog...", M_TEST_DLG);
   fMenuTest->AddEntry("&Message Box...", M_TEST_MSGBOX);
   fMenuTest->AddEntry("&Sliders...", M_TEST_SLIDER);
   fMenuTest->AddEntry("Sh&utter...", M_TEST_SHUTTER);
   fMenuTest->AddEntry("&List Directory...", M_TEST_DIRLIST);
   fMenuTest->AddEntry("&File List...", M_TEST_FILELIST);
   fMenuTest->AddEntry("&Progress...", M_TEST_PROGRESS);
   fMenuTest->AddEntry("&Number Entry...", M_TEST_NUMBERENTRY);
   fMenuTest->AddEntry("F&ont Dialog...", M_TEST_FONTDIALOG);
   fMenuTest->AddSeparator();
   fMenuTest->AddEntry("Add New Menus", M_TEST_NEWMENU);
   fMenuTest->AddSeparator();
   fMenuTest->AddPopup("&Cascaded menus", fCascadeMenu);

   fMenuView = new TGPopupMenu(gClient->GetRoot());
   fMenuView->AddEntry("&Dock", M_VIEW_DOCK);
   fMenuView->AddEntry("&Undock", M_VIEW_UNDOCK);
   fMenuView->AddSeparator();
   fMenuView->AddEntry("Enable U&ndock", M_VIEW_ENBL_DOCK);
   fMenuView->AddEntry("Enable &Hide", M_VIEW_ENBL_HIDE);
   fMenuView->DisableEntry(M_VIEW_DOCK);

   fMenuDock->EnableUndock(kTRUE);
   fMenuDock->EnableHide(kTRUE);
   fMenuView->CheckEntry(M_VIEW_ENBL_DOCK);
   fMenuView->CheckEntry(M_VIEW_ENBL_HIDE);

   // When using the DockButton of the MenuDock,
   // the states 'enable' and 'disable' of menus have to be updated.
   fMenuDock->Connect("Undocked()", "TestMainFrame", this, "HandleMenu(=M_VIEW_UNDOCK)");

   fMenuHelp = new TGPopupMenu(gClient->GetRoot());
   fMenuHelp->AddEntry("&Contents", M_HELP_CONTENTS);
   fMenuHelp->AddEntry("&Search...", M_HELP_SEARCH);
   fMenuHelp->AddSeparator();
   fMenuHelp->AddEntry("&About", M_HELP_ABOUT);

   fMenuNew1 = new TGPopupMenu();
   fMenuNew1->AddEntry("Remove New Menus", M_NEW_REMOVEMENU);

   fMenuNew2 = new TGPopupMenu();
   fMenuNew2->AddEntry("Remove New Menus", M_NEW_REMOVEMENU);

   // Menu button messages are handled by the main frame (i.e. "this")
   // HandleMenu() method.
   fMenuFile->Connect("Activated(Int_t)", "TestMainFrame", this,
                      "HandleMenu(Int_t)");
   fMenuFile->Connect("PoppedUp()", "TestMainFrame", this, "HandlePopup()");
   fMenuFile->Connect("PoppedDown()", "TestMainFrame", this, "HandlePopdown()");
   fMenuTest->Connect("Activated(Int_t)", "TestMainFrame", this,
                      "HandleMenu(Int_t)");
   fMenuView->Connect("Activated(Int_t)", "TestMainFrame", this,
                      "HandleMenu(Int_t)");
   fMenuHelp->Connect("Activated(Int_t)", "TestMainFrame", this,
                      "HandleMenu(Int_t)");
   fCascadeMenu->Connect("Activated(Int_t)", "TestMainFrame", this,
                         "HandleMenu(Int_t)");
   fCascade1Menu->Connect("Activated(Int_t)", "TestMainFrame", this,
                          "HandleMenu(Int_t)");
   fCascade2Menu->Connect("Activated(Int_t)", "TestMainFrame", this,
                          "HandleMenu(Int_t)");
   fMenuNew1->Connect("Activated(Int_t)", "TestMainFrame", this,
                      "HandleMenu(Int_t)");
   fMenuNew2->Connect("Activated(Int_t)", "TestMainFrame", this,
                      "HandleMenu(Int_t)");

   fMenuBar = new TGMenuBar(fMenuDock, 1, 1, kHorizontalFrame);
   fMenuBar->AddPopup("&File", fMenuFile, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Test", fMenuTest, fMenuBarItemLayout);
   fMenuBar->AddPopup("&View", fMenuView, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Help", fMenuHelp, fMenuBarHelpLayout);

   fMenuDock->AddFrame(fMenuBar, fMenuBarLayout);

   // Create TGCanvas and a canvas container which uses a tile layout manager
   fCanvasWindow = new TGCanvas(fMain, 400, 240);
   fContainer = new TileFrame(fCanvasWindow->GetViewPort());
   fContainer->SetCanvas(fCanvasWindow);
   fCanvasWindow->SetContainer(fContainer->GetFrame());

   // use hierarchical cleaning for container
   fContainer->GetFrame()->SetCleanup(kDeepCleanup);

   // Fill canvas with 256 colored frames
   for (int i=0; i < 256; ++i)
      fCanvasWindow->AddFrame(new TGFrame(fCanvasWindow->GetContainer(),
                              32, 32, 0, TColor::RGB2Pixel(0,0,(i+1)&255)),
                              new TGLayoutHints(kLHintsExpandY | kLHintsRight));

   fMain->AddFrame(fCanvasWindow, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
                                                    0, 0, 2, 2));

   // Create status frame containing a button and a text entry widget
   fStatusFrame = new TGCompositeFrame(fMain, 60, 20, kHorizontalFrame |
                                                      kSunkenFrame);

   fTestButton = new TGTextButton(fStatusFrame, "&Open editor...", 150);
   fTestButton->Connect("Clicked()", "TestMainFrame", this, "DoButton()");
   fTestButton->SetToolTipText("Pops up\ntext editor");
   fStatusFrame->AddFrame(fTestButton, new TGLayoutHints(kLHintsTop |
                          kLHintsLeft, 2, 0, 2, 2));
   fTestText = new TGTextEntry(fStatusFrame, new TGTextBuffer(100));
   fTestText->SetToolTipText("This is a text entry widget");
   fTestText->Resize(300, fTestText->GetDefaultHeight());
   fStatusFrame->AddFrame(fTestText, new TGLayoutHints(kLHintsTop | kLHintsLeft,
                                                       10, 2, 2, 2));
   Pixel_t yellow;
   gClient->GetColorByName("yellow", yellow);
   fColorSel = new TGColorSelect(fStatusFrame, yellow, 0);
   fStatusFrame->AddFrame(fColorSel, new TGLayoutHints(kLHintsTop |
                          kLHintsLeft, 2, 0, 2, 2));

   fMain->AddFrame(fStatusFrame, new TGLayoutHints(kLHintsBottom | kLHintsExpandX,
                   0, 0, 1, 0));

   fMain->SetWindowName("GuiTest Signal/Slots");

   fMain->MapSubwindows();

   // we need to use GetDefault...() to initialize the layout algorithm...
   fMain->Resize();
   fMain->MapWindow();
   fMain->Print();
   Connect("Created()", "TestMainFrame", this, "Welcome()");
   Created();
}

TestMainFrame::~TestMainFrame()
{
   // Delete all created widgets.

   delete fMenuFile;
   delete fMenuTest;
   delete fMenuView;
   delete fMenuHelp;
   delete fCascadeMenu;
   delete fCascade1Menu;
   delete fCascade2Menu;
   delete fMenuNew1;
   delete fMenuNew2;

   delete fContainer;
   delete fMain;
}

void TestMainFrame::CloseWindow()
{
   // Got close message for this MainFrame. Terminates the application.

   gApplication->Terminate();
}

void TestMainFrame::DoButton()
{
   // Handle button click.

   Editor *ed = new Editor(fMain, 600, 400);
   ed->LoadBuffer(editortxt1);
   ed->AddBuffer(editortxt2);
   ed->AddBuffer(editortxt3);
   ed->AddBuffer(editortxt4);
   ed->AddBuffer(editortxt5);
   ed->AddBuffer(editortxt6);
   ed->Popup();
}

void TestMainFrame::HandleMenu(Int_t id)
{
   // Handle menu items.

   switch (id) {

      case M_FILE_OPEN:
         {
            static TString dir(".");
            TGFileInfo fi;
            fi.fFileTypes = filetypes;
            fi.SetIniDir(dir);
            printf("fIniDir = %s\n", fi.fIniDir);
            new TGFileDialog(gClient->GetRoot(), fMain, kFDOpen, &fi);
            printf("Open file: %s (dir: %s)\n", fi.fFilename, fi.fIniDir);
            dir = fi.fIniDir;
         }
         break;

      case M_FILE_SAVE:
         printf("M_FILE_SAVE\n");
         break;

      case M_FILE_PRINT:
         printf("M_FILE_PRINT\n");
         printf("Hiding itself, select \"Print Setup...\" to enable again\n");
         fMenuFile->HideEntry(M_FILE_PRINT);
         break;

      case M_FILE_PRINTSETUP:
         printf("M_FILE_PRINTSETUP\n");
         printf("Enabling \"Print\"\n");
         fMenuFile->EnableEntry(M_FILE_PRINT);
         break;

      case M_FILE_EXIT:
         CloseWindow();   // terminate theApp no need to use SendCloseMessage()
         break;

      case M_TEST_DLG:
         new TestDialog(gClient->GetRoot(), fMain, 400, 200);
         break;

      case M_TEST_MSGBOX:
         new TestMsgBox(gClient->GetRoot(), fMain, 400, 200);
         break;

      case M_TEST_SLIDER:
         new TestSliders(gClient->GetRoot(), fMain, 400, 200);
         break;

      case M_TEST_SHUTTER:
         new TestShutter(gClient->GetRoot(), fMain, 400, 200);
         break;

      case M_TEST_DIRLIST:
         new TestDirList(gClient->GetRoot(), fMain, 400, 200);
         break;

     case M_TEST_FILELIST:
         new TestFileList(gClient->GetRoot(), fMain, 400, 200);
         break;

      case M_TEST_PROGRESS:
         new TestProgress(gClient->GetRoot(), fMain, 600, 300);
         break;

      case M_TEST_NUMBERENTRY:
         new EntryTestDlg(gClient->GetRoot(), fMain);
         break;

      case M_TEST_FONTDIALOG:
         {
            TGFontDialog::FontProp_t prop;
            new TGFontDialog(gClient->GetRoot(), fMain, &prop);
            if (prop.fName != "")
               printf("Selected font: %s, size %d, italic %s, bold %s, color 0x%lx, align %u\n",
                      prop.fName.Data(), prop.fSize, prop.fItalic ? "yes" : "no",
                      prop.fBold ? "yes" : "no", prop.fColor, prop.fAlign);
         }
         break;

      case M_TEST_NEWMENU:
         {
            if (fMenuTest->IsEntryChecked(M_TEST_NEWMENU)) {
               HandleMenu(M_NEW_REMOVEMENU);
               return;
            }
            fMenuTest->CheckEntry(M_TEST_NEWMENU);
            TGPopupMenu *p = fMenuBar->GetPopup("Test");
            fMenuBar->AddPopup("New 1", fMenuNew1, fMenuBarItemLayout, p);
            p = fMenuBar->GetPopup("Help");
            fMenuBar->AddPopup("New 2", fMenuNew2, fMenuBarItemLayout, p);
            fMenuBar->MapSubwindows();
            fMenuBar->Layout();

            TGMenuEntry *e = fMenuTest->GetEntry("Add New Menus");
            fMenuTest->AddEntry("Remove New Menus", M_NEW_REMOVEMENU, 0, 0, e);
         }
         break;

      case M_NEW_REMOVEMENU:
         {
            fMenuBar->RemovePopup("New 1");
            fMenuBar->RemovePopup("New 2");
            fMenuBar->Layout();
            fMenuTest->DeleteEntry(M_NEW_REMOVEMENU);
            fMenuTest->UnCheckEntry(M_TEST_NEWMENU);
         }
         break;

      case M_VIEW_ENBL_DOCK:
         fMenuDock->EnableUndock(!fMenuDock->EnableUndock());
         if (fMenuDock->EnableUndock()) {
            fMenuView->CheckEntry(M_VIEW_ENBL_DOCK);
            fMenuView->EnableEntry(M_VIEW_UNDOCK);
         } else {
            fMenuView->UnCheckEntry(M_VIEW_ENBL_DOCK);
            fMenuView->DisableEntry(M_VIEW_UNDOCK);
         }
         break;

      case M_VIEW_ENBL_HIDE:
         fMenuDock->EnableHide(!fMenuDock->EnableHide());
         if (fMenuDock->EnableHide()) {
            fMenuView->CheckEntry(M_VIEW_ENBL_HIDE);
         } else {
            fMenuView->UnCheckEntry(M_VIEW_ENBL_HIDE);
         }
         break;

       case M_VIEW_DOCK:
         fMenuDock->DockContainer();
         fMenuView->EnableEntry(M_VIEW_UNDOCK);
         fMenuView->DisableEntry(M_VIEW_DOCK);
         break;

       case M_VIEW_UNDOCK:
         fMenuDock->UndockContainer();
         fMenuView->EnableEntry(M_VIEW_DOCK);
         fMenuView->DisableEntry(M_VIEW_UNDOCK);
         break;

      default:
         printf("Menu item %d selected\n", id);
         break;
   }
}


TestDialog::TestDialog(const TGWindow *p, const TGWindow *main, UInt_t w,
                       UInt_t h, UInt_t options)
{
   // Create a dialog window. A dialog window pops up with respect to its
   // "main" window.

   fMain = new TGTransientFrame(p, main, w, h, options);
   fMain->Connect("CloseWindow()", "TestDialog", this, "DoClose()");
   fMain->DontCallClose(); // to avoid double deletions.

   // use hierarchical cleaning
   fMain->SetCleanup(kDeepCleanup);

   fFrame1 = new TGHorizontalFrame(fMain, 60, 20, kFixedWidth);

   fOkButton = new TGTextButton(fFrame1, "&Ok", 1);
   fOkButton->Connect("Clicked()", "TestDialog", this, "DoOK()");
   fCancelButton = new TGTextButton(fFrame1, "&Cancel", 2);
   fCancelButton->Connect("Clicked()", "TestDialog", this, "DoCancel()");

   fL1 = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX,
                           2, 2, 2, 2);
   fL2 = new TGLayoutHints(kLHintsBottom | kLHintsRight, 2, 2, 5, 1);

   fFrame1->AddFrame(fOkButton, fL1);
   fFrame1->AddFrame(fCancelButton, fL1);

   fFrame1->Resize(150, fOkButton->GetDefaultHeight());
   fMain->AddFrame(fFrame1, fL2);

   //--------- create Tab widget and some composite frames for Tab testing

   fTab = new TGTab(fMain, 300, 300);
   fTab->Connect("Selected(Int_t)", "TestDialog", this, "DoTab(Int_t)");

   fL3 = new TGLayoutHints(kLHintsTop | kLHintsLeft, 5, 5, 5, 5);

   TGCompositeFrame *tf = fTab->AddTab("Tab 1");
   fF1 = new TGCompositeFrame(tf, 60, 20, kVerticalFrame);
   fF1->AddFrame(new TGTextButton(fF1, "&Test button", 0), fL3);
   fF1->AddFrame(fTxt1 = new TGTextEntry(fF1, new TGTextBuffer(100)), fL3);
   fF1->AddFrame(fTxt2 = new TGTextEntry(fF1, new TGTextBuffer(100)), fL3);
   tf->AddFrame(fF1, fL3);
   fTxt1->Resize(150, fTxt1->GetDefaultHeight());
   fTxt2->Resize(150, fTxt2->GetDefaultHeight());

   tf = fTab->AddTab("Tab 2");
   fL1 = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX,
                           200, 2, 2, 2);
   fF2 = new TGCompositeFrame(tf, 60, 20, kVerticalFrame);
   fF2->AddFrame(fBtn1 = new TGTextButton(fF2, "&Button 1", 61), fL1);
   fF2->AddFrame(fBtn2 = new TGTextButton(fF2, "B&utton 2", 62), fL1);
   fF2->AddFrame(fChk1 = new TGCheckButton(fF2, "C&heck 1", 71), fL1);
   fF2->AddFrame(fChk2 = new TGCheckButton(fF2, "Chec&k 2", 72), fL1);
   fF2->AddFrame(fRad1 = new TGRadioButton(fF2, "&Radio 1", 81), fL1);
   fF2->AddFrame(fRad2 = new TGRadioButton(fF2, "R&adio 2", 82), fL1);
   fCombo = new TGComboBox(fF2, 88);
   fF2->AddFrame(fCombo, fL3);

   tf->AddFrame(fF2, fL3);

   int i;
   char tmp[20];
   for (i = 0; i < 20; i++) {

      sprintf(tmp, "Entry %i", i+1);
      fCombo->AddEntry(tmp, i+1);
   }

   fCombo->Resize(150, 20);

   fBtn1->Connect("Clicked()", "TestDialog", this, "HandleButtons()");
   fBtn2->Connect("Clicked()", "TestDialog", this, "HandleButtons()");
   fChk1->Connect("Clicked()", "TestDialog", this, "HandleButtons()");
   fChk2->Connect("Clicked()", "TestDialog", this, "HandleButtons()");
   fRad1->Connect("Clicked()", "TestDialog", this, "HandleButtons()");
   fRad2->Connect("Clicked()", "TestDialog", this, "HandleButtons()");

   //-------------- embedded canvas demo
   fFillHistos = kFALSE;
   fHpx   = 0;
   fHpxpy = 0;

   tf = fTab->AddTab("Tab 3");
   fF3 = new TGCompositeFrame(tf, 60, 20, kHorizontalFrame);
   fStartB = new TGTextButton(fF3, "Start &Filling Hists", 40);
   fStopB  = new TGTextButton(fF3, "&Stop Filling Hists", 41);
   fStartB->Connect("Clicked()", "TestDialog", this, "HandleButtons()");
   fStopB->Connect("Clicked()", "TestDialog", this, "HandleButtons()");
   fF3->AddFrame(fStartB, fL3);
   fF3->AddFrame(fStopB, fL3);

   fF5 = new TGCompositeFrame(tf, 60, 60, kHorizontalFrame);

   fL4 = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX |
                           kLHintsExpandY, 5, 5, 5, 5);
   fEc1 = new TRootEmbeddedCanvas("ec1", fF5, 100, 100);
   fF5->AddFrame(fEc1, fL4);
   fEc2 = new TRootEmbeddedCanvas("ec2", fF5, 100, 100);
   fF5->AddFrame(fEc2, fL4);

   tf->AddFrame(fF3, fL3);
   tf->AddFrame(fF5, fL4);

   fEc1->GetCanvas()->SetBorderMode(0);
   fEc2->GetCanvas()->SetBorderMode(0);
   fEc1->GetCanvas()->SetBit(kNoContextMenu);
   fEc1->GetCanvas()->Connect("ProcessedEvent(Int_t,Int_t,Int_t,TObject*)",
                              "TestDialog", this,
                              "HandleEmbeddedCanvas(Int_t,Int_t,Int_t,TObject*)");

   // make tab yellow
   Pixel_t yellow;
   gClient->GetColorByName("yellow", yellow);
   TGTabElement *tabel = fTab->GetTabTab("Tab 3");
   tabel->ChangeBackground(yellow);

   //-------------- end embedded canvas demo

   TGTextButton *bt;
   tf = fTab->AddTab("Tab 4");
   fF4 = new TGCompositeFrame(tf, 60, 20, kVerticalFrame);
   fF4->AddFrame(bt = new TGTextButton(fF4, "A&dd Entry", 90), fL3);
   bt->Connect("Clicked()", "TestDialog", this, "HandleButtons()");

   fF4->AddFrame(bt = new TGTextButton(fF4, "Remove &Entry", 91), fL3);
   bt->Connect("Clicked()", "TestDialog", this, "HandleButtons()");

   fF4->AddFrame(fListBox = new TGListBox(fF4, 89), fL3);
   fF4->AddFrame(fCheckMulti = new TGCheckButton(fF4, "&Mutli Selectable", 92), fL3);
   fCheckMulti->Connect("Clicked()", "TestDialog", this, "HandleButtons()");
   tf->AddFrame(fF4, fL3);

   for (i = 0; i < 20; ++i)  {
      sprintf(tmp, "Entry %i", i);
      fListBox->AddEntry(tmp, i);
   }
   fFirstEntry = 0;
   fLastEntry  = 20;

   fListBox->Resize(150, 80);

   //--- tab 5
   tf = fTab->AddTab("Tab 5");
   tf->SetLayoutManager(new TGHorizontalLayout(tf));

   fF6 = new TGGroupFrame(tf, "Options", kVerticalFrame);
   fF6->SetTitlePos(TGGroupFrame::kRight); // right aligned
   tf->AddFrame(fF6, fL3);

   // 2 column, n rows
   fF6->SetLayoutManager(new TGMatrixLayout(fF6, 0, 2, 10));
   char buff[100];
   int j;
   for (j = 0; j < 4; j++) {
      sprintf(buff, "Module %i", j+1);
      fF6->AddFrame(new TGLabel(fF6, new TGHotString(buff)));

      TGTextBuffer *tbuf = new TGTextBuffer(10);
      tbuf->AddText(0, "0.0");

      TGTextEntry  *tent = new TGTextEntry(fF6, tbuf);
      tent->Resize(50, tent->GetDefaultHeight());
      tent->SetFont("-adobe-courier-bold-r-*-*-14-*-*-*-*-*-iso8859-1");
      fF6->AddFrame(tent);
   }
   fF6->Resize();

   // another matrix with text and buttons
   fF7 = new TGGroupFrame(tf, "Tab Handling", kVerticalFrame);
   tf->AddFrame(fF7, fL3);

   fF7->SetLayoutManager(new TGMatrixLayout(fF7, 0, 1, 10));

   fF7->AddFrame(bt = new TGTextButton(fF7, "Remove Tab", 101));
   bt->Connect("Clicked()", "TestDialog", this, "HandleButtons()");
   bt->Resize(90, bt->GetDefaultHeight());

   fF7->AddFrame(bt = new TGTextButton(fF7, "Add Tab", 103));
   bt->Connect("Clicked()", "TestDialog", this, "HandleButtons()");
   bt->Resize(90, bt->GetDefaultHeight());

   fF7->AddFrame(bt = new TGTextButton(fF7, "Remove Tab 5", 102));
   bt->Connect("Clicked()", "TestDialog", this, "HandleButtons()");
   bt->Resize(90, bt->GetDefaultHeight());

   fF7->Resize(fF6->GetDefaultSize());

   //--- end of last tab

   TGLayoutHints *fL5 = new TGLayoutHints(kLHintsBottom | kLHintsExpandX |
                                          kLHintsExpandY, 2, 2, 5, 1);
   fMain->AddFrame(fTab, fL5);

   fMain->MapSubwindows();
   fMain->Resize();

   // position relative to the parent's window
   fMain->CenterOnParent();

   fMain->SetWindowName("Dialog");

   fMain->MapWindow();
   //gClient->WaitFor(fMain);    // otherwise canvas contextmenu does not work
}

TestDialog::~TestDialog()
{
   // Delete test dialog widgets.

   fMain->DeleteWindow();  // deletes fMain
}

void TestDialog::FillHistos()
{
   // Fill histograms till user clicks "Stop Filling" button.

   static int cnt;

   if (!fHpx) {
      fHpx   = new TH1F("hpx","This is the px distribution",100,-4,4);
      fHpxpy = new TH2F("hpxpy","py vs px",40,-4,4,40,-4,4);
      fHpx->SetFillColor(kRed);
      cnt = 0;
   }

   const int kUPDATE = 1000;
   float px, py;
   TCanvas *c1 = fEc1->GetCanvas();
   TCanvas *c2 = fEc2->GetCanvas();

   while (fFillHistos) {
      gRandom->Rannor(px,py); //px and py will be two gaussian random numbers
      fHpx->Fill(px);
      fHpxpy->Fill(px,py);
      cnt++;
      if (!(cnt % kUPDATE)) {
         if (cnt == kUPDATE) {
            c1->cd();
            fHpx->Draw();
            c2->cd();
            fHpxpy->Draw("cont");
         }
         c1->Modified();
         c1->Update();
         c2->Modified();
         c2->Update();
         gSystem->ProcessEvents();  // handle GUI events
      }
   }
}

void TestDialog::DoClose()
{
   printf("\nTerminating dialog: via window manager\n");
   if (fFillHistos) {
      fFillHistos = kFALSE;
      TTimer::SingleShot(150, "TestDialog", this, "CloseWindow()");
   } else
      CloseWindow();

   // Close the Ged editor if it was activated.
   if (TVirtualPadEditor::GetPadEditor(kFALSE) != 0)
      TVirtualPadEditor::Terminate();
}

void TestDialog::CloseWindow()
{
   // Called when window is closed via the window manager.

   delete this;
}

void TestDialog::DoOK()
{
   fFillHistos = kFALSE;
   printf("\nTerminating dialog: OK pressed\n");
   // Add protection against double-clicks
   fOkButton->SetState(kButtonDisabled);
   fCancelButton->SetState(kButtonDisabled);

   // Send a close message to the main frame. This will trigger the
   // emission of a CloseWindow() signal, which will then call
   // TestDialog::CloseWindow(). Calling directly CloseWindow() will cause
   // a segv since the OK button is still accessed after the DoOK() method.
   // This works since the close message is handled synchronous (via
   // message going to/from X server).
   //fMain->SendCloseMessage();

   // The same effect can be obtained by using a singleshot timer:
   TTimer::SingleShot(150, "TestDialog", this, "CloseWindow()");

   // Close the Ged editor if it was activated.
   if (TVirtualPadEditor::GetPadEditor(kFALSE) != 0)
      TVirtualPadEditor::Terminate();
}


void TestDialog::DoCancel()
{
   fFillHistos = kFALSE;
   printf("\nTerminating dialog: Cancel pressed\n");
   // Add protection against double-clicks
   fOkButton->SetState(kButtonDisabled);
   fCancelButton->SetState(kButtonDisabled);
   TTimer::SingleShot(150, "TestDialog", this, "CloseWindow()");
   // Close the Ged editor if it was activated.
   if (TVirtualPadEditor::GetPadEditor(kFALSE) != 0)
      TVirtualPadEditor::Terminate();
}

void TestDialog::HandleButtons(Int_t id)
{
   // Handle different buttons.

   if (id == -1) {
      TGButton *btn = (TGButton *) gTQSender;
      id = btn->WidgetId();
   }

   printf("DoButton: id = %d\n", id);

   char tmp[20];
   static int newtab = 0;

   switch (id) {
      case 40:  // start histogram filling
         fFillHistos = kTRUE;
         FillHistos();
         break;
      case 41:  // stop histogram filling
         fFillHistos = kFALSE;
         break;
      case 61:  // show item 1 in the combo box
         fCombo->Select(1);
         break;
      case 62:  // show item 2 in the combo box
         fCombo->Select(2);
         break;
      case 90:  // add one entry in list box
         fLastEntry++;
         sprintf(tmp, "Entry %i", fLastEntry);
         fListBox->AddEntry(tmp, fLastEntry);
         fListBox->MapSubwindows();
         fListBox->Layout();
         break;
      case 91:  // remove one entry in list box
         if (fFirstEntry <= fLastEntry) {
            fListBox->RemoveEntry(fFirstEntry);
            fListBox->Layout();
            fFirstEntry++;
         }
         break;
      case 101:  // remove tabs
         {
            TString s = fTab->GetTabTab(0)->GetString();
            if ((s == "Tab 3") && (fMain->MustCleanup() != kDeepCleanup)) {
               // Need to delete the embedded canvases
               // since RemoveTab() will Destroy the container
               // window, which in turn will destroy the embedded
               // canvas windows.
               delete fEc1; fEc1 = 0;
               delete fEc2; fEc2 = 0;
            }
            fTab->RemoveTab(0);
            fTab->Layout();
         }
         break;
      case 102:  // remove tab 5
         {
            int nt = fTab->GetNumberOfTabs();
            for (int i = 0 ; i < nt; i++) {
               TString s = fTab->GetTabTab(i)->GetString();
               if (s == "Tab 5") {
                  fTab->RemoveTab(i);
                  fTab->Layout();
                  break;
               }
            }
         }
         break;
      case 103:  // add tabs
         sprintf(tmp, "New Tab %d", ++newtab);
         fTab->AddTab(tmp);
         fTab->MapSubwindows();
         fTab->Layout();
         break;
      case 81:
         fRad2->SetState(kButtonUp);
         break;
      case 82:
         fRad1->SetState(kButtonUp);
         break;
      case 92:
         fListBox->SetMultipleSelections(fCheckMulti->GetState());
         break;
      default:
         break;
   }
}

void TestDialog::DoTab(Int_t id)
{
   printf("Tab item %d activated\n", id);
}

void TestDialog::HandleEmbeddedCanvas(Int_t event, Int_t x, Int_t y,
                                      TObject *sel)
{
   // Handle events in the left embedded canvas.

   if (event == kButton3Down)
      printf("event = %d, x = %d, y = %d, obj = %s::%s\n", event, x, y,
             sel->IsA()->GetName(), sel->GetName());
}

TestMsgBox::TestMsgBox(const TGWindow *p, const TGWindow *main,
                       UInt_t w, UInt_t h, UInt_t options) :
     fRedTextGC(TGButton::GetDefaultGC())
{
   // Create message box test dialog. Use this dialog to select the different
   // message dialog box styles and show the message dialog by clicking the
   // "Test" button.

   fMain = new TGTransientFrame(p, main, w, h, options);
   fMain->Connect("CloseWindow()", "TestMsgBox", this, "CloseWindow()");
   fMain->DontCallClose(); // to avoid double deletions.

   // use hierarchical cleaning
   fMain->SetCleanup(kDeepCleanup);

   //------------------------------
   // Set foreground color in graphics context for drawing of
   // TGlabel and TGButtons with text in red.

   Pixel_t red;
   gClient->GetColorByName("red", red);
   fRedTextGC.SetForeground(red);
   //---------------------------------

   int i;

   fMain->ChangeOptions((fMain->GetOptions() & ~kVerticalFrame) | kHorizontalFrame);

   f1 = new TGCompositeFrame(fMain, 60, 20, kVerticalFrame | kFixedWidth);
   f2 = new TGCompositeFrame(fMain, 60, 20, kVerticalFrame);
   f3 = new TGCompositeFrame(f2, 60, 20, kHorizontalFrame);
   f4 = new TGCompositeFrame(f2, 60, 20, kHorizontalFrame);
   f5 = new TGCompositeFrame(f2, 60, 20, kHorizontalFrame);

   fTestButton = new TGTextButton(f1, "&Test", 1, fRedTextGC());
   fTestButton->Connect("Clicked()", "TestMsgBox", this, "DoTest()");

   // Change background of fTestButton to green
   Pixel_t green;
   gClient->GetColorByName("green", green);
   fTestButton->ChangeBackground(green);

   fCloseButton = new TGTextButton(f1, "&Close", 2);
   fCloseButton->Connect("Clicked()", "TestMsgBox", this, "DoClose()");

   fPictButton = new TGPictureButton(f1, gClient->GetPicture("mb_stop_s.xpm"));

   f1->Resize(fTestButton->GetDefaultWidth()+40, fMain->GetDefaultHeight());

   fL1 = new TGLayoutHints(kLHintsTop | kLHintsExpandX,
                           2, 2, 3, 0);
   fL2 = new TGLayoutHints(kLHintsTop | kLHintsRight | kLHintsExpandX,
                           2, 5, 0, 2);
   fL21 = new TGLayoutHints(kLHintsTop | kLHintsRight,
                            2, 5, 10, 0);

   f1->AddFrame(fTestButton, fL1);
   f1->AddFrame(fCloseButton, fL1);
   f1->AddFrame(fPictButton, fL1);
   fMain->AddFrame(f1, fL21);

   //--------- create check and radio buttons groups

   fG1 = new TGGroupFrame(f3, new TGString("Buttons"),kVerticalFrame|kRaisedFrame);
   fG2 = new TGGroupFrame(f3, new TGString("Icons"),kVerticalFrame|kRaisedFrame);

   fL3 = new TGLayoutHints(kLHintsTop | kLHintsLeft |
                           kLHintsExpandX | kLHintsExpandY,
                           2, 2, 2, 2);
   fL4 = new TGLayoutHints(kLHintsTop | kLHintsLeft,
                           0, 0, 5, 0);

   fC[0]  = new TGCheckButton(fG1, new TGHotString("Yes"),        -1);
   fC[1]  = new TGCheckButton(fG1, new TGHotString("No"),         -1);
   fC[2]  = new TGCheckButton(fG1, new TGHotString("OK"),         -1);
   fC[3]  = new TGCheckButton(fG1, new TGHotString("Apply"),      -1);
   fC[4]  = new TGCheckButton(fG1, new TGHotString("Retry"),      -1);
   fC[5]  = new TGCheckButton(fG1, new TGHotString("Ignore"),     -1);
   fC[6]  = new TGCheckButton(fG1, new TGHotString("Cancel"),     -1);
   fC[7]  = new TGCheckButton(fG1, new TGHotString("Close"),      -1);
   fC[8]  = new TGCheckButton(fG1, new TGHotString("Yes to All"), -1);
   fC[9]  = new TGCheckButton(fG1, new TGHotString("No to All"),  -1);
   fC[10] = new TGCheckButton(fG1, new TGHotString("Newer Only"), -1);
   fC[11] = new TGCheckButton(fG1, new TGHotString("Append"),     -1);
   fC[12] = new TGCheckButton(fG1, new TGHotString("Dismiss"),    -1);

   for (i=0; i<13; ++i) fG1->AddFrame(fC[i], fL4);

   fR[0] = new TGRadioButton(fG2, new TGHotString("Stop"),        21);
   fR[1] = new TGRadioButton(fG2, new TGHotString("Question"),    22);
   fR[2] = new TGRadioButton(fG2, new TGHotString("Exclamation"), 23);
   fR[3] = new TGRadioButton(fG2, new TGHotString("Asterisk"),    24);

   for (i = 0; i < 4; ++i) {
      fG2->AddFrame(fR[i], fL4);
      fR[i]->Connect("Clicked()", "TestMsgBox", this, "DoRadio()");
   }

   fC[2]->SetState(kButtonDown);
   fR[0]->SetState(kButtonDown);

   f3->AddFrame(fG1, fL3);
   f3->AddFrame(fG2, fL3);

   fLtitle = new TGLabel(f4, new TGString("Title:"), fRedTextGC());
   fLmsg   = new TGLabel(f5, new TGString("Message:"));

   fTitle = new TGTextEntry(f4, fTbtitle = new TGTextBuffer(100));
   fMsg   = new TGTextEntry(f5, fTbmsg = new TGTextBuffer(100));

   fTbtitle->AddText(0, "MsgBox");
   fTbmsg->AddText(0, "This is a test message box.");

   fTitle->Resize(300, fTitle->GetDefaultHeight());
   fMsg->Resize(300, fMsg->GetDefaultHeight());

   fL5 = new TGLayoutHints(kLHintsLeft | kLHintsCenterY,
                           3, 5, 0, 0);
   fL6 = new TGLayoutHints(kLHintsRight | kLHintsCenterY,
                           0, 2, 0, 0);

   f4->AddFrame(fLtitle, fL5);
   f4->AddFrame(fTitle, fL6);
   f5->AddFrame(fLmsg, fL5);
   f5->AddFrame(fMsg, fL6);

   f2->AddFrame(f3, fL1);
   f2->AddFrame(f4, fL1);
   f2->AddFrame(f5, fL1);

   fMain->AddFrame(f2, fL2);

   fMain->MapSubwindows();
   fMain->Resize();

   // position relative to the parent's window
   fMain->CenterOnParent();

   fMain->SetWindowName("Message Box Test");

   fMain->MapWindow();
   gClient->WaitFor(fMain);
}

// Order is important when deleting frames. Delete children first,
// parents last.

TestMsgBox::~TestMsgBox()
{
   // Delete widgets created by dialog.

   fMain->DeleteWindow();  // deletes fMain
}

void TestMsgBox::CloseWindow()
{
   // Close dialog in response to window manager close.

   delete this;
}

void TestMsgBox::DoClose()
{
   // Handle Close button.

   CloseWindow();
}

void TestMsgBox::DoTest()
{
   // Handle test button.

   int i, buttons, retval;
   EMsgBoxIcon icontype = kMBIconStop;

   buttons = 0;
   for (i = 0; i < 13; i++)
      if (fC[i]->GetState() == kButtonDown)
         buttons |= mb_button_id[i];

   for (i = 0; i < 4; i++)
      if (fR[i]->GetState() == kButtonDown) {
         icontype = mb_icon[i];
         break;
      }

   // Since the message dialog box is created, we disable the
   // signal/slot communication mechanism, in order to ensure we
   // can't close the fMain window while the message box is open.
   fMain->Disconnect("CloseWindow()");
   fMain->Connect("CloseWindow()", "TestMsgBox", this, "TryToClose()");
   new TGMsgBox(gClient->GetRoot(), fMain,
                fTbtitle->GetString(), fTbmsg->GetString(),
                icontype, buttons, &retval);
   fMain->Disconnect("CloseWindow()");
   fMain->Connect("CloseWindow()", "TestMsgBox", this, "CloseWindow()");

}

void TestMsgBox::TryToClose()
{
   // The user try to close the main window,
   //  while a message dialog box is still open.
   printf("Can't close the window '%s' : a message box is still open\n", fMain->GetWindowName());
}

void TestMsgBox::DoRadio()
{
   // Handle radio buttons.

   TGButton *btn = (TGButton *) gTQSender;
   Int_t id = btn->WidgetId();

   if (id >= 21 && id <= 24) {
      for (int i = 0; i < 4; i++)
         if (fR[i]->WidgetId() != id)
            fR[i]->SetState(kButtonUp);
   }
}


TestSliders::TestSliders(const TGWindow *p, const TGWindow *main,
                         UInt_t w, UInt_t h)
{
   // Dialog used to test the different supported sliders.

   fMain = new TGTransientFrame(p, main, w, h);
   fMain->Connect("CloseWindow()", "TestSliders", this, "CloseWindow()");
   fMain->DontCallClose(); // to avoid double deletions.

   // use hierarchical cleaning
   fMain->SetCleanup(kDeepCleanup);

   fMain->ChangeOptions((fMain->GetOptions() & ~kVerticalFrame) | kHorizontalFrame);

   fVframe1 = new TGVerticalFrame(fMain, 0, 0, 0);

   fTeh1 = new TGTextEntry(fVframe1, fTbh1 = new TGTextBuffer(10), HId1);
   fTev1 = new TGTextEntry(fVframe1, fTbv1 = new TGTextBuffer(10), VId1);
   fTbh1->AddText(0, "0");
   fTbv1->AddText(0, "0");

   fTeh1->Connect("TextChanged(char*)", "TestSliders", this, "DoText(char*)");
   fTev1->Connect("TextChanged(char*)", "TestSliders", this, "DoText(char*)");

   fHslider1 = new TGHSlider(fVframe1, 100, kSlider1 | kScaleBoth, HSId1);
   fHslider1->Connect("PositionChanged(Int_t)", "TestSliders", this, "DoSlider(Int_t)");
   fHslider1->SetRange(0,50);

   fVslider1 = new TGVSlider(fVframe1, 100, kSlider2 | kScaleBoth, VSId1);
   fVslider1->Connect("PositionChanged(Int_t)", "TestSliders", this, "DoSlider(Int_t)");
   fVslider1->SetRange(0,8);

   fVframe1->Resize(100, 100);

   fVframe2 = new TGVerticalFrame(fMain, 0, 0, 0);
   fTeh2 = new TGTextEntry(fVframe2, fTbh2 = new TGTextBuffer(10), HId2);
   fTev2 = new TGTextEntry(fVframe2, fTbv2 = new TGTextBuffer(10), VId2);
   fTbh2->AddText(0, "0");
   fTbv2->AddText(0, "0");

   fTeh2->Connect("TextChanged(char*)", "TestSliders", this, "DoText(char*)");
   fTev2->Connect("TextChanged(char*)", "TestSliders", this, "DoText(char*)");

   fHslider2 = new TGHSlider(fVframe2, 150, kSlider2 | kScaleBoth, HSId2);
   fHslider2->Connect("PositionChanged(Int_t)", "TestSliders", this, "DoSlider(Int_t)");
   fHslider2->SetRange(0,3);

   fVslider2 = new TGDoubleVSlider(fVframe2, 100, kDoubleScaleBoth, VSId2);

   fVslider2->SetRange(-10,10);
   fVslider2->Connect("PositionChanged()", "TestSliders", this, "DoSlider()");
   fVframe2->Resize(100, 100);

   //--- layout for buttons: top align, equally expand horizontally
   fBly = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 3, 0);

   //--- layout for the frame: place at bottom, right aligned
   fBfly1 = new TGLayoutHints(kLHintsTop | kLHintsRight, 20, 10, 15, 0);

   fVframe1->AddFrame(fHslider1, fBly);
   fVframe1->AddFrame(fVslider1, fBly);
   fVframe1->AddFrame(fTeh1, fBly);
   fVframe1->AddFrame(fTev1, fBly);

   fVframe2->AddFrame(fHslider2, fBly);
   fVframe2->AddFrame(fVslider2, fBly);
   fVframe2->AddFrame(fTeh2, fBly);
   fVframe2->AddFrame(fTev2, fBly);

   fMain->AddFrame(fVframe2, fBfly1);
   fMain->AddFrame(fVframe1, fBfly1);

   fMain->SetWindowName("Slider Test");
   TGDimension size = fMain->GetDefaultSize();
   fMain->Resize(size);

   fMain->SetWMSize(size.fWidth, size.fHeight);
   fMain->SetWMSizeHints(size.fWidth, size.fHeight, size.fWidth, size.fHeight, 0, 0);
   fMain->SetMWMHints(kMWMDecorAll | kMWMDecorResizeH  | kMWMDecorMaximize |
                                     kMWMDecorMinimize | kMWMDecorMenu,
                      kMWMFuncAll |  kMWMFuncResize    | kMWMFuncMaximize |
                                     kMWMFuncMinimize,
                      kMWMInputModeless);

   // position relative to the parent's window
   fMain->CenterOnParent();

   fMain->MapSubwindows();
   fMain->MapWindow();

   gClient->WaitFor(fMain);
}

TestSliders::~TestSliders()
{
   // Delete dialog.

   fMain->DeleteWindow();  // deletes fMain
}

void TestSliders::CloseWindow()
{
   // Called when window is closed via the window manager.

   delete this;
}

void TestSliders::DoText(const char * /*text*/)
{
   // Handle text entry widgets.

   TGTextEntry *te = (TGTextEntry *) gTQSender;
   Int_t id = te->WidgetId();

   switch (id) {
      case HId1:
         fHslider1->SetPosition(atoi(fTbh1->GetString()));
         break;
      case VId1:
         fVslider1->SetPosition(atoi(fTbv1->GetString()));
         break;
      case HId2:
         fHslider2->SetPosition(atoi(fTbh2->GetString()));
         break;
      case VId2:
         fVslider2->SetPosition(atoi(fTbv2->GetString()),
                                     atoi(fTbv2->GetString())+2);
         break;
      default:
         break;
   }
}

void TestSliders::DoSlider(Int_t pos)
{
   // Handle slider widgets.

   Int_t id;
   TGFrame *frm = (TGFrame *) gTQSender;
   if (frm->IsA()->InheritsFrom(TGSlider::Class())) {
      TGSlider *sl = (TGSlider*) frm;
      id = sl->WidgetId();
   } else {
      TGDoubleSlider *sd = (TGDoubleSlider *) frm;
      id = sd->WidgetId();
   }

   char buf[32];
   sprintf(buf, "%d", pos);

#ifdef CINT_FIXED
   switch (id) {
   case HSId1:
#else
   if (id == HSId1) {
#endif
      fTbh1->Clear();
      fTbh1->AddText(0, buf);
      // Re-align the cursor with the characters.
      fTeh1->SetCursorPosition(fTeh1->GetCursorPosition());
      fTeh1->Deselect();
      gClient->NeedRedraw(fTeh1);
#ifdef CINT_FIXED
      break;
   case VSId1:
#else
   }
   else if (id == VSId1) {
#endif
      fTbv1->Clear();
      fTbv1->AddText(0, buf);
      fTev1->SetCursorPosition(fTev1->GetCursorPosition());
      fTev1->Deselect();
      gClient->NeedRedraw(fTev1);
#ifdef CINT_FIXED
      break;
   case HSId2:
#else
   }
   else if (id == HSId2) {
#endif
      fTbh2->Clear();
      fTbh2->AddText(0, buf);
      fTeh2->SetCursorPosition(fTeh2->GetCursorPosition());
      fTeh2->Deselect();
      gClient->NeedRedraw(fTeh2);
#ifdef CINT_FIXED
      break;
   case VSId2:
#else
   }
   else if (id == VSId2) {
#endif
      sprintf(buf, "%f", fVslider2->GetMinPosition());
      fTbv2->Clear();
      fTbv2->AddText(0, buf);
      fTev2->SetCursorPosition(fTev2->GetCursorPosition());
      fTev2->Deselect();
      gClient->NeedRedraw(fTev2);
#ifdef CINT_FIXED
      break;
   default:
      break;
#endif
   }
}


TestShutter::TestShutter(const TGWindow *p, const TGWindow *main,
                         UInt_t w, UInt_t h)
{
   // Create transient frame containing a shutter widget.

   fMain = new TGTransientFrame(p, main, w, h);
   fMain->Connect("CloseWindow()", "TestShutter", this, "CloseWindow()");
   fMain->DontCallClose(); // to avoid double deletions.

   // use hierarchical cleaning
   fMain->SetCleanup(kDeepCleanup);

   fDefaultPic = gClient->GetPicture("folder_s.xpm");
   fShutter = new TGShutter(fMain, kSunkenFrame);

   AddShutterItem("Histograms", histo_data);
   AddShutterItem("Functions", function_data);
   AddShutterItem("Trees", tree_data);

   fLayout = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY);
   fMain->AddFrame(fShutter, fLayout);

   fMain->MapSubwindows();
   fMain->Resize(80, 300);

   // position relative to the parent's window
   fMain->CenterOnParent();

   fMain->SetWindowName("Shutter Test");

   fMain->MapWindow();
   //gClient->WaitFor(fMain);
}

void TestShutter::AddShutterItem(const char *name, shutterData_t *data)
{
   TGShutterItem    *item;
   TGCompositeFrame *container;
   TGButton         *button;
   const TGPicture  *buttonpic;
   static int id = 5001;

   TGLayoutHints *l = new TGLayoutHints(kLHintsTop | kLHintsCenterX,
                                        5, 5, 5, 0);

   item = new TGShutterItem(fShutter, new TGHotString(name), id++);
   container = (TGCompositeFrame *) item->GetContainer();

   for (int i=0; data[i].pixmap_name != 0; i++) {
      buttonpic = gClient->GetPicture(data[i].pixmap_name);
      if (!buttonpic) {
         printf("<TestShutter::AddShutterItem>: missing pixmap \"%s\", using default",
                data[i].pixmap_name);
         buttonpic = fDefaultPic;
      }

      button = new TGPictureButton(container, buttonpic, data[i].id);

      container->AddFrame(button, l);
      button->Connect("Clicked()", "TestShutter", this, "HandleButtons()");
      button->SetToolTipText(data[i].tip_text);
      data[i].button = button;
   }

   fShutter->AddItem(item);
}

TestShutter::~TestShutter()
{
   // dtor

   gClient->FreePicture(fDefaultPic);
   fMain->DeleteWindow();  // deletes fMain
}

void TestShutter::CloseWindow()
{
   delete this;
}

void TestShutter::HandleButtons()
{
   TGButton *btn = (TGButton *) gTQSender;
   printf("Shutter button %d\n", btn->WidgetId());
}


TestDirList::TestDirList(const TGWindow *p, const TGWindow *main,
                         UInt_t w, UInt_t h)
{
   // Create transient frame containing a dirlist widget.

   fMain = new TGTransientFrame(p, main, w, h);
   fMain->Connect("CloseWindow()", "TestDirList", this, "CloseWindow()");
   fMain->DontCallClose(); // to avoid double deletions.

   fIcon = gClient->GetPicture("rootdb_t.xpm");
   TGLayoutHints *lo;

   // use hierarchical cleaning
   fMain->SetCleanup(kDeepCleanup);

   TGCanvas* canvas = new TGCanvas(fMain, 500, 300);
   fContents = new TGListTree(canvas, kHorizontalFrame);
   lo = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY | kLHintsBottom);
   fMain->AddFrame(canvas,lo);
   fContents->Connect("DoubleClicked(TGListTreeItem*,Int_t)","TestDirList",this,
                      "OnDoubleClick(TGListTreeItem*,Int_t)");
   fContents->Connect("Clicked(TGListTreeItem*,Int_t)","TestDirList",this,
                      "OnDoubleClick(TGListTreeItem*,Int_t)");
#ifdef G__WIN32
   fContents->AddItem(0,"c:\\");  // browse the upper directory
#else
   fContents->AddItem(0,"/");  // browse the upper directory
#endif

   // position relative to the parent's window
   fMain->CenterOnParent();

   fMain->SetWindowName("List Dir Test");

   fMain->MapSubwindows();
   fMain->Resize();
   fMain->MapWindow();
}

TestDirList::~TestDirList()
{
   // Cleanup.

   gClient->FreePicture(fIcon);
   delete fContents;
   fMain->DeleteWindow();  // delete fMain
}

void TestDirList::CloseWindow()
{
   delete this;
}

TString TestDirList::DirName(TGListTreeItem* item)
{
   // Returns an absolute path.

   TGListTreeItem* parent;
   TString dirname = item->GetText();

   while ((parent = item->GetParent())) {
      gSystem->PrependPathName(parent->GetText(),dirname);
      item = parent;
   }

   return dirname;
}

void TestDirList::OnDoubleClick(TGListTreeItem* item, Int_t btn)
{
   // Show contents of directory.

   if ((btn!=kButton1) || !item || (Bool_t)item->GetUserData()) return;

   // use UserData to indicate that item was already browsed
   item->SetUserData((void*)1);

   TSystemDirectory dir(item->GetText(),DirName(item));

   TList *files = dir.GetListOfFiles();

   if (files) {
      TIter next(files);
      TSystemFile *file;
      TString fname;

      while ((file=(TSystemFile*)next())) {
         fname = file->GetName();
         if (file->IsDirectory()) {
            if ((fname!="..") && (fname!=".")) { // skip it
               fContents->AddItem(item,fname);
            }
         } else if (fname.EndsWith(".root")) {   // add root files
            fContents->AddItem(item,fname,fIcon,fIcon);
         }
      }
      delete files;
   }
}


TestFileList::TestFileList(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h)
{
   // Create transient frame containing a filelist widget.

   TGLayoutHints *lo;

   fMain = new TGTransientFrame(p, main, w, h);
   fMain->Connect("CloseWindow()", "TestDirList", this, "CloseWindow()");
   fMain->DontCallClose(); // to avoid double deletions.

   // use hierarchical cleaning
   fMain->SetCleanup(kDeepCleanup);

   TGMenuBar* mb = new TGMenuBar(fMain);
   lo = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 1, 1);
   fMain->AddFrame(mb, lo);

   fMenu = mb->AddPopup("&View");
   fMenu->AddEntry("Lar&ge Icons",kLVLargeIcons);
   fMenu->AddEntry("S&mall Icons",kLVSmallIcons);
   fMenu->AddEntry("&List",       kLVList);
   fMenu->AddEntry("&Details",    kLVDetails);
   fMenu->AddSeparator();
   fMenu->AddEntry("&Close",      10);
   fMenu->Connect("Activated(Int_t)","TestFileList",this,"DoMenu(Int_t)");

   TGListView* lv = new TGListView(fMain, w, h);
   lo = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY);
   fMain->AddFrame(lv,lo);

   Pixel_t white;
   gClient->GetColorByName("white", white);
   fContents = new TGFileContainer(lv, kSunkenFrame,white);
   fContents->Connect("DoubleClicked(TGFrame*,Int_t)", "TestFileList", this,
                      "OnDoubleClick(TGLVEntry*,Int_t)");

   // position relative to the parent's window
   fMain->CenterOnParent();

   fMain->SetWindowName("File List Test");
   fMain->MapSubwindows();
   fMain->MapWindow();
   fContents->SetDefaultHeaders();
   fContents->DisplayDirectory();
   fContents->AddFile("..");        // up level directory
   fContents->Resize();
   fContents->StopRefreshTimer();   // stop refreshing
   fMain->Resize();
}

TestFileList::~TestFileList()
{
   // Cleanup.

   delete fContents;
   fMain->DeleteWindow();  // deletes fMain
}

void TestFileList::DoMenu(Int_t mode)
{
   // Switch view mode.

   if (mode<10) {
      fContents->SetViewMode((EListViewMode)mode);
   } else {
      delete this;
   }
}

void TestFileList::DisplayFile(const TString &fname)
{
   // Display content of ROOT file.

   TFile file(fname);
   fContents->RemoveAll();
   fContents->AddFile(gSystem->WorkingDirectory());
   fContents->SetPagePosition(0,0);
   fContents->SetColHeaders("Name","Title");

   TIter next(file.GetListOfKeys());
   TKey *key;

   while ((key=(TKey*)next())) {
      TString cname = key->GetClassName();
      TString name = key->GetName();
      TGLVEntry *entry = new TGLVEntry(fContents,name,cname);
      entry->SetSubnames(key->GetTitle());
      fContents->AddItem(entry);

      // user data is a filename
      entry->SetUserData((void*)StrDup(fname));
   }
   fMain->Resize();
}

void TestFileList::DisplayDirectory(const TString &fname)
{
   // Display content of directory.

   fContents->SetDefaultHeaders();
   gSystem->ChangeDirectory(fname);
   fContents->ChangeDirectory(fname);
   fContents->DisplayDirectory();
   fContents->AddFile("..");  // up level directory
   fMain->Resize();
}

void TestFileList::DisplayObject(const TString& fname,const TString& name)
{
   // Browse object located in file.

   TDirectory *sav = gDirectory;

   static TFile *file = 0;
   if (file) delete file;     // close
   file = new TFile(fname);   // reopen

   TObject* obj = file->Get(name);
   if (obj) {
      if (!obj->IsFolder()) {
         obj->Browse(0);
      } else obj->Print();
   }
   gDirectory = sav;
}

void TestFileList::OnDoubleClick(TGLVEntry *f, Int_t btn)
{
   // Handle double click.

   if (btn != kButton1) return;

   // set kWatch cursor
   ULong_t cur = gVirtualX->CreateCursor(kWatch);
   gVirtualX->SetCursor(fContents->GetId(), cur);

   TString name(f->GetTitle());
   const char* fname = (const char*)f->GetUserData();

   if (fname) {
      DisplayObject(fname, name);
   } else if (name.EndsWith(".root")) {
      DisplayFile(name);
   } else {
      DisplayDirectory(name);
   }
   // set kPointer cursor
   cur = gVirtualX->CreateCursor(kPointer);
   gVirtualX->SetCursor(fContents->GetId(), cur);
}

void TestFileList::CloseWindow()
{
   delete this;
}

TestProgress::TestProgress(const TGWindow *p, const TGWindow *main,
                           UInt_t w, UInt_t h)
{
   // Dialog used to test the different supported progress bars.

   fClose = kTRUE;

   fMain = new TGTransientFrame(p, main, w, h);
   fMain->Connect("CloseWindow()", "TestProgress", this, "DoClose()");
   fMain->DontCallClose();

   // use hierarchical cleaning
   fMain->SetCleanup(kDeepCleanup);

   fMain->ChangeOptions((fMain->GetOptions() & ~kVerticalFrame) | kHorizontalFrame);

   fHframe1 = new TGHorizontalFrame(fMain, 0, 0, 0);

   fVProg1 = new TGVProgressBar(fHframe1, TGProgressBar::kFancy, 300);
   fVProg1->SetBarColor("purple");
   fVProg2 = new TGVProgressBar(fHframe1, TGProgressBar::kFancy, 300);
   fVProg2->SetFillType(TGProgressBar::kBlockFill);
   fVProg2->SetBarColor("green");

   fHframe1->Resize(300, 300);

   fVframe1 = new TGVerticalFrame(fMain, 0, 0, 0);

   fHProg1 = new TGHProgressBar(fVframe1, 300);
   fHProg1->ShowPosition();
   fHProg2 = new TGHProgressBar(fVframe1, TGProgressBar::kFancy, 300);
   fHProg2->SetBarColor("lightblue");
   fHProg2->ShowPosition(kTRUE, kFALSE, "%.0f events");
   fHProg3 = new TGHProgressBar(fVframe1, TGProgressBar::kStandard, 300);
   fHProg3->SetFillType(TGProgressBar::kBlockFill);

   fGO = new TGTextButton(fVframe1, "Go", 10);
   fGO->Connect("Clicked()", "TestProgress", this, "DoGo()");

   fVframe1->Resize(300, 300);

   fHint1 = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandY, 5, 10, 5, 5);
   fHint2 = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 5, 5,  5, 10);
   fHint3 = new TGLayoutHints(kLHintsTop | kLHintsRight, 0, 50, 50, 0);
   fHint4 = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandY, 0, 0, 0, 0);
   fHint5 = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX, 0, 0, 0, 0);

   fHframe1->AddFrame(fVProg1, fHint1);
   fHframe1->AddFrame(fVProg2, fHint1);

   fVframe1->AddFrame(fHProg1, fHint2);
   fVframe1->AddFrame(fHProg2, fHint2);
   fVframe1->AddFrame(fHProg3, fHint2);
   fVframe1->AddFrame(fGO,     fHint3);

   fMain->AddFrame(fHframe1, fHint4);
   fMain->AddFrame(fVframe1, fHint5);

   fMain->SetWindowName("Progress Test");
   TGDimension size = fMain->GetDefaultSize();
   fMain->Resize(size);

   // position relative to the parent's window
   fMain->CenterOnParent();

   fMain->MapSubwindows();
   fMain->MapWindow();

   gClient->WaitFor(fMain);
}

TestProgress::~TestProgress()
{
   // Delete dialog.

   fMain->DeleteWindow();   // deletes fMain
}

void TestProgress::CloseWindow()
{
   // Called when window is closed via the window manager.

   delete this;
}

void TestProgress::DoClose()
{
   // If fClose is false we are still in event processing loop in DoGo().
   // In that case, set the close flag true and use a timer to call
   // CloseWindow(). This gives us change to get out of the DoGo() loop.
   // Note: calling SendCloseMessage() will not work since that will
   // bring us back here (CloseWindow() signal is connected to this method)
   // with the fClose flag true, which will cause window deletion while
   // still being in the event processing loop (since SendCloseMessage()
   // is directly processed in ProcessEvents() without exiting DoGo()).

   if (fClose)
      CloseWindow();
   else {
      fClose = kTRUE;
      TTimer::SingleShot(150, "TestProgress", this, "CloseWindow()");
   }
}

void TestProgress::DoGo()
{
   // Handle Go button.

   fClose = kFALSE;
   fVProg1->Reset(); fVProg2->Reset();
   fHProg1->Reset(); fHProg2->Reset(); fHProg3->Reset();
   fVProg2->SetBarColor("green");
   int cnt1 = 0, cnt2 = 0, cnt3 = 0, cnt4 = 0;
   int inc1 = 4, inc2 = 3, inc3 = 2, inc4 = 1;
   while (cnt1 < 100 || cnt2 < 100 || cnt3 < 100 || cnt4 <100) {
      if (cnt1 < 100) {
         cnt1 += inc1;
         fVProg1->Increment(inc1);
      }
      if (cnt2 < 100) {
         cnt2 += inc2;
         fVProg2->Increment(inc2);
         if (cnt2 > 75)
            fVProg2->SetBarColor("red");
      }
      if (cnt3 < 100) {
         cnt3 += inc3;
         fHProg1->Increment(inc3);
      }
      if (cnt4 < 100) {
         cnt4 += inc4;
         fHProg2->Increment(inc4);
         fHProg3->Increment(inc4);
      }
      gSystem->Sleep(100);
      gSystem->ProcessEvents();
      // if user closed window return
      if (fClose) return;
   }
   fClose = kTRUE;
}


// TGNumberEntry widget test dialog
//const char *const EntryTestDlg::numlabel[] = {
const char *numlabel[] = {
   "Integer",
   "One digit real",
   "Two digit real",
   "Three digit real",
   "Four digit real",
   "Real",
   "Degree.min.sec",
   "Min:sec",
   "Hour:min",
   "Hour:min:sec",
   "Day/month/year",
   "Month/day/year",
   "Hex"
};

//const Double_t EntryTestDlg::numinit[] = {
const Double_t numinit[] = {
   12345, 1.0, 1.00, 1.000, 1.0000, 1.2E-12,
   90 * 3600, 120 * 60, 12 * 60, 12 * 3600 + 15 * 60,
   19991121, 19991121, (Double_t) 0xDEADFACEU
};

EntryTestDlg::EntryTestDlg(const TGWindow *p, const TGWindow *main)
{
   // build widgets
   fMain = new TGTransientFrame(p, main, 10, 10, kHorizontalFrame);
   fMain->Connect("CloseWindow()", "EntryTestDlg", this, "CloseWindow()");
   fMain->DontCallClose(); // to avoid double deletions.

   // use hierarchical cleaning
   fMain->SetCleanup(kDeepCleanup);

   TGGC myGC = *gClient->GetResourcePool()->GetFrameGC();
   TGFont *myfont = gClient->GetFont("-adobe-helvetica-bold-r-*-*-12-*-*-*-*-*-iso8859-1");
   if (myfont) myGC.SetFont(myfont->GetFontHandle());

   fF1 = new TGVerticalFrame(fMain, 200, 300);
   fL1 = new TGLayoutHints(kLHintsTop | kLHintsLeft, 2, 2, 2, 2);
   fMain->AddFrame(fF1, fL1);
   fL2 = new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 2, 2, 2, 2);
   for (int i = 0; i < 13; i++) {
      fF[i] = new TGHorizontalFrame(fF1, 200, 30);
      fF1->AddFrame(fF[i], fL2);
      fNumericEntries[i] = new TGNumberEntry(fF[i], numinit[i], 12, i + 20,
                                             (TGNumberFormat::EStyle) i);
      fF[i]->AddFrame(fNumericEntries[i], fL2);
      fLabel[i] = new TGLabel(fF[i], numlabel[i], myGC(), myfont->GetFontStruct());
      fF[i]->AddFrame(fLabel[i], fL2);
   }
   fF2 = new TGVerticalFrame(fMain, 200, 500);
   fL3 = new TGLayoutHints(kLHintsTop | kLHintsLeft, 2, 2, 2, 2);
   fMain->AddFrame(fF2, fL3);
   fLowerLimit = new TGCheckButton(fF2, "lower limit:", 4);
   fF2->AddFrame(fLowerLimit, fL3);
   fLimits[0] = new TGNumberEntry(fF2, 0, 12, 10);
   fLimits[0]->SetLogStep(kFALSE);
   fF2->AddFrame(fLimits[0], fL3);
   fUpperLimit = new TGCheckButton(fF2, "upper limit:", 5);
   fF2->AddFrame(fUpperLimit, fL3);
   fLimits[1] = new TGNumberEntry(fF2, 0, 12, 11);
   fLimits[1]->SetLogStep(kFALSE);
   fF2->AddFrame(fLimits[1], fL3);
   fPositive = new TGCheckButton(fF2, "Positive", 6);
   fF2->AddFrame(fPositive, fL3);
   fNonNegative = new TGCheckButton(fF2, "Non negative", 7);
   fF2->AddFrame(fNonNegative, fL3);
   fSetButton = new TGTextButton(fF2, " Set ", 2);
   fSetButton->Connect("Clicked()", "EntryTestDlg", this, "SetLimits()");
   fF2->AddFrame(fSetButton, fL3);
   fExitButton = new TGTextButton(fF2, " Close ", 1);
   fExitButton->Connect("Clicked()", "EntryTestDlg", this, "DoOK()");
   fF2->AddFrame(fExitButton, fL3);

   // set dialog box title
   fMain->SetWindowName("Number Entry Test");
   fMain->SetIconName("Number Entry Test");
   fMain->SetClassHints("NumberEntryDlg", "NumberEntryDlg");
   // resize & move to center
   fMain->MapSubwindows();
   UInt_t width = fMain->GetDefaultWidth();
   UInt_t height = fMain->GetDefaultHeight();
   fMain->Resize(width, height);
   fMain->CenterOnParent();
   // make the message box non-resizable
   fMain->SetWMSize(width, height);
   fMain->SetWMSizeHints(width, height, width, height, 0, 0);
   fMain->SetMWMHints(kMWMDecorAll | kMWMDecorResizeH | kMWMDecorMaximize |
                      kMWMDecorMinimize | kMWMDecorMenu,
                      kMWMFuncAll | kMWMFuncResize | kMWMFuncMaximize |
                      kMWMFuncMinimize, kMWMInputModeless);

   fMain->MapWindow();
   gClient->WaitFor(fMain);
}

EntryTestDlg::~EntryTestDlg()
{
   // dtor

   fMain->DeleteWindow();
}

void EntryTestDlg::CloseWindow()
{
   delete this;
}

void EntryTestDlg::DoOK()
{
   // Handle ok button.

   fMain->SendCloseMessage();
}

void EntryTestDlg::SetLimits()
{
   Double_t min = fLimits[0]->GetNumber();
   Bool_t low = (fLowerLimit->GetState() == kButtonDown);
   Double_t max = fLimits[1]->GetNumber();
   Bool_t high = (fUpperLimit->GetState() == kButtonDown);
   TGNumberFormat::ELimit lim;
   if (low && high) {
      lim = TGNumberFormat::kNELLimitMinMax;
   } else if (low) {
      lim = TGNumberFormat::kNELLimitMin;
   } else if (high) {
      lim = TGNumberFormat::kNELLimitMax;
   } else {
      lim = TGNumberFormat::kNELNoLimits;
   }
   Bool_t pos = (fPositive->GetState() == kButtonDown);
   Bool_t nneg = (fNonNegative->GetState() == kButtonDown);
   TGNumberFormat::EAttribute attr;
   if (pos) {
      attr = TGNumberFormat::kNEAPositive;
   } else if (nneg) {
      attr = TGNumberFormat::kNEANonNegative;
   } else {
      attr = TGNumberFormat::kNEAAnyNumber;
   }
   for (int i = 0; i < 13; i++) {
      fNumericEntries[i]->SetFormat(fNumericEntries[i]->GetNumStyle(), attr);
      fNumericEntries[i]->SetLimits(lim, min, max);
   }
}


Editor::Editor(const TGWindow *main, UInt_t w, UInt_t h)
{
   // Create an editor in a dialog.

   fMain = new TGTransientFrame(gClient->GetRoot(), main, w, h);
   fMain->Connect("CloseWindow()", "Editor", this, "CloseWindow()");
   fMain->DontCallClose(); // to avoid double deletions.

   // use hierarchical cleaning
   fMain->SetCleanup(kDeepCleanup);

   fEdit = new TGTextEdit(fMain, w, h, kSunkenFrame | kDoubleBorder);
   fL1 = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 3, 3, 3, 3);
   fMain->AddFrame(fEdit, fL1);
   fEdit->Connect("Opened()", "Editor", this, "DoOpen()");
   fEdit->Connect("Saved()",  "Editor", this, "DoSave()");
   fEdit->Connect("Closed()", "Editor", this, "DoClose()");

   // set selected text colors
   Pixel_t pxl;
   gClient->GetColorByName("#3399ff", pxl);
   fEdit->SetSelectBack(pxl);
   fEdit->SetSelectFore(TGFrame::GetWhitePixel());

   fOK = new TGTextButton(fMain, "  &OK  ");
   fOK->Connect("Clicked()", "Editor", this, "DoOK()");
   fL2 = new TGLayoutHints(kLHintsBottom | kLHintsCenterX, 0, 0, 5, 5);
   fMain->AddFrame(fOK, fL2);

   SetTitle();

   fMain->MapSubwindows();

   fMain->Resize();

   // editor covers right half of parent window
   fMain->CenterOnParent(kTRUE, TGTransientFrame::kRight);
}

Editor::~Editor()
{
   // Delete editor dialog.

   fMain->DeleteWindow();  // deletes fMain
}

void Editor::SetTitle()
{
   // Set title in editor window.

   TGText *txt = GetEditor()->GetText();
   Bool_t untitled = !strlen(txt->GetFileName()) ? kTRUE : kFALSE;

   char title[256];
   if (untitled)
      sprintf(title, "ROOT Editor - Untitled");
   else
      sprintf(title, "ROOT Editor - %s", txt->GetFileName());

   fMain->SetWindowName(title);
   fMain->SetIconName(title);
}

void Editor::Popup()
{
   // Show editor.

   fMain->MapWindow();
}

void Editor::LoadBuffer(const char *buffer)
{
   // Load a text buffer in the editor.

   fEdit->LoadBuffer(buffer);
}

void Editor::LoadFile(const char *file)
{
   // Load a file in the editor.

   fEdit->LoadFile(file);
}

void Editor::AddBuffer(const  char *buffer)
{
   // Add text to the editor.

   TGText txt;
   txt.LoadBuffer(buffer);
   fEdit->AddText(&txt);
}

void Editor::CloseWindow()
{
   // Called when closed via window manager action.

   delete this;
}

void Editor::DoOK()
{
   // Handle ok button.

   CloseWindow();
}

void Editor::DoOpen()
{
   SetTitle();
}

void Editor::DoSave()
{
   SetTitle();
}

void Editor::DoClose()
{
   // Handle close button.

   CloseWindow();
}


void guitest()
{
   new TestMainFrame(gClient->GetRoot(), 400, 220);
}

//---- Main program ------------------------------------------------------------
#ifdef STANDALONE
int main(int argc, char **argv)
{
   TApplication theApp("App", &argc, argv);

   if (gROOT->IsBatch()) {
      fprintf(stderr, "%s: cannot run in batch mode\n", argv[0]);
      return 1;
   }

   guitest();

   theApp.Run();

   return 0;
}
#endif

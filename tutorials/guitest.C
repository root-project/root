// @(#)root/tutorials:$Name:  $:$Id: guitest.C,v 1.5 2001/05/28 10:45:14 rdm Exp $
// Author: Fons Rademakers   22/10/2000

// guitest.C: test program for ROOT native GUI classes exactly like
// $ROOTSYS/test/guitest.cxx but using the new signal and slots
// communication mechanism. It is now possible to run this entire
// test program in the interpreter. Do either:
// .x guitest.C
// .x guitest.C++

#include <stdlib.h>

#include <TROOT.h>
#include <TApplication.h>
#include <TVirtualX.h>

#include <TGListBox.h>
#include <TGClient.h>
#include <TGFrame.h>
#include <TGIcon.h>
#include <TGLabel.h>
#include <TGButton.h>
#include <TGTextEntry.h>
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
#include <RQ_OBJECT.h>
#include <TRootEmbeddedCanvas.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TH2.h>
#include <TRandom.h>
#include <TSystem.h>
#include <TEnv.h>


enum ETestCommandIdentifiers {
   M_FILE_OPEN,
   M_FILE_SAVE,
   M_FILE_SAVEAS,
   M_FILE_EXIT,

   M_TEST_DLG,
   M_TEST_MSGBOX,
   M_TEST_SLIDER,
   M_TEST_SHUTTER,
   M_TEST_PROGRESS,

   M_HELP_CONTENTS,
   M_HELP_SEARCH,
   M_HELP_ABOUT,

   M_CASCADE_1,
   M_CASCADE_2,
   M_CASCADE_3,

   VId1,
   HId1,
   VId2,
   HId2,

   VSId1,
   HSId1,
   VSId2,
   HSId2
};


Int_t mb_button_id[9] = { kMBYes, kMBNo, kMBOk, kMBApply,
                          kMBRetry, kMBIgnore, kMBCancel,
                          kMBClose, kMBDismiss };

EMsgBoxIcon mb_icon[4] = { kMBIconStop, kMBIconQuestion,
                           kMBIconExclamation, kMBIconAsterisk };

const char *filetypes[] = { "All files",     "*",
                            "ROOT files",    "*.root",
                            "ROOT macros",   "*.C",
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
"standard X11 text handling application is supported.\n\n"
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
"    Move the cursor to the beginning of the line.\n"
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
"    Move the cursor one character rightwards.\n"
"Control-H\n"
"    Delete the character to the left of the cursor.\n"
"Control-K\n"
"    Delete marked text if any or delete all\n"
"    characters to the right of the cursor."
;
const char *editortxt6 =
"Control-U\n"
"    Delete all characters on the line.\n"
"Control-V\n"
"    Paste the clipboard text into line edit.\n"
"Control-X\n"
"    Cut the marked text, copy to clipboard.\n"
"Control-Y\n"
"    Paste the clipboard text into line edit.\n\n"
"All other keys with valid ASCII codes insert themselves into the line.";


class TileFrame;


class TestMainFrame {

RQ_OBJECT()

private:
   TGMainFrame        *fMain;
   TGCompositeFrame   *fStatusFrame;
   TGCanvas           *fCanvasWindow;
   TileFrame          *fContainer;
   TGTextEntry        *fTestText;
   TGButton           *fTestButton;

   TGMenuBar          *fMenuBar;
   TGPopupMenu        *fMenuFile, *fMenuTest, *fMenuHelp;
   TGPopupMenu        *fCascadeMenu, *fCascade1Menu, *fCascade2Menu;
   TGLayoutHints      *fMenuBarLayout, *fMenuBarItemLayout, *fMenuBarHelpLayout;

public:
   TestMainFrame(const TGWindow *p, UInt_t w, UInt_t h);
   virtual ~TestMainFrame();

   // slots
   void CloseWindow();
   void DoButton();
   void HandleMenu(Int_t id);
};

class TestDialog {

RQ_OBJECT()

private:
   TGTransientFrame    *fMain;
   TGCompositeFrame    *fFrame1, *fF1, *fF2, *fF3, *fF4, *fF5, *fF6, *fF7;
   TGButton            *fOkButton, *fCancelButton, *fStartB, *fStopB;
   TGButton            *fBtn1, *fBtn2, *fChk1, *fChk2, *fRad1, *fRad2;
   TGPictureButton     *fPicBut1;
   TGRadioButton       *fRadio1, *fRadio2;
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
   TList               *fCleanup;

   void FillHistos();

public:
   TestDialog(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h,
               UInt_t options = kVerticalFrame);
   virtual ~TestDialog();

   // slots
   void CloseWindow();
   void DoOK();
   void DoCancel();
   void DoTab(Int_t id);
   void HandleButtons(Int_t id = -1);
};

class TestMsgBox {

RQ_OBJECT()

private:
   TGTransientFrame     *fMain;
   TGCompositeFrame     *f1, *f2, *f3, *f4, *f5;
   TGButton             *fTestButton, *fCloseButton;
   TGPictureButton      *fPictButton;
   TGRadioButton        *fR[4];
   TGCheckButton        *fC[9];
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
   void CloseWindow();
   void DoClose();
   void DoRadio();
   void DoTest();
};


class TestSliders {

RQ_OBJECT()

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

RQ_OBJECT()

private:
   TGTransientFrame *fMain;
   TGShutter        *fShutter;
   TGLayoutHints    *fLayout;
   const TGPicture  *fDefaultPic;
   TList            *fTrash;

public:
   TestShutter(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h);
   ~TestShutter();

   void AddShutterItem(const char *name, shutterData_t data[]);

   // slots
   void CloseWindow();
   void HandleButtons();
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

RQ_OBJECT()

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

   fMain   = new TGMainFrame(p, w, h);
   fMain->Connect("CloseWindow()", "TestMainFrame", this, "CloseWindow()");

   // Create menubar and popup menus. The hint objects are used to place
   // and group the different menu widgets with respect to eachother.
   fMenuBarLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft | kLHintsExpandX,
                                      0, 0, 1, 1);
   fMenuBarItemLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0);
   fMenuBarHelpLayout = new TGLayoutHints(kLHintsTop | kLHintsRight);

   fMenuFile = new TGPopupMenu(gClient->GetRoot());
   fMenuFile->AddEntry("&Open...", M_FILE_OPEN);
   fMenuFile->AddEntry("&Save", M_FILE_SAVE);
   fMenuFile->AddEntry("S&ave as...", M_FILE_SAVEAS);
   fMenuFile->AddEntry("&Close", -1);
   fMenuFile->AddSeparator();
   fMenuFile->AddEntry("&Print", -1);
   fMenuFile->AddEntry("P&rint setup...", -1);
   fMenuFile->AddSeparator();
   fMenuFile->AddEntry("E&xit", M_FILE_EXIT);

   fMenuFile->DisableEntry(M_FILE_SAVEAS);

   fCascade2Menu = new TGPopupMenu(gClient->GetRoot());
   fCascade2Menu->AddEntry("ID = 2&1", M_CASCADE_1);
   fCascade2Menu->AddEntry("ID = 2&2", M_CASCADE_2);
   fCascade2Menu->AddEntry("ID = 2&3", M_CASCADE_3);

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
   fCascadeMenu->AddPopup("C&ascaded 2", fCascade2Menu);

   fMenuTest = new TGPopupMenu(gClient->GetRoot());
   fMenuTest->AddLabel("Test different features...");
   fMenuTest->AddSeparator();
   fMenuTest->AddEntry("&Dialog...", M_TEST_DLG);
   fMenuTest->AddEntry("&Message Box...", M_TEST_MSGBOX);
   fMenuTest->AddEntry("&Sliders...", M_TEST_SLIDER);
   fMenuTest->AddEntry("Sh&utter...", M_TEST_SHUTTER);
   fMenuTest->AddEntry("&Progress...", M_TEST_PROGRESS);
   fMenuTest->AddSeparator();
   fMenuTest->AddPopup("&Cascaded menus", fCascadeMenu);

   fMenuHelp = new TGPopupMenu(gClient->GetRoot());
   fMenuHelp->AddEntry("&Contents", M_HELP_CONTENTS);
   fMenuHelp->AddEntry("&Search...", M_HELP_SEARCH);
   fMenuHelp->AddSeparator();
   fMenuHelp->AddEntry("&About", M_HELP_ABOUT);

   // Menu button messages are handled by the main frame (i.e. "this")
   // HandleMenu() method.
   fMenuFile->Connect("Activated(Int_t)", "TestMainFrame", this,
                      "HandleMenu(Int_t)");
   fMenuTest->Connect("Activated(Int_t)", "TestMainFrame", this,
                      "HandleMenu(Int_t)");
   fMenuHelp->Connect("Activated(Int_t)", "TestMainFrame", this,
                      "HandleMenu(Int_t)");
   fCascadeMenu->Connect("Activated(Int_t)", "TestMainFrame", this,
                         "HandleMenu(Int_t)");
   fCascade1Menu->Connect("Activated(Int_t)", "TestMainFrame", this,
                          "HandleMenu(Int_t)");
   fCascade2Menu->Connect("Activated(Int_t)", "TestMainFrame", this,
                          "HandleMenu(Int_t)");

   fMenuBar = new TGMenuBar(fMain, 1, 1, kHorizontalFrame);
   fMenuBar->AddPopup("&File", fMenuFile, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Test", fMenuTest, fMenuBarItemLayout);
   fMenuBar->AddPopup("&Help", fMenuHelp, fMenuBarHelpLayout);

   fMain->AddFrame(fMenuBar, fMenuBarLayout);

   // Create TGCanvas and a canvas container which uses a tile layout manager
   fCanvasWindow = new TGCanvas(fMain, 400, 240);
   fContainer = new TileFrame(fCanvasWindow->GetViewPort());
   fContainer->SetCanvas(fCanvasWindow);
   fCanvasWindow->SetContainer(fContainer->GetFrame());

   // Fill canvas with 256 colored frames (notice that doing it this way
   // causes a memory leak when deleting the canvas (i.e. the TGFrame's and
   // TGLayoutHints are NOT adopted by the canvas).
   // Best is to create a TList and store all frame and layout pointers
   // in it. When deleting the canvas just delete contents of list.
   for (int i=0; i < 256; ++i)
      fCanvasWindow->AddFrame(new TGFrame(fCanvasWindow->GetContainer(),
                                          32, 32, 0, (i+1)&255),
                              new TGLayoutHints(kLHintsExpandY | kLHintsRight));

   fMain->AddFrame(fCanvasWindow, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,
                                                    0, 0, 2, 2));

   // Create status frame containing a button and a text entry widget
   fStatusFrame = new TGCompositeFrame(fMain, 60, 20, kHorizontalFrame |
                                                      kSunkenFrame);

   fTestButton = new TGTextButton(fStatusFrame, "&Open editor...", 150);
   fTestButton->Connect("Clicked()", "TestMainFrame", this, "DoButton()");
   fTestButton->SetToolTipText("Pops up editor");
   fStatusFrame->AddFrame(fTestButton, new TGLayoutHints(kLHintsTop |
                          kLHintsLeft, 2, 0, 2, 2));
   fTestText = new TGTextEntry(fStatusFrame, new TGTextBuffer(100));
   fTestText->SetToolTipText("This is a text entry widget");
   fTestText->Resize(300, fTestText->GetDefaultHeight());
   fStatusFrame->AddFrame(fTestText, new TGLayoutHints(kLHintsTop | kLHintsLeft,
                                                       10, 2, 2, 2));
   fMain->AddFrame(fStatusFrame, new TGLayoutHints(kLHintsBottom | kLHintsExpandX,
                   0, 0, 1, 0));

   fMain->SetWindowName("GuiTest Signal/Slots");

   fMain->MapSubwindows();

   // we need to use GetDefault...() to initialize the layout algorithm...
   fMain->Resize(fMain->GetDefaultSize());
   //Resize(400, 200);

   fMain->MapWindow();
}

TestMainFrame::~TestMainFrame()
{
   // Delete all created widgets.

   delete fTestButton;
   delete fTestText;

   delete fStatusFrame;
   delete fContainer;
   delete fCanvasWindow;

   delete fMenuBarLayout;
   delete fMenuBarItemLayout;
   delete fMenuBarHelpLayout;

   delete fMenuFile;
   delete fMenuTest;
   delete fMenuHelp;
   delete fCascadeMenu;
   delete fCascade1Menu;
   delete fCascade2Menu;

   delete fMain;
}

void TestMainFrame::CloseWindow()
{
   // Got close message for this MainFrame. Calls parent CloseWindow()
   // (which destroys the window) and terminate the application.
   // The close message is generated by the window manager when its close
   // window menu item is selected.

   gApplication->Terminate(0);
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
            TGFileInfo fi;
            fi.fFileTypes = (char **)filetypes;
            new TGFileDialog(gClient->GetRoot(), fMain, kFDOpen,&fi);
         }
         break;

      case M_TEST_DLG:
         new TestDialog(gClient->GetRoot(), fMain, 400, 200);
         break;

      case M_FILE_SAVE:
         printf("M_FILE_SAVE\n");
         break;

      case M_FILE_EXIT:
         CloseWindow();   // terminate theApp no need to use SendCloseMessage()
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

      case M_TEST_PROGRESS:
         new TestProgress(gClient->GetRoot(), fMain, 600, 300);
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
   fMain->Connect("CloseWindow()", "TestDialog", this, "CloseWindow()");

   // Used to store GUI elements that need to be deleted in the ctor.
   fCleanup = new TList;

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
   fRad1->Connect("Pressed()", "TestDialog", this, "HandleButtons()");
   fRad2->Connect("Pressed()", "TestDialog", this, "HandleButtons()");

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

   // make tab yellow
   ULong_t yellow;
   gClient->GetColorByName("yellow", yellow);
   TGTabElement *tabel = fTab->GetTabTab(2);
   tabel->ChangeBackground(yellow);

   //-------------- end embedded canvas demo

   TGTextButton *bt;
   tf = fTab->AddTab("Tab 4");
   fF4 = new TGCompositeFrame(tf, 60, 20, kVerticalFrame);
   fF4->AddFrame(bt = new TGTextButton(fF4, "A&dd Entry", 90), fL3);
   bt->Connect("Clicked()", "TestDialog", this, "HandleButtons()");
   fCleanup->Add(bt);
   fF4->AddFrame(bt = new TGTextButton(fF4, "Remove &Entry", 91), fL3);
   bt->Connect("Clicked()", "TestDialog", this, "HandleButtons()");
   fCleanup->Add(bt);
   fF4->AddFrame(fListBox = new TGListBox(fF4, 89), fL3);
   fF4->AddFrame(fCheckMulti = new TGCheckButton(fF4, "&Mutli Selectable", 92), fL3);
   fCheckMulti->Connect("Clicked()", "TestDialog", this, "HandleButtons()");
   tf->AddFrame(fF4, fL3);

   for (i = 0; i < 20; ++i) {
      sprintf(tmp, "Entry %i", i+1);
      fListBox->AddEntry(tmp, i+1);
   }
   fFirstEntry = 1;
   fLastEntry  = 20;

   fListBox->Resize(150, 80);

   //--- tab 5
   tf = fTab->AddTab("Tab 5");
   tf->SetLayoutManager(new TGHorizontalLayout(tf));

   fF6 = new TGGroupFrame(tf, "Options", kVerticalFrame);
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
      fCleanup->Add(tent);
      fF6->AddFrame(tent);
   }
   fF6->Resize(fF6->GetDefaultSize());

   // another matrix with text and buttons
   fF7 = new TGGroupFrame(tf, "Tab Handling", kVerticalFrame);
   tf->AddFrame(fF7, fL3);

   fF7->SetLayoutManager(new TGMatrixLayout(fF7, 0, 1, 10));

   fF7->AddFrame(bt = new TGTextButton(fF7, "Remove Tab", 101));
   bt->Connect("Clicked()", "TestDialog", this, "HandleButtons()");
   bt->Resize(90, bt->GetDefaultHeight());
   fCleanup->Add(bt);

   fF7->AddFrame(bt = new TGTextButton(fF7, "Add Tab", 103));
   bt->Connect("Clicked()", "TestDialog", this, "HandleButtons()");
   bt->Resize(90, bt->GetDefaultHeight());
   fCleanup->Add(bt);

   fF7->AddFrame(bt = new TGTextButton(fF7, "Remove Tab 5", 102));
   bt->Connect("Clicked()", "TestDialog", this, "HandleButtons()");
   bt->Resize(90, bt->GetDefaultHeight());
   fCleanup->Add(bt);

   fF7->Resize(fF6->GetDefaultSize());

   //--- end of last tab

   TGLayoutHints *fL5 = new TGLayoutHints(kLHintsBottom | kLHintsExpandX |
                                          kLHintsExpandY, 2, 2, 5, 1);
   fMain->AddFrame(fTab, fL5);

   fMain->MapSubwindows();
   fMain->Resize(fMain->GetDefaultSize());

   // position relative to the parent's window
   Window_t wdum;
   int ax, ay;
   gVirtualX->TranslateCoordinates(main->GetId(), fMain->GetParent()->GetId(),
                     (((TGFrame *) main)->GetWidth() - fMain->GetWidth()) >> 1,
                     (((TGFrame *) main)->GetHeight() - fMain->GetHeight()) >> 1,
                     ax, ay, wdum);
   fMain->Move(ax, ay);

   fMain->SetWindowName("Dialog");

   fMain->MapWindow();
   //gClient->WaitFor(fMain);    // otherwise canvas contextmenu does not work
}

TestDialog::~TestDialog()
{
   // Delete test dialog widgets.

   fCleanup->Delete();
   delete fCleanup;
   delete fOkButton;
   delete fCancelButton;
   delete fStartB; delete fStopB;
   delete fFrame1;
   delete fBtn1; delete fBtn2;
   delete fChk1; delete fChk2;
   delete fRad1; delete fRad2;
   delete fTxt1; delete fTxt2;
   delete fEc1; delete fEc2;
   delete fListBox; delete fCheckMulti;
   delete fF1; delete fF2; delete fF3; delete fF4; delete fF5;
   delete fF6; delete fF7;
   delete fCombo;
   delete fTab;
   delete fL3; delete fL4;
   delete fL1; delete fL2;
   delete fHpx; delete fHpxpy;

   delete fMain;
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

void TestDialog::CloseWindow()
{
   // Called when window is closed via the window manager.

   delete this;
}

void TestDialog::DoOK()
{
   printf("\nTerminating dialog: OK pressed\n");

   // Send a close message to the main frame. This will trigger the
   // emission of a CloseWindow() signal, which will then call
   // TestDialog::CloseWindow(). Calling directly CloseWindow() will cause
   // a segv since the OK button is still accessed after the DoOK() method.
   // This works since the close message is handled synchronous (via
   // message going to/from X server).
   fMain->SendCloseMessage();

   // The same effect can be obtained by using a singleshot timer:
   //TTimer::SingleShot(50, "TestDialog", this, "CloseWindow()");
}

void TestDialog::DoCancel()
{
   printf("\nTerminating dialog: Cancel pressed\n");
   fMain->SendCloseMessage();
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
         if (fFirstEntry < fLastEntry) {
            fListBox->RemoveEntry(fFirstEntry);
            fListBox->Layout();
            fFirstEntry++;
         }
         break;
      case 101:  // remove tabs
         {
            TString s = fTab->GetTabTab(0)->GetString();
            if (s == "Tab 3") {
               // Need to delete the embedded canvases
               // since RemoveTab() will Destroy the container
               // window, which in turn will destroy the embedded
               // canvas windows.
               SafeDelete(fEc1);
               SafeDelete(fEc2);
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


TestMsgBox::TestMsgBox(const TGWindow *p, const TGWindow *main,
                       UInt_t w, UInt_t h, UInt_t options) :
     fRedTextGC(TGButton::GetDefaultGC())
{
   // Create message box test dialog. Use this dialog to select the different
   // message dialog box styles and show the message dialog by clicking the
   // "Test" button.

   fMain = new TGTransientFrame(p, main, w, h, options);
   fMain->Connect("CloseWindow()", "TestMsgBox", this, "CloseWindow()");

   //------------------------------
   // Set foreground color in graphics context for drawing of
   // TGlabel and TGButtons with text in red.

   ULong_t red;
   gClient->GetColorByName("red", red);
   fRedTextGC.SetForeground(red);
   //---------------------------------

   int i, ax, ay;

   fMain->ChangeOptions((fMain->GetOptions() & ~kVerticalFrame) | kHorizontalFrame);

   f1 = new TGCompositeFrame(fMain, 60, 20, kVerticalFrame | kFixedWidth);
   f2 = new TGCompositeFrame(fMain, 60, 20, kVerticalFrame);
   f3 = new TGCompositeFrame(f2, 60, 20, kHorizontalFrame);
   f4 = new TGCompositeFrame(f2, 60, 20, kHorizontalFrame);
   f5 = new TGCompositeFrame(f2, 60, 20, kHorizontalFrame);

   fTestButton = new TGTextButton(f1, "&Test", 1, fRedTextGC());
   fTestButton->Connect("Clicked()", "TestMsgBox", this, "DoTest()");

   // Change background of fTestButton to green
   ULong_t green;
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

   fG1 = new TGGroupFrame(f3, new TGString("Buttons"));
   fG2 = new TGGroupFrame(f3, new TGString("Icons"));

   fL3 = new TGLayoutHints(kLHintsTop | kLHintsLeft |
                           kLHintsExpandX | kLHintsExpandY,
                           2, 2, 2, 2);
   fL4 = new TGLayoutHints(kLHintsTop | kLHintsLeft,
                           0, 0, 5, 0);

   fC[0] = new TGCheckButton(fG1, new TGHotString("Yes"),     -1);
   fC[1] = new TGCheckButton(fG1, new TGHotString("No"),      -1);
   fC[2] = new TGCheckButton(fG1, new TGHotString("OK"),      -1);
   fC[3] = new TGCheckButton(fG1, new TGHotString("Apply"),   -1);
   fC[4] = new TGCheckButton(fG1, new TGHotString("Retry"),   -1);
   fC[5] = new TGCheckButton(fG1, new TGHotString("Ignore"),  -1);
   fC[6] = new TGCheckButton(fG1, new TGHotString("Cancel"),  -1);
   fC[7] = new TGCheckButton(fG1, new TGHotString("Close"),   -1);
   fC[8] = new TGCheckButton(fG1, new TGHotString("Dismiss"), -1);

   for (i=0; i<9; ++i) fG1->AddFrame(fC[i], fL4);

   fR[0] = new TGRadioButton(fG2, new TGHotString("Stop"),        21);
   fR[1] = new TGRadioButton(fG2, new TGHotString("Question"),    22);
   fR[2] = new TGRadioButton(fG2, new TGHotString("Exclamation"), 23);
   fR[3] = new TGRadioButton(fG2, new TGHotString("Asterisk"),    24);

   for (i = 0; i < 4; ++i) {
      fG2->AddFrame(fR[i], fL4);
      fR[i]->Connect("Pressed()", "TestMsgBox", this, "DoRadio()");
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
   fMain->Resize(fMain->GetDefaultSize());

   // position relative to the parent's window
   Window_t wdum;
   gVirtualX->TranslateCoordinates(main->GetId(), fMain->GetParent()->GetId(),
                          (((TGFrame *) main)->GetWidth() - fMain->GetWidth()) >> 1,
                          (((TGFrame *) main)->GetHeight() - fMain->GetHeight()) >> 1,
                          ax, ay, wdum);
   fMain->Move(ax, ay);

   fMain->SetWindowName("Message Box Test");

   fMain->MapWindow();
   gClient->WaitFor(fMain);
}

// Order is important when deleting frames. Delete children first,
// parents last.

TestMsgBox::~TestMsgBox()
{
   // Delete widgets created by dialog.

   delete fTestButton; delete fCloseButton;
   delete fC[0]; delete fC[1]; delete fC[2]; delete fC[3];
   delete fC[4]; delete fC[5]; delete fC[6]; delete fC[7]; delete fC[8];
   delete fR[0]; delete fR[1]; delete fR[2]; delete fR[3];
   delete fTitle; delete fLtitle;
   delete fMsg; delete fLmsg;
   delete f3; delete f4; delete f5; delete f2; delete f1;
   delete fL1; delete fL2; delete fL3; delete fL4; delete fL5; delete fL6;
   delete fL21;
   delete fMain;
}

void TestMsgBox::CloseWindow()
{
   // Close dialog in response to window manager close.

   delete this;
}

void TestMsgBox::DoClose()
{
   // Handle Close button.

   fMain->SendCloseMessage();
}

void TestMsgBox::DoTest()
{
   // Handle test button.

   int i, buttons, retval;
   EMsgBoxIcon icontype = kMBIconStop;

   buttons = 0;
   for (i = 0; i < 9; i++)
      if (fC[i]->GetState() == kButtonDown)
         buttons |= mb_button_id[i];

   for (i = 0; i < 4; i++)
      if (fR[i]->GetState() == kButtonDown) {
         icontype = mb_icon[i];
         break;
      }

   new TGMsgBox(gClient->GetRoot(), fMain,
                fTbtitle->GetString(), fTbmsg->GetString(),
                icontype, buttons, &retval);
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
   Window_t wdummy;
   int ax, ay;
   gVirtualX->TranslateCoordinates(main->GetId(), fMain->GetParent()->GetId(),
                          (((TGFrame *) main)->GetWidth() - size.fWidth) >> 1,
                          (((TGFrame *) main)->GetHeight() - size.fHeight) >> 1,
                          ax, ay, wdummy);
   fMain->Move(ax, ay);

   fMain->MapSubwindows();
   fMain->MapWindow();

   gClient->WaitFor(fMain);
}

TestSliders::~TestSliders()
{
   // Delete dialog.

   delete fVframe1;
   delete fVframe2;
   delete fBfly1; delete fBly;
   delete fHslider1; delete fHslider2;
   delete fVslider1; delete fVslider2;
   delete fTeh1; delete fTeh2; delete fTev1; delete fTev2;
   delete fMain;
}

void TestSliders::CloseWindow()
{
   // Called when window is closed via the window manager.

   delete this;
}

void TestSliders::DoText(const char *text)
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

   switch (id) {
      case HSId1:
         fTbh1->Clear();
         fTbh1->AddText(0, buf);
         gClient->NeedRedraw(fTeh1);
         break;
      case VSId1:
         fTbv1->Clear();
         fTbv1->AddText(0, buf);
         gClient->NeedRedraw(fTev1);
         break;
      case HSId2:
         fTbh2->Clear();
         fTbh2->AddText(0, buf);
         gClient->NeedRedraw(fTeh2);
         break;
      case VSId2:
         sprintf(buf, "%f", fVslider2->GetMinPosition());
         fTbv2->Clear();
         fTbv2->AddText(0, buf);
         gClient->NeedRedraw(fTev2);
         break;
      default:
         break;
   }
}


TestShutter::TestShutter(const TGWindow *p, const TGWindow *main,
                         UInt_t w, UInt_t h)
{
   // Create transient frame containing a shutter widget.

   fMain = new TGTransientFrame(p, main, w, h);
   fMain->Connect("CloseWindow()", "TestShutter", this, "CloseWindow()");

   fTrash = new TList;

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
   Window_t wdum;
   int ax, ay;
   gVirtualX->TranslateCoordinates(main->GetId(), fMain->GetParent()->GetId(),
                          (((TGFrame *) main)->GetWidth() - fMain->GetWidth()) >> 1,
                          (((TGFrame *) main)->GetHeight() - fMain->GetHeight()) >> 1,
                          ax, ay, wdum);
   fMain->Move(ax, ay);

   fMain->SetWindowName("Shutter Test");

   fMain->MapWindow();
   //gClient->WaitFor(fMain);
}

void TestShutter::AddShutterItem(const char *name, shutterData_t data[])
{
   TGShutterItem    *item;
   TGCompositeFrame *container;
   TGButton         *button;
   const TGPicture  *buttonpic;
   static int id = 5001;

   TGLayoutHints *l = new TGLayoutHints(kLHintsTop | kLHintsCenterX,
                                        5, 5, 5, 0);
   fTrash->Add(l);

   item = new TGShutterItem(fShutter, new TGHotString(name), id++);
   container = (TGCompositeFrame *) item->GetContainer();
   fTrash->Add(item);

   for (int i=0; data[i].pixmap_name != 0; i++) {
      buttonpic = gClient->GetPicture(data[i].pixmap_name);
      if (!buttonpic) {
         printf("<TestShutter::AddShutterItem>: missing pixmap \"%s\", using default",
                data[i].pixmap_name);
         buttonpic = fDefaultPic;
      }

      button = new TGPictureButton(container, buttonpic, data[i].id);
      fTrash->Add(button);
      container->AddFrame(button, l);
      button->Connect("Clicked()", "TestShutter", this, "HandleButtons()");
      button->SetToolTipText(data[i].tip_text);
      data[i].button = button;
   }

   fShutter->AddItem(item);
}

TestShutter::~TestShutter()
{
   delete fLayout;
   delete fShutter;
   fTrash->Delete();
   delete fTrash;
   delete fMain;
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


TestProgress::TestProgress(const TGWindow *p, const TGWindow *main,
                           UInt_t w, UInt_t h)
{
   // Dialog used to test the different supported progress bars.

   fClose = kTRUE;

   fMain = new TGTransientFrame(p, main, w, h);
   fMain->Connect("CloseWindow()", "TestProgress", this, "DoClose()");
   fMain->DontCallClose();

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
   Window_t wdummy;
   int ax, ay;
   gVirtualX->TranslateCoordinates(main->GetId(), fMain->GetParent()->GetId(),
                          (((TGFrame *) main)->GetWidth() - size.fWidth) >> 1,
                          (((TGFrame *) main)->GetHeight() - size.fHeight) >> 1,
                          ax, ay, wdummy);
   fMain->Move(ax, ay);

   fMain->MapSubwindows();
   fMain->MapWindow();

   gClient->WaitFor(fMain);
}

TestProgress::~TestProgress()
{
   // Delete dialog.

   delete fHframe1;
   delete fVframe1;
   delete fVProg1; delete fVProg2;
   delete fHProg1; delete fHProg2; delete fHProg3;
   delete fHint1; delete fHint2; delete fHint3; delete fHint4; delete fHint5;
   delete fGO;
   delete fMain;
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
   // is directly processed in ProcessEvents() without exiting DoGO()).

   if (fClose)
      CloseWindow();
   else {
      fClose = kTRUE;
      TTimer::SingleShot(50, "TestProgress", this, "CloseWindow()");
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


Editor::Editor(const TGWindow *main, UInt_t w, UInt_t h)
{
   // Create an editor in a dialog.

   fMain = new TGTransientFrame(gClient->GetRoot(), main, w, h);
   fMain->Connect("CloseWindow()", "Editor", this, "CloseWindow()");

   fEdit = new TGTextEdit(fMain, w, h, kSunkenFrame | kDoubleBorder);
   fL1 = new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 3, 3, 3, 3);
   fMain->AddFrame(fEdit, fL1);
   fEdit->Connect("Opened()", "Editor", this, "DoOpen()");
   fEdit->Connect("Saved()",  "Editor", this, "DoSave()");
   fEdit->Connect("Closed()", "Editor", this, "DoClose()");

   fOK = new TGTextButton(fMain, "  &OK  ");
   fOK->Connect("Clicked()", "Editor", this, "DoOK()");
   fL2 = new TGLayoutHints(kLHintsBottom | kLHintsCenterX, 0, 0, 5, 5);
   fMain->AddFrame(fOK, fL2);

   SetTitle();

   fMain->MapSubwindows();

   fMain->Resize(fMain->GetDefaultSize());

   // position relative to the parent's window
   Window_t wdum;
   int ax, ay;
   gVirtualX->TranslateCoordinates(main->GetId(), fMain->GetParent()->GetId(),
                          ((TGFrame *) main)->GetWidth() - (fMain->GetWidth() >> 1),
                          (((TGFrame *) main)->GetHeight() - fMain->GetHeight()) >> 1,
                          ax, ay, wdum);
   fMain->Move(ax, ay);
   fMain->SetWMPosition(ax, ay);
}

Editor::~Editor()
{
   // Delete editor dialog.

   delete fEdit;
   delete fOK;
   delete fL1;
   delete fL2;
   delete fMain;
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

   fMain->SendCloseMessage();
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

   fMain->SendCloseMessage();
}


void guitest()
{
   new TestMainFrame(gClient->GetRoot(), 400, 220);
}

//---- Main program ------------------------------------------------------------
#ifdef STANDALONE
TROOT root("GUI", "GUI test environement");


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

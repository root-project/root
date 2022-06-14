/// \file
/// \ingroup tutorial_gui
/// GUI MDI features
///
/// \macro_code
///
/// \authors Ilka Antcheva, Fons Rademakers

#include <stdio.h>
#include <stdlib.h>

#include <TApplication.h>
#include <TGClient.h>
#include <TGFrame.h>
#include <TGButton.h>
#include <TGTextEntry.h>
#include <TGCanvas.h>
#include <TGMenu.h>
#include <TGMdi.h>
#include <TGMsgBox.h>
#include <TGSlider.h>
#include <TGListBox.h>
#include <RQ_OBJECT.h>

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


//----------------------------------------------------------------------

class TGMdiTestSubclass {

RQ_OBJECT("TGMdiTestSubclass")

protected:
   TGMdiFrame       *fMdiFrame;
   TGCanvas         *fCanvasWindow;
   TGCompositeFrame *fContainer;

public:
   TGMdiTestSubclass(TGMdiMainFrame *main, int w, int h);

   TGMdiFrame *GetMdiFrame() const { return fMdiFrame; }
   virtual Bool_t CloseWindow();
};

class TGMdiHintTest {

RQ_OBJECT("TGMdiHintTest")

protected:
   TGMdiFrame    *fMdiFrame;
   TGTextEntry   *fWName;
   TGCheckButton *fClose, *fMenu, *fMin, *fMax, *fSize, *fHelp;

public:
   TGMdiHintTest(TGMdiMainFrame *main, int w, int h);

   void HandleButtons();
   void HandleText(const char *);
};

class TGAppMainFrame  {

RQ_OBJECT("TGAppMainFrame")

protected:
   TGMainFrame     *fMain;
   TGMdiMainFrame  *fMainFrame;
   TGMdiMenuBar    *fMenuBar;
   TGLayoutHints   *fMenuBarItemLayout;
   TGPopupMenu     *fMenuFile, *fMenuWindow, *fMenuHelp;

   void InitMenu();
   void CloseWindow();

public:
   TGAppMainFrame(const TGWindow *p, int w, int h);

   void HandleMenu(Int_t id);
};

//----------------------------------------------------------------------

TGAppMainFrame::TGAppMainFrame(const TGWindow *p, int w, int h)
{
   fMain = new TGMainFrame(p, w, h, kVerticalFrame);
   fMenuBar = new TGMdiMenuBar(fMain, 10, 10);
   fMain->AddFrame(fMenuBar, new TGLayoutHints(kLHintsTop | kLHintsExpandX));

   fMainFrame = new TGMdiMainFrame(fMain, fMenuBar, 300, 300);
   fMain->AddFrame(fMainFrame, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   const TGPicture *pbg = gClient->GetPicture("mditestbg.xpm");
   if (pbg)
      fMainFrame->GetContainer()->SetBackgroundPixmap(pbg->GetPicture());

   TGMdiFrame *mdiFrame;

   //--- 1

   TGMdiTestSubclass *t = new TGMdiTestSubclass(fMainFrame, 320, 240);
   mdiFrame = t->GetMdiFrame();
   mdiFrame->SetMdiHints(kMdiClose | kMdiMenu);
   mdiFrame->SetWindowName("One");
   mdiFrame->MapSubwindows();
   mdiFrame->Layout();

   //--- 2

   ULong_t ic;
   gClient->GetColorByName("red", ic);
   mdiFrame = new TGMdiFrame(fMainFrame, 200, 200,
                             kOwnBackground, ic);
   mdiFrame->AddFrame(new TGTextButton(mdiFrame, new TGHotString("&Press me!"), 1),
                      new TGLayoutHints(kLHintsCenterX | kLHintsCenterY));
   mdiFrame->SetMdiHints(kMdiDefaultHints | kMdiHelp);
   mdiFrame->SetWindowName("Two");
   mdiFrame->MapSubwindows();
   mdiFrame->Layout();
   mdiFrame->Move(150, 200);

   //--- 3

   gClient->GetColorByName("green", ic);
   mdiFrame = new TGMdiFrame(fMainFrame, 200, 200, kOwnBackground, ic);
   mdiFrame->AddFrame(new TGTextButton(mdiFrame, new TGHotString("Button 1"), 11),
                      new TGLayoutHints(kLHintsCenterX | kLHintsCenterY));
   mdiFrame->AddFrame(new TGTextButton(mdiFrame, new TGHotString("Button 2"), 12),
                      new TGLayoutHints(kLHintsCenterX | kLHintsCenterY));
   mdiFrame->SetMdiHints(kMdiDefaultHints | kMdiHelp);
   mdiFrame->SetWindowName("Three");
   mdiFrame->MapSubwindows();
   mdiFrame->Layout();
   mdiFrame->Move(180, 220);

   //--- 4

   gClient->GetColorByName("blue", ic);
   mdiFrame = new TGMdiFrame(fMainFrame, 200, 400, kOwnBackground, ic);

   TGListBox *fListBox = new TGListBox(mdiFrame,1);
   fListBox->AddEntry("Entry   1", 1);
   fListBox->AddEntry("Entry   2", 2);
   fListBox->AddEntry("Entry   3", 3);
   fListBox->AddEntry("Entry   4", 4);
   fListBox->AddEntry("Entry   5", 5);
   fListBox->AddEntry("Entry   6", 6);
   fListBox->AddEntry("Entry   7", 7);
   fListBox->AddEntry("Entry   8", 8);
   fListBox->AddEntry("Entry   9", 9);
   fListBox->Resize(100,70);
   fListBox->SetMultipleSelections(kFALSE);
   mdiFrame->AddFrame(fListBox,
                      new TGLayoutHints(kLHintsCenterX | kLHintsCenterY));
   mdiFrame->AddFrame(new TGHSlider(mdiFrame, 50, kSlider1, 1,
                      kHorizontalFrame, ic),
                      new TGLayoutHints(kLHintsCenterX | kLHintsCenterY));
   mdiFrame->Move(400, 300);
   mdiFrame->SetWindowName("Four");
   mdiFrame->MapSubwindows();
   mdiFrame->Layout();

   //--- 5

   new TGMdiHintTest(fMainFrame, 200, 200);

   InitMenu();

   fMain->SetWindowName("MDI test");
   fMain->SetClassHints("mdi test", "mdi test");

   if (pbg && pbg->GetWidth() > 600 && pbg->GetHeight() > 400)
      fMain->Resize(pbg->GetWidth(), pbg->GetHeight()+25);
   else
      fMain->Resize(640, 400);

   fMain->MapSubwindows();
   fMain->MapWindow();
   fMain->Layout();
}

void TGAppMainFrame::HandleMenu(Int_t id)
{
   // Handle menu items.

   switch (id) {
      case M_FILE_NEW:
         new TGMdiFrame(fMainFrame, 200, 100);
         break;

      case M_FILE_CLOSE:
         fMainFrame->Close(fMainFrame->GetCurrent());
         break;

      case M_FILE_EXIT:
         CloseWindow();
         break;

      case M_WINDOW_HOR:
         fMainFrame->TileHorizontal();
         break;

      case M_WINDOW_VERT:
         fMainFrame->TileVertical();
         break;

      case M_WINDOW_CASCADE:
         fMainFrame->Cascade();
         break;

      case M_WINDOW_ARRANGE:
         fMainFrame->ArrangeMinimized();
         break;

      case M_WINDOW_OPAQUE:
         if (fMenuWindow->IsEntryChecked(M_WINDOW_OPAQUE)) {
            fMenuWindow->UnCheckEntry(M_WINDOW_OPAQUE);
            fMainFrame->SetResizeMode(kMdiNonOpaque);
         } else {
            fMenuWindow->CheckEntry(M_WINDOW_OPAQUE);
            fMainFrame->SetResizeMode(kMdiOpaque);
         }
         break;

      default:
         fMainFrame->SetCurrent(id);
         break;
   }
}

void TGAppMainFrame::InitMenu()
{
   fMenuBarItemLayout = new TGLayoutHints(kLHintsTop | kLHintsLeft, 0, 4, 0, 0);

   fMenuFile = new TGPopupMenu(gClient->GetRoot());
   fMenuFile->AddEntry(new TGHotString("&New Window"), M_FILE_NEW);
   fMenuFile->AddEntry(new TGHotString("&Close Window"), M_FILE_CLOSE);
   fMenuFile->AddSeparator();
   fMenuFile->AddEntry(new TGHotString("E&xit"), M_FILE_EXIT);

   fMenuWindow = new TGPopupMenu(gClient->GetRoot());
   fMenuWindow->AddEntry(new TGHotString("Tile &Horizontally"), M_WINDOW_HOR);
   fMenuWindow->AddEntry(new TGHotString("Tile &Vertically"), M_WINDOW_VERT);
   fMenuWindow->AddEntry(new TGHotString("&Cascade"), M_WINDOW_CASCADE);
   fMenuWindow->AddSeparator();
   fMenuWindow->AddPopup(new TGHotString("&Windows"), fMainFrame->GetWinListMenu());
   fMenuWindow->AddSeparator();
   fMenuWindow->AddEntry(new TGHotString("&Arrange icons"), M_WINDOW_ARRANGE);
   fMenuWindow->AddSeparator();
   fMenuWindow->AddEntry(new TGHotString("&Opaque resize"), M_WINDOW_OPAQUE);

   fMenuWindow->CheckEntry(M_WINDOW_OPAQUE);

   fMenuHelp = new TGPopupMenu(gClient->GetRoot());
   fMenuHelp->AddEntry(new TGHotString("&Contents"), M_HELP_CONTENTS);
   fMenuHelp->AddSeparator();
   fMenuHelp->AddEntry(new TGHotString("&About"), M_HELP_ABOUT);

   fMenuHelp->DisableEntry(M_HELP_CONTENTS);
   fMenuHelp->DisableEntry(M_HELP_ABOUT);

   // menu message are handled by the class' HandleMenu() method
   fMenuFile->Connect("Activated(Int_t)", "TGAppMainFrame", this,
                      "HandleMenu(Int_t)");
   fMenuWindow->Connect("Activated(Int_t)", "TGAppMainFrame", this,
                        "HandleMenu(Int_t)");
   fMenuHelp->Connect("Activated(Int_t)", "TGAppMainFrame", this,
                      "HandleMenu(Int_t)");

   fMenuBar->AddPopup(new TGHotString("&File"), fMenuFile, fMenuBarItemLayout);
   fMenuBar->AddPopup(new TGHotString("&Windows"),fMenuWindow,fMenuBarItemLayout);
   fMenuBar->AddPopup(new TGHotString("&Help"), fMenuHelp, fMenuBarItemLayout);
}

void TGAppMainFrame::CloseWindow()
{
   gApplication->Terminate(0);
}

//----------------------------------------------------------------------

TGMdiTestSubclass::TGMdiTestSubclass(TGMdiMainFrame *main, int w, int h)
{
   fMdiFrame = new TGMdiFrame(main, w, h);
   fMdiFrame->Connect("CloseWindow()", "TGMdiTestSubclass", this, "CloseWindow()");
   fMdiFrame->DontCallClose();

   fCanvasWindow = new TGCanvas(fMdiFrame, 400, 240);
   fContainer = new TGCompositeFrame(fCanvasWindow->GetViewPort(), 10, 10,
                                     kHorizontalFrame | kOwnBackground,
                                     fMdiFrame->GetWhitePixel());
   fContainer->SetLayoutManager(new TGTileLayout(fContainer, 8));
   fCanvasWindow->SetContainer(fContainer);

   for (int i = 0; i < 256; ++i)
      fCanvasWindow->AddFrame(new TGFrame(fCanvasWindow->GetContainer(),
                              32, 32, kOwnBackground, (i+1) & 255),
                              new TGLayoutHints(kLHintsNormal));

   fMdiFrame->AddFrame(fCanvasWindow, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   fMdiFrame->SetWindowIcon(gClient->GetPicture("ofolder_t.xpm"));
}

Bool_t TGMdiTestSubclass::CloseWindow()
{
   int ret = 0;

   new TGMsgBox(gClient->GetRoot(), fMdiFrame,
                fMdiFrame->GetWindowName(), "Really want to close the window?",
                kMBIconExclamation, kMBYes | kMBNo, &ret);

   if (ret == kMBYes) return fMdiFrame->CloseWindow();

   return kFALSE;
}


//----------------------------------------------------------------------

TGMdiHintTest::TGMdiHintTest(TGMdiMainFrame *main, int w, int h)
{
   fMdiFrame = new TGMdiFrame(main, w, h);

   fClose = new TGCheckButton(fMdiFrame, new TGHotString("Close"), 11);
   fMenu  = new TGCheckButton(fMdiFrame, new TGHotString("Menu (left icon)"), 12);
   fMin   = new TGCheckButton(fMdiFrame, new TGHotString("Minimize"), 13);
   fMax   = new TGCheckButton(fMdiFrame, new TGHotString("Maximize"), 14);
   fSize  = new TGCheckButton(fMdiFrame, new TGHotString("Resize"), 15);
   fHelp  = new TGCheckButton(fMdiFrame, new TGHotString("Help"), 16);

   TGLayoutHints *lh = new TGLayoutHints(kLHintsLeft | kLHintsTop, 5, 100, 5, 0);

   fMdiFrame->AddFrame(fClose, lh);
   fMdiFrame->AddFrame(fMenu, lh);
   fMdiFrame->AddFrame(fMin, lh);
   fMdiFrame->AddFrame(fMax, lh);
   fMdiFrame->AddFrame(fSize, lh);
   fMdiFrame->AddFrame(fHelp, lh);

   fClose->SetState(kButtonDown);
   fMin->SetState(kButtonDown);
   fMenu->SetState(kButtonDown);
   fMax->SetState(kButtonDown);
   fSize->SetState(kButtonDown);

   fClose->Connect("Clicked()", "TGMdiHintTest", this, "HandleButtons()");
   fMenu->Connect("Clicked()", "TGMdiHintTest", this, "HandleButtons()");
   fMin->Connect("Clicked()", "TGMdiHintTest", this, "HandleButtons()");
   fMax->Connect("Clicked()", "TGMdiHintTest", this, "HandleButtons()");
   fSize->Connect("Clicked()", "TGMdiHintTest", this, "HandleButtons()");
   fHelp->Connect("Clicked()", "TGMdiHintTest", this, "HandleButtons()");

   fWName = new TGTextEntry(fMdiFrame, (const char *)"", 20);
   fMdiFrame->AddFrame(fWName, new TGLayoutHints(kLHintsTop | kLHintsExpandX,
                                                 5, 5, 5, 5));

   fWName->GetBuffer()->AddText(0, "MDI hints test");
   fWName->Connect("TextChanged(char*)", "TGMdiHintTest", this, "HandleText(char*)");

   fMdiFrame->SetMdiHints(kMdiDefaultHints);
   fMdiFrame->SetWindowName(fWName->GetBuffer()->GetString());

   fMdiFrame->SetWindowIcon(gClient->GetPicture("app_t.xpm"));

   fMdiFrame->MapSubwindows();
   fMdiFrame->Layout();
}

void TGMdiHintTest::HandleButtons()
{
   int hints = 0;

   if (fClose->GetState() != kButtonUp) hints |= kMdiClose;
   if (fMenu->GetState() != kButtonUp) hints |= kMdiMenu;
   if (fMin->GetState() != kButtonUp) hints |= kMdiMinimize;
   if (fMax->GetState() != kButtonUp) hints |= kMdiMaximize;
   if (fSize->GetState() != kButtonUp) hints |= kMdiSize;
   if (fHelp->GetState() != kButtonUp) hints |= kMdiHelp;

   fMdiFrame->SetMdiHints(hints);
}

void TGMdiHintTest::HandleText(const char *)
{
   fMdiFrame->SetWindowName(fWName->GetBuffer()->GetString());
}

void mditest()
{
   new TGAppMainFrame(gClient->GetRoot(), 640, 400);
}

//----------------------------------------------------------------------

#ifdef STANDALONE
int main(int argc, char **argv)
{
   TApplication theApp("MdiTest", &argc, argv);

   mditest();

   theApp.Run();

   return 0;
}
#endif

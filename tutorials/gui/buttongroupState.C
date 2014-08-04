// A simple example that shows the enabled and disabled state
// of a button group with radio and check buttons.
//
// Author: Roel Aaij   4/07/2007

#include <TApplication.h>
#include <TGClient.h>
#include <TGButton.h>
#include <TGFrame.h>
#include <TGLayout.h>
#include <TGWindow.h>
#include <TGLabel.h>
#include <TString.h>
#include <TGButtonGroup.h>

class IDList {

private:
   Int_t nID;   // creates unique widget's IDs

public:
   IDList() : nID(0) {}
   ~IDList() {}
   Int_t GetUnID(void) { return ++nID; }
};

class MyButtonTest : public TGMainFrame {

private:
   TGTextButton        *fExit;         // Exit text button
   TGVButtonGroup      *fButtonGroup;  // Button group
   TGCheckButton       *fCheckb[4];    // Check buttons
   TGRadioButton       *fRadiob[2];    // Radio buttons
   IDList               IDs;           // Widget IDs generator

public:
   MyButtonTest(const TGWindow *p, UInt_t w, UInt_t h);
   virtual ~MyButtonTest();

   void DoExit(void);
   void SetGroupEnabled(Bool_t);

   ClassDef(MyButtonTest, 0)
};

MyButtonTest::MyButtonTest(const TGWindow *p, UInt_t w, UInt_t h)
   : TGMainFrame(p, w, h)
{
   SetCleanup(kDeepCleanup);

   Connect("CloseWindow()", "MyButtonTest", this, "DoExit()");
   DontCallClose();

   TGHorizontalFrame *fHL2 = new TGHorizontalFrame(this, 70, 100);
   fCheckb[0] = new TGCheckButton(fHL2, new TGHotString("Enable BG"),
                                  IDs.GetUnID());
   fCheckb[0]->SetToolTipText("Enable/Disable the button group");
   fHL2->AddFrame(fCheckb[0], new TGLayoutHints(kLHintsCenterX|kLHintsCenterY,
                                                1, 1, 1, 1));
   fButtonGroup = new TGVButtonGroup(fHL2, "My Button Group");
   fCheckb[1] = new TGCheckButton(fButtonGroup, new TGHotString("CB 2"),
                                  IDs.GetUnID());
   fCheckb[2] = new TGCheckButton(fButtonGroup, new TGHotString("CB 3"),
                                  IDs.GetUnID());
   fCheckb[3] = new TGCheckButton(fButtonGroup, new TGHotString("CB 4"),
                                  IDs.GetUnID());
   fRadiob[0] = new TGRadioButton(fButtonGroup, new TGHotString("RB 1"),
                                  IDs.GetUnID());
   fRadiob[1] = new TGRadioButton(fButtonGroup, new TGHotString("RB 2"),
                                  IDs.GetUnID());
   fButtonGroup->Show();

   fHL2->AddFrame(fButtonGroup, new TGLayoutHints(kLHintsCenterX|kLHintsCenterY,
                                                  1, 1, 1, 1));
   AddFrame(fHL2);

   fCheckb[0]->Connect("Toggled(Bool_t)", "MyButtonTest", this,
                       "SetGroupEnabled(Bool_t)");

   TGHorizontalFrame *fHL3 = new TGHorizontalFrame(this, 70, 100, kFixedWidth);
   fExit = new TGTextButton(fHL3, "&Exit", IDs.GetUnID());
   fExit->Connect("Clicked()", "MyButtonTest", this, "DoExit()");
   fHL3->AddFrame(fExit, new TGLayoutHints(kLHintsExpandX));
   AddFrame(fHL3, new TGLayoutHints(kLHintsCenterX | kLHintsCenterY,1,1,1,1));

   //Default state
   fCheckb[0]->SetOn();
   fButtonGroup->SetState(kTRUE);

   SetWindowName("My Button Group");
   MapSubwindows();
   Resize(GetDefaultSize());
   MapWindow();

   fButtonGroup->SetRadioButtonExclusive(kTRUE);
   fRadiob[1]->SetOn();
};

MyButtonTest::~MyButtonTest()
{
   // Destructor.
   Cleanup();
}

void MyButtonTest::DoExit()
{
   // Exit this application via the Exit button or Window Manager.
   // Use one of the both lines according to your needs.
   // Please note to re-run this macro in the same ROOT session,
   // you have to compile it to get signals/slots 'on place'.

   //DeleteWindow();            // to stay in the ROOT session
   gApplication->Terminate();   // to exit and close the ROOT session
}

void MyButtonTest::SetGroupEnabled(Bool_t on)
{
   fButtonGroup->SetState(on);
}

void buttongroupState()
{
   new MyButtonTest(gClient->GetRoot(),100,100);
}


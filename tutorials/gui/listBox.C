/// \file
/// \ingroup tutorial_gui
/// This macro gives an example of how to create a list box and how to set and use its multiple selection feature.
/// To run it do either:
/// ~~~ 
/// .x listBox.C
/// .x listBox.C++
/// ~~~ 
///
/// \macro_code
///
/// \author Ilka Antcheva   1/12/2006


#include <TApplication.h>
#include <TGClient.h>
#include <TGButton.h>
#include <TGListBox.h>
#include <TList.h>

class MyMainFrame : public TGMainFrame {

private:
   TGListBox           *fListBox;
   TGCheckButton       *fCheckMulti;
   TList               *fSelected;

public:
   MyMainFrame(const TGWindow *p, UInt_t w, UInt_t h);
   virtual ~MyMainFrame();
   void DoExit();
   void DoSelect();
   void HandleButtons();
   void PrintSelected();

   ClassDef(MyMainFrame, 0)
};

void MyMainFrame::DoSelect()
{
   Printf("Slot DoSelect()");
}

void MyMainFrame::DoExit()
{
   Printf("Slot DoExit()");
   gApplication->Terminate(0);
}

MyMainFrame::MyMainFrame(const TGWindow *p, UInt_t w, UInt_t h) :
   TGMainFrame(p, w, h)
{
   // Create main frame

   fListBox = new TGListBox(this, 89);
   fSelected = new TList;
   char tmp[20];
   for (int i = 0; i < 20; ++i) {
      sprintf(tmp, "Entry %i", i+1);
      fListBox->AddEntry(tmp, i+1);
   }
   fListBox->Resize(100,150);
   AddFrame(fListBox, new TGLayoutHints(kLHintsTop | kLHintsLeft |
                                        kLHintsExpandX | kLHintsExpandY,
                                        5, 5, 5, 5));

   fCheckMulti = new TGCheckButton(this, "&Mutliple selection", 10);
   AddFrame(fCheckMulti, new TGLayoutHints(kLHintsTop | kLHintsLeft,
                                           5, 5, 5, 5));
   fCheckMulti->Connect("Clicked()", "MyMainFrame", this, "HandleButtons()");
   // Create a horizontal frame containing button(s)
   TGHorizontalFrame *hframe = new TGHorizontalFrame(this, 150, 20, kFixedWidth);
   TGTextButton *show = new TGTextButton(hframe, "&Show");
   show->SetToolTipText("Click here to print the selection you made");
   show->Connect("Pressed()", "MyMainFrame", this, "PrintSelected()");
   hframe->AddFrame(show, new TGLayoutHints(kLHintsExpandX, 5, 5, 3, 4));
   TGTextButton *exit = new TGTextButton(hframe, "&Exit ");
   exit->Connect("Pressed()", "MyMainFrame", this, "DoExit()");
   hframe->AddFrame(exit, new TGLayoutHints(kLHintsExpandX, 5, 5, 3, 4));
   AddFrame(hframe, new TGLayoutHints(kLHintsExpandX, 2, 2, 5, 1));

   // Set a name to the main frame
   SetWindowName("List Box");
   MapSubwindows();

   // Initialize the layout algorithm via Resize()
   Resize(GetDefaultSize());

   // Map main frame
   MapWindow();
   fListBox->Select(1);
}

MyMainFrame::~MyMainFrame()
{
   // Clean up main frame...
   Cleanup();
   if (fSelected) {
      fSelected->Delete();
      delete fSelected;
   }
}

void MyMainFrame::HandleButtons()
{
   // Handle check button.
   Int_t id;
   TGButton *btn = (TGButton *) gTQSender;
   id = btn->WidgetId();

   printf("HandleButton: id = %d\n", id);

   if (id == 10)
      fListBox->SetMultipleSelections(fCheckMulti->GetState());
}


void MyMainFrame::PrintSelected()
{
   // Writes selected entries in TList if multiselection.

   fSelected->Clear();

   if (fListBox->GetMultipleSelections()) {
      Printf("Selected entries are:\n");
      fListBox->GetSelectedEntries(fSelected);
      fSelected->ls();
   } else {
      Printf("Selected entries is: %d\n", fListBox->GetSelected());
   }
}

void listBox()
{
   // Popup the GUI...
   new MyMainFrame(gClient->GetRoot(), 200, 200);
}

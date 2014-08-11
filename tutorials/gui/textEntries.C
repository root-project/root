// Author: Valeriy Onuchin   25/08/2007
//
// This macro gives an example of how to set/change text entry attributes.
//
// To run it do either:
// .x textEntries.C
// .x textEntries.C++

#include <TGTextEntry.h>
#include <TGButtonGroup.h>
#include <TGLabel.h>
#include <TGComboBox.h>
#include <TApplication.h>


//////////// auxilary class ///////////////////////////////////////////////////
class GroupBox : public TGGroupFrame {
private:
   TGComboBox  *fCombo; // combo box
   TGTextEntry *fEntry; // text entry

public:
   GroupBox(const TGWindow *p, const char *name, const char *title);
   TGTextEntry *GetEntry() const { return fEntry; }
   TGComboBox  *GetCombo() const { return fCombo; }

   ClassDef(GroupBox, 0)
};

//______________________________________________________________________________
GroupBox::GroupBox(const TGWindow *p, const char *name, const char *title) :
   TGGroupFrame(p, name)
{
   // Group frame containing combobox and text entry.

   TGHorizontalFrame *horz = new TGHorizontalFrame(this);
   AddFrame(horz, new TGLayoutHints(kLHintsExpandX | kLHintsCenterY));
   TGLabel *label = new TGLabel(horz, title);
   horz->AddFrame(label, new TGLayoutHints(kLHintsLeft | kLHintsCenterY));

   fCombo = new TGComboBox(horz);
   horz->AddFrame(fCombo, new TGLayoutHints(kLHintsRight | kLHintsExpandY,
                                            5, 0, 5, 5));
   fCombo->Resize(100, 20);

   fEntry = new TGTextEntry(this);
   AddFrame(fEntry, new TGLayoutHints(kLHintsExpandX | kLHintsCenterY));
}

////////////////////////////////////////////////////////////////////////////////
class TextEntryWindow {

protected:
   TGMainFrame *fMain;     // main frame
   GroupBox    *fEcho;     // echo mode (echo, password, no echo)
   GroupBox    *fAlign;    // alignment (left, right, center)
   GroupBox    *fAccess;   // read-only mode
   GroupBox    *fBorder;   // border mode

public:
   TextEntryWindow();
   virtual ~TextEntryWindow() { delete fMain; }

   ClassDef(TextEntryWindow, 0);
};


//______________________________________________________________________________
TextEntryWindow::TextEntryWindow()
{
   // Main test window.

   TGComboBox  *combo;
   TGTextEntry *entry;

   fMain = new TGMainFrame(gClient->GetRoot(), 10, 10, kVerticalFrame);

   // recusively delete all subframes on exit
   fMain->SetCleanup(kDeepCleanup);

   fEcho = new GroupBox(fMain, "Echo", "Mode:");
   fMain->AddFrame(fEcho, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 5, 5));
   combo = fEcho->GetCombo();
   entry = fEcho->GetEntry();
   // add entries
   combo->AddEntry("Normal", TGTextEntry::kNormal);
   combo->AddEntry("Password", TGTextEntry::kPassword);
   combo->AddEntry("No Echo", TGTextEntry::kNoEcho);
   combo->Connect("Selected(Int_t)", "TGTextEntry", entry, "SetEchoMode(Int_t)");
   combo->Select(TGTextEntry::kNormal);

   fAlign = new GroupBox(fMain, "Alignment", "Type:");
   fMain->AddFrame(fAlign, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 5, 5));
   combo = fAlign->GetCombo();
   entry = fAlign->GetEntry();
   // add entries
   combo->AddEntry("Left", kTextLeft);
   combo->AddEntry("Centered", kTextCenterX);
   combo->AddEntry("Right", kTextRight);
   combo->Connect("Selected(Int_t)", "TGTextEntry", entry, "SetAlignment(Int_t)");
   combo->Select(kTextLeft);

   fAccess = new GroupBox(fMain, "Access", "Read-only:");
   fMain->AddFrame(fAccess, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 5, 5));
   combo = fAccess->GetCombo();
   entry = fAccess->GetEntry();
   // add entries
   combo->AddEntry("False", 1);
   combo->AddEntry("True", 0);
   combo->Connect("Selected(Int_t)", "TGTextEntry", entry, "SetEnabled(Int_t)");
   combo->Select(1);

   fBorder = new GroupBox(fMain, "Border", "Drawn:");
   fMain->AddFrame(fBorder, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 5, 5));
   combo = fBorder->GetCombo();
   entry = fBorder->GetEntry();
   // add entries
   combo->AddEntry("False", 0);
   combo->AddEntry("True", 1);
   combo->Connect("Selected(Int_t)", "TGTextEntry", entry, "SetFrameDrawn(Int_t)");
   combo->Select(1);

   // terminate ROOT session when window is closed
   fMain->Connect("CloseWindow()", "TApplication", gApplication, "Terminate()");
   fMain->DontCallClose();

   fMain->MapSubwindows();
   fMain->Resize();

   // set minimum width, height
   fMain->SetWMSizeHints(fMain->GetDefaultWidth(), fMain->GetDefaultHeight(),
                         1000, 1000, 0, 0);
   fMain->SetWindowName("Text Entries");
   fMain->MapRaised();
}


////////////////////////////////////////////////////////////////////////////////
void textEntries()
{
   // Main program.

   new TextEntryWindow();
}

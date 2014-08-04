// Author: Valeriy Onuchin   17/07/2007
//
// This macro gives an example of how to set/change text button attributes.
//
// To run it do either:
// .x buttonTest.C
// .x buttonTest.C++

#include <TGButton.h>
#include <TGButtonGroup.h>
#include <TGLabel.h>
#include <TGNumberEntry.h>
#include <TG3DLine.h>
#include <TApplication.h>



//////////// auxilary class ///////////////////////////////////////////////////
class TextMargin : public TGHorizontalFrame {

protected:
   TGNumberEntry *fEntry;

public:
   TextMargin(const TGWindow *p, const char *name) : TGHorizontalFrame(p)
   {
      fEntry = new TGNumberEntry(this, 0, 6, -1, TGNumberFormat::kNESInteger);
      AddFrame(fEntry, new TGLayoutHints(kLHintsLeft));
      TGLabel *label = new TGLabel(this, name);
      AddFrame(label, new TGLayoutHints(kLHintsLeft, 10));
   }
   TGTextEntry *GetEntry() const { return fEntry->GetNumberEntry(); }

   ClassDef(TextMargin, 0)
};

////////////////////////////////////////////////////////////////////////////////
class ButtonWindow : public TGMainFrame {

protected:
   TGTextButton *fButton;   // button being tested

public:
   ButtonWindow();
   void DoHPosition(Int_t);
   void DoVPosition(Int_t);
   void DoLeftMargin(char*);
   void DoRightMargin(char*);
   void DoTopMargin(char*);
   void DoBottomMargin(char*);

   ClassDef(ButtonWindow, 0)
};


//______________________________________________________________________________
ButtonWindow::ButtonWindow() : TGMainFrame(gClient->GetRoot(), 10, 10, kHorizontalFrame)
{
   // Main test window.

   SetCleanup(kDeepCleanup);

   // Controls on right
   TGVerticalFrame *controls = new TGVerticalFrame(this);
   AddFrame(controls, new TGLayoutHints(kLHintsRight | kLHintsExpandY,
                                        5, 5, 5, 5));

   // Separator
   TGVertical3DLine *separator = new TGVertical3DLine(this);
   AddFrame(separator, new TGLayoutHints(kLHintsRight | kLHintsExpandY));

   // Contents
   TGHorizontalFrame *contents = new TGHorizontalFrame(this);
   AddFrame(contents, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY,5,5));

   // The button for test
   fButton = new TGTextButton(contents,
      "&This button has a multi-line label\nand shows features\n"
      "available in the button classes");
   fButton->Resize(300, 200);
   fButton->ChangeOptions(fButton->GetOptions() | kFixedSize);
   fButton->SetToolTipText("The assigned tooltip\ncan be multi-line also",200);
   contents->AddFrame(fButton, new TGLayoutHints(kLHintsCenterX|kLHintsCenterY,
                      20, 20, 20, 20));

   TGGroupFrame *group = new TGGroupFrame(controls, "Enable/Disable");
   group->SetTitlePos(TGGroupFrame::kCenter);
   TGCheckButton *disable = new TGCheckButton(group, "Switch state\nEnable/Disable");
   disable->SetOn();
   disable->Connect("Toggled(Bool_t)", "TGButton", fButton, "SetEnabled(Bool_t)");
   group->AddFrame(disable, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
   controls->AddFrame(group, new TGLayoutHints(kLHintsExpandX));


   // control horizontal position of the text
   TGButtonGroup *horizontal = new TGButtonGroup(controls, "Horizontal Position");
   horizontal->SetTitlePos(TGGroupFrame::kCenter);
   new TGRadioButton(horizontal, "Center", kTextCenterX);
   new TGRadioButton(horizontal, "Left", kTextLeft);
   new TGRadioButton(horizontal, "Right", kTextRight);
   horizontal->SetButton(kTextCenterX);
   horizontal->Connect("Pressed(Int_t)", "ButtonWindow", this,
                       "DoHPosition(Int_t)");
   controls->AddFrame(horizontal, new TGLayoutHints(kLHintsExpandX));


   // control vertical position of the text
   TGButtonGroup *vertical = new TGButtonGroup(controls, "Vertical Position");
   vertical->SetTitlePos(TGGroupFrame::kCenter);
   new TGRadioButton(vertical, "Center", kTextCenterY);
   new TGRadioButton(vertical, "Top", kTextTop);
   new TGRadioButton(vertical, "Bottom", kTextBottom);
   vertical->SetButton(kTextCenterY);
   vertical->Connect("Pressed(Int_t)", "ButtonWindow", this,
                     "DoVPosition(Int_t)");
   controls->AddFrame(vertical, new TGLayoutHints(kLHintsExpandX));


   // control margins of the text
   TGGroupFrame *margins = new TGGroupFrame(controls, "Text Margins");
   margins->SetTitlePos(TGGroupFrame::kCenter);

   TextMargin *left = new TextMargin(margins, "Left");
   margins->AddFrame(left, new TGLayoutHints(kLHintsExpandX, 0, 0, 2, 2));
   left->GetEntry()->Connect("TextChanged(char*)", "ButtonWindow",
                             this, "DoLeftMargin(char*)");

   TextMargin *right = new TextMargin(margins, "Right");
   margins->AddFrame(right, new TGLayoutHints(kLHintsExpandX, 0, 0, 2, 2));
   right->GetEntry()->Connect("TextChanged(char*)", "ButtonWindow",
                               this, "DoRightMargin(char*)");

   TextMargin *top = new TextMargin(margins, "Top");
   margins->AddFrame(top, new TGLayoutHints(kLHintsExpandX, 0, 0, 2, 2));
   top->GetEntry()->Connect("TextChanged(char*)", "ButtonWindow",
                             this, "DoTopMargin(char*)");

   TextMargin *bottom = new TextMargin(margins, "Bottom");
   margins->AddFrame(bottom, new TGLayoutHints(kLHintsExpandX, 0, 0, 2, 2));
   bottom->GetEntry()->Connect("TextChanged(char*)", "ButtonWindow",
                               this, "DoBottomMargin(char*)");

   controls->AddFrame(margins, new TGLayoutHints(kLHintsExpandX));

   TGTextButton *quit = new TGTextButton(controls, "Quit");
   controls->AddFrame(quit, new TGLayoutHints(kLHintsBottom | kLHintsExpandX,
                                              0, 0, 0, 5));
   quit->Connect("Pressed()", "TApplication", gApplication, "Terminate()");

   Connect("CloseWindow()", "TApplication", gApplication, "Terminate()");
   DontCallClose();

   MapSubwindows();
   Resize();

   SetWMSizeHints(GetDefaultWidth(), GetDefaultHeight(), 1000, 1000, 0 ,0);
   SetWindowName("Button Test");
   MapRaised();
}

//______________________________________________________________________________
void ButtonWindow::DoHPosition(Int_t id)
{
   // Horizontal position handler.

   Int_t tj = fButton->GetTextJustify();
   tj &= ~kTextCenterX;
   tj &= ~kTextLeft;
   tj &= ~kTextRight;
   tj |= id;
   fButton->SetTextJustify(tj);
}

//______________________________________________________________________________
void ButtonWindow::DoVPosition(Int_t id)
{
   // Vertical position handler.

   Int_t tj = fButton->GetTextJustify();

   tj &= ~kTextCenterY;
   tj &= ~kTextTop;
   tj &= ~kTextBottom;
   tj |= id;
   fButton->SetTextJustify(tj);
}

//______________________________________________________________________________
void ButtonWindow::DoLeftMargin(char *val)
{
   // Set left text margin.

   fButton->SetLeftMargin(atoi(val));
   gClient->NeedRedraw(fButton);
}

//______________________________________________________________________________
void ButtonWindow::DoRightMargin(char *val)
{
   // Set right text margin.

   fButton->SetRightMargin(atoi(val));
   gClient->NeedRedraw(fButton);
}

//______________________________________________________________________________
void ButtonWindow::DoTopMargin(char *val)
{
   // Set top text margin.

   fButton->SetTopMargin(atoi(val));
   gClient->NeedRedraw(fButton);
}

//______________________________________________________________________________
void ButtonWindow::DoBottomMargin(char *val)
{
   // Set bottom text margin.

   fButton->SetBottomMargin(atoi(val));
   gClient->NeedRedraw(fButton);
}


////////////////////////////////////////////////////////////////////////////////
void buttonTest()
{
   // Main program.

   new ButtonWindow();
}

#include "TTextEditor.h"
#include "TText.h"
#include "TGTextEntry.h"
#include "TGNumberEntry.h"
#include "TGLabel.h"

ClassImp(TTextEditor)


enum ELatexID{
   kText_Text = 0, kText_Xpos, kText_Ypos, kText_Angle, kText_Size
};


//______________________________________________________________________________
TTextEditor::TTextEditor(const TGWindow *p,
                  Int_t width, Int_t height,
                  UInt_t options, Pixel_t back)
                  : TGedFrame(p, width, height, options | kVerticalFrame, back)
{
   // TTextEditor constructor.

   fText = 0;

   // start initializing the window components
   MakeTitle("Text String");

   fText = new TGTextEntry(this, new TGTextBuffer(50), kText_Text);
   fText->Resize(135, fText->GetDefaultHeight());
   fText->SetToolTipText("Enter the text string");
   AddFrame(fText, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));

   TGCompositeFrame *f1 = new TGCompositeFrame(this, 120, 20, kHorizontalFrame);
   TGLabel *lbl1 = new TGLabel(f1,"X Position");
   fXpos = new TGNumberEntry(f1, 4, 2, kText_Xpos, TGNumberEntry::kNESRealTwo,
   TGNumberEntry::kNEAAnyNumber);
   fXpos->Resize(50, 20);
   f1->AddFrame(lbl1, new TGLayoutHints(kLHintsLeft,1, 1, 1, 1));
   f1->AddFrame(fXpos, new TGLayoutHints(kLHintsLeft, 7, 1, 1, 1));
   AddFrame(f1, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 120, 20, kHorizontalFrame);
   TGLabel *lbl2 = new TGLabel(f2,"Y Position");
   fYpos = new TGNumberEntry(f2, 4, 2, kText_Ypos, TGNumberEntry::kNESRealTwo,
   TGNumberEntry::kNEAAnyNumber);
   fYpos->Resize(50, 20);
   f2->AddFrame(lbl2, new TGLayoutHints(kLHintsLeft,1, 1, 1, 1));
   f2->AddFrame(fYpos, new TGLayoutHints(kLHintsLeft, 7, 1, 1, 1));
   AddFrame(f2, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));

   TGCompositeFrame *f3 = new TGCompositeFrame(this, 120, 20, kHorizontalFrame);
   TGLabel *lbl3 = new TGLabel(f3,"Text Angle");
   fAngle = new TGNumberEntry(f3, 4, 2, kText_Angle, TGNumberEntry::kNESInteger,
   TGNumberEntry::kNEANonNegative);
   fAngle->Resize(50, 20);
   f3->AddFrame(lbl3, new TGLayoutHints(kLHintsLeft,1, 1, 1, 1));
   f3->AddFrame(fAngle, new TGLayoutHints(kLHintsLeft, 7, 1, 1, 1));
   AddFrame(f3, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));

   TGCompositeFrame *f4 = new TGCompositeFrame(this, 120, 20, kHorizontalFrame);
   TGLabel *lbl4 = new TGLabel(f4,"Text Size");
   fSize = new TGNumberEntry(f4, 4, 2, kText_Size, TGNumberEntry::kNESRealTwo,
   TGNumberEntry::kNEANonNegative);
   fSize->Resize(50, 20);
   f4->AddFrame(lbl4, new TGLayoutHints(kLHintsLeft,1, 1, 1, 1));
   f4->AddFrame(fSize, new TGLayoutHints(kLHintsLeft, 7, 1, 1, 1));
   AddFrame(f4, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
}


//______________________________________________________________________________
TTextEditor::~TTextEditor()
{
   // TTextEditor destructor.
}


//______________________________________________________________________________
void TTextEditor::SetModel(TObject *obj)
{
   // Set model.

   fEditedText = (TText*) (obj);

   fAvoidSignal = kTRUE;
   fText->SetText(fEditedText->GetTitle());
   fXpos->SetNumber(fEditedText->GetX());
   fYpos->SetNumber(fEditedText->GetY());
   fAngle->SetNumber(fEditedText->GetTextAngle());
   fSize->SetNumber(fEditedText->GetTextSize());

   if (fInit) ConnectSignals2Slots();
   fAvoidSignal = kFALSE;
}


//_____________________________________________________________________________
void TTextEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   fText->Connect("TextChanged(const char *)","TTextEditor",this,"DoText(const char *)");
   fXpos->Connect("ValueSet(Long_t)", "TTextEditor", this, "DoXpos()");
   fYpos->Connect("ValueSet(Long_t)", "TTextEditor", this, "DoYpos()");
   fAngle->Connect("ValueSet(Long_t)", "TTextEditor", this, "DoAngle()");
   fSize->Connect("ValueSet(Long_t)", "TTextEditor", this, "DoSize()");

   fInit = kFALSE;  // connect the slots to the signals only once
}

//______________________________________________________________________________
void TTextEditor::DoAngle()
{
   // Slot for setting the text Angle.

   if (fAvoidSignal) return;
   fEditedText->SetTextAngle(fAngle->GetNumber());
   Update();
}


//______________________________________________________________________________
void TTextEditor::DoSize()
{
   // Slot for setting the text Size.

   if (fAvoidSignal) return;
   fEditedText->SetTextSize(fSize->GetNumber());
   Update();
}


//______________________________________________________________________________
void TTextEditor::DoText(const char *text)
{
   // Slot for setting the text string.

   if (fAvoidSignal) return;
   fEditedText->SetTitle(text);
   Update();
}


//______________________________________________________________________________
void TTextEditor::DoXpos()
{
   // Slot for setting the text X position.

   if (fAvoidSignal) return;
   fEditedText->SetX(fXpos->GetNumber());
   Update();
}


//______________________________________________________________________________
void TTextEditor::DoYpos()
{
   // Slot for setting the text Y position.

   if (fAvoidSignal) return;
   fEditedText->SetY(fYpos->GetNumber());
   Update();
}

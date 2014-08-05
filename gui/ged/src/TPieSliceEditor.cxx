#include "TPieSliceEditor.h"
#include "TPieSlice.h"
#include "TGTextEntry.h"
#include "TGNumberEntry.h"
#include "TGLabel.h"

ClassImp(TPieSliceEditor)


enum EPieSliceID{
   kPieSlice_Title = 0, kPieSlice_Value, kPieSlice_Offset
};


//______________________________________________________________________________
TPieSliceEditor::TPieSliceEditor(const TGWindow *p,
                  Int_t width, Int_t height,
                  UInt_t options, Pixel_t back)
                  : TGedFrame(p, width, height, options | kVerticalFrame, back)
{
   // TPieSliceEditor constructor.

   fPieSlice = 0;

   // start initializing the window components
   MakeTitle("Title");

   fTitle = new TGTextEntry(this, new TGTextBuffer(50), kPieSlice_Title);
   fTitle->Resize(135, fTitle->GetDefaultHeight());
   fTitle->SetToolTipText("Enter the pie-slice label");
   // better take kLHintsLeft and Right - Right is not working at the moment
   AddFrame(fTitle, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));

   TGCompositeFrame *f1 = new TGCompositeFrame(this, 120, 20, kHorizontalFrame);
   TGLabel *lbl1 = new TGLabel(f1,"Value");
   fValue = new TGNumberEntry(f1, 2, 2, kPieSlice_Value, TGNumberEntry::kNESReal, TGNumberEntry::kNEANonNegative);
   //fValue->SetToolTipText("Set the slice absolute value")
   fValue->Resize(50, 20);
   f1->AddFrame(lbl1, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   f1->AddFrame(fValue, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   AddFrame(f1, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));

   TGCompositeFrame *f2 = new TGCompositeFrame(this, 120, 20, kHorizontalFrame);
   TGLabel *lbl2 = new TGLabel(f2,"Rad Offset");
   fOffset = new TGNumberEntry(f2, 4, 2, kPieSlice_Offset, TGNumberEntry::kNESRealTwo, TGNumberEntry::kNEANonNegative);
   //fOffset->SetToolTipText("Set the slice radial offset")
   fOffset->Resize(50, 20);
   f2->AddFrame(lbl2, new TGLayoutHints(kLHintsLeft,1, 1, 1, 1));
   f2->AddFrame(fOffset, new TGLayoutHints(kLHintsLeft, 7, 1, 1, 1));
   AddFrame(f2, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
}


//______________________________________________________________________________
TPieSliceEditor::~TPieSliceEditor()
{
   // TPieSliceEditor destructor.
}


//______________________________________________________________________________
void TPieSliceEditor::SetModel(TObject *obj)
{
   // Set model.

   fPieSlice = (TPieSlice*) (obj);

   fAvoidSignal = kTRUE;
   fTitle->SetText(fPieSlice->GetTitle());
   fValue->SetNumber(fPieSlice->GetValue());
   fOffset->SetNumber(fPieSlice->GetRadiusOffset());

   if (fInit) ConnectSignals2Slots();
   fAvoidSignal = kFALSE;
}


//_____________________________________________________________________________
void TPieSliceEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   fTitle->Connect("TextChanged(const char *)","TPieSliceEditor",this,"DoTitle(const char *)");
   fValue->Connect("ValueSet(Long_t)", "TPieSliceEditor", this, "DoValue()");
   fOffset->Connect("ValueSet(Long_t)", "TPieSliceEditor", this, "DoOffset()");

   fInit = kFALSE;  // connect the slots to the signals only once
}


//______________________________________________________________________________
void TPieSliceEditor::DoTitle(const char *text)
{
   // Slot for setting the graph title.

   if (fAvoidSignal) return;
   fPieSlice->SetTitle(text);
   Update();
}


//______________________________________________________________________________
void TPieSliceEditor::DoValue()
{
   // Slot for setting the graph title.

   if (fAvoidSignal) return;

   fPieSlice->SetValue(fValue->GetNumber());
   Update();
}


//______________________________________________________________________________
void TPieSliceEditor::DoOffset()
{
   // Slot for setting the graph title.

   if (fAvoidSignal) return;

   fPieSlice->SetRadiusOffset(fOffset->GetNumber());
   Update();
}


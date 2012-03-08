// @(#)root/treeviewer:$Id$
// Author: Bastien Dalla Piazza  02/08/2007

/*************************************************************************
 * Copyright (C) 1995-2007, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TParallelCoordEditor.h"
#include "TParallelCoord.h"
#include "TParallelCoordRange.h"
#include "TParallelCoordVar.h"

#include "TGFrame.h"
#include "TGButton.h"
#include "TGButtonGroup.h"
#include "TGNumberEntry.h"
#include "TGLabel.h"
#include "TGTextEntry.h"
#include "TGComboBox.h"
#include "TGColorSelect.h"
#include "TColor.h"
#include "TG3DLine.h"
#include "TGSlider.h"
#include "TGComboBox.h"
#include "TGDoubleSlider.h"
#include "TTree.h"
#include "TGListBox.h"
#include "TGedPatternSelect.h"

#include "Riostream.h"

ClassImp(TParallelCoordEditor)

//______________________________________________________________________________
/* Begin_Html
<center><h2>The TParallelCoord editor class</h2></center>
This is the TParallelCoord editor. It brings tools to explore datas
Using parallel coordinates. The main tools are:
</ul>
<li> Dots spacing : Set the dots spacing with whichone the lines
  must be drawn. This tool is useful to reduce the image
  cluttering.
<li> The Selections section : Set the current edited selection and
  allows to apply it to the tree through a generated entry list.
<li> The Entries section : Set how many events must be drawn.
  A weight cut can be defioned here (see TParallelCoord for a
  a description of the weight cut).
<li> The Variables tab : To define the global settings to display
  the axes. It is also possible to add a variable from its
  expression or delete a selected one (also possible using right
  click on the pad.
</ul>
End_Html */


enum EParallelWid {
   kGlobalLineColor,
   kLineTypeBgroup,
   kLineTypePoly,
   kLineTypeCurves,
   kGlobalLineWidth,
   kDotsSpacing,
   kDotsSpacingField,
   kSelectionSelect,
   kSelectLineColor,
   kSelectLineWidth,
   kActivateSelection,
   kDeleteSelection,
   kAddSelection,
   kAddSelectionEntry,
   kShowRanges,
   kPaintEntries,
   kEntriesToDraw,
   kFirstEntry,
   kNentries,
   kApplySelect,
   kUnApply,
   kDelayDrawing,
   kHideAllRanges,
   kVariables,
   kDeleteVar,
   kHistHeight,
   kHistWidth,
   kHistBinning,
   kRenameVar,
   kWeightCut,
   kHistColorSelect,
   kHistPatternSelect
};

//______________________________________________________________________________
TParallelCoordEditor::TParallelCoordEditor(const TGWindow* /*p*/,
                                           Int_t/*width*/, Int_t /*height*/,
                                           UInt_t /*options*/, Pixel_t /*back*/)
{
   // Normal constructor.

   fPriority = 1;
   fDelay = kTRUE;

   //**Line**_________________________________________
   MakeTitle("Line");

   TGHorizontalFrame *f1 = new TGHorizontalFrame(this);
   fGlobalLineColor = new TGColorSelect(f1,0,kGlobalLineColor);
   f1->AddFrame(fGlobalLineColor,new TGLayoutHints(kLHintsLeft | kLHintsTop));
   fGlobalLineWidth = new TGLineWidthComboBox(f1, kGlobalLineWidth);
   fGlobalLineWidth->Resize(91, 20);
   f1->AddFrame(fGlobalLineWidth, new TGLayoutHints(kLHintsLeft, 3, 1, 1, 1));
   AddFrame(f1, new TGLayoutHints(kLHintsLeft | kLHintsTop));

   AddFrame(new TGLabel(this,"Dots spacing"),
            new TGLayoutHints(kLHintsLeft | kLHintsCenterY));

   TGHorizontalFrame *f2 = new TGHorizontalFrame(this);
   fDotsSpacing = new TGHSlider(f2,100,kSlider1|kScaleBoth,kDotsSpacing);
   fDotsSpacing->SetRange(0,60);
   f2->AddFrame(fDotsSpacing,new TGLayoutHints(kLHintsLeft | kLHintsCenterY));
   fDotsSpacingField = new TGNumberEntryField(f2, kDotsSpacingField, 0,
                                              TGNumberFormat::kNESInteger,
                                              TGNumberFormat::kNEANonNegative);
   fDotsSpacingField->Resize(40,20);
   f2->AddFrame(fDotsSpacingField,new TGLayoutHints(kLHintsLeft | kLHintsCenterY));
   AddFrame(f2, new TGLayoutHints(kLHintsLeft | kLHintsCenterY));

   fLineTypeBgroup = new TGButtonGroup(this,2,1,0,0, "Line type");
   fLineTypeBgroup->SetRadioButtonExclusive(kTRUE);
   fLineTypePoly = new TGRadioButton(fLineTypeBgroup,"Polyline", kLineTypePoly);
   fLineTypePoly->SetToolTipText("Draw the entries with a polyline");
   fLineTypeCurves = new TGRadioButton(fLineTypeBgroup,"Curves",
                                       kLineTypeCurves);
   fLineTypeCurves->SetToolTipText("Draw the entries with a curve");
   fLineTypeBgroup->ChangeOptions(kChildFrame|kVerticalFrame);
   AddFrame(fLineTypeBgroup, new TGLayoutHints(kLHintsCenterY | kLHintsLeft));

   //**Selections**___________________________________
   MakeTitle("Selections");

   fHideAllRanges = new TGCheckButton(this,"Hide all ranges",kHideAllRanges);
   AddFrame(fHideAllRanges);

   fSelectionSelect = new TGComboBox(this,kSelectionSelect);
   fSelectionSelect->Resize(140,20);
   AddFrame(fSelectionSelect, new TGLayoutHints(kLHintsCenterY | kLHintsLeft,0,0,3,0));

   TGHorizontalFrame *f3 = new TGHorizontalFrame(this);
   fSelectLineColor = new TGColorSelect(f3,0,kSelectLineColor);
   f3->AddFrame(fSelectLineColor,new TGLayoutHints(kLHintsLeft | kLHintsTop));
   fSelectLineWidth = new TGLineWidthComboBox(f3, kSelectLineWidth);
   fSelectLineWidth->Resize(94, 20);
   f3->AddFrame(fSelectLineWidth, new TGLayoutHints(kLHintsLeft, 3, 1, 1, 1));
   AddFrame(f3, new TGLayoutHints(kLHintsLeft | kLHintsTop,0,0,3,0));

   fActivateSelection = new TGCheckButton(this,"Activate",kActivateSelection);
   fActivateSelection->SetToolTipText("Activate the current selection");
   AddFrame(fActivateSelection);
   fShowRanges = new TGCheckButton(this,"Show ranges",kShowRanges);
   AddFrame(fShowRanges);

   TGHorizontalFrame *f5 = new TGHorizontalFrame(this);
   fAddSelectionField = new TGTextEntry(f5);
   fAddSelectionField->Resize(57,20);
   f5->AddFrame(fAddSelectionField, new TGLayoutHints(kLHintsLeft | kLHintsCenterY));
   fAddSelection = new TGTextButton(f5,"Add");
   fAddSelection->SetToolTipText("Add a new selection (Right click on the axes to add a range).");
   f5->AddFrame(fAddSelection, new TGLayoutHints(kLHintsLeft | kLHintsCenterY,3,0,0,0));
   fDeleteSelection = new TGTextButton(f5,"Delete",kDeleteSelection);
   fDeleteSelection->SetToolTipText("Delete the current selection");
   f5->AddFrame(fDeleteSelection, new TGLayoutHints(kLHintsLeft | kLHintsCenterY,3,0,0,0));
   AddFrame(f5, new TGLayoutHints(kLHintsLeft | kLHintsCenterY,3,0,0,0));

   TGHorizontalFrame *f7 = new TGHorizontalFrame(this);
   fApplySelect = new TGTextButton(f7,"Apply to tree",kApplySelect);
   fApplySelect->SetToolTipText("Generate an entry list for the current selection and apply it to the tree.");
   f7->AddFrame(fApplySelect);
   fUnApply = new TGTextButton(f7,"Reset tree",kUnApply);
   fUnApply->SetToolTipText("Reset the tree entry list");
   f7->AddFrame(fUnApply, new TGLayoutHints(kLHintsLeft | kLHintsCenterY,10,0,0,0));
   AddFrame(f7, new TGLayoutHints(kLHintsLeft | kLHintsCenterY,0,0,3,0));

   //**Entries**___________________________________
   MakeTitle("Entries");

   fPaintEntries = new TGCheckButton(this,"Draw entries",kPaintEntries);
   AddFrame(fPaintEntries);
   fDelayDrawing = new TGCheckButton(this,"Delay Drawing", kDelayDrawing);
   AddFrame(fDelayDrawing);

   fEntriesToDraw = new TGDoubleHSlider(this,140,kDoubleScaleBoth,kEntriesToDraw);
   AddFrame(fEntriesToDraw);

   TGHorizontalFrame *f6 = new TGHorizontalFrame(this);
   TGVerticalFrame *v1 = new TGVerticalFrame(f6);
   TGVerticalFrame *v2 = new TGVerticalFrame(f6);
   v1->AddFrame(new TGLabel(v1,"First entry:"));
   fFirstEntry = new TGNumberEntryField(v1, kFirstEntry, 0,
                                        TGNumberFormat::kNESInteger,
                                        TGNumberFormat::kNEANonNegative);
   fFirstEntry->Resize(68,20);
   v1->AddFrame(fFirstEntry);
   v2->AddFrame(new TGLabel(v2,"# of entries:"));
   fNentries = new TGNumberEntryField(v2, kFirstEntry, 0,
                                     TGNumberFormat::kNESInteger,
                                     TGNumberFormat::kNEANonNegative);
   fNentries->Resize(68,20);
   v2->AddFrame(fNentries);
   f6->AddFrame(v1);
   f6->AddFrame(v2, new TGLayoutHints(kLHintsLeft,4,0,0,0));
   AddFrame(f6);

   AddFrame(new TGLabel(this,"Weight cut:"));

   TGHorizontalFrame *f8 = new TGHorizontalFrame(this);
   fWeightCut = new TGHSlider(f8,100,kSlider1|kScaleBoth,kDotsSpacing);
   fWeightCutField = new TGNumberEntryField(f8,kDotsSpacingField, 0,
                                            TGNumberFormat::kNESInteger,
                                            TGNumberFormat::kNEANonNegative);
   fWeightCutField->Resize(40,20);
   f8->AddFrame(fWeightCut);
   f8->AddFrame(fWeightCutField);
   AddFrame(f8);

   MakeVariablesTab();
}


//______________________________________________________________________________
void TParallelCoordEditor::MakeVariablesTab()
{
   // Make the "variable" tab.

   fVarTab = CreateEditorTabSubFrame("Variables");
   //**Variable**_________________________________

   TGHorizontalFrame *f9 = new TGHorizontalFrame(fVarTab);
   fAddVariable = new TGTextEntry(f9);
   fAddVariable->Resize(71,20);
   f9->AddFrame(fAddVariable, new TGLayoutHints(kLHintsCenterY));
   fButtonAddVar = new TGTextButton(f9,"Add");
   fButtonAddVar->SetToolTipText("Add a new variable from the tree (must be a valid expression).");
   f9->AddFrame(fButtonAddVar, new TGLayoutHints(kLHintsCenterY,4,0,0,0));
   fVarTab->AddFrame(f9);

   TGHorizontalFrame *f10 = new TGHorizontalFrame(fVarTab);
   fVariables = new TGComboBox(f10,kVariables);
   fVariables->Resize(105,20);
   f10->AddFrame(fVariables, new TGLayoutHints(kLHintsCenterY));
   fVarTab->AddFrame(f10,new TGLayoutHints(kLHintsLeft,0,0,2,0));

   TGHorizontalFrame *f12 = new TGHorizontalFrame(fVarTab);
   fDeleteVar = new TGTextButton(f12,"Delete",kDeleteVar);
   fDeleteVar->SetToolTipText("Delete the current selected variable");
   f12->AddFrame(fDeleteVar, new TGLayoutHints(kLHintsCenterY,1,0,0,0));
   fRenameVar = new TGTextButton(f12,"Rename",kRenameVar);
   fRenameVar->SetToolTipText("Rename the current selected variable");
   f12->AddFrame(fRenameVar, new TGLayoutHints(kLHintsCenterY,4,0,0,0));
   fVarTab->AddFrame(f12,new TGLayoutHints(kLHintsLeft,0,0,2,0));

   fVarTab->AddFrame(new TGLabel(fVarTab,"Axis histograms:"));

   TGHorizontalFrame *f11 = new TGHorizontalFrame(fVarTab);
   TGVerticalFrame *v3 = new TGVerticalFrame(f11);
   TGVerticalFrame *v4 = new TGVerticalFrame(f11);
   v3->AddFrame(new TGLabel(v3,"Binning:"));
   fHistBinning = new TGNumberEntryField(v3, kHistWidth, 0,
                                         TGNumberFormat::kNESInteger,
                                         TGNumberFormat::kNEANonNegative);
   fHistBinning->Resize(68,20);
   v3->AddFrame(fHistBinning);
   v4->AddFrame(new TGLabel(v4,"Width:"));
   fHistWidth = new TGNumberEntryField(v4, kHistWidth, 0,
                                       TGNumberFormat::kNESInteger,
                                       TGNumberFormat::kNEANonNegative);
   fHistWidth->Resize(68,20);
   v4->AddFrame(fHistWidth, new TGLayoutHints(kLHintsLeft,4,0,0,0));
   f11->AddFrame(v3);
   f11->AddFrame(v4);
   fVarTab->AddFrame(f11);

   fHistShowBoxes = new TGCheckButton(fVarTab,"Show box histograms");
   fVarTab->AddFrame(fHistShowBoxes);
   
   fVarTab->AddFrame(new TGLabel(fVarTab,"Bar histograms style:"));
   
   TGCompositeFrame *f13 = new TGCompositeFrame(fVarTab, 80, 20, kHorizontalFrame);
   fHistColorSelect = new TGColorSelect(f13, 0, kHistColorSelect);
   f13->AddFrame(fHistColorSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fHistColorSelect->Associate(this);
   fHistPatternSelect = new TGedPatternSelect(f13, 1, kHistPatternSelect);
   f13->AddFrame(fHistPatternSelect, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 1));
   fHistPatternSelect->Associate(this);
   fVarTab->AddFrame(f13, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
}


//______________________________________________________________________________
TParallelCoordEditor::~TParallelCoordEditor()
{
   // Destructor.

   delete fLineTypePoly;
   delete fLineTypeCurves;
}


//______________________________________________________________________________
void TParallelCoordEditor::CleanUpSelections()
{
   // Clean up the selection combo box.

   TList *list = fParallel->GetSelectList();
   fSelectionSelect->RemoveAll();
   Bool_t enable = list->GetSize() > 0;
   fSelectionSelect->SetEnabled(enable);
   fSelectLineColor->SetEnabled(enable);
   fSelectLineWidth->SetEnabled(enable);
   fActivateSelection->SetEnabled(enable);
   fShowRanges->SetEnabled(enable);
   fDeleteSelection->SetEnabled(enable);
   if (list->GetSize() > 0) {
      Int_t i = 0;
      TIter next(list);
      TParallelCoordSelect* sel;
      while ((sel = (TParallelCoordSelect*)next())) {
         fSelectionSelect->AddEntry(sel->GetTitle(),i);
         TGLBEntry *entry = fSelectionSelect->GetListBox()->GetEntry(i);
         if (entry)
            entry->SetBackgroundColor(TColor::Number2Pixel(sel->GetLineColor()));
         ++i;
      }
      sel = fParallel->GetCurrentSelection();
      if (sel) {
         fSelectionSelect->Select(list->IndexOf(sel),kFALSE);
         Color_t c;
         Pixel_t p;
         c = sel->GetLineColor();
         p = TColor::Number2Pixel(c);
         fSelectLineColor->SetColor(p);
         fSelectLineWidth->Select(sel->GetLineWidth());
         fActivateSelection->SetOn(sel->TestBit(TParallelCoordSelect::kActivated));
         fShowRanges->SetOn(sel->TestBit(TParallelCoordSelect::kShowRanges));
      }
   }
}


//______________________________________________________________________________
void TParallelCoordEditor::CleanUpVariables()
{
   // Clean up the variables combo box.
   
   TList *list = fParallel->GetVarList();
   fVariables->RemoveAll();
   Bool_t enable = list->GetSize() > 0;
   fVariables->SetEnabled(enable);
   fDeleteVar->SetEnabled(enable);
   fHistShowBoxes->SetEnabled(enable);
   fHistWidth->SetEnabled(enable);
   fHistBinning->SetEnabled(enable);
   if (list->GetSize() > 0) {
      Int_t i = 0;
      TIter next(list);
      TParallelCoordVar* var;
      while ((var = (TParallelCoordVar*)next())) {
         fVariables->AddEntry(var->GetTitle(),i);
         ++i;
      }
      var = (TParallelCoordVar*)list->First();
      fVariables->Select(0,kFALSE);
      fHistShowBoxes->SetOn(var->TestBit(TParallelCoordVar::kShowBarHisto));
      fHistWidth->SetNumber(var->GetHistLineWidth());
      fHistBinning->SetNumber(var->GetHistBinning());
   }
}


//______________________________________________________________________________
void TParallelCoordEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   fGlobalLineColor->Connect("ColorSelected(Pixel_t)","TParallelCoordEditor",
                             this, "DoGlobalLineColor(Pixel_t)");
   fGlobalLineWidth->Connect("Selected(Int_t)","TParallelCoordEditor",
                             this, "DoGlobalLineWidth(Int_t)");
   fDotsSpacing->Connect("Released()","TParallelCoordEditor",
                        this, "DoDotsSpacing()");
   fDotsSpacing->Connect("PositionChanged(Int_t)","TParallelCoordEditor",
                        this, "DoLiveDotsSpacing(Int_t)");
   fDotsSpacingField->Connect("ReturnPressed()","TParallelCoordEditor",
                              this, "DoDotsSpacingField()");
   fLineTypeBgroup->Connect("Clicked(Int_t)", "TParallelCoordEditor",
                            this, "DoLineType()");
   fSelectionSelect->Connect("Selected(const char*)","TParallelCoordEditor",
                            this, "DoSelectionSelect(const char*)");
   fSelectLineColor->Connect("ColorSelected(Pixel_t)","TParallelCoordEditor",
                             this, "DoSelectLineColor(Pixel_t)");
   fSelectLineWidth->Connect("Selected(Int_t)","TParallelCoordEditor",
                             this, "DoSelectLineWidth(Int_t)");
   fActivateSelection->Connect("Toggled(Bool_t)","TParallelCoordEditor",
                               this, "DoActivateSelection(Bool_t)");
   fShowRanges->Connect("Toggled(Bool_t)","TParallelCoordEditor",
                        this, "DoShowRanges(Bool_t)");
   fDeleteSelection->Connect("Clicked()","TParallelCoordEditor",
                             this, "DoDeleteSelection()");
   fAddSelection->Connect("Clicked()","TParallelCoordEditor",
                             this, "DoAddSelection()");
   fPaintEntries->Connect("Toggled(Bool_t)","TParallelCoordEditor",
                               this, "DoPaintEntries(Bool_t)");
   fEntriesToDraw->Connect("Released()","TParallelCoordEditor",
                          this, "DoEntriesToDraw()");
   fEntriesToDraw->Connect("PositionChanged()","TParallelCoordEditor",
                          this, "DoLiveEntriesToDraw()");
   fFirstEntry->Connect("ReturnPressed()","TParallelCoordEditor",
                        this, "DoFirstEntry()");
   fNentries->Connect("ReturnPressed()","TParallelCoordEditor",
                     this, "DoNentries()");
   fApplySelect->Connect("Clicked()","TParallelCoordEditor",
                         this, "DoApplySelect()");
   fUnApply->Connect("Clicked()","TParallelCoordEditor",
                     this, "DoUnApply()");
   fDelayDrawing->Connect("Toggled(Bool_t)","TParallelCoordEditor",
                     this, "DoDelayDrawing(Bool_t)");
   fAddVariable->Connect("ReturnPressed()","TParallelCoordEditor",
                     this, "DoAddVariable()");
   fButtonAddVar->Connect("Clicked()","TParallelCoordEditor",
                     this, "DoAddVariable()");
   fHideAllRanges->Connect("Toggled(Bool_t)","TParallelCoordEditor",
                           this, "DoHideAllRanges(Bool_t)");
   fVariables->Connect("Selected(const char*)","TParallelCoordEditor",
                      this, "DoVariableSelect(const char*)");
   fDeleteVar->Connect("Clicked()","TParallelCoordEditor",
                      this, "DoDeleteVar()");
   fHistWidth->Connect("ReturnPressed()","TParallelCoordEditor",
                       this, "DoHistWidth()");
   fHistBinning->Connect("ReturnPressed()","TParallelCoordEditor",
                         this, "DoHistBinning()");
   fWeightCut->Connect("Released()","TParallelCoordEditor",
                       this, "DoWeightCut()");
   fWeightCut->Connect("PositionChanged(Int_t)","TParallelCoordEditor",
                       this, "DoLiveWeightCut(Int_t)");
   fWeightCutField->Connect("ReturnPressed()","TParallelCoordEditor",
                            this, "DoWeightCut()");
   fHistColorSelect->Connect("ColorSelected(Pixel_t)", "TParallelCoordEditor",
                             this, "DoHistColorSelect(Pixel_t)");
   fHistPatternSelect->Connect("PatternSelected(Style_t)", "TParallelCoordEditor",
                               this, "DoHistPatternSelect(Style_t)");
   fHistShowBoxes->Connect("Toggled(Bool_t)","TParallelCoordEditor",
                               this, "DoHistShowBoxes(Bool_t)");

   fInit = kFALSE;
}


//______________________________________________________________________________
void TParallelCoordEditor::DoActivateSelection(Bool_t on)
{
   // Slot to activate or not a selection.

   if (fAvoidSignal) return;

   TParallelCoordSelect* sel = fParallel->GetCurrentSelection();
   if (sel) {
      sel->SetActivated(on);
      Update();
   }
}


//______________________________________________________________________________
void TParallelCoordEditor::DoAddSelection()
{
   // Slot to add a selection.

   TString title = fAddSelectionField->GetText();
   if(title == "") title = "Selection";
   TString titlebis = title;
   Bool_t found = kTRUE;
   Int_t i=1;
   while (found){
      if(fSelectionSelect->FindEntry(titlebis)) {
         titlebis = title;
         titlebis.Append(Form("(%d)",i));
      }
      else found = kFALSE;
      ++i;
   }

   fParallel->AddSelection(titlebis.Data());

   CleanUpSelections();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoAddVariable()
{
   // Slot to add a variable.

   if (fAvoidSignal) return;

   fParallel->AddVariable(fAddVariable->GetText());
   CleanUpVariables();
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoApplySelect()
{
   // Slot to apply a selection to the tree.

   //FIXME I forgot to update the slider over the entries
   //      (nentries and firstentry might have changed after applying the selection)
   
   if (fAvoidSignal) return;

   fParallel->ApplySelectionToTree();
   Update();
   SetModel(fParallel);
}


//______________________________________________________________________________
void TParallelCoordEditor::DoDelayDrawing(Bool_t on)
{
   // Slot to delay the drawing.

   if (fAvoidSignal) return;

   fDelay = on;
   fParallel->SetLiveRangesUpdate(!on);
}


//______________________________________________________________________________
void TParallelCoordEditor::DoDeleteSelection()
{
   // Slot to delete a selection.

   if (fAvoidSignal) return;

   fParallel->DeleteSelection(fParallel->GetCurrentSelection());

   CleanUpSelections();
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoDeleteVar()
{
   // Slot to delete a variable().

   if (fAvoidSignal) return;

   TParallelCoordVar* var = fParallel->RemoveVariable(((TGTextLBEntry*)fVariables->GetSelectedEntry())->GetTitle());
   CleanUpVariables();
   if(var) Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoDotsSpacing()
{
   // Slot to set the line dotspacing.

   if (fAvoidSignal) return;

   fParallel->SetDotsSpacing(fDotsSpacing->GetPosition());
   fDotsSpacingField->SetNumber((Int_t)fDotsSpacing->GetPosition());
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoDotsSpacingField()
{
   // Slot to set the line dotspacing from the entry field.

   if (fAvoidSignal) return;

   fParallel->SetDotsSpacing((Int_t)fDotsSpacingField->GetNumber());
   fDotsSpacing->SetPosition((Int_t)fDotsSpacingField->GetNumber());
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoEntriesToDraw()
{
   // Slot to select the entries to be drawn.

   if (fAvoidSignal) return;

   Long64_t nentries,firstentry;
   firstentry = (Long64_t)fEntriesToDraw->GetMinPosition();
   nentries = (Long64_t)(fEntriesToDraw->GetMaxPosition() - fEntriesToDraw->GetMinPosition() + 1);

   fParallel->SetCurrentFirst(firstentry);
   fParallel->SetCurrentN(nentries);
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoFirstEntry()
{
   // Slot to set the first entry.

   if (fAvoidSignal) return;

   fParallel->SetCurrentFirst((Long64_t)fFirstEntry->GetNumber());
   fEntriesToDraw->SetPosition((Long64_t)fFirstEntry->GetNumber(),(Long64_t)fFirstEntry->GetNumber()+fParallel->GetCurrentN());
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoGlobalLineColor(Pixel_t a)
{
   // Slot to set the global line color.

   if (fAvoidSignal) return;

   fParallel->SetLineColor(TColor::GetColor(a));
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoGlobalLineWidth(Int_t wid)
{
   // Slot to set the global line width.

   if (fAvoidSignal) return;

   fParallel->SetLineWidth(wid);
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoHideAllRanges(Bool_t on)
{
   // Slot to hide all the ranges.

   if (fAvoidSignal) return;

   TIter next(fParallel->GetSelectList());
   TParallelCoordSelect* sel;
   while((sel = (TParallelCoordSelect*)next())) sel->SetShowRanges(!on);
   fShowRanges->SetOn(!on);
   fShowRanges->SetEnabled(!on);
   fShowRanges->SetOn(!on);
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoHistBinning()
{
   // Slot to set the axes histogram binning.

   if (fAvoidSignal) return;

   fParallel->SetAxisHistogramBinning((Int_t)fHistBinning->GetNumber());
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoHistColorSelect(Pixel_t p)
{
   // Slot to set the histograms color.

   if (fAvoidSignal) return;
   
   Color_t col = TColor::GetColor(p);
   TIter next(fParallel->GetVarList());
   TParallelCoordVar *var = NULL;
   while ((var = (TParallelCoordVar*)next())) var->SetFillColor(col);
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoHistShowBoxes(Bool_t s)
{
   // Slot to set histogram height.

   if (fAvoidSignal) return;

   TIter next(fParallel->GetVarList());
   TParallelCoordVar* var;
   while ((var = (TParallelCoordVar*)next())) var->SetBit(TParallelCoordVar::kShowBarHisto,s);
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoHistPatternSelect(Style_t sty)
{
   // Slot to set the histograms fill style.

   if (fAvoidSignal) return;
   
   TIter next(fParallel->GetVarList());
   TParallelCoordVar *var = NULL;
   while ((var = (TParallelCoordVar*)next())) var->SetFillStyle(sty);
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoHistWidth()
{
   // Slot to set histogram width.

   if (fAvoidSignal) return;

   fParallel->SetAxisHistogramLineWidth((Int_t)fHistWidth->GetNumber());
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoLineType()
{
   // Slot to set the line type.

   if (fAvoidSignal) return;

   if(fLineTypePoly->GetState() == kButtonDown) fParallel->SetCurveDisplay(kFALSE);
   else fParallel->SetCurveDisplay(kTRUE);
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoLiveDotsSpacing(Int_t a)
{
   // Slot to set the dots spacing online.

   if (fAvoidSignal) return;
   fDotsSpacingField->SetNumber(a);
   fParallel->SetDotsSpacing(a);
   if(!fDelay) Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoLiveEntriesToDraw()
{
   // Slot to update the entries fields from the slider position.

   if (fAvoidSignal) return;

   Long64_t nentries,firstentry;
   firstentry = (Long64_t)fEntriesToDraw->GetMinPosition();
   nentries = (Long64_t)(fEntriesToDraw->GetMaxPosition() - fEntriesToDraw->GetMinPosition() + 1);

   fFirstEntry->SetNumber(firstentry);
   fNentries->SetNumber(nentries);

   if (!fDelay) {
      fParallel->SetCurrentFirst(firstentry);
      fParallel->SetCurrentN(nentries);
      Update();
   }
}


//______________________________________________________________________________
void TParallelCoordEditor::DoLiveWeightCut(Int_t n)
{
   // Slot to update the wieght cut entry field from the slider position.

   if (fAvoidSignal) return;

   fWeightCutField->SetNumber(n);
   if (!fDelay) {
      fParallel->SetWeightCut(n);
      Update();
   }
}


//______________________________________________________________________________
void TParallelCoordEditor::DoNentries()
{
   // Slot to set the number of entries to display.

   if (fAvoidSignal) return;

   fParallel->SetCurrentN((Long64_t)fNentries->GetNumber());
   fEntriesToDraw->SetPosition(fParallel->GetCurrentFirst(),fParallel->GetCurrentFirst()+fParallel->GetCurrentN());
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoPaintEntries(Bool_t on)
{
   // Slot to postpone the entries drawing.

   if (fAvoidSignal) return;

   fParallel->SetBit(TParallelCoord::kPaintEntries,on);
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoSelectLineColor(Pixel_t a)
{
   // Slot to set the global line color.

   if (fAvoidSignal) return;

   TParallelCoordSelect* sel = fParallel->GetCurrentSelection();
   if (sel) sel->SetLineColor(TColor::GetColor(a));
   fSelectionSelect->GetSelectedEntry()->SetBackgroundColor(a);
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::DoSelectLineWidth(Int_t wid)
{
   // Slot to set the global line width.

   if (fAvoidSignal) return;

   TParallelCoordSelect* sel = fParallel->GetCurrentSelection();
   if (sel) {
      sel->SetLineWidth(wid);
      Update();
   }
}


//______________________________________________________________________________
void TParallelCoordEditor::DoSelectionSelect(const char* title)
{
   // Slot to set the selection beeing edited.

   if (fAvoidSignal) return;

   if (!fParallel->SetCurrentSelection(title)) return;

   Color_t c = fParallel->GetCurrentSelection()->GetLineColor();
   Pixel_t p = TColor::Number2Pixel(c);
   fSelectLineColor->SetColor(p,kFALSE);

   fSelectLineWidth->Select(fParallel->GetCurrentSelection()->GetLineWidth(),kFALSE);

   fActivateSelection->SetOn(fParallel->GetCurrentSelection()->TestBit(TParallelCoordSelect::kActivated));
   fShowRanges->SetOn(fParallel->GetCurrentSelection()->TestBit(TParallelCoordSelect::kShowRanges));
}


//______________________________________________________________________________
void TParallelCoordEditor::DoShowRanges(Bool_t s)
{
   // Slot to show or not the ranges on the pad.

   if (fAvoidSignal) return;

   TParallelCoordSelect *select = fParallel->GetCurrentSelection();
   if (select) {
      select->SetShowRanges(s);
      Update();
   }
}


//______________________________________________________________________________
void TParallelCoordEditor::DoUnApply()
{
   // Slot to reset the tree entry list to the original one.

   if (fAvoidSignal) return;

   fParallel->ResetTree();
   Update();
   SetModel(fParallel);
}


//______________________________________________________________________________
void TParallelCoordEditor::DoVariableSelect(const char* /*var*/)
{
   // Slot to select a variable.

   //cout<<((TGTextLBEntry*)fVariables->GetSelectedEntry())->GetTitle()<<endl;
}


//______________________________________________________________________________
void TParallelCoordEditor::DoWeightCut()
{
   // Slot to update the weight cut.

   if (fAvoidSignal) return;

   Int_t n = (Int_t)fWeightCutField->GetNumber();
   fParallel->SetWeightCut(n);
   Update();
}


//______________________________________________________________________________
void TParallelCoordEditor::SetModel(TObject* obj)
{
   // Pick up the used parallel coordinates plot attributes.

   if (!obj) return;
   fParallel = dynamic_cast<TParallelCoord*>(obj);
   if (!fParallel) return;
   fAvoidSignal = kTRUE;

   Color_t c = fParallel->GetLineColor();
   Pixel_t p = TColor::Number2Pixel(c);
   fGlobalLineColor->SetColor(p);

   fGlobalLineWidth->Select(fParallel->GetLineWidth());

   fPaintEntries->SetOn(fParallel->TestBit(TParallelCoord::kPaintEntries));

   fDotsSpacing->SetPosition(fParallel->GetDotsSpacing());

   fDotsSpacingField->SetNumber(fParallel->GetDotsSpacing());

   Bool_t cur = fParallel->GetCurveDisplay();
   if (cur) fLineTypeBgroup->SetButton(kLineTypeCurves,kTRUE);
   else     fLineTypeBgroup->SetButton(kLineTypePoly,kTRUE);

   if(fInit) fHideAllRanges->SetOn(kFALSE);
   
   CleanUpSelections();
   CleanUpVariables();

   if (fInit) fEntriesToDraw->SetRange(0,fParallel->GetNentries());
   fEntriesToDraw->SetPosition(fParallel->GetCurrentFirst(), fParallel->GetCurrentFirst()+fParallel->GetCurrentN());

   fFirstEntry->SetNumber(fParallel->GetCurrentFirst());
   fNentries->SetNumber(fParallel->GetCurrentN());

   fDelayDrawing->SetOn(fDelay);

   fWeightCut->SetRange(0,(Int_t)(fParallel->GetNentries()/10)); // Maybe search here for better boundaries.
   fWeightCut->SetPosition(fParallel->GetWeightCut());
   fWeightCutField->SetNumber(fParallel->GetWeightCut());
   
   fHistColorSelect->SetColor(TColor::Number2Pixel(((TParallelCoordVar*)fParallel->GetVarList()->Last())->GetFillColor()), kFALSE);
   fHistPatternSelect->SetPattern(((TParallelCoordVar*)fParallel->GetVarList()->Last())->GetFillStyle(),kFALSE);
   
   if (fInit) ConnectSignals2Slots();

   fAvoidSignal = kFALSE;
}

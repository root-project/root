// @(#)root/ged:$Name:  TH2Editor.cxx
// Author: Carsten Hof   09/08/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TH2Editor                                                           //
//                                                                      //
//  Editor for histogram attributes.                                    //
//	 ?????????? all parts are missing !!!!!!!!                      //
//                                                                      //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/TH2Editor.gif">
*/
//End_Html

#include "TH2Editor.h"
#include "TGedFrame.h"
#include "TGComboBox.h"
#include "TGTextEntry.h"
#include "TGToolTip.h"
#include "TGLabel.h"
#include "TGClient.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TString.h"
#include "TGButtonGroup.h"
#include "TGNumberEntry.h"
#include "TG3DLine.h"
#include "TGDoubleSlider.h"
#include "TView.h"
#include "TCanvas.h"
#include "TGedPatternSelect.h"
#include "TGColorSelect.h"
#include "TGColorDialog.h"
#include "TColor.h"


ClassImp(TH2Editor)

enum {
   kTH2_TITLE,
   kDIM_SIMPLE,
   kDIM_COMPLEX,
   kHIST_TYPE,
   kTYPE_LEGO,
   kTYPE_LEGO1,
   kTYPE_LEGO2,
   kTYPE_SURF,
   kTYPE_SURF1,
   kTYPE_SURF2,
   kTYPE_SURF3,
   kTYPE_SURF4,
   kTYPE_SURF5,
   kCOORD_TYPE,
   kCOORDS_CAR,
   kCOORDS_CYL,
   kCOORDS_POL,
   kCOORDS_PSR,
   kCOORDS_SPH,
   kCONT_TYPE,  
   kERROR_ONOFF,
   kPALETTE_ONOFF,
   kPALETTE_ONOFF1,
   kARROW_ONOFF,
   kBOX_ONOFF,
   kSCAT_ONOFF,
   kCOL_ONOFF,
   kTEXT_ONOFF,
   kFRONTBOX_ONOFF,
   kBACKBOX_ONOFF,
   kBAR_WIDTH,
   kBAR_OFFSET,
   kCONT_NONE,
   kCONT_0,
   kCONT_1,
   kCONT_2,
   kCONT_3,
   kCONT_4,
   kCONT_LEVELS,
   kCONT_LEVELS1,
   kSLIDERX_MIN,
   kSLIDERX_MAX,
   kSLIDERY_MIN,
   kSLIDERY_MAX,
   kDELAYED_DRAWING,
   kCOLOR, 
   kPATTERN
};

//______________________________________________________________________________

TH2Editor::TH2Editor(const TGWindow *p, Int_t id, Int_t width,
                         Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of histogram attribute GUI.

   fHist = 0;
   
   MakeTitle("Title");
  
   fTitlePrec = 2;
   fTitle = new TGTextEntry(this, new TGTextBuffer(50), kTH2_TITLE);
   fTitle->Resize(135, fTitle->GetDefaultHeight());
   fTitle->SetToolTipText("Enter the histogram title string");
   AddFrame(fTitle, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));

   
   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fdimgroup = new TGHButtonGroup(f2,"Plot");
   fDim = new TGRadioButton(fdimgroup,"2-D",kDIM_SIMPLE);
   fDim->SetToolTipText("A 2-d plot of the histogram is dawn");
   fDim0 = new TGRadioButton(fdimgroup,"3-D",kDIM_COMPLEX);
   fDim0->SetToolTipText("A 3-d plot of the histogram is dawn");
   fdimgroup->SetLayoutHints(new TGLayoutHints(kLHintsLeft ,-2,3,3,-7),fDim);
   fdimgroup->SetLayoutHints(new TGLayoutHints(kLHintsLeft ,16,-1,3,-7),fDim0);   
   fdimgroup->Show();
   fdimgroup->ChangeOptions(kFitWidth|kChildFrame|kHorizontalFrame);
   f2->AddFrame(fdimgroup, new TGLayoutHints(kLHintsTop, 4, 1, 0, 0));
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 4));


   f3 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGLabel *fType = new TGLabel(f3, "Type:"); 
   f3->AddFrame(fType, new TGLayoutHints(kLHintsLeft, 6, 1, 4, 1));
   fTypeCombo = BuildHistTypeComboBox(f3, kHIST_TYPE);
   f3->AddFrame(fTypeCombo, new TGLayoutHints(kLHintsLeft, 15, 1, 2, 1));
   fTypeCombo->Resize(86, 20);
   fTypeCombo->Associate(this);
   AddFrame(f3, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   
   f4 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGLabel *fCoords = new TGLabel(f4, "Coords:"); 
   f4->AddFrame(fCoords, new TGLayoutHints(kLHintsLeft, 6, 1, 4, 1));
   fCoordsCombo = BuildHistCoordsComboBox(f4, kCOORD_TYPE);
   f4->AddFrame(fCoordsCombo, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 1));
   fCoordsCombo->Resize(86, 20);
   fCoordsCombo->Associate(this);
   AddFrame(f4, new TGLayoutHints(kLHintsTop, 1, 1, 0, 3));

   f5 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGLabel *fAddLabel = new TGLabel(f5, "Contour:"); 
   f5->AddFrame(fAddLabel, new TGLayoutHints(kLHintsLeft, 11, 1, 4, 1));
   fContCombo = BuildHistContComboBox(f5, kCONT_TYPE);
   f5->AddFrame(fContCombo, new TGLayoutHints(kLHintsLeft, 18, 1, 2, 1));
   fContCombo->Resize(61, 20);
   fContCombo->Associate(this);
   AddFrame(f5, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   f16 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fColContLbl = new TGLabel(f16, "Cont #:");  
   f16->AddFrame(fColContLbl, new TGLayoutHints( kLHintsLeft, 11, 1, 4, 1));                            
   fContLevels = new TGNumberEntry(f16, 20, 0, kCONT_LEVELS, 
                                      TGNumberFormat::kNESInteger,
                                      TGNumberFormat::kNEANonNegative, 
                                      TGNumberFormat::kNELLimitMinMax, 1, 99);
   fContLevels->GetNumberEntry()->SetToolTipText("Set number of contours (1..99)");
   fContLevels->Resize(60,20);
   f16->AddFrame(fContLevels, new TGLayoutHints(kLHintsLeft, 25, 1, 2, 1));
   AddFrame(f16, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   f6 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);

   TGCompositeFrame *f7 = new TGCompositeFrame(f6, 40, 20, kVerticalFrame);
   fAddArr = new TGCheckButton(f7, "Arrow", kARROW_ONOFF);
   fAddArr ->SetToolTipText("Shows gradient between adjacent cells");
   f7->AddFrame(fAddArr, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   fAddCol = new TGCheckButton(f7, "Col", kCOL_ONOFF);
   fAddCol ->SetToolTipText("A box is drawn for each cell with a color scale varying with contents");
   f7->AddFrame(fAddCol, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   fAddText = new TGCheckButton(f7, "Text", kTEXT_ONOFF);
   fAddText ->SetToolTipText("Draw bin contents as text");
   f7->AddFrame(fAddText, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 3));

   TGCompositeFrame *f8 = new TGCompositeFrame(f6, 40, 20, kVerticalFrame);
   fAddBox = new TGCheckButton(f8, "Box", kBOX_ONOFF);
   fAddBox ->SetToolTipText("A box is drawn for each cell with surface proportional to contents");
   f8->AddFrame(fAddBox, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   fAddScat = new TGCheckButton(f8, "Scat", kSCAT_ONOFF);
   fAddScat ->SetToolTipText("Draw a scatter-plot");
   f8->AddFrame(fAddScat, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   fAddPalette = new TGCheckButton(f8, "Palette", kPALETTE_ONOFF);
   fAddPalette ->SetToolTipText("Add color palette beside the histogram");
   f8->AddFrame(fAddPalette, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   
   f6->AddFrame(f7, new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f6->AddFrame(f8, new TGLayoutHints(kLHintsLeft, 5, 1, 0, 0));   
   AddFrame(f6, new TGLayoutHints(kLHintsTop, 3, 1, 0, 0));

   f9 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame); 
   TGCompositeFrame *f10 = new TGCompositeFrame(f9, 40, 20, kVerticalFrame);
   fAddError = new TGCheckButton(f10, "Errors", kERROR_ONOFF);
   fAddError ->SetToolTipText("Add color palette beside the histogram");
   f10->AddFrame(fAddError, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   fAddPalette1 = new TGCheckButton(f10, "Palette", kPALETTE_ONOFF1);
   fAddPalette1 ->SetToolTipText("Add color palette beside the histogram");
   f10->AddFrame(fAddPalette1, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));

   TGCompositeFrame *f11 = new TGCompositeFrame(f9, 40, 20, kVerticalFrame);
   fAddFB = new TGCheckButton(f11, "Front", kFRONTBOX_ONOFF);
   fAddFB ->SetToolTipText("Supress the drawing of the front box");
   f11->AddFrame(fAddFB, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   fAddBB = new TGCheckButton(f11, "Back", kBACKBOX_ONOFF);
   fAddBB ->SetToolTipText("Supress the drawing of the back box");
   f11->AddFrame(fAddBB, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   
   f9->AddFrame(f10, new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f9->AddFrame(f11, new TGLayoutHints(kLHintsLeft, 5, 1, 0, 0));   
   AddFrame(f9, new TGLayoutHints(kLHintsTop, 3, 1, 0, 2));

   f19 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fColContLbl1 = new TGLabel(f19, "Cont #:");  
   f19->AddFrame(fColContLbl1, new TGLayoutHints( kLHintsLeft, 29, 1, 4, 1));                            
   fContLevels1 = new TGNumberEntry(f19, 20, 0, kCONT_LEVELS1, 
                                      TGNumberFormat::kNESInteger,
                                      TGNumberFormat::kNEANonNegative, 
                                      TGNumberFormat::kNELLimitMinMax, 1, 99);
   fContLevels1->GetNumberEntry()->SetToolTipText("Set number of contours (1..99)");
   fContLevels1->Resize(55,20);
   f19->AddFrame(fContLevels1, new TGLayoutHints(kLHintsLeft, 12, 1, 2, 1));
   AddFrame(f19, new TGLayoutHints(kLHintsTop, 1, 1, 1, 0));

   f12 = new TGCompositeFrame(this, 145, 10, kHorizontalFrame | kLHintsExpandX | kFixedWidth | kOwnBackground);
   f12->AddFrame(new TGLabel(f12,"Bar"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f12->AddFrame(new TGHorizontal3DLine(f12), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   AddFrame(f12, new TGLayoutHints(kLHintsTop));
        
   f13 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGLabel *fWidthLbl = new TGLabel(f13, "W:");                              
   f13->AddFrame(fWidthLbl, new TGLayoutHints( kLHintsLeft, 1, 3, 4, 1));
   fBarWidth = new TGNumberEntry(f13, 1.00, 6, kBAR_WIDTH, 
                                      TGNumberFormat::kNESRealTwo,
                                      TGNumberFormat::kNEANonNegative, 
                                      TGNumberFormat::kNELLimitMinMax, 0.01, 1.);
   fBarWidth->GetNumberEntry()->SetToolTipText("Set bar chart width");
   fBarWidth->Resize(45,20);
   f13->AddFrame(fBarWidth, new TGLayoutHints(kLHintsLeft, 1, 1, 2, 1));

   TGLabel *fOffsetLbl = new TGLabel(f13, "O:");                              
   f13->AddFrame(fOffsetLbl, new TGLayoutHints(kLHintsLeft, 6,3, 4, 1));
   fBarOffset = new TGNumberEntry(f13, 0.00, 5, kBAR_OFFSET, 
                                      TGNumberFormat::kNESRealTwo,
                                      TGNumberFormat::kNEAAnyNumber, 
                                      TGNumberFormat::kNELLimitMinMax, -1., 1.);
   fBarOffset->GetNumberEntry()->SetToolTipText("Set bar chart offset");
   fBarOffset->Resize(50,20);
   f13->AddFrame(fBarOffset, new TGLayoutHints(kLHintsLeft, 1, 1, 2, 1));
   AddFrame(f13, new TGLayoutHints(kLHintsTop, 1, 1, 0, 3));


   MakeTitle("Axis Range");
   
   TGCompositeFrame *f14 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGLabel *fSliderXLbl = new TGLabel(f14,"x:");
   f14->AddFrame(fSliderXLbl, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 4,3, 4, 1)); 
   fSliderX = new TGDoubleHSlider(f14, 1, 2);
   f14->AddFrame(fSliderX, new TGLayoutHints(kLHintsExpandX));
   AddFrame(f14, new TGLayoutHints(kLHintsExpandX, 3, 7, 3, 0));

   TGCompositeFrame *f17 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fSldXMin = new TGNumberEntryField(f17, kSLIDERX_MIN, 0.0,  
                                      TGNumberFormat::kNESRealTwo,
                                      TGNumberFormat::kNEAAnyNumber);
   ((TGTextEntry*)fSldXMin)->SetToolTipText("Set the minimum value of the x-axis");
   fSldXMin->Resize(58,20);
   f17->AddFrame(fSldXMin, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));
   fSldXMax = new TGNumberEntryField(f17, kSLIDERX_MAX, 0.0,  
                                      TGNumberFormat::kNESRealTwo,
                                      TGNumberFormat::kNEAAnyNumber);
   ((TGTextEntry*)fSldXMax)->SetToolTipText("Set the maximum value of the x-axis");
   fSldXMax->Resize(58,20);
   f17->AddFrame(fSldXMax, new TGLayoutHints(kLHintsLeft, 2, 0, 0, 0));
   AddFrame(f17, new TGLayoutHints(kLHintsTop, 20, 3, 3, 3));

   TGCompositeFrame *f15 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGLabel *fSliderYLbl = new TGLabel(f15,"y:");
   f15->AddFrame(fSliderYLbl, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 4,3, 4, 1)); 
   fSliderY = new TGDoubleHSlider(f15, 1, 2);
   f15->AddFrame(fSliderY, new TGLayoutHints(kLHintsExpandX));
   AddFrame(f15, new TGLayoutHints(kLHintsExpandX, 3, 7, 3, 0));

   TGCompositeFrame *f18 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fSldYMin = new TGNumberEntryField(f18, kSLIDERY_MIN, 0.0,  
                                      TGNumberFormat::kNESRealTwo,
                                      TGNumberFormat::kNEAAnyNumber);
   ((TGTextEntry*)fSldYMin)->SetToolTipText("Set the minimum value of the y-axis");
   fSldYMin->Resize(58,20);
   f18->AddFrame(fSldYMin, new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));
   fSldYMax = new TGNumberEntryField(f18, kSLIDERY_MAX, 0.0,  
                                      TGNumberFormat::kNESRealTwo,
                                      TGNumberFormat::kNEAAnyNumber);
   ((TGTextEntry*)fSldYMax)->SetToolTipText("Set the maximum value of the y-axis");
   fSldYMax->Resize(58,20);
   f18->AddFrame(fSldYMax, new TGLayoutHints(kLHintsLeft, 2, 0, 0, 0));
   AddFrame(f18, new TGLayoutHints(kLHintsTop, 20, 3, 3, 0));

   TGCompositeFrame *f20 = new TGCompositeFrame(this, 80, 20, kVerticalFrame); 
   fDelaydraw = new TGCheckButton(f20, "Delayed drawing", kDELAYED_DRAWING);
   fDelaydraw ->SetToolTipText("Draw the new axis range when the Slider is released");
   f20->AddFrame(fDelaydraw, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   AddFrame(f20, new TGLayoutHints(kLHintsTop, 1, 1, 5, 3)); 

   MakeTitle("Frame Fill");
   
   TGCompositeFrame *f21 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   fFrameColor = new TGColorSelect(f21, 0, kCOLOR);
   f21->AddFrame(fFrameColor, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 0));
   fFrameColor->Associate(this);
   fFramePattern = new TGedPatternSelect(f21, 1, kPATTERN);
   f21->AddFrame(fFramePattern, new TGLayoutHints(kLHintsLeft, 1, 1, 1, 0));
   fFramePattern->Associate(this);
   AddFrame(f21, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   
   MapSubwindows();
   Layout();
   MapWindow();
   
   TClass *cl = TH2::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________

TH2Editor::~TH2Editor()
{
   // Destructor of TH2 editor.

   TGFrameElement *el;
   TIter next(GetList());
   
   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();
}

//______________________________________________________________________________

void TH2Editor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   fTitle->Connect("TextChanged(const char *)", "TH2Editor", this, "DoTitle(const char *)");
   fDim->Connect("Pressed()","TH2Editor",this,"DoHistSimple()");
   fDim0->Connect("Pressed()","TH2Editor",this,"DoHistComplex()");   
   fTypeCombo->Connect("Selected(Int_t)", "TH2Editor", this, "DoHistChanges()");   
   fCoordsCombo->Connect("Selected(Int_t)", "TH2Editor", this, "DoHistChanges()");
   fContCombo->Connect("Selected(Int_t)", "TH2Editor", this, "DoHistChanges()");   
   fAddArr->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddArr(Bool_t)");
   fAddBox->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddBox(Bool_t)");
   fAddCol->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddCol(Bool_t)");
   fAddScat->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddScat(Bool_t)");
   fAddText->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddText(Bool_t)");
   fAddError->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddError(Bool_t)");
   fAddPalette->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddPalette(Bool_t)");
   fAddPalette1->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddPalette(Bool_t)");   
   fAddFB->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddFB()");
   fAddBB->Connect("Toggled(Bool_t)", "TH2Editor", this, "DoAddBB()");
   fContLevels->Connect("ValueSet(Long_t)", "TH2Editor", this, "DoContLevel()");
   (fContLevels->GetNumberEntry())->Connect("ReturnPressed()", "TH2Editor", this,"DoContLevel()");   
   fContLevels1->Connect("ValueSet(Long_t)", "TH2Editor", this, "DoContLevel1()");
   (fContLevels1->GetNumberEntry())->Connect("ReturnPressed()", "TH2Editor", this,"DoContLevel1()");   
   fBarWidth->Connect("ValueSet(Long_t)", "TH2Editor", this, "DoBarWidth()");
   (fBarWidth->GetNumberEntry())->Connect("ReturnPressed()", "TH2Editor", this, "DoBarWidth()");   
   fBarOffset->Connect("ValueSet(Long_t)", "TH2Editor", this, "DoBarOffset()");
   (fBarOffset->GetNumberEntry())->Connect("ReturnPressed()", "TH2Editor", this, "DoBarOffset()");
   fSliderX->Connect("PositionChanged()","TH2Editor",this, "DoSliderXMoved()");  
   fSliderX->Connect("Pressed()","TH2Editor",this, "DoSliderXPressed()"); 
   fSliderX->Connect("Released()","TH2Editor",this, "DoSliderXReleased()");     
   fSldXMin->Connect("ReturnPressed()", "TH2Editor", this, "DoXAxisRange()");
   fSldXMax->Connect("ReturnPressed()", "TH2Editor", this, "DoXAxisRange()");   
   fSliderY->Connect("PositionChanged()","TH2Editor",this, "DoSliderYMoved()");  
   fSliderY->Connect("Pressed()","TH2Editor",this, "DoSliderYPressed()"); 
   fSliderY->Connect("Released()","TH2Editor",this, "DoSliderYReleased()");     
   fSldYMin->Connect("ReturnPressed()", "TH2Editor", this, "DoYAxisRange()");
   fSldYMax->Connect("ReturnPressed()", "TH2Editor", this, "DoYAxisRange()");   
   fFrameColor->Connect("ColorSelected(Pixel_t)", "TH2Editor", this, "DoFillColor(Pixel_t)");
   fFramePattern->Connect("PatternSelected(Style_t)", "TH2Editor", this, "DoFillPattern(Style_t)");

   fInit = kFALSE;
} 

//______________________________________________________________________________

void TH2Editor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Pick up the used values of histogram attributes.
   
   fModel = 0;
   fPad = 0;

   if (obj == 0 || !obj->InheritsFrom("TH2")) {
      SetActive(kFALSE);
      return;
   }

   fModel = obj;
   fPad = pad;
   fHist = (TH2*) fModel;
   const char *text = fHist->GetTitle();
   fTitle->SetText(text);
   if (!fInit) {
      DisconnectAllSlots();
      fInit=kTRUE;
   }
   TString str = GetDrawOption();
   str.ToUpper();
   
   if (str == "") {
      // default options = Scatter-Plot
      HideFrame(f3);
      HideFrame(f4);
      ShowFrame(f5);
      ShowFrame(f6);
      HideFrame(f9);
      HideFrame(f12);
      HideFrame(f13);          
      ShowFrame(f16);
      HideFrame(f19);
      
      fdimgroup->SetButton(kDIM_SIMPLE, kTRUE);  
      fTypeCombo->Select(kTYPE_LEGO);
      fCoordsCombo->Select(kCOORDS_CAR);
      fContCombo->Select(kCONT_NONE);
      
      fAddArr->SetState(kButtonUp);
      fAddBox->SetState(kButtonUp);
      fAddCol->SetState(kButtonUp);
      fAddScat->SetState(kButtonDisabled);      
      fAddText->SetState(kButtonUp);
      fAddError->SetState(kButtonUp);
      fAddPalette->SetState(kButtonDisabled);
      fAddPalette1->SetState(kButtonUp);      
      fAddFB->SetState(kButtonDown);      
      fAddBB->SetState(kButtonDown);      
   } else if (!str.Contains("LEGO") && !str.Contains("SURF")) {
      HideFrame(f3);
      HideFrame(f4);
      ShowFrame(f5);
      ShowFrame(f6);
      HideFrame(f9);
      HideFrame(f12);
      HideFrame(f13);
      ShowFrame(f16);
      HideFrame(f19);

      fdimgroup->SetButton(kDIM_SIMPLE, kTRUE);  
      fTypeCombo->Select(kTYPE_LEGO);
      fCoordsCombo->Select(kCOORDS_CAR);
      if (str.Contains("CONT")){
         if (str.Contains("CONT1")) fContCombo->Select(kCONT_1);
	 else if (str.Contains("CONT2")) fContCombo->Select(kCONT_2);
	 else if (str.Contains("CONT3")) fContCombo->Select(kCONT_3);
	 else if (str.Contains("CONT4")) fContCombo->Select(kCONT_4);
	 else if (str.Contains("CONT0") || str.Contains("CONT")) fContCombo->Select(kCONT_0);
      } else fContCombo->Select(kCONT_NONE);

      if (str.Contains("ARR")) fAddArr->SetState(kButtonDown);
      else fAddArr->SetState(kButtonUp);
      if (str.Contains("BOX")) fAddBox->SetState(kButtonDown);
      else if (str.Contains("COL")) fAddBox->SetState(kButtonDisabled);
      else fAddBox->SetState(kButtonUp);
      if (str.Contains("COL")) fAddCol->SetState(kButtonDown);
      else fAddCol->SetState(kButtonUp);
      if (str.Contains("SCAT")) {
         if (str=="SCAT") fAddScat->SetState(kButtonDisabled);
	 else fAddScat->SetState(kButtonDown);
      } else fAddScat->SetState(kButtonUp);            
      if (str.Contains("TEXT")) fAddText->SetState(kButtonDown);
      else fAddText->SetState(kButtonUp);
 
      fAddError->SetState(kButtonUp);
      if (str.Contains("COL") || (str.Contains("CONT") && !str.Contains("CONT2") && !str.Contains("CONT3"))) {
         if (str.Contains("Z")) fAddPalette->SetState(kButtonDown);
         else fAddPalette->SetState(kButtonUp);
      } else fAddPalette->SetState(kButtonDisabled);
      fAddPalette1->SetState(kButtonUp);
      fAddFB->SetState(kButtonDown);      
      fAddBB->SetState(kButtonDown);      

   } else if (str.Contains("LEGO") || str.Contains("SURF")) {
      ShowFrame(f3);
      ShowFrame(f4);
      HideFrame(f5);
      HideFrame(f6);
      ShowFrame(f9);
      ShowFrame(f12);
      ShowFrame(f13);
      HideFrame(f16);
      ShowFrame(f19);

      fdimgroup->SetButton(kDIM_COMPLEX, kTRUE);  
      if (str.Contains("LEGO2")) fTypeCombo->Select(kTYPE_LEGO2);
      else if (str.Contains("LEGO1")) fTypeCombo->Select(kTYPE_LEGO1);
      else if (str.Contains("LEGO")) fTypeCombo->Select(kTYPE_LEGO);
      else if (str.Contains("SURF5")) fTypeCombo->Select(kTYPE_SURF5);
      else if (str.Contains("SURF4")) fTypeCombo->Select(kTYPE_SURF4);
      else if (str.Contains("SURF3")) fTypeCombo->Select(kTYPE_SURF3);
      else if (str.Contains("SURF2")) fTypeCombo->Select(kTYPE_SURF2);
      else if (str.Contains("SURF1")) fTypeCombo->Select(kTYPE_SURF1);
      else if (str.Contains("SURF")) fTypeCombo->Select(kTYPE_SURF);
       
      if (str.Contains("CYL")) fCoordsCombo->Select(kCOORDS_CYL);
      else if (str.Contains("POL")) fCoordsCombo->Select(kCOORDS_POL);
      else if (str.Contains("SPH")) fCoordsCombo->Select(kCOORDS_SPH);
      else if (str.Contains("PSR")) fCoordsCombo->Select(kCOORDS_PSR);
      else fCoordsCombo->Select(kCOORDS_CAR); //default
      
      fContCombo->Select(kCONT_NONE);
      fAddArr->SetState(kButtonUp);
      fAddBox->SetState(kButtonUp);
      fAddCol->SetState(kButtonUp);
      fAddScat->SetState(kButtonDisabled);      
      fAddText->SetState(kButtonUp);   
      
      if (fCoordsCombo->GetSelected()!=kCOORDS_CAR) {
         if (fAddFB->GetState()!=kButtonDisabled) fAddFB->SetState(kButtonDisabled);
	 if (fAddBB->GetState()!=kButtonDisabled) fAddBB->SetState(kButtonDisabled);
	 if (fAddError->GetState()!=kButtonDisabled) fAddError->SetState(kButtonDisabled);
      } else {
         if (str.Contains("FB")) fAddFB->SetState(kButtonUp);      
         else fAddFB->SetState(kButtonDown);      
         if (str.Contains("BB")) fAddBB->SetState(kButtonUp);
         else fAddBB->SetState(kButtonDown);
         if (str.Contains("E")){
            TString dum = str;  
            if (str.Contains("LEGO")) dum.Remove(strstr(dum.Data(),"LEGO")-dum.Data(),4);
            if (str.Contains("TEXT")) dum.Remove(strstr(dum.Data(),"TEXT")-dum.Data(),4);
            if (dum.Contains("E")) fAddError->SetState(kButtonDown);
            else fAddError->SetState(kButtonUp);
         } else fAddError->SetState(kButtonUp);
      }
      if ((fTypeCombo->GetSelected()==kTYPE_LEGO) || (fTypeCombo->GetSelected()==kTYPE_LEGO1) || (fTypeCombo->GetSelected()==kTYPE_SURF)|| (fTypeCombo->GetSelected()==kTYPE_SURF4)) fAddPalette1->SetState(kButtonDisabled);
      else if (str.Contains("Z")) fAddPalette1->SetState(kButtonDown);
      else fAddPalette1->SetState(kButtonUp);
   }
   
   fBarWidth->SetNumber(fHist->GetBarWidth());
   fBarOffset->SetNumber(fHist->GetBarOffset());
   
   Int_t nx = fHist -> GetXaxis() -> GetNbins();
   Int_t nxbinmin = fHist -> GetXaxis() -> GetFirst();
   Int_t nxbinmax = fHist -> GetXaxis() -> GetLast();
   fSliderX->SetRange(1,nx);
   fSliderX->SetPosition((Double_t)nxbinmin,(Double_t)nxbinmax);
   fSldXMin->SetNumber(fHist->GetXaxis()->GetBinLowEdge(nxbinmin));
   fSldXMax->SetNumber(fHist->GetXaxis()->GetBinUpEdge(nxbinmax));
    

   Int_t ny = fHist -> GetYaxis() -> GetNbins();
   Int_t nybinmin = fHist -> GetYaxis() -> GetFirst();
   Int_t nybinmax = fHist -> GetYaxis() -> GetLast();
   fSliderY->SetRange(1,ny);
   fSliderY->SetPosition((Double_t)nybinmin,(Double_t)nybinmax);
   fSldYMin->SetNumber(fHist->GetYaxis()->GetBinLowEdge(nybinmin));
   fSldYMax->SetNumber(fHist->GetYaxis()->GetBinUpEdge(nybinmax));
   
   if (fDelaydraw->GetState()!=kButtonDown) fDelaydraw->SetState(kButtonUp);
   
   if (str.Contains("COL") || fContCombo->GetSelected()!= kCONT_NONE) fColContLbl->Enable() ;
   else fColContLbl->Disable();
   
   if (str.Contains("LEGO2") || str.Contains("SURF1") || str.Contains("SURF2") || str.Contains("SURF3") || str.Contains("SURF5")) fColContLbl1->Enable() ;
   else fColContLbl1->Disable();

   if (!gPad) return;
   gPad->cd();
   fFrameColor->SetColor(TColor::Number2Pixel(gPad->GetFrameFillColor()));
   fFramePattern->SetPattern(gPad->GetFrameFillStyle());
   
   SetActive();        
   Update();
   if (fInit) ConnectSignals2Slots();
   SetActive();
}
  
//______________________________________________________________________________

void TH2Editor::DoTitle(const char *text)
{
   // Slot connected to the title of the histogram .
  
   fHist->SetTitle(text);
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoHistSimple()
{
   // Slot connected to the 2D-Plot RadioButton
   
   TString str = "";
   HideFrame(f3);
   HideFrame(f4);
   ShowFrame(f5);
   ShowFrame(f6);
   HideFrame(f9);
   HideFrame(f12);
   HideFrame(f13);   
   ShowFrame(f16);
   HideFrame(f19);
   if (fContCombo->GetSelected()==-1) fContCombo->Select(kCONT_NONE);
   if ((fContCombo->GetSelected()!=kCONT_NONE) && fAddPalette->GetState()==kButtonDisabled) fAddPalette->SetState(kButtonUp);
   str = GetHistContLabel()+GetHistAdditiveLabel();
   if (str=="" || str=="SCAT") {
      fAddScat->SetState(kButtonDisabled); 
      fAddPalette->SetState(kButtonDisabled);
   } else if (fAddScat->GetState()==kButtonDisabled) fAddScat->SetState(kButtonUp);
   if (str.Contains("COL") || fContCombo->GetSelected()!= kCONT_NONE) fColContLbl->Enable() ;
   else fColContLbl->Disable() ; 
   SetDrawOption(str);
   SetActive();
   Update();
}
   
//______________________________________________________________________________

void TH2Editor::DoHistComplex()
{
   // Slot connected to the 3D-Plot RadioButton
   
   TString str = "";
   ShowFrame(f3);
   ShowFrame(f4);
   HideFrame(f5);   
   HideFrame(f6);   
   ShowFrame(f9);
   HideFrame(f16);   
   ShowFrame(f19);

   if (GetHistTypeLabel().Contains("LEGO")) {
      ShowFrame(f12);   
      ShowFrame(f13);   
   } else {
      HideFrame(f12);
      HideFrame(f13);
   }
   if (fTypeCombo->GetSelected()==-1) fTypeCombo->Select(kTYPE_LEGO);
   if (fCoordsCombo->GetSelected()==-1) fCoordsCombo->Select(kCOORDS_CAR);
 
   str = GetHistTypeLabel()+GetHistCoordsLabel()+GetHistAdditiveLabel(); 

   if (str.Contains("LEGO2") || str.Contains("SURF1") || str.Contains("SURF2") || str.Contains("SURF3") || str.Contains("SURF5")) fColContLbl1->Enable() ;
   else fColContLbl1->Disable() ;
   
   SetDrawOption(str);
   SetActive();
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoHistChanges()
{
   // Slot connected to the histogram type, the coordinate system and the Contour ComboBox
   
   TString str = "";
   if (fDim->GetState() == kButtonDown) {
      str = GetHistContLabel()+GetHistAdditiveLabel();
      if ((fContCombo->GetSelected()!=kCONT_NONE && fContCombo->GetSelected()!=kCONT_2 && fContCombo->GetSelected()!=kCONT_3) ||
      str.Contains("COL")) {
         if (str.Contains("Z")) fAddPalette->SetState(kButtonDown);
	 else fAddPalette->SetState(kButtonUp);
      } else fAddPalette->SetState(kButtonDisabled);
      if (str=="" || str=="SCAT") {
         fAddScat->SetState(kButtonDisabled);
	 fAddPalette->SetState(kButtonDisabled);
      } else if (fAddScat->GetState()==kButtonDisabled) fAddScat->SetState(kButtonUp);
      str = GetHistContLabel()+GetHistAdditiveLabel();
      if (str.Contains("COL") || fContCombo->GetSelected()!= kCONT_NONE) fColContLbl->Enable();
      else fColContLbl->Disable();
   } else if (fDim0->GetState() == kButtonDown) {
      if (fCoordsCombo->GetSelected()!=kCOORDS_CAR) {
         if (fAddFB->GetState()!=kButtonDisabled) fAddFB->SetState(kButtonDisabled);
	 if (fAddBB->GetState()!=kButtonDisabled) fAddBB->SetState(kButtonDisabled);
	 if (fAddError->GetState()!=kButtonDisabled) fAddError->SetState(kButtonDisabled);
      } else {
         if (fAddFB->GetState()==kButtonDisabled) fAddFB->SetState(kButtonDown);
	 if (fAddBB->GetState()==kButtonDisabled) fAddBB->SetState(kButtonDown);
	 if (fAddError->GetState()==kButtonDisabled) fAddError->SetState(kButtonDown);
      }
      if ((fTypeCombo->GetSelected()==kTYPE_LEGO) || (fTypeCombo->GetSelected()==kTYPE_LEGO1) ||
           (fTypeCombo->GetSelected()==kTYPE_SURF)|| (fTypeCombo->GetSelected()==kTYPE_SURF4)) fAddPalette1->SetState(kButtonDisabled);
      else if (fAddPalette1->GetState()==kButtonDisabled) fAddPalette1->SetState(kButtonUp);
      if (GetHistTypeLabel().Contains("LEGO")) {
         ShowFrame(f12);
	 ShowFrame(f13);
      } else {
         HideFrame(f12);
	 HideFrame(f13);
      }
      SetActive();
      str = GetHistTypeLabel()+GetHistCoordsLabel()+GetHistAdditiveLabel();
      if (str.Contains("LEGO2") || str.Contains("SURF1") || str.Contains("SURF2") || str.Contains("SURF3") || str.Contains("SURF5")) fColContLbl1->Enable() ;
      else fColContLbl1->Disable() ;
   }
   SetDrawOption(str);
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoAddArr(Bool_t on)
{
   // Slot connected to the "Arrow Draw Option"-CheckButton

   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   if (on) {
      if (!str.Contains("ARR")) {
         str += "ARR";
	 if (fAddScat->GetState()==kButtonDisabled) fAddScat->SetState(kButtonUp);	 
/*	 if ((!str.Contains("COL")) && (fContCombo->GetSelected()==kCONT_NONE) && !str.Contains("BOX")) {
	    if  ((str.Contains("SCAT")) && (fAddScat->GetState()!=kButtonDown)) fAddScat->SetState(kButtonDown);
	    else if (fAddScat->GetState()!=kButtonUp)fAddScat->SetState(kButtonUp);
	 }*/
	 make=kTRUE;
      }
   } else if (fAddArr->GetState()==kButtonUp) {
      if (str.Contains("ARR")) {
         str.Remove(strstr(str.Data(),"ARR")-str.Data(),3);
         if (str=="" || str=="SCAT") {
	    fAddScat->SetState(kButtonDisabled);
	    fAddPalette->SetState(kButtonDisabled);
	 }
	 make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();      
}

//______________________________________________________________________________

void TH2Editor::DoAddBox(Bool_t on)
{
   // Slot connected to the "Box Draw Option"-CheckButton

   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   if (on) {
      if (!str.Contains("BOX")) {
         str += "BOX";
	 if (fAddScat->GetState()==kButtonDisabled) fAddScat->SetState(kButtonUp);
//	 if (fAddScat->GetState()!=kButtonDisabled) fAddScat->SetState(kButtonDisabled);
	 make=kTRUE;
      }
   } else if (fAddBox->GetState()==kButtonUp) {
      if (str.Contains("BOX")) {
         str.Remove(strstr(str.Data(),"BOX")-str.Data(),3);
	 if (str=="" || str=="SCAT") {
	    fAddScat->SetState(kButtonDisabled);
	    fAddPalette->SetState(kButtonDisabled);
	 }
	 make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();   
}

//______________________________________________________________________________

void TH2Editor::DoAddCol(Bool_t on)
{
   // Slot connected to the "Col Draw Option"-CheckButton

   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   if (on) {
      if (!str.Contains("COL")) {
         str += "COL";
/*	 if (fAddBox->GetState()!=kButtonDisabled) {
	    fAddBox->SetState(kButtonDisabled);
	    if (str.Contains("BOX")) str.Remove(strstr(str.Data(),"BOX")-str.Data(),3);
	 }*/
	 fColContLbl->Enable() ;
	 if (fAddScat->GetState()==kButtonDisabled) fAddScat->SetState(kButtonUp);
//	 if (fAddScat->GetState()!=kButtonDisabled) fAddScat->SetState(kButtonDisabled);
	 if (fAddPalette->GetState()==kButtonDisabled) fAddPalette->SetState(kButtonUp);
	 make=kTRUE;
      }
   } else if (fAddCol->GetState()==kButtonUp) {
      if (str.Contains("COL")) {
         str.Remove(strstr(str.Data(),"COL")-str.Data(),3);
	 if (fAddBox->GetState()==kButtonDisabled) fAddBox->SetState(kButtonUp);
	 if (fContCombo->GetSelected()==kCONT_NONE) {
	    fAddPalette->SetState(kButtonDisabled);
	    if (str.Contains("Z")) str.Remove(strstr(str.Data(),"Z")-str.Data(),1);
	 }
         if (str=="" || str=="SCAT" /*|| str.Contains("TEXT")*/) fAddScat->SetState(kButtonDisabled);
         if (fContCombo->GetSelected()!= kCONT_NONE) fColContLbl->Enable() ;
         else fColContLbl->Disable();
	 make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();   
}

//______________________________________________________________________________

void TH2Editor::DoAddScat(Bool_t on)
{
   // Slot connected to the "Scat Draw Option"-CheckButton

   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   if (on) {
      if (!str.Contains("SCAT")) {
         str += "SCAT";
	 make=kTRUE;
      }
   } else if (fAddScat->GetState()==kButtonUp) {
      if (str.Contains("SCAT")) {
         str.Remove(strstr(str.Data(),"SCAT")-str.Data(),4);
	 make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoAddText(Bool_t on)
{
   // Slot connected to the "Text Draw Option"-CheckButton

   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   if (on) {
      if (!str.Contains("TEXT")) {
         str += "TEXT";
	 if (fAddScat->GetState()==kButtonDisabled) fAddScat->SetState(kButtonUp);
//	 if (fAddScat->GetState()!=kButtonDisabled) fAddScat->SetState(kButtonDisabled);
	 make=kTRUE;
      }
   } else if (fAddText->GetState()==kButtonUp) {
      if (str.Contains("TEXT")) {
         str.Remove(strstr(str.Data(),"TEXT")-str.Data(),4);
         if (str=="" || str=="SCAT" /*|| str.Contains("COL")*/) fAddScat->SetState(kButtonDisabled);
//	 else if ((fContCombo->GetSelected()==kCONT_NONE) && (fAddScat->GetState()==kButtonDisabled)) fAddScat->SetState(kButtonUp);
 	 make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoAddError(Bool_t on)
{
   // Slot connected to the "Error"-CheckButton

   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   TString dum = str;
   if (str.Contains("LEGO")) dum.Remove(strstr(dum.Data(),"LEGO")-dum.Data(),4);
   if (str.Contains("TEXT")) dum.Remove(strstr(dum.Data(),"TEXT")-dum.Data(),4);
   if (on) {
      if (!dum.Contains("E")) {
         str += "E";
         make=kTRUE;
      }
   } else if (fAddError->GetState()==kButtonUp) {
      if (str.Contains("E")) {
         str= GetHistTypeLabel()+GetHistCoordsLabel()+GetHistContLabel()+GetHistAdditiveLabel(); 
         make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoAddPalette(Bool_t on)
{
   // Slot connected to the Color Palette 

   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   if (on) {
      if (!str.Contains("Z")) {
         str += "Z";
	 make=kTRUE;
      }
   } else if (fAddPalette->GetState()==kButtonUp || fAddPalette1->GetState()==kButtonUp) {
      if (str.Contains("Z")) {
         str.Remove(strstr(str.Data(),"Z")-str.Data(),1);
	 make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoAddFB()
{
   // Slot connected to the "FB Front-Box Draw Option"-CheckButton
   
   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   if (fAddFB->GetState()==kButtonDown) {
      if (str.Contains("FB")) {
         if (str.Contains("SURF") && !(str.Contains("1") || str.Contains("2") ||  str.Contains("3") || str.Contains("4") ||
	 str.Contains("5"))) {
	    TString dum = str;
	    dum.Remove(strstr(dum.Data(),"SURF")-dum.Data(),4); 
	    if (dum.Contains("FB")) dum.Remove(strstr(dum.Data(),"FB")-dum.Data(),2); 
	    str = "SURF" + dum;
	 } else str.Remove(strstr(str.Data(),"FB")-str.Data(),2);
	 make = kTRUE;
      }
   } else if (fAddFB->GetState()==kButtonUp){
      if (!str.Contains("FB")) {
         str += "FB";
	 make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoAddBB()
{
   // Slot connected to the "BB Back-Box Draw Option"-CheckButton
   
   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   if (fAddBB->GetState()==kButtonDown) {
      if (str.Contains("BB")) {
         if (str.Contains("FB")) {
	    TString dum = str;
	    dum.Remove(strstr(dum.Data(),"FB")-dum.Data(),2);
	    dum.Remove(strstr(dum.Data(),"BB")-dum.Data(),2);
	    str=dum+"FB";
	 } else str.Remove(strstr(str.Data(),"BB")-str.Data(),2);
	 make = kTRUE;
      }
   } else if (fAddBB->GetState()==kButtonUp){
      if (!str.Contains("BB")) {
         str += "BB";
	 make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoContLevel()
{
   // Slot connected to the Contour Level TGNumberEntry 
   
   fHist->SetContour((Int_t)fContLevels->GetNumber());
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoContLevel1()
{
   // Slot connected to the Contour Level TGNumberEntry 
   
   fHist->SetContour((Int_t)fContLevels1->GetNumber());
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoBarWidth()
{
   // Slot connected to the Bar Width of the Bar Chart
   
   fHist->SetBarWidth(fBarWidth->GetNumber());
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoBarOffset()
{
   // Slot connected to the Bar Offset of the Bar Chart
   
   fHist->SetBarOffset((Float_t)fBarOffset->GetNumber());
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoSliderXMoved()
{
   // Slot connected to the x-Slider
   // Redraws the Histogram with the new Slider Range
   
   if (fDelaydraw->GetState()==kButtonDown && fDim->GetState()==kButtonDown) {
      static Int_t px1,py1,px2,py2;
      static Float_t ymin,ymax,xleft,xright;
      xleft = fHist->GetXaxis()->GetBinLowEdge((Int_t)((fSliderX->GetMinPosition())+0.5));
      xright =  fHist->GetXaxis()->GetBinUpEdge((Int_t)((fSliderX->GetMaxPosition())+0.5));
      ymin  = gPad->GetUymin();
      ymax  = gPad->GetUymax();
      px1   = gPad->XtoAbsPixel(xleft);
      py1   = gPad->YtoAbsPixel(ymin);
      px2   = gPad->XtoAbsPixel(xright);
      py2   = gPad->YtoAbsPixel(ymax);
      gPad->GetCanvas()->FeedbackMode(kTRUE);
      gPad->cd();
      gVirtualX->SetLineWidth(1);
      gVirtualX->SetLineColor(2);
      gVirtualX->DrawBox(fPx1old, fPy1old, fPx2old, fPy2old, TVirtualX::kHollow);
      gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      fPx1old = px1;
      fPy1old = py1;
      fPx2old = px2 ;
      fPy2old = py2;
      gVirtualX->Update(0);
      fSldXMin->SetNumber(xleft);
      fSldXMax->SetNumber(xright);
   }  else  if (fDelaydraw->GetState()==kButtonDown && fDim0->GetState()==kButtonDown && fCoordsCombo->GetSelected()==kCOORDS_CAR) {
      static Float_t p1[3], p2[3], p3[3], p4[3], p5[3], p6[3], p7[3], p8[3];
      gPad->GetCanvas()->FeedbackMode(kTRUE); 
      gPad->cd();
      TView *fView = gPad->GetView();
      Double_t *rmin = fView->GetRmin();
      Double_t *rmax = fView->GetRmax();
      p1[0] = p4[0] = p5[0] = p8[0] =fHist->GetXaxis()->GetBinLowEdge((Int_t)((fSliderX->GetMinPosition())+0.5));
      p2[0] = p3[0] = p6[0] = p7[0] = fHist->GetXaxis()->GetBinUpEdge((Int_t)((fSliderX->GetMaxPosition())+0.5));
      p1[1] = p2[1] = p3[1] = p4[1] = rmin[1];
      p5[1] = p6[1] = p7[1] = p8[1] = rmax[1];
      p1[2] = p2[2] = p5[2] = p6[2] = rmin[2];
      p3[2] = p4[2] = p7[2] = p8[2] = rmax[2];
      gPad->SetLineWidth(1);
      gPad->SetLineColor(2);
      PaintBox3D(fP2oldx, fP3oldx, fP7oldx, fP6oldx);
      PaintBox3D(fP1oldx, fP4oldx, fP8oldx, fP5oldx);
      PaintBox3D(p2, p3, p7, p6);
      PaintBox3D(p1, p4, p8, p5);
      for (Int_t i = 0; i<3; i++){
         fP1oldx[i] = p1[i];
         fP2oldx[i] = p2[i];      
         fP3oldx[i] = p3[i];      
         fP4oldx[i] = p4[i];                              
         fP5oldx[i] = p5[i];
         fP6oldx[i] = p6[i];      
         fP7oldx[i] = p7[i];      
         fP8oldx[i] = p8[i];                              
      }
      fSldXMin->SetNumber(p1[0]);
      fSldXMax->SetNumber(p2[0]);
   } else  if (fDelaydraw->GetState()==kButtonDown && fDim0->GetState()==kButtonDown) {
      fSldXMin->SetNumber(fHist->GetXaxis()->GetBinLowEdge((Int_t)((fSliderX->GetMinPosition())+0.5)));
      fSldXMax->SetNumber(fHist->GetXaxis()->GetBinUpEdge((Int_t)((fSliderX->GetMaxPosition())+0.5)));
   } else {
      fHist->GetXaxis()->SetRange((Int_t)((fSliderX->GetMinPosition())+0.5),(Int_t)((fSliderX->GetMaxPosition())+0.5));
      fSldXMin->SetNumber(fHist->GetXaxis()->GetBinLowEdge(fHist->GetXaxis()->GetFirst()));
      fSldXMax->SetNumber(fHist->GetXaxis()->GetBinUpEdge(fHist->GetXaxis()->GetLast())); 
      Update();   
   }

}

//______________________________________________________________________________

void TH2Editor::DoSliderXPressed()
{
   // ...
   
   static Float_t ymin,ymax,xleft,xright;
   if (fDelaydraw->GetState()==kButtonDown && fDim->GetState()==kButtonDown) {
      if (!gPad) return;
      gPad->cd();
      gPad->GetCanvas()->FeedbackMode(kFALSE);
      gVirtualX->SetLineWidth(1);
      gVirtualX->SetLineColor(2);
      xleft = fHist->GetXaxis()->GetBinLowEdge((Int_t)((fSliderX->GetMinPosition())+0.5));
      xright =  fHist->GetXaxis()->GetBinUpEdge((Int_t)((fSliderX->GetMaxPosition())+0.5));
      ymin  = gPad->GetUymin();
      ymax  = gPad->GetUymax();
      fPx1old   = gPad->XtoAbsPixel(xleft);
      fPy1old   = gPad->YtoAbsPixel(ymin);
      fPx2old   = gPad->XtoAbsPixel(xright);
      fPy2old   = gPad->YtoAbsPixel(ymax);
      gVirtualX->DrawBox(fPx1old, fPy1old, fPx2old, fPy2old, TVirtualX::kHollow);
   } else if (fDelaydraw->GetState()==kButtonDown && fDim0->GetState()==kButtonDown && fCoordsCombo->GetSelected()==kCOORDS_CAR) {
      if (!gPad) return;
      gPad->cd();
      TView *fView = gPad->GetView();
      Double_t *rmin = fView->GetRmin();
      Double_t *rmax = fView->GetRmax();
      fP1oldx[0] = fP4oldx[0] = fP5oldx[0] = fP8oldx[0] = fHist->GetXaxis()->GetBinLowEdge((Int_t)((fSliderX->GetMinPosition())+0.5));
      fP2oldx[0] = fP3oldx[0] = fP6oldx[0] = fP7oldx[0] = fHist->GetXaxis()->GetBinUpEdge((Int_t)((fSliderX->GetMaxPosition())+0.5));
      fP1oldx[1] = fP2oldx[1] = fP3oldx[1] = fP4oldx[1] = rmin[1];
      fP5oldx[1] = fP6oldx[1] = fP7oldx[1] = fP8oldx[1] = rmax[1];
      fP1oldx[2] = fP2oldx[2] = fP5oldx[2] = fP6oldx[2] = rmin[2]; 
      fP3oldx[2] = fP4oldx[2] = fP7oldx[2] = fP8oldx[2] = rmax[2];
      gPad->GetCanvas()->FeedbackMode(kTRUE); 
      gPad->SetLineWidth(1);
      gPad->SetLineColor(2);
      PaintBox3D(fP2oldx, fP3oldx, fP7oldx, fP6oldx);
      PaintBox3D(fP1oldx, fP4oldx, fP8oldx, fP5oldx);
   }
   Update();
}
   
//______________________________________________________________________________

void TH2Editor::DoSliderXReleased()
{

   if (fDelaydraw->GetState()==kButtonDown) {
      fHist->GetXaxis()->SetRange((Int_t)((fSliderX->GetMinPosition())+0.5),(Int_t)((fSliderX->GetMaxPosition())+0.5));
      fSldXMin->SetNumber(fHist->GetXaxis()->GetBinLowEdge(fHist->GetXaxis()->GetFirst()));
      fSldXMax->SetNumber(fHist->GetXaxis()->GetBinUpEdge(fHist->GetXaxis()->GetLast()));
      Update();
   }
}

//______________________________________________________________________________

void TH2Editor::DoXAxisRange()
{
   // Slot connected to the TextNumberEntryFields which contain the Max/Min value of the x-axis

   Int_t nx = fHist->GetXaxis()->GetNbins();
   Axis_t width = fHist->GetXaxis()->GetBinWidth(1);
   if ((fSldXMin->GetNumber()+width/2) < (fHist->GetXaxis()->GetBinLowEdge(1))) fSldXMin->SetNumber(fHist->GetXaxis()->GetBinLowEdge(1)); 
   if ((fSldXMax->GetNumber()-width/2) > (fHist->GetXaxis()->GetBinUpEdge(nx))) fSldXMax->SetNumber(fHist->GetXaxis()->GetBinUpEdge(nx)); 
   fHist->GetXaxis()->SetRangeUser(fSldXMin->GetNumber()+width/2, fSldXMax->GetNumber()-width/2);
   Int_t nxbinmin = fHist -> GetXaxis() -> GetFirst();
   Int_t nxbinmax = fHist -> GetXaxis() -> GetLast();
   fSliderX->SetPosition((Double_t)(nxbinmin),(Double_t)(nxbinmax));
   Update();
}

//______________________________________________________________________________


void TH2Editor::DoSliderYMoved()
{
   // Slot connected to the x-Slider
   // Redraws the Histogram with the new Slider Range
   
   if (fDelaydraw->GetState()==kButtonDown && fDim->GetState()==kButtonDown) {
      static Int_t px1,py1,px2,py2;
      static Float_t xmin,xmax,ybottom,ytop;
      ybottom = fHist->GetYaxis()->GetBinLowEdge((Int_t)((fSliderY->GetMinPosition())+0.5));
      ytop =  fHist->GetYaxis()->GetBinUpEdge((Int_t)((fSliderY->GetMaxPosition())+0.5));
      xmin  = gPad->GetUxmin();
      xmax  = gPad->GetUxmax();
      px1   = gPad->XtoAbsPixel(xmin);
      py1   = gPad->YtoAbsPixel(ybottom);
      px2   = gPad->XtoAbsPixel(xmax);
      py2   = gPad->YtoAbsPixel(ytop);
      gPad->GetCanvas()->FeedbackMode(kTRUE);
      gPad->cd();
      gVirtualX->SetLineWidth(1);
      gVirtualX->SetLineColor(2);
      gVirtualX->DrawBox(fPx1old, fPy1old, fPx2old, fPy2old, TVirtualX::kHollow);
      gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
      fPx1old = px1;
      fPy1old = py1;
      fPx2old = px2 ;
      fPy2old = py2; 
      gVirtualX->Update(0);
      fSldYMin->SetNumber(ybottom);
      fSldYMax->SetNumber(ytop);
   } else  if (fDelaydraw->GetState()==kButtonDown && fDim0->GetState()==kButtonDown && fCoordsCombo->GetSelected()==kCOORDS_CAR) {
      static Float_t p1[3], p2[3], p3[3], p4[3], p5[3], p6[3], p7[3], p8[3];
      gPad->GetCanvas()->FeedbackMode(kTRUE); 
      gPad->cd();
      TView *fView = gPad->GetView();
      Double_t *rmin = fView->GetRmin();
      Double_t *rmax = fView->GetRmax();
      p1[0] = p2[0] = p3[0] = p4[0] = rmin[0];
      p5[0] = p6[0] = p7[0] = p8[0] = rmax[0];
      p1[1] = p4[1] = p5[1] = p8[1] = fHist->GetYaxis()->GetBinLowEdge((Int_t)((fSliderY->GetMinPosition())+0.5));
      p2[1] = p3[1] = p6[1] = p7[1] = fHist->GetYaxis()->GetBinUpEdge((Int_t)((fSliderY->GetMaxPosition())+0.5));
      p1[2] = p2[2] = p5[2] = p6[2] = rmin[2];
      p3[2] = p4[2] = p7[2] = p8[2] = rmax[2];
      gPad->SetLineWidth(1);
      gPad->SetLineColor(2);
      PaintBox3D(fP2oldy, fP3oldy, fP7oldy, fP6oldy);
      PaintBox3D(fP1oldy, fP4oldy, fP8oldy, fP5oldy);
      PaintBox3D(p2, p3, p7, p6);
      PaintBox3D(p1, p4, p8, p5);
      for (Int_t i = 0; i<3; i++) { 
         fP1oldy[i] = p1[i];
         fP2oldy[i] = p2[i];      
         fP3oldy[i] = p3[i];      
         fP4oldy[i] = p4[i];     
         fP5oldy[i] = p5[i];
         fP6oldy[i] = p6[i];      
         fP7oldy[i] = p7[i];      
         fP8oldy[i] = p8[i];                                
      }   
      fSldYMin->SetNumber(p1[1]);
      fSldYMax->SetNumber(p2[1]);
   } else  if (fDelaydraw->GetState()==kButtonDown && fDim0->GetState()==kButtonDown) {
      fSldYMin->SetNumber(fHist->GetYaxis()->GetBinLowEdge((Int_t)((fSliderY->GetMinPosition())+0.5)));
      fSldYMax->SetNumber(fHist->GetYaxis()->GetBinUpEdge((Int_t)((fSliderY->GetMaxPosition())+0.5)));
   } else {
      fHist->GetYaxis()->SetRange((Int_t)((fSliderY->GetMinPosition())+0.5),(Int_t)((fSliderY->GetMaxPosition())+0.5));
      fSldYMin->SetNumber(fHist->GetYaxis()->GetBinLowEdge(fHist->GetYaxis()->GetFirst()));
      fSldYMax->SetNumber(fHist->GetYaxis()->GetBinUpEdge(fHist->GetYaxis()->GetLast())); 
      Update();   
   }

}

//______________________________________________________________________________

void TH2Editor::DoSliderYPressed()
{
   // ...
   
   static Float_t xmin,xmax,ytop,ybottom;
   if (fDelaydraw->GetState()==kButtonDown && fDim->GetState()==kButtonDown) {
      if (!gPad) return;
      gPad->cd();
      gPad->GetCanvas()->FeedbackMode(kFALSE);
      gVirtualX->SetLineWidth(1);
      gVirtualX->SetLineColor(2);
      ybottom = fHist->GetYaxis()->GetBinLowEdge((Int_t)((fSliderY->GetMinPosition())+0.5));
      ytop =  fHist->GetYaxis()->GetBinUpEdge((Int_t)((fSliderY->GetMaxPosition())+0.5));
      xmin  = gPad->GetUxmin();
      xmax  = gPad->GetUxmax();
      fPx1old   = gPad->XtoAbsPixel(xmin);
      fPy1old   = gPad->YtoAbsPixel(ybottom);
      fPx2old   = gPad->XtoAbsPixel(xmax);
      fPy2old   = gPad->YtoAbsPixel(ytop);
      gVirtualX->DrawBox(fPx1old, fPy1old, fPx2old, fPy2old, TVirtualX::kHollow);
   }  else if (fDelaydraw->GetState()==kButtonDown && fDim0->GetState()==kButtonDown && fCoordsCombo->GetSelected()==kCOORDS_CAR) {
      if (!gPad) return;
      gPad->cd();
      TView *fView = gPad->GetView();
      Double_t *rmin = fView->GetRmin();
      Double_t *rmax = fView->GetRmax();

      fP1oldy[0] = fP2oldy[0] = fP3oldy[0] = fP4oldy[0] = rmin[0];
      fP5oldy[0] = fP6oldy[0] = fP7oldy[0] = fP8oldy[0] = rmax[0];
      fP1oldy[1] = fP4oldy[1] = fP5oldy[1] = fP8oldy[1] = fHist->GetYaxis()->GetBinLowEdge((Int_t)((fSliderY->GetMinPosition())+0.5));
      fP2oldy[1] = fP3oldy[1] = fP6oldy[1] = fP7oldy[1] = fHist->GetYaxis()->GetBinUpEdge((Int_t)((fSliderY->GetMaxPosition())+0.5));
      fP1oldy[2] = fP2oldy[2] = fP5oldy[2] = fP6oldy[2] = rmin[2]; 
      fP3oldy[2] = fP4oldy[2] = fP7oldy[2] = fP8oldy[2] = rmax[2];
      gPad->GetCanvas()->FeedbackMode(kTRUE); 
      gPad->SetLineWidth(1);
      gPad->SetLineColor(2);
      PaintBox3D(fP2oldy, fP3oldy, fP7oldy, fP6oldy);
      PaintBox3D(fP1oldy, fP4oldy, fP8oldy, fP5oldy);
   }
   Update();
}
   
//______________________________________________________________________________

void TH2Editor::DoSliderYReleased()
{

   if (fDelaydraw->GetState()==kButtonDown) {
      fHist->GetYaxis()->SetRange((Int_t)((fSliderY->GetMinPosition())+0.5),(Int_t)((fSliderY->GetMaxPosition())+0.5));
      fSldYMin->SetNumber(fHist->GetYaxis()->GetBinLowEdge(fHist->GetYaxis()->GetFirst()));
      fSldYMax->SetNumber(fHist->GetYaxis()->GetBinUpEdge(fHist->GetYaxis()->GetLast()));
      Update();
   }
}

//______________________________________________________________________________

void TH2Editor::DoYAxisRange()
{
   // Slot connected to the TextNumberEntryFields which contain the Max/Min value of the y-axis

   Int_t ny = fHist->GetYaxis()->GetNbins();
   Axis_t width = fHist->GetYaxis()->GetBinWidth(1);
   if ((fSldYMin->GetNumber()+width/2) < (fHist->GetYaxis()->GetBinLowEdge(1))) fSldYMin->SetNumber(fHist->GetYaxis()->GetBinLowEdge(1)); 
   if ((fSldYMax->GetNumber()-width/2) > (fHist->GetYaxis()->GetBinUpEdge(ny))) fSldYMax->SetNumber(fHist->GetYaxis()->GetBinUpEdge(ny)); 
   fHist->GetYaxis()->SetRangeUser(fSldYMin->GetNumber()+width/2, fSldYMax->GetNumber()-width/2);
   Int_t nybinmin = fHist -> GetYaxis() -> GetFirst();
   Int_t nybinmax = fHist -> GetYaxis() -> GetLast();
   fSliderY->SetPosition((Double_t)(nybinmin),(Double_t)(nybinmax));
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoFillColor(Pixel_t color)
{
   // Slot connected to the fill area color.

   if (!gPad) return;
   gPad->cd();
   gPad->SetFrameFillColor(TColor::GetColor(color));
   Update();
}

//______________________________________________________________________________

void TH2Editor::DoFillPattern(Style_t pattern)
{
   // Slot connected to the fill area pattern.

   if (!gPad) return;
   gPad->cd();
   gPad->SetFrameFillStyle(pattern);
   Update();
}

//______________________________________________________________________________

TString TH2Editor::GetHistTypeLabel()
{
   // Returns the immediate histogram type (HIST, LEGO1-2, SURF1-5)

   TString s="";
   switch (fTypeCombo->GetSelected()){
      case (-1)         : {s = ""; break;}
      case (kTYPE_LEGO ): {s = "LEGO"; break;}
      case (kTYPE_LEGO1): {s = "LEGO1"; break;}
      case (kTYPE_LEGO2): {s = "LEGO2"; break;}
      case (kTYPE_SURF ): {s = "SURF"; break;}
      case (kTYPE_SURF1): {s = "SURF1"; break;}
      case (kTYPE_SURF2): {s = "SURF2"; break;}
      case (kTYPE_SURF3): {s = "SURF3"; break;}
      case (kTYPE_SURF4): {s = "SURF4"; break;}
      case (kTYPE_SURF5): {s = "SURF5"; break;}
      default:  break;
   }
   return s;
}

//______________________________________________________________________________

TString TH2Editor::GetHistCoordsLabel()
{
   // Returns the immediate coordinate system of the histogram (POL, CYL, SPH,PSR)

   TString s="";
   switch (fCoordsCombo->GetSelected()){
      case (-1)         : {s = ""; break;}
      case (kCOORDS_CAR): {s = ""; break;}
      case (kCOORDS_POL): {s = "POL"; break;}
      case (kCOORDS_CYL): {s = "CYL"; break;}
      case (kCOORDS_SPH): {s = "SPH"; break;}
      case (kCOORDS_PSR): {s = "PSR"; break;}
      default:  break;
   }
   return s;
}

//______________________________________________________________________________

TString TH2Editor::GetHistContLabel()
{
   // Returns histogram contour option (None,Cont0..5)

   TString s="";
   switch (fContCombo->GetSelected()){
      case (-1)         : {s = ""; break;}
      case (kCONT_NONE) : {s = ""; break;}
      case (kCONT_0)    : {s = "CONT0"; break;}
      case (kCONT_1)    : {s = "CONT1"; break;}
      case (kCONT_2)    : {s = "CONT2"; break;}
      case (kCONT_3)    : {s = "CONT3"; break;}
      case (kCONT_4)    : {s = "CONT4"; break;}
      default:  break;
   }
   return s;
}

//______________________________________________________________________________

TString TH2Editor::GetHistAdditiveLabel()
{
   // Returns histogram additive options (Arr,Box,Col,Scat,Col,Text,E,Z,FB,BB)

   TString s="";
   if (fDim->GetState()==kButtonDown) {
      if (fAddArr->GetState()==kButtonDown) s+="ARR";
      if (fAddBox->GetState()==kButtonDown) s+="BOX";
      if (fAddCol->GetState()==kButtonDown) s+="COL";  
      if (fAddScat->GetState()==kButtonDown) s+="SCAT";
      if (fAddText->GetState()==kButtonDown) s+="TEXT";
      if (fAddPalette->GetState()==kButtonDown) s+="Z";
   } else if (fDim0->GetState()==kButtonDown){
      if (fAddPalette1->GetState()==kButtonDown) s+="Z";
      if (fAddError->GetState()==kButtonDown) s+="E";
      if (fAddFB->GetState()==kButtonUp) s+="FB";      
      if (fAddBB->GetState()==kButtonUp) s+="BB";    
   }
   
   return s;
}

//______________________________________________________________________________

TGComboBox* TH2Editor::BuildHistTypeComboBox(TGFrame* parent, Int_t id)
{
   // Create histogram type combo box.

   TGComboBox *c = new TGComboBox(parent, id);
   
   c->AddEntry("Lego" , kTYPE_LEGO);
   c->AddEntry("Lego1", kTYPE_LEGO1);
   c->AddEntry("Lego2", kTYPE_LEGO2);
   c->AddEntry("Surf" , kTYPE_SURF);
   c->AddEntry("Surf1", kTYPE_SURF1);
   c->AddEntry("Surf2", kTYPE_SURF2);
   c->AddEntry("Surf3", kTYPE_SURF3);   
   c->AddEntry("Surf4", kTYPE_SURF4);   
   c->AddEntry("Surf5", kTYPE_SURF5); 

   return c;
}

//______________________________________________________________________________

TGComboBox* TH2Editor::BuildHistCoordsComboBox(TGFrame* parent, Int_t id)
{
   // Create histogram coordinate system type combo box.

   TGComboBox *c = new TGComboBox(parent, id);

   c->AddEntry("Cartesian", kCOORDS_CAR);
   c->AddEntry("Cylindric", kCOORDS_CYL);
   c->AddEntry("Polar", kCOORDS_POL);
   c->AddEntry("Rapidity", kCOORDS_PSR);   
   c->AddEntry("Spheric", kCOORDS_SPH);   
   TGListBox* lb = c->GetListBox();
   lb->Resize(lb->GetWidth(), 83);

   return c;
}

//______________________________________________________________________________

TGComboBox* TH2Editor::BuildHistContComboBox(TGFrame* parent, Int_t id)
{
   // Create contour combo box.

   TGComboBox *c = new TGComboBox(parent, id);
   
   c->AddEntry("None" , kCONT_NONE);
   c->AddEntry("Cont0", kCONT_0);
   c->AddEntry("Cont1", kCONT_1);
   c->AddEntry("Cont2", kCONT_2);
   c->AddEntry("Cont3", kCONT_3);
   c->AddEntry("Cont4", kCONT_4);
 
   return c;
}

//______________________________________________________________________________

void TH2Editor::DisconnectAllSlots()
{
   // Disconnects all Signals from the Slots

   Disconnect(fDim, "Pressed()", this, "DoHistSimple()");
   Disconnect(fDim0, "Pressed()", this, "DoHistComplex()");   
   Disconnect(fTitle,"TextChanged(const char *)", this, "DoTitle(const char *)");
   Disconnect(fTypeCombo, "Selected(Int_t)",this, "DoHistChanges()");   
   Disconnect(fCoordsCombo, "Selected(Int_t)", this, "DoHistChanges()");
   Disconnect(fContCombo, "Selected(Int_t)", this, "DoHistChanges()");   
   Disconnect(fAddArr, "Toggled(Bool_t)", this, "DoAddArr(Bool_t)");
   Disconnect(fAddBox, "Toggled(Bool_t)", this, "DoAddBox(Bool_t)");
   Disconnect(fAddCol, "Toggled(Bool_t)", this, "DoAddCol(Bool_t)");
   Disconnect(fAddScat, "Toggled(Bool_t)", this, "DoAddScat(Bool_t)");
   Disconnect(fAddText, "Toggled(Bool_t)", this, "DoAddText(Bool_t)");
   Disconnect(fAddError, "Toggled(Bool_t)", this, "DoAddError(Bool_t)");
   Disconnect(fAddPalette, "Toggled(Bool_t)", this, "DoAddPalette(Bool_t)");
   Disconnect(fAddPalette1, "Toggled(Bool_t)", this, "DoAddPalette(Bool_t)");   
   Disconnect(fAddFB, "Toggled(Bool_t)", this, "DoAddFB()");
   Disconnect(fAddBB, "Toggled(Bool_t)", this, "DoAddBB()");
   Disconnect(fContLevels, "ValueSet(Long_t)", this, "DoContLevel()");
   Disconnect((fContLevels->GetNumberEntry()), "ReturnPressed()", this,"DoContLevel()");   
   Disconnect(fContLevels1, "ValueSet(Long_t)", this, "DoContLevel1()");
   Disconnect((fContLevels1->GetNumberEntry()), "ReturnPressed()", this,"DoContLevel1()");   
   Disconnect(fBarWidth, "ValueSet(Long_t)", this, "DoBarWidth()");
   Disconnect((fBarWidth->GetNumberEntry()), "ReturnPressed()", this, "DoBarWidth()");   
   Disconnect(fBarOffset, "ValueSet(Long_t)", this, "DoBarOffset()");
   Disconnect((fBarOffset->GetNumberEntry()), "ReturnPressed()", this, "DoBarOffset()");
   Disconnect(fSliderX, "PositionChanged()", this, "DoSliderX()");  
   Disconnect(fSliderX, "Pressed()", this, "DoSliderXPressed()"); 
   Disconnect(fSliderX, "Released()", this, "DoSliderXReleased()");     
   Disconnect(fSliderY, "PositionChanged()", this, "DoSliderY()");  
   Disconnect(fSliderY, "Pressed()", this, "DoSliderYPressed()"); 
   Disconnect(fSliderY, "Released()", this, "DoSliderYReleased()");     
   Disconnect(fSldYMin, "ReturnPressed()", this, "DoYAxisRange()");
   Disconnect(fSldYMax, "ReturnPressed()", this, "DoYAxisRange()");   
   Disconnect(fFrameColor, "ColorSelected(Pixel_t)", this, "DoFillColor(Pixel_t)");
   Disconnect(fFramePattern, "PatternSelected(Style_t)", this, "DoFillPattern(Style_t)"); 
}

//______________________________________________________________________________

void TH2Editor::PaintBox3D(Float_t *p1, Float_t *p2,Float_t *p3, Float_t *p4) 
{
   // Paints a square in 3D

   gPad->PaintLine3D(p1, p2);
   gPad->PaintLine3D(p2, p3);
   gPad->PaintLine3D(p3, p4);
   gPad->PaintLine3D(p4, p1);
}

// @(#)root/ged:$Name:  TH1Editor.cxx
// Author: Carsten Hof   16/08/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TH1Editor                                                           //
//                                                                      //
//  Editor for histogram attributes.                                    //
//      changing title, histogram type, coordinate system, shown Errors //
//      and histogram shape                                             //
//	'Title': set the title of the histogram                         //
//  Histogram option:                                                   //
//      'Type': define the type of histogram                            //
//              "hist"   : When an histogram has errors it is visualized//
//                         by default with error bars: default          //
//              "Lego"   : Draw a lego plot with hidden line removal	//
//              "Lego1"  : Draw a lego plot with hidden surface removal //
//              "Lego2"  : Draw a lego plot using colors to show the   	//
//                         cell contents				//
//              "SURF"   : Draw a surface plot with hidden line removal	//
//              "SURF1"  : Draw a surface plot with hidden surface 	//
//                         removal					//
//              "SURF2"  : Draw a surface plot using colors to show the //
//                         cell contents				//
//              "SURF3"  : same as SURF with in addition a contour view	//
//                         drawn on the top				//
//              "SURF4"  : Draw a surface using Gouraud shading		//
//              "SURF5"  : Same as SURF3 but only the colored contour 	//
//                         is drawn. Used with option CYL, SPH or PSR 	//
//                         it allows to draw colored contours on a 	//
//                         sphere, a cylinder or a in pseudo rapidy 	//
//                         space. In cartesian or polar coordinates, 	//
//			   option SURF3 is used.			//
//	'Coords': define the coordinate system				//
//		"Cartesian": use Cartesian coordinates  		//
//		"Polar"    : Use Polar coordinates			//
//   		"Cylindric": Use Cylindrical coordinates		//
//    		"Spheric"  : Use Spherical coordinates			//
//    		"Rapidity" : Use PseudoRapidity/Phi coordinates		//
//	'Error': define error drawing					//
//		"No Error": no error bars are drawn			//
//		"Simple"  : Draw simple error bars			//
//    		"Edges"   : Draw error bars with perpendicular lines at //
//			    the edges					//
//    		"Rectangles": Draw error bars with rectangles		//
//   		"Fill"     : Draw a fill area througth the end points 	//
//			     of the vertical error bars			//
//    		"Contour"  : Draw a smoothed filled area through the 	//
//			     end points of the error bars		//
//	'Style'	Changing the draw style (line/bar/fill)			//
//		"Default"  : Default layout				//
//		"Simple Line": Draw a line througth the bin contents	//
//		"Smooth Line": Draw a smooth curve througth bin contents//
//		"Bar Chart"  : Bar chart option				//
//    		"Fill Area"  : Draw histogram like with option "L" but 	//
//			       with a fill area. Note that "L" draws 	//
//			       also a fill area if the hist fillcolor is// 
//			       set but the fill area corresponds to the //	
//			       histogram contour.Draw a smooth Curve 	//
//			       througth the histogram bins		//
//	 ?????????? some parts are missing !!!!!!!!                     //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/TH1Editor.gif">
*/
//End_Html


#include "TH1Editor.h"
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
#include <stdlib.h>
#include "TG3DLine.h"
#include "TGDoubleSlider.h"

#include "Riostream.h"
#include "TRegexp.h"


ClassImp(TH1Editor)

enum {
   kTH1_TITLE,
   kTYPE_HIST,
   kTYPE_LEGO,
   kTYPE_LEGO1,
   kTYPE_LEGO2,
   kTYPE_SURF,
   kTYPE_SURF1,
   kTYPE_SURF2,
   kTYPE_SURF3,
   kTYPE_SURF4,
   kTYPE_SURF5,
   kCOORDS_CAR,
   kCOORDS_CYL,
   kCOORDS_POL,
   kCOORDS_PSR,
   kCOORDS_SPH,
   kERRORS_NO,
   kERRORS_SIMPLE,
   kERRORS_EDGES,
   kERRORS_REC,  
   kERRORS_FILL,
   kERRORS_CONTOUR,
   kHIST_TYPE,
   kCOORD_TYPE,
   kERROR_TYPE,
   kMARKER_ONOFF,
   kHIST_ONOFF,
   kB_ONOFF,
   kBAR_ONOFF,
   kADD_TYPE,
   kADD_NONE,
   kADD_SIMPLE,
   kADD_SMOOTH,
   kADD_FILL,
   kADD_BAR,
   kADD_LINE,
   kDIM_SIMPLE,
   kDIM_COMPLEX,
   kPERCENT_TYPE,
   kPER_0,
   kPER_10,
   kPER_20,
   kPER_30,         
   kPER_40,
   kBAR_H,
   kBAR_WIDTH,
   kBAR_OFFSET   
};

//______________________________________________________________________________

TH1Editor::TH1Editor(const TGWindow *p, Int_t id, Int_t width,
                         Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of histogram attribute GUI.
   
   fHist = 0;
   
   MakeTitle("Title");
  
   fTitlePrec = 2;
   fTitle = new TGTextEntry(this, new TGTextBuffer(50), kTH1_TITLE);
   fTitle->Resize(135, fTitle->GetDefaultHeight());
   fTitle->SetToolTipText("Enter the histogram title string");
   AddFrame(fTitle, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));
  
   TGCompositeFrame *fHistLbl = new TGCompositeFrame(this, 145, 10, kHorizontalFrame | kLHintsExpandX | kFixedWidth | kOwnBackground);
   fHistLbl->AddFrame(new TGLabel(fHistLbl,"Histogram"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   fHistLbl->AddFrame(new TGHorizontal3DLine(fHistLbl), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 0));
   AddFrame(fHistLbl, new TGLayoutHints(kLHintsTop));
   
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
   AddFrame(f4, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   
   TGCompositeFrame *f5 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGLabel *fError = new TGLabel(f5, "Error:"); 
   f5->AddFrame(fError, new TGLayoutHints(kLHintsLeft, 6, 1, 4, 1));
   fErrorCombo = BuildHistErrorComboBox(f5, kERROR_TYPE);
   f5->AddFrame(fErrorCombo, new TGLayoutHints(kLHintsLeft, 16, 1, 2, 1));
   fErrorCombo->Resize(86, 20);
   fErrorCombo->Associate(this);
   AddFrame(f5, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));
   
   f6 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGLabel *fAddLabel = new TGLabel(f6, "Style:"); 
   f6->AddFrame(fAddLabel, new TGLayoutHints(kLHintsLeft, 6, 1, 4, 1));
   fAddCombo = BuildHistAddComboBox(f6, kADD_TYPE);
   f6->AddFrame(fAddCombo, new TGLayoutHints(kLHintsLeft, 15, 1, 2, 1));
   fAddCombo->Resize(86, 20);
   fAddCombo->Associate(this);
   AddFrame(f6, new TGLayoutHints(kLHintsTop, 1, 1, 0, 3));
   
   f7 = new TGCompositeFrame(this, 80, 20, kVerticalFrame);
   fAddMarker = new TGCheckButton(f7, "Show markers", kMARKER_ONOFF);
   fAddMarker ->SetToolTipText("Make marker visible/invisible");
   f7->AddFrame(fAddMarker, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   AddFrame(f7, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   f8 = new TGCompositeFrame(this, 80, 20, kVerticalFrame); 
   fAddB = new TGCheckButton(f8, "Draw bar chart", kB_ONOFF);
   fAddB ->SetToolTipText("Draw a bar chart");
   f8->AddFrame(fAddB, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   AddFrame(f8, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));

   f9 = new TGCompositeFrame(this, 80, 20, kVerticalFrame); 
   fAddBar = new TGCheckButton(f9, "Bar option", kBAR_ONOFF);
   fAddBar ->SetToolTipText("Draw bar chart with bar-option");
   f9->AddFrame(fAddBar, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   AddFrame(f9, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0)); 
   
   f15 = new TGCompositeFrame(this, 80, 20, kVerticalFrame); 
   fAddLine = new TGCheckButton(f15, "Add outer line", kADD_LINE);
   fAddLine ->SetToolTipText("Draw an outer line on the histogram");
   f15->AddFrame(fAddLine, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   AddFrame(f15, new TGLayoutHints(kLHintsTop, 1, 1, 0, 3)); 
   
   f10 = new TGCompositeFrame(this, 145, 10, kHorizontalFrame | kLHintsExpandX | kFixedWidth | kOwnBackground);
   f10->AddFrame(new TGLabel(f10,"Bar"), new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   f10->AddFrame(new TGHorizontal3DLine(f10), new TGLayoutHints(kLHintsExpandX, 5, 5, 7, 7));
   AddFrame(f10, new TGLayoutHints(kLHintsTop));
   
   f11 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGLabel *fWidthLbl = new TGLabel(f11, "W:");                              
   f11->AddFrame(fWidthLbl, new TGLayoutHints(kLHintsLeft, 1, 3, 4, 1));
   fBarWidth = new TGNumberEntry(f11, 1.00, 6, kBAR_WIDTH, 
                                      TGNumberFormat::kNESRealTwo,
                                      TGNumberFormat::kNEANonNegative, 
                                      TGNumberFormat::kNELLimitMinMax, 0.01, 1.);
   fBarWidth->GetNumberEntry()->SetToolTipText("Set bar chart width");
   fBarWidth->Resize(45,20);
   f11->AddFrame(fBarWidth, new TGLayoutHints(kLHintsLeft, 1, 1, 2, 1));

   TGLabel *fOffsetLbl = new TGLabel(f11, "O:");                              
   f11->AddFrame(fOffsetLbl, new TGLayoutHints(/*kLHintsCenterY | */kLHintsLeft, 6,3, 4, 1));
   fBarOffset = new TGNumberEntry(f11, 0.00, 5, kBAR_OFFSET, 
                                      TGNumberFormat::kNESRealTwo,
                                      TGNumberFormat::kNEAAnyNumber, 
                                      TGNumberFormat::kNELLimitMinMax, -1., 1.);
   fBarOffset->GetNumberEntry()->SetToolTipText("Set bar chart offset");
   fBarOffset->Resize(50,20);
   f11->AddFrame(fBarOffset, new TGLayoutHints(kLHintsLeft, 1, 1, 2, 1));
   AddFrame(f11, new TGLayoutHints(kLHintsTop, 1, 1, 0, 3));
   
   f12 = new TGCompositeFrame(this, 80, 20, kVerticalFrame);
   TGCompositeFrame *f13 = new TGCompositeFrame(f12, 80, 20, kHorizontalFrame);
   TGLabel *fPercentLabel = new TGLabel(f13, "Percentage:"); 
   f13->AddFrame(fPercentLabel, new TGLayoutHints(kLHintsLeft, 6, 1, 4, 1));
   fPercentCombo = BuildPercentComboBox(f13, kPERCENT_TYPE);
   f13->AddFrame(fPercentCombo, new TGLayoutHints(kLHintsLeft, 13, 1, 2, 1));
   fPercentCombo->Resize(52, 20);
   fPercentCombo->Associate(f12);
   f12->AddFrame(f13,new TGLayoutHints(kLHintsLeft, 0, 0, 0, 0));
   
   fMakeHBar = new TGCheckButton(f12, "Horizontal Bar", kBAR_H);
   fMakeHBar ->SetToolTipText("Draw a horizontal bar chart with hBar-Option");
   f12->AddFrame(fMakeHBar, new TGLayoutHints(kLHintsLeft, 6, 1, 1, 0));
   AddFrame(f12, new TGLayoutHints(kLHintsTop, 1, 1, 0, 3)); 

   MakeTitle("Axis Range");

   TGCompositeFrame *f14 = new TGCompositeFrame(this, 80, 20, kHorizontalFrame);
   TGLabel *fSliderYLbl = new TGLabel(f14,"x:");
   f14->AddFrame(fSliderYLbl, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 6,3, 4, 1)); 
   fSlider = new TGDoubleHSlider(f14, 1, 2);
   f14->AddFrame(fSlider, new TGLayoutHints(kLHintsExpandX));
   AddFrame(f14, new TGLayoutHints(kLHintsExpandX, 3, 7, 3, 0));
   
   makeB=kTRUE;
   make=kTRUE;


   MapSubwindows();
   Layout();
   MapWindow();
   
   TClass *cl = TH1::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________

TH1Editor::~TH1Editor()
{
   // Destructor of TH1 editor.

   TGFrameElement *el;
   TIter next(GetList());
   
   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();
}

//______________________________________________________________________________

void TH1Editor::ConnectSignals2Slots()
{
   // Connect signals to slots.

   fAddB->Connect("Toggled(Bool_t)", "TH1Editor", this, "DoAddB(Bool_t)");
   fAddBar->Connect("Toggled(Bool_t)", "TH1Editor", this, "DoAddBar(Bool_t)");
   fTitle->Connect("TextChanged(const char *)", "TH1Editor", this, "DoTitle(const char *)");
   fTypeCombo->Connect("Selected(Int_t)", "TH1Editor", this, "DoHistChanges()");
   fCoordsCombo->Connect("Selected(Int_t)", "TH1Editor", this, "DoHistChanges()");
   fErrorCombo->Connect("Selected(Int_t)", "TH1Editor", this, "DoHistChanges()");
   fAddCombo->Connect("Selected(Int_t)", "TH1Editor", this, "DoHistChanges()");
   fAddMarker->Connect("Toggled(Bool_t)", "TH1Editor", this, "DoAddMarker(Bool_t)");
   fAddLine->Connect("Toggled(Bool_t)", "TH1Editor", this, "DoAddLine(Bool_t)");
   fDim->Connect("Pressed()","TH1Editor",this,"DoHistSimple()");
   fDim0->Connect("Pressed()","TH1Editor",this,"DoHistComplex()");   
   fBarWidth->Connect("ValueSet(Long_t)", "TH1Editor", this, "DoBarWidth()");
   (fBarWidth->GetNumberEntry())->Connect("ReturnPressed()", "TH1Editor", this, "DoBarWidth()");   
   fBarOffset->Connect("ValueSet(Long_t)", "TH1Editor", this, "DoBarOffset()");
   (fBarOffset->GetNumberEntry())->Connect("ReturnPressed()", "TH1Editor", this, "DoBarOffset()");
   fPercentCombo->Connect("Selected(Int_t)", "TH1Editor", this, "DoPercent()");
   fMakeHBar-> Connect("Toggled(Bool_t)","TH1Editor",this,"DoHBar(Bool_t))"); 
   fSlider->Connect("PositionChanged()","TH1Editor",this,"DoSlider()");         
   fInit = kFALSE;
}

//______________________________________________________________________________

void TH1Editor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Pick up the used values of histogram attributes.
   
   fModel = 0;
   fPad = 0;

   if (obj == 0 || !obj->InheritsFrom("TH1") /*|| obj->InheritsFrom("TH2")  || obj->InheritsFrom("TProfile")*/) {
      SetActive(kFALSE);
      return;
   } else if (((TH1*)obj)->GetDimension()!=1) {
      SetActive(kFALSE);
      return;
   }


   fModel = obj;
   fPad = pad;

   fHist = (TH1*)fModel;
   const char *text = fHist->GetTitle();
   fTitle->SetText(text);
   
   make=kFALSE;
   if (!fInit) {
      DisconnectAllSlots();
      fInit=kTRUE;
   }
   TString str = GetDrawOption();
   str.ToUpper();
   Bool_t errorset = kFALSE;
   if (str.IsNull() || str=="" ) {        
      HideFrame(f3);  // Hiding the histogram type combo box
      HideFrame(f4);  // Hiding the histogram coord type combo box
      ShowFrame(f6);
      ShowFrame(f7);
      ShowFrame(f8);
      ShowFrame(f9);
      HideFrame(f10);
      HideFrame(f11);
      HideFrame(f12);
      ShowFrame(f15);
      fCoordsCombo->Select(kCOORDS_CAR);
      if (fErrorCombo->GetSelected()!=kERRORS_NO) fErrorCombo->Select(kERRORS_NO);
      errorset=kTRUE;
      fAddCombo->Select(kADD_NONE);
      fdimgroup->SetButton(kDIM_SIMPLE, kTRUE);
      fAddMarker->SetState(kButtonUp);
      fAddB->SetState(kButtonUp);
      SetDrawOption("HIST");
      fAddLine->SetState(kButtonDisabled);
      ChangeErrorCombo(1);

   } else if (!str.Contains("LEGO") && !str.Contains("SURF")){
      HideFrame(f3);  // Hiding the histogram type combo box
      HideFrame(f4);  // Hiding the histogram coord type combo box
      ShowFrame(f7);
      ShowFrame(f8);
      ShowFrame(f9);
      ShowFrame(f15);
      fCoordsCombo->Select(kCOORDS_CAR);
      if (str.Contains("C")) {
         if (str.Contains("CYL")) {
	    TString dum = str;
	    dum.Remove(strstr(dum.Data(),"CYL")-dum.Data(),3);
	    if (dum.Contains("C")) fAddCombo->Select(kADD_SMOOTH);
	 } else fAddCombo->Select(kADD_SMOOTH);
      } 
      else if (str.Contains("LF2")) fAddCombo->Select(kADD_FILL);
      else if (str.Contains("L")){
         TString dum = str;
         if (str.Contains("CYL")) {
	    dum.Remove(strstr(dum.Data(),"CYL")-dum.Data(),3);
	    if (dum.Contains("L")) fAddCombo->Select(kADD_SIMPLE);
         }
         if (str.Contains("POL")) {
	    dum.Remove(strstr(dum.Data(),"POL")-dum.Data(),3);
	    if (dum.Contains("L")) fAddCombo->Select(kADD_SIMPLE);
         } else fAddCombo->Select(kADD_SIMPLE);
      } else fAddCombo->Select(kADD_NONE);

      if (fAddCombo->GetSelected()!=kADD_NONE) fAddLine->SetState(kButtonDisabled);
      else if (str.Contains("HIST")) {
         if (str=="HIST") fAddLine->SetState(kButtonDisabled);
	 else fAddLine->SetState(kButtonDown);
      } else fAddLine->SetState(kButtonUp);
      
      fdimgroup->SetButton(kDIM_SIMPLE,kTRUE);
      if (str.Contains("B")) {
         TString dum = str;
	 if (str.Contains("BAR")) {
	    fAddBar->SetState(kButtonDown);
            fAddB->SetState(kButtonDisabled);
	    ShowFrame(f10);
            ShowFrame(f11);
	    ShowFrame(f12);
	 } else {
	    fAddB->SetState(kButtonDown);
            fAddBar->SetState(kButtonDisabled);
	    fAddLine->SetState(kButtonDisabled);
	    ShowFrame(f10);
            ShowFrame(f11);      
            HideFrame(f12);
	 }
      } else {
         fAddB->SetState(kButtonUp);
	 fAddBar->SetState(kButtonUp);
	 HideFrame(f10);
         HideFrame(f11);
         HideFrame(f12);
      }
      if (str.Contains("P") ) {
         fAddMarker->SetState(kButtonDown);
	 fAddLine->SetState(kButtonDisabled);
      } else if (!str.Contains("BAR")) fAddMarker->SetState(kButtonUp);
     
      ChangeErrorCombo(1);
      
   } else if (str.Contains("LEGO") || str.Contains("SURF")){
      ChangeErrorCombo(0);
      if (str.Contains("SURF")){ 
         fCoordsCombo->RemoveEntry(kCOORDS_SPH);
         fCoordsCombo->RemoveEntry(kCOORDS_CAR);
      } else {
         if (((TGLBContainer*)((TGListBox*)fCoordsCombo->GetListBox())->GetContainer())->GetPos(kCOORDS_SPH)==-1) fCoordsCombo->AddEntry("Spheric", kCOORDS_SPH);
         if (((TGLBContainer*)((TGListBox*)fCoordsCombo->GetListBox())->GetContainer())->GetPos(kCOORDS_CAR)==-1) fCoordsCombo->AddEntry("Cartesian", kCOORDS_CAR);
      }
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

      HideFrame(f6); 
      HideFrame(f7); 
      HideFrame(f8); 
      HideFrame(f9);
      HideFrame(f15);
      if (str.Contains("LEGO")) {
         ShowFrame(f10);
         ShowFrame(f11); 
	 HideFrame(f12);
      } else {
         HideFrame(f10);
         HideFrame(f11); 
	 HideFrame(f12);
      }
      fdimgroup->SetButton(kDIM_COMPLEX,kTRUE);
      fAddMarker->SetState(kButtonDisabled);
      fAddB->SetState(kButtonDisabled);
   }
   
   if (!errorset) {   
      if (str.Contains("E1")) fErrorCombo->Select(kERRORS_EDGES);
      else if (str.Contains("E2")) fErrorCombo->Select(kERRORS_REC);
      else if (str.Contains("E3")) fErrorCombo->Select(kERRORS_FILL);
      else if (str.Contains("E4")) fErrorCombo->Select(kERRORS_CONTOUR);
      else if (str.Contains("E")) {
	 if (str.Contains("LEGO")) {
            TString dum=str;
	    dum.Remove(strstr(dum.Data(),"LEGO")-dum.Data(),4);
  	    if (dum.Contains("E")) fErrorCombo->Select(kERRORS_SIMPLE);
	 } else fErrorCombo->Select(kERRORS_SIMPLE); 
      } else fErrorCombo->Select(kERRORS_NO); //default
   }     
    
   if (fErrorCombo->GetSelected() != kERRORS_NO){
      HideFrame(f7);
      HideFrame(f8);
   }
   if (str.Contains("BAR") || (fAddBar->GetState()==kButtonDown) && (fDim->GetState()==kButtonDown)) {
      ShowFrame(f10);
      ShowFrame(f11);
      ShowFrame(f12);
      fBarWidth->SetNumber(fHist->GetBarWidth());
      fBarOffset->SetNumber(fHist->GetBarOffset());
      if (str.Contains("HBAR")) fMakeHBar->SetState(kButtonDown);
      else fMakeHBar->SetState(kButtonUp);
      
      if (str.Contains("BAR4")) fPercentCombo->Select(kPER_40);
      else if (str.Contains("BAR3")) fPercentCombo->Select(kPER_30);
      else if (str.Contains("BAR2")) fPercentCombo->Select(kPER_20);
      else if (str.Contains("BAR1")) fPercentCombo->Select(kPER_10);
      else fPercentCombo->Select(kPER_0);
   }

   Int_t nx = fHist -> GetXaxis() -> GetNbins();
   Int_t nxbinmin = fHist -> GetXaxis() -> GetFirst()-1;
   Int_t nxbinmax = fHist -> GetXaxis() -> GetLast()+1;
   fSlider->SetPosition((Float_t)nxbinmin/nx,(Float_t)nxbinmax/nx);
   
   Update();
   if (fInit) ConnectSignals2Slots();
   SetActive();
   make=kTRUE;
}

//______________________________________________________________________________

void TH1Editor::DoTitle(const char *text)
{
   // Slot connected to the title of the histogram .
  
   fHist->SetTitle(text);
   Update();
}

//______________________________________________________________________________

void TH1Editor::DoAddMarker(Bool_t on)
{
   // Slot connected to the marker add checkbox .
   
   TString str = GetDrawOption();
   str.ToUpper(); 
   TString dum = str;
   
   if (dum.Contains("POL")) dum.Remove(strstr(dum.Data(),"POL")-dum.Data(),3);
   if (dum.Contains("SPH")) dum.Remove(strstr(dum.Data(),"SPH")-dum.Data(),3); 
   if (dum.Contains("PSR")) dum.Remove(strstr(dum.Data(),"PSR")-dum.Data(),3);      
   if (on) {
      if (!dum.Contains("P")) str += "P"; 
      fAddLine->SetState(kButtonDisabled);
      if (str.Contains("HIST")) str.Remove(strstr(str.Data(),"HIST")-str.Data(),4);
   } else if (fAddMarker->GetState()==kButtonUp) {
      if (str.Contains("POL") || str.Contains("SPH")) {
         while (dum.Contains("P")) dum.Remove(strstr(dum.Data(),"P")-dum.Data(),1);
	 if (str.Contains("POL")) str = dum + "POL";
         if (str.Contains("SPH")) str = dum + "SPH";
         if (str.Contains("PSR")) str = dum + "PSR";	 
      } else if (str.Contains("P")) str.Remove(str.First("P"),1); 
      if ((str=="HIST") || (str=="") || (fAddB->GetState()==kButtonDown) || fAddCombo->GetSelected()!=kADD_NONE) fAddLine->SetState(kButtonDisabled);
      else if (str.Contains("HIST")) fAddLine->SetState(kButtonDown);
      else fAddLine->SetState(kButtonUp);
   }
   if (make) SetDrawOption(str);
   Update();
}

//______________________________________________________________________________

void TH1Editor::DoAddB(Bool_t on)
{
   // Slot connected to the bar add checkbox .
   
   TString str = GetDrawOption();
   str.ToUpper();
   if (makeB) {
      makeB=kFALSE;
      if (on) {
         if (!str.Contains("B")) str += "B";
         ShowFrame(f10);
         ShowFrame(f11);
         HideFrame(f12);
         fAddBar->SetState(kButtonDisabled);
	 fAddLine->SetState(kButtonDisabled);
         fBarOffset->SetNumber(fHist->GetBarOffset());
         fBarWidth->SetNumber(fHist->GetBarWidth());      
      } else if (fAddB->GetState()==kButtonUp) {
         while (str.Contains("B")) str.Remove(str.First("B"),1);
         HideFrame(f10);
	 HideFrame(f11);
	 HideFrame(f12);
         fAddBar->SetState(kButtonUp);
	 if (fAddMarker->GetState()!=kButtonDown && !(str=="" || str=="HIST" || fAddCombo->GetSelected()!=kADD_NONE)) fAddLine->SetState(kButtonUp);
      }
      if (make) SetDrawOption(str);
      Update(); 
      SetActive();
      makeB=kTRUE;
   }
}

//______________________________________________________________________________

void TH1Editor::DoAddBar(Bool_t on)
{
   // Slot connected to the bar add checkbox .
   
   Disconnect(fAddMarker);
   TString str = GetDrawOption();
   str.ToUpper();
   if (makeB) {
      makeB=kFALSE;
      Int_t o = 0;
      if (str.Contains("HBAR")) o=1;
      if (str.Contains("BAR4")) str.Remove(strstr(str.Data(),"BAR4")-str.Data()-o,4+o);
      else if (str.Contains("BAR3")) str.Remove(strstr(str.Data(),"BAR3")-str.Data()-o,4+o);
      else if (str.Contains("BAR2")) str.Remove(strstr(str.Data(),"BAR2")-str.Data()-o,4+o);
      else if (str.Contains("BAR1")) str.Remove(strstr(str.Data(),"BAR1")-str.Data()-o,4+o);            
      else if (str.Contains("BAR0")) str.Remove(strstr(str.Data(),"BAR0")-str.Data()-o,4+o);      
      else if (str.Contains("BAR")) str.Remove(strstr(str.Data(),"BAR")-str.Data()-o,3+o);      
      if (on) {
//         fAddMarker->SetState(kButtonUp);
//	 if (str.Contains("P")) str.Remove(str.First("P"),1);
         if ((fAddMarker->GetState()==kButtonDown) && (fErrorCombo->GetSelected()==kERRORS_NO) && (fAddLine->GetState()!=kButtonDisabled)) fAddLine->SetState(kButtonDisabled);
	 else if ((fAddMarker->GetState()!=kButtonDown) && (fAddLine->GetState()==kButtonDisabled)) {
	    if (str.Contains("HIST")) fAddLine->SetState(kButtonDown);  
	    else if (fAddCombo->GetSelected()!=kADD_NONE) fAddLine->SetState(kButtonDisabled);
	    else fAddLine->SetState(kButtonUp);
	 }
	 
         switch (fPercentCombo->GetSelected()){
            case(-1): { str += "BAR";
	       fPercentCombo->Select(kPER_0);
	       break;
	    }
            case(kPER_0): { 
	       str += "BAR"; 
	       break;
            }
            case(kPER_10): { 
               str += "BAR1"; 
               break;
            }
            case(kPER_20): { 
               str += "BAR2"; 
               break;
            }
            case(kPER_30): { 
	       str += "BAR3"; 
	       break;
            }
            case(kPER_40): { 
	       str += "BAR4"; 
	       break;
            }	 
         }
         ShowFrame(f10);
         ShowFrame(f11);
         ShowFrame(f12);
         if (fMakeHBar->GetState()==kButtonDown) str.Insert(strstr(str.Data(),"BAR")-str.Data(),"H");
         fBarOffset->SetNumber(fHist->GetBarOffset());
         fBarWidth->SetNumber(fHist->GetBarWidth());      
         fAddB->SetState(kButtonDisabled);
      } else if (fAddBar->GetState()==kButtonUp) {
         HideFrame(f10);
         HideFrame(f11); 
         HideFrame(f12);
         fAddB->SetState(kButtonUp);
         if (fAddMarker->GetState()==kButtonDisabled) fAddMarker->SetState(kButtonUp);
	 if (str=="" || str=="HIST" || fAddCombo->GetSelected()!=kADD_NONE || ((fAddMarker->GetState()==kButtonDown) && fErrorCombo->GetSelected()==kERRORS_NO)) fAddLine->SetState(kButtonDisabled);
      }
      if (make) SetDrawOption(str);
      Update(); 
      SetActive();
      makeB=kTRUE;
   }
   fAddMarker->Connect("Toggled(Bool_t)", "TH1Editor", this, "DoAddMarker(Bool_t)");
}

//______________________________________________________________________________

void TH1Editor::DoAddLine(Bool_t on)
{
   // Slot: Draws/Removes an outer line on the histogram

   Disconnect(fAddMarker);
   Bool_t make=kFALSE;
   TString str = GetDrawOption();
   str.ToUpper();
   if (on) {
      if (!str.Contains("HIST")) {
         str += "HIST";
	 fAddMarker->SetState(kButtonDisabled);
	 make=kTRUE;
      }
   } else if (fAddLine->GetState()==kButtonUp) {
      if (str.Contains("HIST")) {
         str.Remove(strstr(str.Data(),"HIST")-str.Data(),4);
         fAddMarker->SetState(kButtonUp);	 
	 make=kTRUE;
      }
   }
   if (make) SetDrawOption(str);
   fAddMarker->Connect("Toggled(Bool_t)", "TH1Editor", this, "DoAddMarker(Bool_t)");
   Update();   
}    

//______________________________________________________________________________

void TH1Editor::DoHistSimple()
{
   // Slot connected to the simple histogram radiobutton

   if (fDim->GetState()==kButtonDown){
      TString str ="";
      make=kFALSE;
      HideFrame(f3);
      HideFrame(f4);
      ShowFrame(f6);
      ShowFrame(f9);
      ShowFrame(f15);
      ChangeErrorCombo(1);
      if ((fAddBar->GetState() !=kButtonDown || fAddMarker->GetState()==kButtonDown ) && (fErrorCombo->GetSelected()==kERRORS_NO)) fAddLine->SetState(kButtonDisabled);
      else if ((fAddLine->GetState()==kButtonDisabled) && (fAddMarker->GetState()!=kButtonDown) ) fAddLine->SetState(kButtonUp);
      else if (fAddLine->GetState()!=kButtonUp) fAddLine->SetState(kButtonDown);
      if (fAddMarker->GetState()==kButtonDisabled && fAddLine->GetState()!=kButtonDown) fAddMarker->SetState(kButtonUp);

      if (fErrorCombo->GetSelected()==kERRORS_NO){   
         ShowFrame(f7);
         ShowFrame(f8);
      } else {
         HideFrame(f7);
         HideFrame(f8);
         if (fAddBar->GetState()==kButtonDisabled)  fAddBar->SetState(kButtonUp);
      } 

      if ((fAddB->GetState() == kButtonDisabled)){
         ShowFrame(f10);
         ShowFrame(f11);
         ShowFrame(f12);
      } 
      if (fAddBar->GetState() == kButtonDisabled){
         ShowFrame(f10);  
         ShowFrame(f11);
         HideFrame(f12);
	 SetActive();
      } 
     if ((fAddBar->GetState() == kButtonUp) && (fAddB->GetState() == kButtonUp)){
         HideFrame(f10);  
         HideFrame(f11);
         HideFrame(f12);
	 SetActive();
      }

      if (fAddCombo->GetSelected()== -1 )fAddCombo->Select(kADD_NONE);
      if (fErrorCombo->GetSelected()!=kERRORS_NO) {
         fAddCombo->RemoveEntries(kADD_SIMPLE,kADD_FILL);
	 Disconnect(fAddCombo);
	 fAddCombo->Select(kADD_NONE);
	 fAddCombo->Connect("Selected(Int_t)", "TH1Editor", this, "DoHistChanges()"); 
      } else {
         if (((TGLBContainer*)((TGListBox*)fAddCombo->GetListBox())->GetContainer())->GetPos(kADD_SIMPLE)==-1) ((TGListBox*)fAddCombo->GetListBox())->AddEntry("Simple Line", kADD_SIMPLE);
	 if (((TGLBContainer*)((TGListBox*)fAddCombo->GetListBox())->GetContainer())->GetPos(kADD_SMOOTH)==-1) ((TGListBox*)fAddCombo->GetListBox())->AddEntry("Smooth Line", kADD_SMOOTH);
	 if (((TGLBContainer*)((TGListBox*)fAddCombo->GetListBox())->GetContainer())->GetPos(kADD_FILL)==-1) ((TGListBox*)fAddCombo->GetListBox())->AddEntry("Fill Area",kADD_FILL);
      }
      if (fAddLine->GetState()==kButtonDown) str+="HIST";
      str += GetHistErrorLabel()+GetHistAddLabel();
      
      SetDrawOption(str);
      Update();
      SetActive();
      make=kTRUE;
   }
}

//______________________________________________________________________________

void TH1Editor::DoHistComplex()
{
   // Slot connected to the complex histogram radiobutton

   if (fDim0->GetState()==kButtonDown) {
      TString str ="";
      make=kFALSE;
      ShowFrame(f3);
      ShowFrame(f4);
      HideFrame(f6);
      HideFrame(f7);
      HideFrame(f8);
      HideFrame(f9);
      HideFrame(f15);  
      ChangeErrorCombo(0); 
      if (fTypeCombo->GetSelected()==-1 && fCoordsCombo->GetSelected()==-1) {
         str = "LEGO"+GetHistErrorLabel();
         fTypeCombo->Select(kTYPE_LEGO);
         fCoordsCombo->Select(kCOORDS_CAR);
      } else if (fTypeCombo->GetSelected()==-1){
         str = "LEGO"+GetHistErrorLabel();
         fTypeCombo->Select(kTYPE_LEGO);
      } else if (fCoordsCombo->GetSelected()==-1) {
         str = GetHistTypeLabel()+GetHistErrorLabel();
         fCoordsCombo->Select(kCOORDS_CAR);
      } else {
         str = GetHistTypeLabel()+GetHistCoordsLabel()+GetHistErrorLabel();
      }
     if (str.Contains("LEGO")) {
         ShowFrame(f10);
         ShowFrame(f11); 
         HideFrame(f12);
	 SetActive();
      } else {
         HideFrame(f10);
         HideFrame(f11); 
         HideFrame(f12);
	 SetActive();
      }
      SetDrawOption(str);
      Update();
      SetActive();
      make=kTRUE;
   }
}    

//______________________________________________________________________________

void TH1Editor::DoHistChanges()
{
   // Slot connected to the histogram type, the coordinate type, the error type
   // and the AddCombobox
   
   makeB= kFALSE;
   if (GetHistTypeLabel().Contains("SURF")){ 
      if (fCoordsCombo->GetSelected()==kCOORDS_CAR || fCoordsCombo->GetSelected()==kCOORDS_SPH) fCoordsCombo->Select(kCOORDS_POL); 
      fCoordsCombo->RemoveEntry(kCOORDS_SPH);
      fCoordsCombo->RemoveEntry(kCOORDS_CAR);
   } else {
      if (((TGLBContainer*)((TGListBox*)fCoordsCombo->GetListBox())->GetContainer())->GetPos(kCOORDS_SPH)==-1) ((TGListBox*)fCoordsCombo->GetListBox())->AddEntrySort("Spheric", kCOORDS_SPH);
      if (((TGLBContainer*)((TGListBox*)fCoordsCombo->GetListBox())->GetContainer())->GetPos(kCOORDS_CAR)==-1) ((TGListBox*)fCoordsCombo->GetListBox())->AddEntrySort("Cartesian", kCOORDS_CAR);
   }
   if (fDim->GetState()!=kButtonUp){
      if (fErrorCombo->GetSelected() != kERRORS_NO){
         HideFrame(f7);
         HideFrame(f8);
         ShowFrame(f9);
	 fAddMarker->SetState(kButtonDisabled);
	 fAddB->SetState(kButtonDisabled);
	 if (fAddBar->GetState()==kButtonDisabled) fAddBar->SetState(kButtonUp);
	 if (fAddLine->GetState()==kButtonDisabled) fAddLine->SetState(kButtonUp);
	 fAddCombo->RemoveEntries(kADD_SIMPLE,kADD_FILL);
	 Disconnect(fAddCombo);
	 fAddCombo->Select(kADD_NONE);
	 fAddCombo->Connect("Selected(Int_t)", "TH1Editor", this, "DoHistChanges()");
         if (fAddBar->GetState()==kButtonDown) {
            ShowFrame(f10);	
	    ShowFrame(f11);
            ShowFrame(f12);
	 } else {
            HideFrame(f10);
            HideFrame(f11);
            HideFrame(f12);
         }	    
      } else {
         Bool_t on = make;
         make=kFALSE;
         ShowFrame(f7);
         ShowFrame(f8);
         ShowFrame(f9);
	 if (fAddMarker->GetState()==kButtonDisabled) fAddMarker->SetState(kButtonUp);
	 if (fAddBar->GetState()!=kButtonDown && fAddB->GetState()==kButtonDisabled) fAddB->SetState(kButtonUp);
	 if (((TGLBContainer*)((TGListBox*)fAddCombo->GetListBox())->GetContainer())->GetPos(kADD_SIMPLE)==-1) ((TGListBox*)fAddCombo->GetListBox())->AddEntry("Simple Line", kADD_SIMPLE);
	 if (((TGLBContainer*)((TGListBox*)fAddCombo->GetListBox())->GetContainer())->GetPos(kADD_SMOOTH)==-1) ((TGListBox*)fAddCombo->GetListBox())->AddEntry("Smooth Line", kADD_SMOOTH);
	 if (((TGLBContainer*)((TGListBox*)fAddCombo->GetListBox())->GetContainer())->GetPos(kADD_FILL)==-1) ((TGListBox*)fAddCombo->GetListBox())->AddEntry("Fill Area",kADD_FILL);
         make=on;
      }
      if (fAddCombo->GetSelected()!=kADD_NONE) {
         fAddLine->SetState(kButtonDisabled);
      } else {
         if (fAddMarker->GetState()==kButtonDown) fAddLine->SetState(kButtonDisabled);
	 else if (fAddLine->GetState()==kButtonDisabled) fAddLine->SetState(kButtonUp);
      }
   } else if (fDim0->GetState()==kButtonDown) {
      if (GetHistTypeLabel().Contains("LEGO")) {
         ShowFrame(f10);
         ShowFrame(f11);
         HideFrame(f12);
      } else {
         HideFrame(f10);
         HideFrame(f11);
         HideFrame(f12);
      }
   }        
   if (make) {
      TString str = "";
      if (fDim->GetState()==kButtonDown) str = GetHistErrorLabel()+GetHistAddLabel();
      else if (fDim0->GetState()==kButtonDown) str = GetHistTypeLabel()+GetHistCoordsLabel()+GetHistErrorLabel();
      if (fAddLine->GetState()==kButtonDown) str += "HIST";   
      SetDrawOption(str);
      if (str=="" || str=="HIST") fAddLine->SetState(kButtonDisabled);
      Update();
   }
   SetActive(); 
   makeB=kTRUE;
}

//______________________________________________________________________________

void TH1Editor::DoBarWidth()
{
   // Slot connected to the Bar Width of the Bar Chart
   
   fHist->SetBarWidth(fBarWidth->GetNumber());
   Update();
}
   
//______________________________________________________________________________

void TH1Editor::DoBarOffset()
{
   // Slot connected to the Bar Offset of the Bar Chart
   
   Float_t f = fBarOffset->GetNumber();
   fHist->SetBarOffset(f);
   Update();
}

//______________________________________________________________________________

void TH1Editor::DoPercent()
{
   // Slot connected percentage of ???
      
   TString str = GetDrawOption();
   str.ToUpper();
   Int_t o = 0;
   if (str.Contains("HBAR")) o=1;
   if (str.Contains("BAR4")) str.Remove(strstr(str.Data(),"BAR4")-str.Data()-1,4+o);
   else if (str.Contains("BAR3")) str.Remove(strstr(str.Data(),"BAR3")-str.Data()-o,4+o);
   else if (str.Contains("BAR2")) str.Remove(strstr(str.Data(),"BAR2")-str.Data()-o,4+o);   
   else if (str.Contains("BAR1")) str.Remove(strstr(str.Data(),"BAR1")-str.Data()-o,4+o);
   else if (str.Contains("BAR0")) str.Remove(strstr(str.Data(),"BAR0")-str.Data()-o,4+o);
   else if (str.Contains("BAR")) str.Remove(strstr(str.Data(),"BAR")-str.Data()-o,3+o);
   
   if (fMakeHBar->GetState()==kButtonDown) str+="H";
   switch (fPercentCombo->GetSelected()){
      case (kPER_0) :{ str += "BAR"; break;}
      case (kPER_10):{ str += "BAR1"; break;}
      case (kPER_20):{ str += "BAR2"; break;}            
      case (kPER_30):{ str += "BAR3"; break;} 
      case (kPER_40):{ str += "BAR4"; break;}                  
   }
   if (make) SetDrawOption(str);
   Update();
}

//______________________________________________________________________________

void TH1Editor::DoHBar(Bool_t on)
{
   // Slot connected to the Horizontal Bar CheckButton
   
   TString str = GetDrawOption();
   str.ToUpper();
   if (on) {
      if (!str.Contains("HBAR")) str.Insert(strstr(str.Data(),"BAR")-str.Data(),"H");
   }
   else if (fMakeHBar->GetState()==kButtonUp) {
      if(str.Contains("HBAR")) str.Remove(strstr(str.Data(),"BAR")-str.Data()-1,1);
   }
   if (make) SetDrawOption(str);
   Update();
}

//______________________________________________________________________________

void TH1Editor::DoSlider()
{
   // Slot connected to the x-Slider
   // Redraws the Histogram with the new Slider Range
   
   Int_t nx = fHist->GetXaxis()->GetNbins();
   Int_t binxmin = (Int_t)(nx*fSlider->GetMinPosition());
   Int_t binxmax = (Int_t)(nx*fSlider->GetMaxPosition());
   if (binxmin==(fHist->GetXaxis()->GetNbins())) binxmin-=1;
   else if (binxmax==binxmin) binxmax+=1;
   if (binxmax==1) binxmax+=1;   
   fHist->GetXaxis()->SetRange(binxmin,binxmax);
   Update();
}
   
//______________________________________________________________________________

TString TH1Editor::GetHistTypeLabel()
{
   // Returns the immediate histogram type (HIST, LEGO1-2, SURF1-5)

   TString s="";
   switch (fTypeCombo->GetSelected()){
      case (-1)         : {s = "LEGO"; break;}
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

TString TH1Editor::GetHistCoordsLabel()
{
   // Returns the immediate coordinate system of the histogram (POL, CYL, SPH,PSR)

   TString s="";
   if (fDim->GetState()!=kButtonDown) {
      switch (fCoordsCombo->GetSelected()){
         case (-1)         : {s = "POL"; break;}
         case (kCOORDS_CAR): {s = ""; break;}
         case (kCOORDS_POL): {s = "POL"; break;}
         case (kCOORDS_CYL): {s = "CYL"; break;}
         case (kCOORDS_SPH): {s = "SPH"; break;}
         case (kCOORDS_PSR): {s = "PSR"; break;}
         default:  break;
      }
   }

   return s;
}

//______________________________________________________________________________

TString TH1Editor::GetHistErrorLabel()
{
   // Returns the immediate error type (E,E1-5)
   
   TString s="";
   switch (fErrorCombo->GetSelected()){
      case (-1)             : {s = ""; break;}
      case (kERRORS_NO)     : {s = ""; break;}
      case (kERRORS_SIMPLE) : {s = "E"; break;}
      case (kERRORS_EDGES)  : {s = "E1"; break;}
      case (kERRORS_REC)    : {s = "E2"; break;}
      case (kERRORS_FILL)   : {s = "E3"; break;}
      case (kERRORS_CONTOUR): {s = "E4"; break;}
      default:  break;
   }

   return s;
}

//______________________________________________________________________________

TString TH1Editor::GetHistAddLabel()
{
   // Returns the immediate shape of the histogram (C, L, LF2)
   
   TString s="";
   switch (fAddCombo->GetSelected()){
      case (-1)         : {s = "" ; break;}
      case (kADD_NONE)  : {s = "" ; break;}
      case (kADD_SMOOTH): {s = "C"; break;}
      case (kADD_SIMPLE): {s = "L"; break;}
      case (kADD_FILL)  : {s = "LF2"; break;}
      default           :  break;
   }
   if (fAddMarker->GetState()==kButtonDown) s += "P";
   if (fAddB->GetState()==kButtonDown) s += "B";
   if (fAddBar->GetState()==kButtonDown){
      if (fMakeHBar->GetState()==kButtonDown) s+="H";
      switch (fPercentCombo->GetSelected()){
         case (kPER_0) : { s += "BAR" ; break;}
         case (kPER_10): { s += "BAR1"; break;}
         case (kPER_20): { s += "BAR2"; break;}            
         case (kPER_30): { s += "BAR3"; break;} 
         case (kPER_40): { s += "BAR4"; break;}                  
      }
   }    

   return s;
}

//______________________________________________________________________________

TGComboBox* TH1Editor::BuildHistTypeComboBox(TGFrame* parent, Int_t id)
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

TGComboBox* TH1Editor::BuildHistCoordsComboBox(TGFrame* parent, Int_t id)
{
   // Create histogram coordinate system type combo box.

   TGComboBox *c = new TGComboBox(parent, id);

   c->AddEntry("Cartesian", kCOORDS_CAR);
   c->AddEntry("Cylindric", kCOORDS_CYL);
   c->AddEntry("Polar", kCOORDS_POL);
   c->AddEntry("Rapidity", kCOORDS_PSR);   
   c->AddEntry("Spheric", kCOORDS_SPH);   
   
   return c;
}

//______________________________________________________________________________

TGComboBox* TH1Editor::BuildHistErrorComboBox(TGFrame* parent, Int_t id)
{
   // Create histogram error type combo box.

   TGComboBox *c = new TGComboBox(parent, id);

   c->AddEntry("No Errors", kERRORS_NO);
   c->AddEntry("Simple", kERRORS_SIMPLE);
   c->AddEntry("Edges", kERRORS_EDGES);
   c->AddEntry("Rectangles",kERRORS_REC);
   c->AddEntry("Fill", kERRORS_FILL);   
   c->AddEntry("Contour", kERRORS_CONTOUR);   
 
   return c;
}

//______________________________________________________________________________

TGComboBox* TH1Editor::BuildHistAddComboBox(TGFrame* parent, Int_t id)
{
   // Create histogram Line/Bar adding combo box.

   TGComboBox *c = new TGComboBox(parent, id);

   c->AddEntry("No Line", kADD_NONE);
   c->AddEntry("Simple Line", kADD_SIMPLE);
   c->AddEntry("Smooth Line", kADD_SMOOTH);
   c->AddEntry("Fill Area",kADD_FILL);
 
   return c;
}

//______________________________________________________________________________

TGComboBox* TH1Editor::BuildPercentComboBox(TGFrame* parent, Int_t id)
{
   // Create Percentage Combo Box for Bar Option
   
   TGComboBox *c = new TGComboBox(parent, id);
   
   c->AddEntry(" 0 %", kPER_0);   
   c->AddEntry("10 %", kPER_10);   
   c->AddEntry("20 %", kPER_20);
   c->AddEntry("30 %", kPER_30);
   c->AddEntry("40 %", kPER_40);   
   
   return c;
}

//______________________________________________________________________________

void TH1Editor::DisconnectAllSlots()
{
   // Disconnects all Slots

   Disconnect(fAddB,"Toggled(Bool_t)", this, "DoAddB(Bool_t)");
   Disconnect(fAddBar,"Toggled(Bool_t)", this, "DoAddBar(Bool_t)");
   Disconnect(fTitle, "TextChanged(const char *)", this, "DoTitle(const char *)");
   Disconnect(fTypeCombo, "Selected(Int_t)", this, "DoHistChanges()");
   Disconnect(fCoordsCombo, "Selected(Int_t)", this, "DoHistChanges()");
   Disconnect(fErrorCombo, "Selected(Int_t)", this, "DoHistChanges()");
   Disconnect(fAddCombo, "Selected(Int_t)", this, "DoHistChanges()");
   Disconnect(fAddMarker, "Toggled(Bool_t)", this, "DoAddMarker(Bool_t)");
   Disconnect(fAddLine, "Toggled(Bool_t)", this, "DoAddLine(Bool_t)"); 
   Disconnect(fDim, "Pressed()", this, "DoHistSimple()");
   Disconnect(fDim0, "Pressed()", this, "DoHistComplex()");   
   Disconnect(fBarWidth, "ValueSet(Long_t)", this, "DoBarWidth()");
   Disconnect((fBarWidth->GetNumberEntry()), "ReturnPressed()", this, "DoBarWidth()");   
   Disconnect(fBarOffset, "ValueSet(Long_t)", this, "DoBarOffset()");
   Disconnect((fBarOffset->GetNumberEntry()), "ReturnPressed()", this, "DoBarOffset()");
   Disconnect(fPercentCombo, "Selected(Int_t)", this, "DoPercent()");
   Disconnect(fMakeHBar, "Toggled(Bool_t)", this, "DoHBar(Bool_t))"); 
   Disconnect(fSlider, "PositionChanged()", this, "DoSlider()");         

}

//______________________________________________________________________________

void TH1Editor::ChangeErrorCombo(Int_t i)
{
   // Changes the display of the error combobox
   
  switch (i){
     case 0: {
        if (((TGLBContainer*)((TGListBox*)fErrorCombo->GetListBox())->GetContainer())->GetPos(kERRORS_EDGES)!=-1) fErrorCombo->RemoveEntries(kERRORS_EDGES,kERRORS_CONTOUR);
	if (!((fErrorCombo->GetSelected()== kERRORS_NO) || (fErrorCombo->GetSelected()== kERRORS_SIMPLE))) fErrorCombo->Select(kERRORS_NO);
        break;
     }
     case 1: {   
        if (((TGLBContainer*)((TGListBox*)fErrorCombo->GetListBox())->GetContainer())->GetPos(kERRORS_EDGES)==-1){
           fErrorCombo->AddEntry("Edges", kERRORS_EDGES);
           fErrorCombo->AddEntry("Rectangles",kERRORS_REC);
           fErrorCombo->AddEntry("Fill", kERRORS_FILL);   
           fErrorCombo->AddEntry("Contour", kERRORS_CONTOUR);
        }
	break;
      }
   }
}
//______________________________________________________________________________

 


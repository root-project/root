// @(#)root/ged:$Name:  $:$Id: TGraphEditor.cxx,
// Author: Carsten Hof   28/07/04

/*************************************************************************
 * Copyright (C) 1995-2004, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TGraphEditor                                                        //
//                                                                      //
//  Implements GUI for graph attributes.                                //
//     'Title': set the title of the graph                              //
//     Change the Shape of the graph:                                   //
//     'No Line'     = "": just draw unconnected points                 // 
//     'Simple Line' = "L":simple poly line between every point is drawn//
//     'Smooth Line' = "C":smooth curve is drawn                        //
//     'Bar Chart'   = "B": A bar chart is drawn at each point          //
//     'Fill Area'   = "F": A fill area is drawn                        //
//     Check box: 'Axis On/Off' Set graph axis visible/invisible        //
//     Check box: 'Marker On/Off' Set Marker visible/invisible          //
//////////////////////////////////////////////////////////////////////////
//Begin_Html
/*
<img src="gif/TGraphEditor.gif">
*/
//End_Html

#include "TGButton.h"
#include "TGButtonGroup.h"
#include "TGraphEditor.h"
#include "TGedFrame.h"
#include "TGTextEntry.h"
#include "TGToolTip.h"
#include "TGLabel.h"
#include "TGClient.h"
#include "TColor.h"
#include "TVirtualPad.h"
#include "TStyle.h"

ClassImp(TGraphEditor)

enum 
{ 
   kShape = 1,
   kSHAPE_NOLINE,
   kSHAPE_SIMPLE,
   kSHAPE_SMOOTH,
   kSHAPE_BAR,
   kSHAPE_FILL,
   kAXIS_ONOFF,
   kMARKER_ONOFF,
   kGRAPH_TITLE
};

//______________________________________________________________________________

TGraphEditor::TGraphEditor(const TGWindow *p, Int_t id, Int_t width,
                         Int_t height, UInt_t options, Pixel_t back)
   : TGedFrame(p, id, width, height, options | kVerticalFrame, back)
{
   // Constructor of graph attribute GUI.
   
   fGraph = 0;
   
   MakeTitle("Title");
   
   fTitlePrec = 2;
   fTitle = new TGTextEntry(this, new TGTextBuffer(50), kGRAPH_TITLE);
   fTitle->Resize(135, fTitle->GetDefaultHeight());
   fTitle->SetToolTipText("Enter the graph title string");
   AddFrame(fTitle, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));
   
   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kVerticalFrame);
   fgr = new TGButtonGroup(f2,3,1,3,0,"Shape");
   fShape = new TGRadioButton(fgr,"No Line",kSHAPE_NOLINE);
   fShape->SetToolTipText("The points are not connected by a line");
   fShape0 = new TGRadioButton(fgr,"Smooth Line  ",kSHAPE_SMOOTH);
   fShape0->SetToolTipText("Draw a smooth graph curve");
   fShape1 = new TGRadioButton(fgr,"Simple Line",kSHAPE_SIMPLE);
   fShape1->SetToolTipText("Draw a simple poly-line between the graph points");
   fShape2 = new TGRadioButton(fgr,"Bar Chart",kSHAPE_BAR);
   fShape2->SetToolTipText("Draw a bar chart at each graph point");  
   fShape3 = new TGRadioButton(fgr,"Fill area",kSHAPE_FILL);
   fShape3->SetToolTipText("A fill area is drawn");  

   fgr->SetLayoutHints(new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 0,5,3,0), fShape);
   fgr->Show();
   fgr->ChangeOptions(kFitWidth|kChildFrame|kVerticalFrame);
   f2->AddFrame(fgr, new TGLayoutHints(kLHintsCenterY | kLHintsLeft, 4, 1, 0, 0));
   fAxisOnOff = new TGCheckButton(f2,"Axis visible",kAXIS_ONOFF);
   fAxisOnOff->SetToolTipText("Make Axis visible/invisible");
   f2->AddFrame(fAxisOnOff, new TGLayoutHints(kLHintsTop, 5, 1, 0, 0)); 
   fMarkerOnOff = new TGCheckButton(f2,"Show Marker",kMARKER_ONOFF);
   fMarkerOnOff->SetToolTipText("Make Marker visible/invisible");
   f2->AddFrame(fMarkerOnOff, new TGLayoutHints(kLHintsTop, 5, 1, 0, 0));
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));   
   
   MapSubwindows();
   Layout();
   MapWindow();
      
   TClass *cl = TGraph::Class();
   TGedElement *ge = new TGedElement;
   ge->fGedFrame = this;
   ge->fCanvas = 0;
   cl->GetEditorList()->Add(ge);
}

//______________________________________________________________________________

TGraphEditor::~TGraphEditor()
{
   // Destructor of graph editor.

   TGFrameElement *el;
   TIter next(GetList());
   
   while ((el = (TGFrameElement *)next())) {
      if (!strcmp(el->fFrame->ClassName(), "TGCompositeFrame"))
         ((TGCompositeFrame *)el->fFrame)->Cleanup();
   }
   Cleanup();
}

//______________________________________________________________________________

void TGraphEditor::ConnectSignals2Slots()
{
   // Connect signals to slots.
 
   fTitle->Connect("TextChanged(const char *)", "TGraphEditor", this, "DoTitle(const char *)");;
   fShape->Connect("Pressed()","TGraphEditor",this,"DoShape()");
   fShape0->Connect("Pressed()","TGraphEditor",this,"DoShape0()");
   fShape1->Connect("Pressed()","TGraphEditor",this,"DoShape1()");
   fShape2->Connect("Pressed()","TGraphEditor",this,"DoShape2()");
   fShape3->Connect("Pressed()","TGraphEditor",this,"DoShape3()");   
   fAxisOnOff->Connect("Toggled(Bool_t)","TGraphEditor",this,"DoAxisOnOff()");
   fMarkerOnOff->Connect("Toggled(Bool_t)","TGraphEditor",this,"DoMarkerOnOff()");
   fInit = kFALSE;
}

//______________________________________________________________________________

void TGraphEditor::SetModel(TVirtualPad* pad, TObject* obj, Int_t)
{
   // Pick up the used values of graph attributes.
   
   fModel = 0;
   fPad = 0;

   if (obj == 0 || !obj->InheritsFrom("TGraph")) {
      SetActive(kFALSE);
      return;
   }

   fModel = obj;
   fPad = pad;
   fGraph = (TGraph *)(obj);

   const char *text = fGraph->GetTitle();
   fTitle->SetText(text);

   TString opt = GetDrawOption();
   opt.ToUpper();
   Int_t i=0;
   TString dum  = opt;
   Int_t l = opt.Length()-1;
   while (i < l) { 
      dum.Remove(dum.First(opt[i]),1);
      if (dum.Contains(opt[i])){ opt.Remove(opt.First(opt[i]),1); l--; i--;}
      i++;
   }
   if (opt.Contains("C")) {
      fgr->SetButton(kSHAPE_SMOOTH, kTRUE); 
      fDrawShape='C';
   }
   else if (opt.Contains("L")) {
      fgr->SetButton(kSHAPE_SIMPLE, kTRUE);  
      fDrawShape='L';
   }
   else if (opt.Contains("B")){
       fgr->SetButton(kSHAPE_BAR, kTRUE);  
       fDrawShape='B';
   }
   else if (opt.Contains("F")){
       fgr->SetButton(kSHAPE_FILL, kTRUE);  
       fDrawShape='F';
   }
   else {
      fgr->SetButton(kSHAPE_NOLINE, kTRUE); 
      fDrawShape=' ';
   }

   if (opt.Contains("A")) fAxisOnOff->SetState(kButtonDown); 
   else fAxisOnOff->SetState(kButtonUp);

   if (opt=="A" || opt=="AP" || opt=="PA" || opt == "P") {
         if (!opt.Contains("P")) opt +="P"; 
	 fMarkerOnOff->SetState(kButtonDisabled);
      }
   else if (opt.Contains("P")) fMarkerOnOff->SetState(kButtonDown);
   else fMarkerOnOff->SetState(kButtonUp);

   if (fInit) ConnectSignals2Slots();
   SetActive();
}

//______________________________________________________________________________

void TGraphEditor::DoTitle(const char *text)
{
   // Slot connected to the title of the graph.
  
   fGraph->SetTitle(text);
   Update();
}

//______________________________________________________________________________

void TGraphEditor::DoShape()
{
   // Slot connected to the draw options (no line, simple/smooth line, bar chart, fill area).
   
   TString opt = GetDrawOption();
   opt.ToUpper();
   Int_t first = opt.First(fDrawShape);
   if (first < 0) return;
   opt.Remove(first,1); 
   fDrawShape = ' ';
   if (opt=="A") opt += "P";
   fMarkerOnOff->SetState(kButtonDisabled);
   SetDrawOption(opt);
}

//______________________________________________________________________________

void TGraphEditor::DoShape0()
{
   // Slot connected to the draw options (no line, simple/smooth line, bar chart, fill area).
   
   TString opt = GetDrawOption();
   opt.ToUpper();
   if (opt.Contains("P")) fMarkerOnOff->SetState(kButtonDown);
   else fMarkerOnOff->SetState(kButtonUp);
   if (fDrawShape == ' ') opt +="C";
   else {
      Int_t first = opt.First(fDrawShape);
      if (first < 0) return;
      opt.Replace(first,1,'C'); 
      if (opt=="A" || opt=="AP" || opt=="PA" || opt == "P") {
         if (!opt.Contains("P")) opt +="P"; 
	 fMarkerOnOff->SetState(kButtonDisabled);
      }
   }
   fDrawShape = 'C';
   SetDrawOption(opt);  
}

//______________________________________________________________________________

void TGraphEditor::DoShape1()
{
   // Slot connected to the draw options (no line, simple/smooth line, bar chart, fill area).
   
   TString opt = GetDrawOption();
   opt.ToUpper();
   if (opt.Contains("P")) fMarkerOnOff->SetState(kButtonDown);
   else fMarkerOnOff->SetState(kButtonUp);
   if (fDrawShape == ' ') opt +="L";
   else {
      Int_t first = opt.First(fDrawShape);
      if (first < 0) return;
      opt.Replace(first,1,'L'); 
      if (opt=="A" || opt=="AP" || opt=="PA" || opt == "P") {
         if (!opt.Contains("P")) opt +="P"; 
	 fMarkerOnOff->SetState(kButtonDisabled);
      }
   }
   fDrawShape='L';
   SetDrawOption(opt);
}

//______________________________________________________________________________

void TGraphEditor::DoShape2()
{
   // Slot connected to the draw options (no line, simple/smooth line, bar chart, fill area).
   
   TString opt = GetDrawOption();
   opt.ToUpper();
   if (opt.Contains("P")) fMarkerOnOff->SetState(kButtonDown);
   else fMarkerOnOff->SetState(kButtonUp);
   if (fDrawShape == ' ') opt +="B";
   else { 
      Int_t first = opt.First(fDrawShape);
      if (first < 0) return;
      opt.Replace(first,1,'B'); 
      if (opt=="A" || opt=="AP" || opt=="PA" || opt == "P") {
         if (!opt.Contains("P")) opt +="P"; 
	 fMarkerOnOff->SetState(kButtonDisabled);
      }
   }
   fDrawShape='B';
   SetDrawOption(opt);
}

//______________________________________________________________________________

void TGraphEditor::DoShape3()
{
   // Slot connected to the draw options (no line, simple/smooth line, bar chart, fill area).
   
   TString opt = fGraph->GetDrawOption();
   opt.ToUpper();
   if (opt.Contains("P")) fMarkerOnOff->SetState(kButtonDown);
   else fMarkerOnOff->SetState(kButtonUp);
   if (fDrawShape == ' ') opt +="F";
   else {
      Int_t first = opt.First(fDrawShape);
      if (first < 0) return;
      opt.Replace(first,1,'F'); 
      if (opt=="A" || opt=="AP" || opt=="PA" || opt == "P") {
         if (!opt.Contains("P")) opt +="P"; 
	 fMarkerOnOff->SetState(kButtonDisabled);
      }
   }
   fDrawShape='F';
   SetDrawOption(opt);
}

//______________________________________________________________________________

void TGraphEditor::DoAxisOnOff()
{
   // Slot connected to axis: Set axis visible/invisible.
   
   TString t = GetDrawOption();
   t.ToUpper();
   if (fAxisOnOff->GetState()==kButtonDown) fGraph->SetDrawOption((t+="A"));
   else if (fAxisOnOff->GetState()==kButtonUp)fGraph->SetDrawOption((t.Remove(t.First("A"),1)));   
   Update();
}
   
//______________________________________________________________________________

void TGraphEditor::DoMarkerOnOff()
{
   // Slot connected to Marker: Set marker visible/invisible.
   
   TString t = GetDrawOption();
   t.ToUpper();
   if (fMarkerOnOff->GetState()==kButtonDown) {
      if (!t.Contains("P")) t+="P";
      fShape->SetState(kButtonEngaged);
   }
   else if (fMarkerOnOff->GetState()==kButtonUp) {
      while(t.Contains("P")) t.Remove(t.First("P"),1);
      fShape->SetState(kButtonDisabled);
   }   
   SetDrawOption(t);
}
   
//______________________________________________________________________________


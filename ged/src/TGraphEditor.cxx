// @(#)root/ged:$Name:  $:$Id: TGraphEditor.cxx,
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
//  TGraphEditor                                                        //
//                                                                      //
//  Implements GUI for graph attributes.                                //
//                                                                      //
//  Title': set the title of the graph                                  //
//  Change the Shape of the graph:                                      //
//     'No Line'     = " ": just draw unconnected points                // 
//     'Simple Line' = "L":simple poly line between every point is drawn//
//     'Smooth Line' = "C":smooth curve is drawn                        //
//     'Bar Chart'   = "B": A bar chart is drawn at each point          //
//     'Fill Area'   = "F": A fill area is drawn                        //
//  Check box: 'Marker On/Off' Set Marker visible/invisible             //
//                                                                      //
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
   kShape = 0,
   kSHAPE_NOLINE,
   kSHAPE_SMOOTH,
   kSHAPE_SIMPLE,
   kSHAPE_BAR,
   kSHAPE_FILL,
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

// TextEntry to change the title   
   MakeTitle("Title");
 
   fTitlePrec = 2;
   fTitle = new TGTextEntry(this, new TGTextBuffer(50), kGRAPH_TITLE);
   fTitle->Resize(135, fTitle->GetDefaultHeight());
   fTitle->SetToolTipText("Enter the graph title string");
// better take kLHintsLeft and Right - Right is not working at the moment
   AddFrame(fTitle, new TGLayoutHints(kLHintsLeft, 3, 1, 2, 5));

// Radio Buttons to change the draw options of the graph   
   TGCompositeFrame *f2 = new TGCompositeFrame(this, 80, 20, kVerticalFrame);
   fgr = new TGButtonGroup(f2,3,1,3,5,"Shape");
   fgr->SetRadioButtonExclusive(kTRUE);
   fShape = new TGRadioButton(fgr,"No Line",kSHAPE_NOLINE);   // no draw option
   fShape->SetToolTipText("The points are not connected by a line");
   fShape0 = new TGRadioButton(fgr,"Smooth Line  ",kSHAPE_SMOOTH);  // option C
   fShape0->SetToolTipText("Draw a smooth graph curve");
   fShape1 = new TGRadioButton(fgr,"Simple Line   ",kSHAPE_SIMPLE); // option L
   fShape1->SetToolTipText("Draw a simple poly-line between the graph points");
   fShape2 = new TGRadioButton(fgr,"Bar Chart",kSHAPE_BAR);         // option B
   fShape2->SetToolTipText("Draw a bar chart at each graph point");  
   fShape3 = new TGRadioButton(fgr,"Fill area",kSHAPE_FILL);        // option F
   fShape3->SetToolTipText("A fill area is drawn");  

   fgr->SetLayoutHints(new TGLayoutHints(kLHintsLeft, 0,3,0,0), fShape1);
   fgr->Show();
   fgr->ChangeOptions(kFitWidth|kChildFrame|kVerticalFrame);
   f2->AddFrame(fgr, new TGLayoutHints(kLHintsLeft, 4, 0, 0, 3));
// CheckBox to activate/deactivate the drawing of the Marker
   fMarkerOnOff = new TGCheckButton(f2,"Show Marker",kMARKER_ONOFF);
   fMarkerOnOff->SetToolTipText("Make Marker visible/invisible");
   f2->AddFrame(fMarkerOnOff, new TGLayoutHints(kLHintsTop, 5, 1, 0, 0));
   AddFrame(f2, new TGLayoutHints(kLHintsTop, 1, 1, 0, 0));   

// initialises the window layout   
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
 
   fTitle->Connect("TextChanged(const char *)","TGraphEditor",this,"DoTitle(const char *)");
   fgr->Connect("Pressed(Int_t)","TGraphEditor",this,"DoShape(Int_t)"); 
   fMarkerOnOff->Connect("Toggled(Bool_t)","TGraphEditor",this,"DoMarkerOnOff(Bool_t)");
   fInit = kFALSE;  // connect the slots to the signals only once
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
   fGraph = (TGraph *)fModel;

// set the Title TextEntry
   const char *text = fGraph->GetTitle();
   fTitle->SetText(text);

   TString opt = GetDrawOption();
   opt.ToUpper();
   Int_t i=0;
   Bool_t make=kFALSE;
// Remove characters which appear twice in the draw option   
   TString dum  = opt;
   Int_t l = opt.Length()-1;
   while (i < l) { 
      dum.Remove(dum.First(opt[i]),1);
      if (dum.Contains(opt[i])){ 
         opt.Remove(opt.First(opt[i]),1); 
         l--; 
         i--;
         make=kTRUE;
      }
      i++;
   }
// initialise the RadioButton group which shows the drawoption
   if (opt.Contains("C")) {
      fgr->SetButton(kSHAPE_SMOOTH, kTRUE); 
      fDrawShape='C';
   } else if (opt.Contains("L")) {
      fgr->SetButton(kSHAPE_SIMPLE, kTRUE);  
      fDrawShape='L';
   } else if (opt.Contains("B")){
      fgr->SetButton(kSHAPE_BAR, kTRUE);  
      fDrawShape='B';
   } else if (opt.Contains("F")){
      fgr->SetButton(kSHAPE_FILL, kTRUE);  
      fDrawShape='F';
   } else {
      fgr->SetButton(kSHAPE_NOLINE, kTRUE); 
      fDrawShape=' ';
   }
   if (make) SetDrawOption(opt);
// if the draw option is A, P, AP the P option cannot be removed, 
// we deactivate the CheckBox
// also initialising the MarkerOnOff checkbutton (== P option)
   if (opt=="A" || opt=="AP" || opt=="PA" || opt == "P") {
      if (!opt.Contains("P")) opt +="P"; 
	     fMarkerOnOff->SetState(kButtonDisabled);
   } else if (opt.Contains("P")) {
      fMarkerOnOff->SetState(kButtonDown);
   } else fMarkerOnOff->SetState(kButtonUp);

   if (fInit) ConnectSignals2Slots();
   SetActive();  // activates this Editor
}

//______________________________________________________________________________

void TGraphEditor::DoTitle(const char *text)
{
   // Slot connected to the title textentry
   // sets the Title of the graph
  
   fGraph->SetTitle(text);
   Update();
}

//______________________________________________________________________________

void TGraphEditor::DoShape(Int_t s)
{
   // Slot connected to the draw options.

   TString opt = GetDrawOption();
   opt.ToUpper();
   
   switch (s) {
      
      // change draw option to No Line:
      case kSHAPE_NOLINE: { 
         if (opt.Contains(fDrawShape)) 
            opt.Remove(opt.First(fDrawShape),1);
         fDrawShape = ' '; 
	        fMarkerOnOff->SetState(kButtonDisabled);
         break;
      }  
      
      // change draw option to Smooth Line (C)
      case kSHAPE_SMOOTH: { 
         if (fDrawShape == ' ') 
            opt +="C";
         else if (opt.Contains(fDrawShape)) 
            opt.Replace(opt.First(fDrawShape),1,'C'); 
	        fDrawShape = 'C';
	        break;
      }

      // change draw option to Simple Line (L) 
      case kSHAPE_SIMPLE: { 
         if (fDrawShape == ' ') 
            opt +="L";
         else if (opt.Contains(fDrawShape)) 
            opt.Replace(opt.First(fDrawShape),1,'L'); 
         fDrawShape='L';
         break;
      }

      // change draw option to Bar Chart (B) 
      case kSHAPE_BAR: { 
         if (fDrawShape == ' ') 
            opt +="B";
         else if (opt.Contains(fDrawShape)) 
            opt.Replace(opt.First(fDrawShape),1,'B'); 
         fDrawShape='B';
         break;
      }

      // change draw option to Fill Area (F) 
      case kSHAPE_FILL: { 
         if (fDrawShape == ' ') 
            opt +="F";
         else if (opt.Contains(fDrawShape)) 
            opt.Replace(opt.First(fDrawShape),1,'F'); 
         fDrawShape='F';
         break;
      }
   }

   if (gPad) gPad->GetVirtCanvas()->SetCursor(kWatch);
   gVirtualX->SetCursor(GetId(), gVirtualX->CreateCursor(kWatch));
// set/reset the Marker CheckBox
   if (opt.Contains("P")) 
      fMarkerOnOff->SetState(kButtonDown);
   else 
      fMarkerOnOff->SetState(kButtonUp);
   if (opt=="A" || opt=="AP" || opt=="PA" || opt == "P") {
      if (!opt.Contains("P")) 
         opt +="P"; 
      fMarkerOnOff->SetState(kButtonDisabled);
   }
   SetDrawOption(opt);
   if (gPad) gPad->GetVirtCanvas()->SetCursor(kPointer);
   gVirtualX->SetCursor(GetId(), gVirtualX->CreateCursor(kPointer));
}

//______________________________________________________________________________

void TGraphEditor::DoMarkerOnOff(Bool_t on)
{
   // Slot connected to MarkerOnOff CheckBox: 
   // Set marker visible/invisible.
   TString t = GetDrawOption();
   t.ToUpper();

   // showing the marker:
   if (on) {
      if  (!t.Contains("P")) t+="P";
      fShape->SetState(kButtonEngaged);
   } else {
   // remove the marker option P
      while (t.Contains("P")) t.Remove(t.First("P"),1);
      fShape->SetState(kButtonDisabled);
   }   
   SetDrawOption(t);
}

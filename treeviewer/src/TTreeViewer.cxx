// @(#)root/treeviewer:$Name$:$Id$
// Author: Rene Brun   08/12/98

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TSystem.h"
#include "TTreeViewer.h"
#include "TPaveVar.h"
#include "TLeaf.h"
#include "TH1.h"
#include "TSliderBox.h"
#include "Getline.h"


ClassImp(TTreeViewer)

//______________________________________________________________________________//*-*
//*-*   A TreeViewer is a canvas specialized to view TTree attributes.
//
// The TTreeViewer canvas (see example below) shows the list
//   of TTree variables represented by TPaveVar objects.
//   A set of buttons allows to execute the following operations:
//    "Draw"     display the active variable(s) below the X,Y,Z buttons
//    "Scan"     same with TTree::Scan output style
//    "Break"    interrupt current event loop
//    "List Out" to create a new TEventList using the current selection (see below)
//    "List In"  to activate a TEventList as an input selection
//    "Hist"     to redefine the default histogram (default htemp)
//    "Cpad"     to specify the output pad (default c1)
//    "Gopt"     to specify a graphics option
//    "X"        to select the X variable
//    "Y"        to select the Y variable
//    "Z"        to select the Z variable
//    "W"        to select the Weight/Selection expression
//    "Echo"     to toggle the current command echo in the terminal
//
//    A vertical slider can be used to select the events range (min/max)
//    When the "Draw" button is clicked, The TTreeViewer object assembles
//    the information from the above widgets and call TTree::Draw with
//    the corresponding arguments.
//    While the event loop is executing, a red box inside the slider bar
//    indicates the progress in the loop. The event loop can be interrupted
//    by clicking on the "Break" button.
//
//    A new variable can be created by clicking on the canvas with the right
//    button and selecting "CreateNewVar".
//    New variables can be created by ANDing/ORing a stack of overlapping
//    TPaveVars on the screen. Click on a TPaveVar and select "Merge".
//
//    The selection list TPaveVar (empty by default) can be edited
//    by clicking with the right button and selecting the item "SetLabel".
//
//    TPaveVar objects may be dragged/droped to their destination (X,Y,X
//    or W/Selection).
//    Clicking on the canvas and selecting "MakeClass" generates the C++ code
//    corresponding to the current selections in the canvas.
//
//    While the "Draw" button is executed, the event loop can be interrupted
//    by pressing the button "Break". The current histogram is shown.
//
//    While the "Draw" button is executed, one can display the status
//    of the histogram by clicking on the button "Hist".
//
//    Instead of clicking on the "Draw" button, one can also double-click
//    on a TPaveVar. This will automatically invoke "Draw" with the
//    current setup (cuts, events range, graphics options,etc).
//
//    At any time the canvas history can be saved in a Root file via the
//    standard tool bar menu "File" and item "Save as Canvas.root".
//    Assuming the current canvas is called TreeViewer, this generates
//    the file TreeViewer.root. One can continue a previous session via
//       Root > TFile f("TreeViewer.root");
//       Root > TreeViewer.Draw();
//
//    Assume an existing file f containing a TTree T, one can start
//    the TreeViewer with the sequence:
//       Root > TFile f("Event.root");
//       Root > TTreeViewer TV("T");
//
//Begin_Html
/*
<img src="gif/treeviewer.gif">
*/
//End_Html
//


//______________________________________________________________________________
TTreeViewer::TTreeViewer() : TCanvas()
{
//*-*-*-*-*-*-*-*-*-*-*-*TreeViewer default constructor*-*-*-*-*-*-*-*-*-*-*
//*-*                    ================================

   fTree    = 0;
   fDraw    = 0;
   fScan    = 0;
   fBreak   = 0;
   fGopt    = 0;
   fIList   = 0;
   fOList   = 0;
   fX       = 0;
   fY       = 0;
   fZ       = 0;
   fW       = 0;
   fHist    = 0;
   fRecord  = 0;
   fSlider  = 0;
   fTimer   = 0;
   SetTimerInterval(50);
   fRecordFlag = kTRUE;
}

//_____________________________________________________________________________
TTreeViewer::TTreeViewer(const char *treename, const char *title, UInt_t ww, UInt_t wh)
            : TCanvas(title,title,ww,wh)
{
//*-*-*-*-*-*-*-*-*-*-*-*TreeViewer constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                    ========================

   fTree    = 0;
   fDraw    = 0;
   fScan    = 0;
   fBreak   = 0;
   fGopt    = 0;
   fIList   = 0;
   fOList   = 0;
   fX       = 0;
   fY       = 0;
   fZ       = 0;
   fW       = 0;
   fHist    = 0;
   fRecord  = 0;
   fSlider  = 0;
   SetBorderMode(0);
   SetFillColor(21);
   SetTextSize(0.65);
   SetTextFont(61);
   SetTimerInterval(50);
   fRecordFlag = kTRUE;

   SetTreeName(treename);

   fTimer = new TTimer(this,fTimerInterval,kTRUE);
}

//______________________________________________________________________________
TTreeViewer::~TTreeViewer()
{
//*-*-*-*-*-*-*-*-*-*-*TreeViewer default destructor*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ===============================

   delete fDraw;
   delete fScan;
   delete fBreak;
   delete fGopt;
   delete fIList;
   delete fOList;
   delete fX;
   delete fY;
   delete fZ ;
   delete fW ;
   delete fHist;
   delete fSlider;
   delete fTimer;
   delete fRecord;
}

//______________________________________________________________________________
void TTreeViewer::BuildInterface()
{
//*-*-*-*-*-*-*-*-*Create all viewer control buttons*-*-*-*-*-*-*-*-*-*-*
//*-*              =================================

   // Clear viewer is buttons already drawn
   delete fDraw;
   delete fScan;
   delete fBreak;
   delete fGopt;
   delete fIList;
   delete fOList;
   delete fX;
   delete fY;
   delete fZ ;
   delete fW ;
   delete fHist;
   delete fSlider;
   delete fRecord;

   // compute optmum canvas size
   TObjArray *leaves = fTree->GetListOfLeaves();
   Int_t nleaves = leaves->GetEntries();
   Int_t npixvar = 75; //Number of pixels along x for each variable
   Int_t npiyvar = 25; //Number of pixels along y for each variable
   Int_t winh = GetWindowHeight();
   Int_t   nlines  = (winh -2*npiyvar)/npiyvar;
   if (nlines < 1) nlines = 1;
   Int_t   nl  = 1 + nleaves/nlines;
   Int_t winw = nl*npixvar+150;
   if (winw < (Int_t)GetWindowWidth()) winw = GetWindowWidth();
   SetCanvasSize(winw,winh);

   Clear();

   Int_t nbuttons = 10;
   Float_t dy  = 1/Float_t(nbuttons);
   Float_t dyb = 13./GetWh();  //choose 13 pixels for the button half height
   if (dyb > 0.45*dy) dyb = 0.45*dy;
   Float_t xbl = PixeltoX(25); // give 25 pixels for slider
   Float_t xbr = PixeltoX(25+2+50); // 25 for slider, 50 for buttons
   Float_t dxx = PixeltoX(2) - PixeltoX(0);
   Float_t dxm = PixeltoX(150) - PixeltoX(0);
   Float_t dxp = xbr-xbl;
   Float_t xin = 0.3*dxp;
   Float_t dyp = 0.6*dyb;
   Float_t y   = 0.5*dy;
   Float_t xll = 2*xbr - xbl;

   // Draw slider on left side of canvas
   fSlider = new TSlider("SlideEvents", "events",0.001,0.01,xbl-dxx,0.99);
   fSlider->SetFillColor(33);

   // Draw control buttons or functions paves
   char *action = new char[100];
   sprintf(action,"((TTreeViewer*)gROOT->GetListOfCanvases()->FindObject(\"%s\"))->ExecuteDraw(\"Scan\")",GetName());
   fScan = new TButton("Scan",action,xbl,y-dyb,xbr,y+dyb);
   fScan->SetName("Scan");
   fScan->SetFillColor(38);
   fScan->SetToolTipText("Scan Tree with current variables and selection");
   fScan->Draw(); y += dy;

   sprintf(action,"((TTreeViewer*)gROOT->GetListOfCanvases()->FindObject(\"%s\"))->ExecuteDraw()",GetName());
   fDraw = new TButton("Draw",action,xbl,y-dyb,xbr,y+dyb);
   fDraw->SetName("Draw");
   fDraw->SetFillColor(38);
   fDraw->SetToolTipText("Draw current variables in the selected pad");
   fDraw->Draw(); y += dy;

   TPaveVar *pl;
   fGopt = new TButton("Gopt","",xbl,y-dyb,xbr,y+dyb);
   fGopt->SetName("Gopt");
   fGopt->SetFillColor(16);
   fGopt->Draw();
   fGopt->SetToolTipText("To set the graphics options when drawing variables");
   pl = new TPaveVar(xbr-xin,y-dyp,xbr+dxp,y+dyp,"",this);
   pl->Draw();
   y += dy;

   fOList = new TButton("OList","",xbl,y-dyb,xbr,y+dyb);
   fOList->SetName("ListOut");
   fOList->SetFillColor(16);
   fOList->SetToolTipText("Output Selection List");
   fOList->Draw();
   pl = new TPaveVar(xbr-xin,y-dyp,xbr+dxp,y+dyp,"",this);
   pl->Draw();
   y += dy;

   sprintf(action,"((TTreeViewer*)gROOT->GetListOfCanvases()->FindObject(\"%s\"))->ExecuteDraw(\"Hist\")",GetName());
   fHist = new TButton("Hist",action,xbl,y-dyb,xbr,y+dyb);
   fHist->SetName("Hist");
   fHist->SetFillColor(16);
   fHist->SetToolTipText("To specify the name of the output histogram");
   fHist->Draw();
   pl = new TPaveVar(xbr-xin,y-dyp,xbr+dxp,y+dyp,"htemp",this);
   pl->Draw();
   y += dy;

   fZ = new TButton("Z","",xbl,y-dyb,xbr,y+dyb);
   fZ->SetName("Z");
   fZ->SetFillColor(45);
   fZ->SetToolTipText("Variable to be drawn along Z");
   fZ->Draw(); y += dy;

   fY = new TButton("Y","",xbl,y-dyb,xbr,y+dyb);
   fY->SetName("Y");
   fY->SetFillColor(45);
   fY->SetToolTipText("Variable to be drawn along Y");
   fY->Draw(); y += dy;

   fX = new TButton("X","",xbl,y-dyb,xbr,y+dyb);
   fX->SetName("X");
   fX->SetFillColor(45);
   fX->SetToolTipText("Variable to be drawn along X");
   fX->Draw(); y += dy;

   fIList = new TButton("IList","",xbl,y-dyb,xbr,y+dyb);
   fIList->SetName("IList");
   fIList->SetFillColor(16);
   fIList->SetToolTipText("Input Selection List");
   fIList->Draw();
   pl = new TPaveVar(xbr-xin,y-dyp,xbr+dxp,y+dyp,"",this);
   pl->Draw();
   y += dy;

   fBreak= new TButton("Break","gROOT->SetInterrupt()",xbl,y-dyb,xbr,y+dyb);
   fBreak->SetName("Break");
   fBreak->SetToolTipText("To stop the current transaction");
   fBreak->SetFillColor(38);
   fBreak->Draw(); y += dy;

   // Draw Selection/weight pave
   Float_t xbwl = xbr+0.1;
   Float_t xbwr = 0.99;
   y  = 0.1+dyb;
   fW = new TButton("Weight - Selection","",xbwl,y-dyb,xbwr,y+dyb);
   fW->SetName("W");
   fW->SetFillColor(45);
   fW->SetToolTipText("Place your selections under this box");
   fW->Draw();
   y += dyb-0.1*dyb;
   pl = new TPaveVar(xbwl+0.1,y,xbwr-0.1,y+2.5*dyp,"",this);
   pl->Draw();

   // Draw Command Text
   y  = 0.001+dyb;
   sprintf(action,"((TTreeViewer*)gROOT->GetListOfCanvases()->FindObject(\"%s\"))->ToggleRecordCommand()",GetName());
   fRecord = new TButton("Rec.",action,xbwl,y-dyb,xbwr,y+dyb);
   fRecord->SetName("Rec.");
   fRecord->SetFillColor(45);
   fRecord->SetToolTipText("Press here to toggle the recording of the Draw command");
   fRecord->Draw();
   delete [] action;

   // Draw leaves
   Float_t xlr = 0.99;
   Float_t dlx = (xlr-xll)/nl;
   if (dlx > dxm) dlx = dxm;
   Float_t dleaf = 0.8*dlx;
   xll += 0.2*dleaf;
   Int_t i, k = 0;
   y  = 0.98-dyp;
   dy = 3*dyp;
   for (i=0;i<nleaves;i++) {
      TLeaf *leaf = (TLeaf*)leaves->At(i);
      TBranch *branch = leaf->GetBranch();
      TString name = branch->GetName();
      if ( branch->GetNleaves() > 1) {
         name.Append(".").Append(leaf->GetName());
      }
      pl = new TPaveVar(xll+k*dlx,y-dyp,xll+k*dlx+dleaf,y+dyp,name,this);
      if (branch->InheritsFrom("TBranchObject")) {
         pl->SetFillColor(16);
         pl->SetBit(TPaveVar::kBranchObject);
      }
      pl->SetToolTipText("Double-Click to draw this variable");
      pl->Draw();
      k++;
      if (k == nl) {
         k = 0;
         y -= dy; if (y < 0.3) break;
      }
   }
   cd();
   Update();
}

//______________________________________________________________________________
TPaveVar *TTreeViewer::CreateNewVar(const char *varname)
{
//*-*-*-*-*-*-*-*-*Create a new variable TPaveVar*-*-*-*-*-*-*-*-*-*-*
//*-*              ==============================

   Int_t px  = GetEventX();
   Int_t py  = GetEventY();
   Float_t x = AbsPixeltoX(px);
   Float_t y = AbsPixeltoY(py);
   Float_t dyp = 0.6*13./GetWh();
   Float_t dxx = 0.6/8;
   TPaveVar *pl = new TPaveVar(x-dxx,y-dyp,x+dxx,y+dyp,varname,this);
   pl->Draw();
   return pl;
}

//______________________________________________________________________________
void TTreeViewer::ExecuteDraw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*Called when the DRAW button is executed*-*-*-*-*-*-*-*-*-*-*
//*-*              ========================================
//
// Look for TPaveVar objects below the action buttons X,Y,Z,W
// Check if an Entry range is given via the slider
// Check if a new histogram name is selected (instead of default "htemp")
// Check if there is an Output Event List or/and an Input Event List
//
// Special cases when option contains:
//   -"Break"    the event loop is interrupted. Show current histogram status
//   -"Hist"     show current histogram without interrupting the event loop
//   -"VarDraw:" User has double clicked on one variable
//
   TString opt = option;
   if (opt.Contains("Break")) {
      printf("Breaking event loop\n");
      return;
   }
   char *VarDraw = 0;
   if (opt.Contains("VarDraw:")) { //we double clicked on a TPaveVar
      VarDraw = (char*)strstr(option,":");
   }

   char varexp[80];
   varexp[0] = 0;
   // find label under X
   TPaveVar *plx = IsUnder(fX);
   // find label under Y
   TPaveVar *ply = IsUnder(fY);
   // find label under Z
   TPaveVar *plz = IsUnder(fZ);
   if (VarDraw) {
      plx = (TPaveVar*)GetListOfPrimitives()->FindObject(VarDraw+1);
      if (plx == 0) return;
      ply = 0;
      plz = 0;
   }
   if (plz) sprintf(varexp,"%s",plz->GetLabel());
   if (plz && (ply || plx)) strcat(varexp,":");
   if (ply) strcat(varexp,ply->GetLabel());
   if (ply && plx) strcat(varexp,":");
   if (plx) strcat(varexp,plx->GetLabel());
   // find ListIn
   fTree->SetEventList(0);
   TPaveVar *plin = IsUnder(fIList);
   TEventList *elist = 0;
   if (plin) {
      if (strlen(plin->GetLabel()) == 0) plin = 0;
      if (plin) elist = (TEventList*)gROOT->FindObject(plin->GetLabel());
      if (elist) fTree->SetEventList(elist);
   }
   // find ListOut
   TPaveVar *plout = IsUnder(fOList);
   if (plout) {
      if (strlen(plout->GetLabel()) == 0) plout = 0;
      if (plout) sprintf(varexp,">>%s",plout->GetLabel());
   }
   // find histogram name
   TPaveVar *plhist = IsUnder(fHist);
   char histname[80];
   strcpy(histname,"htemp");
   if (plout) plhist = 0;
   if (plhist && strcmp("htemp",plhist->GetLabel())) {
      strcat(varexp,">>");
      strcat(varexp,plhist->GetLabel());
      strcpy(histname,plhist->GetLabel());
   }
   // find selection/weight
   char select[500];
   select[0] = 0;
   TPaveVar *plw = IsUnderW(fW);
 //  if (plw) strncat(select,plw->GetLabel(),500);
   if (plw) sprintf(select,plw->GetLabel());

   // find graphics option
   char gopt[80];
   gopt[0] = 0;
   TPaveVar *plgopt = IsUnder(fGopt);
   if (plgopt) sprintf(gopt,"%s",plgopt->GetLabel());

   // find slider to get number of events and first event
   Float_t smin = fSlider->GetMinimum();
   Float_t smax = fSlider->GetMaximum();
   Int_t nentries = Int_t(fTree->GetEntries());
   Int_t nevents  = Int_t(nentries*(smax-smin));
   Int_t firstEntry = Int_t(nentries*smin);

   // find canvas/pad where to draw
   TPad *pad = (TPad*)gROOT->GetSelectedPad();
   if (pad == this) {
      pad = (TPad*)gROOT->GetListOfCanvases()->FindObject("c1");
   }
   if (pad) {
      pad->cd();
   } else {
      new TCanvas("c1");
   }

   gROOT->SetInterrupt(kFALSE); // just in case a BREAK had been set

   TH1 *hist = 0;
   if (opt.Contains("Hist")) {
      hist = fTree->GetHistogram();
      if (hist) {
         hist->Draw(gopt);
         gPad->Update();
      }
      return;
   }
   if (opt.Contains("Scan")) {
      fTree->Scan(varexp,select,gopt,nevents,firstEntry);
      return;
   }

   // Draw mode
   if (TestBit(kDrawExecuting)) return;
   SetBit(kDrawExecuting);
   fTree->SetTimerInterval(fTimerInterval);
   fTimer->TurnOn();
   fTree->Draw(varexp,select,gopt,nevents,firstEntry);
   HandleTimer(fTimer); //call necessary to show last slider status
   fTimer->TurnOff();
   fTree->SetTimerInterval(0);
   ResetBit(kDrawExecuting);
   if (gROOT->IsInterrupted()) {
      printf("Break event loop at event: %d\n",fTree->GetReadEntry());
   }
	
   // Print and save the draw command
   char command[512];
   const char *treeName = GetTreeName();
	
   // show the command on the Viewer
   sprintf(command,"%s->Draw(\"%s\", \"%s\", \"%s\", %d, %d);\n",
           treeName, varexp, select, gopt, nevents, firstEntry);

   // clear the label
   fRecord->SetTitle(command);
   fRecord->Modified();
   fRecord->Update();
   	
   // record the draw command if the fRecordFlag is on
   if (fRecordFlag) {
        // show the command on the command line
        printf("%s", command);

        // print the command to the history file
        Gl_histadd(command);
   }

   gPad->Update();
}

//__________________________________________________________
Bool_t TTreeViewer::HandleTimer(TTimer *timer)
{
// This function is called by the fTimer object
// Paint a vertical red box in the slider box to show the progress
// in the event loop

   if (TestBit(kDrawExecuting)) {
      Float_t smin = fSlider->GetMinimum();
      Float_t ymax = Float_t(fTree->GetReadEntry())/fTree->GetEntries();
      Int_t px1 = fSlider->XtoAbsPixel(0)+3;
      Int_t px2 = fSlider->XtoAbsPixel(1)-2;
      Int_t py1 = fSlider->YtoAbsPixel(smin)-1;
      Int_t py2 = fSlider->YtoAbsPixel(ymax);
      gVirtualX->SelectWindow(GetCanvasID());
      gVirtualX->SetFillColor(kRed);
      gVirtualX->DrawBox(px1,py1,px2,py2,TVirtualX::kFilled);
      gVirtualX->UpdateWindow(1);
      gPad->SetCursor(kWatch);
   }

   timer->Reset();
   return kFALSE;
}

//__________________________________________________________
TPaveVar *TTreeViewer::IsUnder(TButton *button)
{
   //look for a TPaveVar under button
   Float_t xmin = button->GetXlowNDC();
   Float_t xmax = xmin + button->GetWNDC();
   Float_t ymin = button->GetYlowNDC();
   Float_t ymax = ymin + button->GetHNDC();
   TIter next(GetListOfPrimitives());
   TPaveVar *pl;
   TObject *obj;
   while ((obj=next())) {
      if (obj->InheritsFrom(TPaveVar::Class())) {
         pl = (TPaveVar*)obj;
         if (pl->GetX1() < xmax) {
            Float_t ymean = 0.5*(pl->GetY1()+pl->GetY2());
            if (ymean > ymin && ymean < ymax) return pl;
         }
      }
   }
   return 0;
}

//__________________________________________________________
TPaveVar *TTreeViewer::IsUnderW(TButton *button)
{
   //look for a TPaveVar under button W
   Float_t xmin = button->GetXlowNDC();
   Float_t xmax = xmin + button->GetWNDC();
   Float_t ymin = button->GetYlowNDC();
   Float_t ymax = ymin + button->GetHNDC();
   TIter next(GetListOfPrimitives());
   TPaveVar *pl;
   TObject *obj;
   while ((obj=next())) {
      if (obj->InheritsFrom(TPaveVar::Class())) {
         pl = (TPaveVar*)obj;
         if (pl->GetY1() < ymax && pl->GetY2() > ymax) {
            Float_t xmean = 0.5*(pl->GetX1()+pl->GetX2());
            if (xmean > xmin && xmean < xmax) return pl;
         }
      }
   }
   return 0;
}

//______________________________________________________________________________
void TTreeViewer::MakeClass(const char *classname)
{
//*-*-*-*-*-*-*-*-*Create all viewer control buttons*-*-*-*-*-*-*-*-*-*-*
//*-*              =================================

   printf("TTreeViewer::MakeClass for %s not yet implemented\n",classname);
}

//______________________________________________________________________________
void TTreeViewer::Reorganize()
{

   BuildInterface();

}

//______________________________________________________________________________
void TTreeViewer::SetTreeName(const char *treename)
{
   // Set the current TTree to treename


   fTree = (TTree*)gROOT->FindObject(treename);
   if (!fTree) {
      printf("ERROR: Cannot find TTree with name: %s in current file\n",treename);
      return;
   }
   fTreeName = treename;
   Int_t nch = 30;
   nch  += strlen(fTree->GetName())+strlen(fTree->GetTitle());
   char *name = new char[nch];
   sprintf(name,"%s","TreeViewer: ");
   sprintf(&name[13],"%s : %s",fTree->GetName(),fTree->GetTitle());
   SetTitle(name);
   delete [] name;
   fIsEditable = kFALSE;

   BuildInterface();
}

//______________________________________________________________________________
void TTreeViewer::Streamer(TBuffer &R__b)
{
   // Stream an object of class TTreeViewer.

   if (R__b.IsReading()) {
      Version_t R__v = R__b.ReadVersion(); if (R__v) { }
      TCanvas::Streamer(R__b);
      TAttText::Streamer(R__b);
      fTreeName.Streamer(R__b);
      R__b >> fDraw;
      R__b >> fScan;
      R__b >> fBreak;
      R__b >> fGopt;
      R__b >> fIList;
      R__b >> fOList;
      R__b >> fX;
      R__b >> fY;
      R__b >> fZ;
      R__b >> fW;
      R__b >> fHist;
      R__b >> fSlider;
      R__b >> fRecord;
      fTimer = new TTimer(this,40,kTRUE);
   } else {
      R__b.WriteVersion(TTreeViewer::IsA());
      TCanvas::Streamer(R__b);
      TAttText::Streamer(R__b);
      fTreeName.Streamer(R__b);
      R__b << fDraw;
      R__b << fScan;
      R__b << fBreak;
      R__b << fGopt;
      R__b << fIList;
      R__b << fOList;
      R__b << fX;
      R__b << fY;
      R__b << fZ;
      R__b << fW;
      R__b << fHist;
      R__b << fSlider;
      R__b << fRecord;
   }
}

//______________________________________________________________________________
void TTreeViewer::ToggleRecordCommand()
{
//*-*-*-*-*-*-*-*-*Toggle the recording of the Draw command*-*-*-*-*-*-*-*-*-*-*
//*-*              ========================================
  fRecordFlag = !fRecordFlag;
  if (!fRecordFlag){
     fRecord->SetFillColor(16);
  } else {
     fRecord->SetFillColor(45);
  }
}


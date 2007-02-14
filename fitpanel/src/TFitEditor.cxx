// @(#)root/fitpanel:$Name:  $:$Id: TFitEditor.cxx,v 1.24 2007/02/09 09:58:40 antcheva Exp $
// Author: Ilka Antcheva, Lorenzo Moneta 10/08/2006

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TFitEditor                                                           //
//                                                                      //
// Allows to perform, explore and compare various fits.                 //
//                                                                      //
// To display the new Fit panel interface right click on a histogram    //
// or a graph to pop up the context menu and then select the menu       //
// entry 'Fit Panel'.                                                   //
//                                                                      //
// "General" Tab                                                        //
//                                                                      //
// The first set of GUI elements is related to the function choice      //
// and settings. The status bar on the bottom provides information      //
// about the current minimization settings using the following          //
// abbreviations:                                                       //
// LIB - shows the current choice between Minuit/Minuit2/Fumili         //
// MIGRAD or FUMILI points to the current minimization method in use.   //
// Itr: - shows the maximum number of iterations nnnn set for the fit.  //
// Prn: - can be DEF/VER/QT and shows the current print option in use.  //
//                                                                      //
// "Predefined" combo box - contains a list of predefined functions     //
// in ROOT. The default one is Gaussian.                                //
//                                                                      //
// "Operation" radio button group defines selected operational mode     //
// between functions: NOP - no operation (default); ADD - addition      //
// CONV - convolution (will be implemented in the future).              //
//                                                                      //
// Users can enter the function expression in a text entry field.       //
// The entered string is checked after Enter key was pressed. An        //
// error message shows up if the string is not accepted. The current    //
// prototype is limited and users have no freedom to enter file/user    //
// function names in this field.                                        //
//                                                                      //
// "Set Parameters" button opens a dialog for parameters settings.      //
//                                                                      //
// "Fit Settings" provides user interface elements related to the       //
// fitter. Currently there are two method choices: Chi-square and       //
// Binned Likelihood.                                                   //
//                                                                      //
// "Linear Fit" check button sets the use of Linear fitter is it is     //
// selected. Otherwise the option 'F' is applied if polN is selected.   //
// "Robust" number entry sets the robust value when fitting graphs.     //
// "No Chi-square" check button sets ON/OFF option 'C' - do not         //
// calculate Chi-square (for Linear fitter).                            //
//                                                                      //
// Fit options:                                                         //
// "Integral" check button switch ON/OFF option 'I' - use integral      //
// of function instead of value in bin center.                          //
// "Best Errors" sets ON/OFF option 'E' - better errors estimation      //
// using Minos technique.                                               //
// "All weights = 1" sets ON/OFF option 'W' - all weights set to 1,     //
// excluding empty bins and ignoring error bars.                        //
// "Empty bins, weights=1" sets ON/OFF option 'WW' -  all weights       //
// equal to 1, including  empty bins, error bars ignored.               //
// "Use range" sets ON/OFF option 'R' - fit only data within the        //
// specified function range with the slider.                            //
// "Improve fit results" sets ON/OFF option 'M' - after minimum is      //
// found, search for a new one.                                         //
// "Add to list" sets On/Off option '+'- add function to the list       //
// without deleting the previous.                                       //
//                                                                      //
// Draw options:                                                        //
// "SAME" sets On/Off function drawing on the same pad.                 //
// "No drawing" sets On/Off option '0'- do not draw function graphics.  //
// "Do not store/draw" sets On/Off option 'N'- do not store the         //
// function, do not draw it.                                            //
//                                                                      //
// Sliders settings are used if option 'R' - use range is active.       //
// Users can change min/max values by pressing the left mouse button    //
// near to the left/right slider edges. It is possible o change both    //
// values simultaneously by pressing the left mouse button near to its  //
// center and moving it to a new desire position.                       //
//                                                                      //
// "Minimization" Tab                                                   //
//                                                                      //
// "Library" group allows you to use Minuit, Minuit2 or Fumili          //
// minimization packages for your fit.                                  //
//  "Minuit" - the popular Minuit minimization package.                 //
//  "Minuit2" - a new object-oriented implementation of Minuit in C++.  //
//  "Fumili" - the popular Fumili minimization package.                 //
//                                                                      //
// "Method" group has currently restricted functionality.               //
//  "MIGRAD" method is available for Minuit and Minuit2                 //
//  "FUMILI" method is available for Fumili and Minuit2                 //
//  "SIMPLEX" method is disabled (will come with the new fitter design) //
//                                                                      //
// "Minimization Settings' group allows users to set values for:        //
//  "Error definition" - between 0.0 and 100.0  (default is 1.0).       //
//  "Maximum tolerance" - the fit relative precision in use.            //
//  "Maximum number of iterations" - default is 5000.                   //
//                                                                      //
// Print options:                                                       //
//  "Default" - between Verbose and Quiet.                              //
//  "Verbose" - prints results after each iteration.                    //
//  "Quiet" - no fit information is printed.                            //
//                                                                      //
// Fit button - performs a fit.                                         //
// Reset - resets all GUI elements and related fit settings to the      //
// default ones.                                                        //
// Close - closes this window.                                          //
//                                                                      //
// Begin_Html                                                           //
/*
<img src="gif/TFitEditor.gif">
*/
//End_Html
//////////////////////////////////////////////////////////////////////////

#include "TFitEditor.h"
#include "TROOT.h"
#include "TClass.h"
#include "TCanvas.h"
#include "TGTab.h"
#include "TGLabel.h"
#include "TG3DLine.h"
#include "TGComboBox.h"
#include "TGTextEntry.h"
#include "TGFont.h"
#include "TGGC.h"
#include "TGButtonGroup.h"
#include "TGNumberEntry.h"
#include "TGDoubleSlider.h"
#include "TGStatusBar.h"
#include "TFitParametersDialog.h"
#include "TGMsgBox.h"
#include "TAxis.h"
#include "TGraph.h"
#include "TH1.h"
#include "TF1.h"
#include "TTimer.h"
#include "THStack.h"
#include "TVirtualFitter.h"

enum EFitPanel {
   kFP_FLIST, kFP_GAUS,  kFP_GAUSN, kFP_EXPO,  kFP_LAND,  kFP_LANDN,
   kFP_POL0,  kFP_POL1,  kFP_POL2,  kFP_POL3,  kFP_POL4,  kFP_POL5,
   kFP_POL6,  kFP_POL7,  kFP_POL8,  kFP_POL9,  kFP_USER,
   kFP_NONE,  kFP_ADD,   kFP_CONV,  kFP_FILE,  kFP_PARS,  kFP_RBUST, kFP_EMPW1,
   kFP_INTEG, kFP_IMERR, kFP_USERG, kFP_ADDLS, kFP_ALLW1, kFP_IFITR, kFP_NOCHI,
   kFP_MLIST, kFP_MCHIS, kFP_MBINL, kFP_MUBIN, kFP_MUSER, kFP_MLINF, kFP_MUSR,
   kFP_DSAME, kFP_DNONE, kFP_DADVB, kFP_DNOST, kFP_PDEF,  kFP_PVER,  kFP_PQET,
   kFP_XMIN,  kFP_XMAX,  kFP_YMIN,  kFP_YMAX,  kFP_ZMIN,  kFP_ZMAX,
   
   kFP_LMIN,  kFP_LMIN2, kFP_LFUM,  kFP_MIGRAD,kFP_SIMPLX,kFP_FUMILI,
   kFP_MERR,  kFP_MTOL,  kFP_MITR,
   
   kFP_FIT,   kFP_RESET, kFP_CLOSE
};

ClassImp(TFitEditor)

TFitEditor *TFitEditor::fgFitDialog = 0;

//______________________________________________________________________________
void TFitEditor::Open(TVirtualPad* pad, TObject *obj)
{
   // Static method - opens the fit panel.

   if (!fgFitDialog) {
      TFitEditor::GetFP() = new TFitEditor(pad, obj);
   } else {
      fgFitDialog->Show(pad, obj);
   }
}

//______________________________________________________________________________
TFitEditor::TFitEditor(TVirtualPad* pad, TObject *obj) :
   TGMainFrame(gClient->GetRoot(), 20, 20),
   fCanvas      (0),
   fParentPad   (0),
   fFitObject   (0),
   fDim         (0),
   fXaxis       (0),
   fYaxis       (0),
   fZaxis       (0),
   fXmin        (0),
   fXmax        (0),
   fYmin        (0),
   fYmax        (0),
   fZmin        (0),
   fZmax        (0),
   fPlus        ('+'),
   fFunction    (""),
   fFitOption   ("R"),
   fDrawOption  (""),
   fFitFunc     (0)

{
   // Constructor of fit editor.

   SetCleanup(kDeepCleanup);

   TString name = obj->GetName();
   name.Append("::");
   name.Append(obj->ClassName());
   fObjLabelParent = new TGHorizontalFrame(this, 80, 20);
   TGLabel *label = new TGLabel(fObjLabelParent,"Current selection: ");
   fObjLabelParent->AddFrame(label, new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   fObjLabel = new TGLabel(fObjLabelParent, Form("%s", name.Data()));
   fObjLabelParent->AddFrame(fObjLabel, new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   AddFrame(fObjLabelParent, new TGLayoutHints(kLHintsTop, 1, 1, 10, 10));
   // set red color for the name
   Pixel_t color;
   gClient->GetColorByName("#ff0000", color);
   fObjLabel->SetTextColor(color, kFALSE);
   fObjLabel->SetTextJustify(kTextLeft);

   fTab = new TGTab(this, 10, 10);
   AddFrame(fTab, new TGLayoutHints(kLHintsExpandY | kLHintsExpandX));
   fTab->SetCleanup(kDeepCleanup);
   fTab->Associate(this);
   
   TGHorizontalFrame *cf1 = new TGHorizontalFrame(this, 250, 20, kFixedWidth);
   cf1->SetCleanup(kDeepCleanup);
   fFitButton = new TGTextButton(cf1, "&Fit", kFP_FIT);
   fFitButton->Associate(this);
   cf1->AddFrame(fFitButton, new TGLayoutHints(kLHintsTop |
                                               kLHintsExpandX, 2, 2, 2, 2));

   fResetButton = new TGTextButton(cf1, "&Reset", kFP_RESET);
   fResetButton->Associate(this);
   cf1->AddFrame(fResetButton, new TGLayoutHints(kLHintsTop |
                                                 kLHintsExpandX, 3, 2, 2, 2));

   fCloseButton = new TGTextButton(cf1, "&Close", kFP_CLOSE);
   fCloseButton->Associate(this);
   cf1->AddFrame(fCloseButton, new TGLayoutHints(kLHintsTop |
                                                 kLHintsExpandX, 3, 2, 2, 2));
   AddFrame(cf1, new TGLayoutHints(kLHintsNormal |
                                   kLHintsRight, 0, 5, 5, 5));

   // Create status bar
   int parts[] = { 20, 20, 20, 20, 20 };
   fStatusBar = new TGStatusBar(this, 10, 10);
   fStatusBar->SetParts(parts, 5);
   AddFrame(fStatusBar, new TGLayoutHints(kLHintsBottom | 
                                          kLHintsLeft   | 
                                          kLHintsExpandX));

   CreateGeneralTab();
   CreateMinimizationTab();

   gROOT->GetListOfCleanups()->Add(this);

   MapSubwindows();
   // not ready yet for 2 & 3 dim
   fGeneral->HideFrame(fSliderYParent);
   fGeneral->HideFrame(fSliderZParent);

   // do not allow resizing
   TGDimension size = GetDefaultSize();
   SetWMSize(size.fWidth, size.fHeight);
   SetWMSizeHints(size.fWidth, size.fHeight, size.fWidth, size.fHeight, 0, 0);
   SetWindowName("New Fit Panel");
   SetIconName("New Fit Panel");
   SetClassHints("New Fit Panel", "New Fit Panel");

   SetMWMHints(kMWMDecorAll | kMWMDecorResizeH  | kMWMDecorMaximize |
                              kMWMDecorMinimize | kMWMDecorMenu,
               kMWMFuncAll  | kMWMFuncResize    | kMWMFuncMaximize |
                              kMWMFuncMinimize,
               kMWMInputModeless);
   if (pad && obj) {
      fParentPad = (TPad *)pad;
      fFitObject = (TObject *)obj;
      fDrawOption = GetDrawOption();
      SetCanvas(pad->GetCanvas());
      pad->GetCanvas()->Selected(pad, obj, kButton1Down);
   } else {
      Error("FitPanel", "need to have an object drawn first");
      return;
   }
   UInt_t dw = fClient->GetDisplayWidth();
   UInt_t cw = pad->GetCanvas()->GetWindowWidth();
   UInt_t cx = (UInt_t)pad->GetCanvas()->GetWindowTopX();
   UInt_t cy = (UInt_t)pad->GetCanvas()->GetWindowTopY();

   if (cw + size.fWidth < dw) {
      Int_t gedx = 0, gedy = 0;
      gedx = cx+cw+4;
      gedy = cy-20;
      MoveResize(gedx, gedy,size.fWidth, size.fHeight);
      SetWMPosition(gedx, gedy);
   } 
   
   Resize(size);
   MapWindow();
   gVirtualX->RaiseWindow(GetId());
}

//______________________________________________________________________________
TFitEditor::~TFitEditor()
{
   // Fit editor destructor.

   DisconnectSlots();
   fCloseButton->Disconnect("Clicked()");
   TQObject::Disconnect("TCanvas", "Selected(TVirtualPad *, TObject *, Int_t)");
   gROOT->GetListOfCleanups()->Remove(this);

   if (fFitFunc) delete fFitFunc;
   Cleanup();
   delete fLayoutNone;
   delete fLayoutAdd;
   delete fLayoutConv;
   fgFitDialog = 0;
}

//______________________________________________________________________________
void TFitEditor::CreateGeneralTab()
{
   // Create 'General' tab.
   
   fTabContainer = fTab->AddTab("General");
   fGeneral = new TGCompositeFrame(fTabContainer, 10, 10, kVerticalFrame);
   fTabContainer->AddFrame(fGeneral, new TGLayoutHints(kLHintsTop |
                                                       kLHintsExpandX,
                                                       5, 5, 2, 2));

   TGGroupFrame *gf1 = new TGGroupFrame(fGeneral, "Function", kFitWidth);
   TGCompositeFrame *tf1 = new TGCompositeFrame(gf1, 350, 26,
                                                kHorizontalFrame);
   TGVerticalFrame *tf11 = new TGVerticalFrame(tf1);
   TGLabel *label1 = new TGLabel(tf11,"Predefined:");
   tf11->AddFrame(label1, new TGLayoutHints(kLHintsNormal, 0, 0, 2, 0));
   fFuncList = BuildFunctionList(tf11, kFP_FLIST);
   fFuncList->Resize(80, 20);
   fFuncList->Select(1, kFALSE);

   TGListBox *lb = fFuncList->GetListBox();
   lb->Resize(lb->GetWidth(), 200);
   tf11->AddFrame(fFuncList, new TGLayoutHints(kLHintsNormal, 0, 0, 5, 0));
   fFuncList->Associate(this);
   tf1->AddFrame(tf11);

   TGHButtonGroup *bgr = new TGHButtonGroup(tf1,"Operation");
   bgr->SetRadioButtonExclusive();
   fNone = new TGRadioButton(bgr, "Nop", kFP_NONE);
   fNone->SetToolTipText("No operation defined");
   fNone->SetState(kButtonDown, kFALSE);
   fAdd = new TGRadioButton(bgr, "Add", kFP_ADD);
   fAdd->SetToolTipText("Addition");
   fConv = new TGRadioButton(bgr, "Conv", kFP_CONV);
   fConv->SetToolTipText("Convolution (not implemented yet)");
   fConv->SetState(kButtonDisabled);
   fLayoutNone = new TGLayoutHints(kLHintsLeft,0,5,3,-10);
   fLayoutAdd  = new TGLayoutHints(kLHintsLeft,10,5,3,-10);
   fLayoutConv = new TGLayoutHints(kLHintsLeft,10,5,3,-10);
   bgr->SetLayoutHints(fLayoutNone,fNone);
   bgr->SetLayoutHints(fLayoutAdd,fAdd);
   bgr->SetLayoutHints(fLayoutConv,fConv);
   bgr->Show();
   bgr->ChangeOptions(kFitWidth | kHorizontalFrame);
   tf1->AddFrame(bgr, new TGLayoutHints(kLHintsNormal, 15, 0, 3, 0));

   gf1->AddFrame(tf1, new TGLayoutHints(kLHintsNormal | kLHintsExpandX));

   TGCompositeFrame *tf2 = new TGCompositeFrame(gf1, 350, 26,
                                                kHorizontalFrame);
   fEnteredFunc = new TGTextEntry(tf2, new TGTextBuffer(50), kFP_FILE);
   fEnteredFunc->SetMaxLength(250);
   fEnteredFunc->SetAlignment(kTextLeft);
   TGTextLBEntry *te = (TGTextLBEntry *)fFuncList->GetSelectedEntry();
   fFunction = te->GetTitle();
   fEnteredFunc->SetText(fFunction.Data());
   fEnteredFunc->SetToolTipText("Enter file_name/function_name or a function expression");
   fEnteredFunc->Resize(250,fEnteredFunc->GetDefaultHeight());
   tf2->AddFrame(fEnteredFunc, new TGLayoutHints(kLHintsLeft    |
                                                 kLHintsCenterY |
                                                 kLHintsExpandX, 2, 2, 2, 2));
   gf1->AddFrame(tf2, new TGLayoutHints(kLHintsNormal |
                                        kLHintsExpandX, 0, 0, 2, 0));

   TGHorizontalFrame *s1 = new TGHorizontalFrame(gf1);
   TGLabel *label21 = new TGLabel(s1, "Selected: ");
   s1->AddFrame(label21, new TGLayoutHints(kLHintsNormal |
                                           kLHintsCenterY, 2, 2, 2, 0));
   TGHorizontal3DLine *hlines = new TGHorizontal3DLine(s1);
   s1->AddFrame(hlines, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX));
   gf1->AddFrame(s1, new TGLayoutHints(kLHintsExpandX));

   TGCompositeFrame *tf4 = new TGCompositeFrame(gf1, 350, 26,
                                                kHorizontalFrame);
   TGTextLBEntry *txt = (TGTextLBEntry *)fFuncList->GetSelectedEntry();
   fSelLabel = new TGLabel(tf4, Form("%s", txt->GetTitle()));
   tf4->AddFrame(fSelLabel, new TGLayoutHints(kLHintsNormal |
                                              kLHintsCenterY, 0, 6, 2, 0));
   Pixel_t color;
   gClient->GetColorByName("#336666", color);
   fSelLabel->SetTextColor(color, kFALSE);
   TGCompositeFrame *tf5 = new TGCompositeFrame(tf4, 120, 20,
                                                kHorizontalFrame | kFixedWidth);
   fSetParam = new TGTextButton(tf5, "Set Parameters...", kFP_PARS);
   tf5->AddFrame(fSetParam, new TGLayoutHints(kLHintsRight   |
                                              kLHintsCenterY |
                                              kLHintsExpandX));
   fSetParam->SetToolTipText("Open a dialog for parameter(s) settings");
   tf4->AddFrame(tf5, new TGLayoutHints(kLHintsRight |
                                        kLHintsTop, 5, 0, 2, 2));
   gf1->AddFrame(tf4, new TGLayoutHints(kLHintsNormal |
                                             kLHintsExpandX, 5, 0, 0, 0));

   fGeneral->AddFrame(gf1, new TGLayoutHints(kLHintsExpandX, 5, 5, 0, 0));


   // 'options' group frame
   TGGroupFrame *gf = new TGGroupFrame(fGeneral, "Fit Settings", kFitWidth);

   // 'method' sub-group
   TGHorizontalFrame *h1 = new TGHorizontalFrame(gf);
   TGLabel *label4 = new TGLabel(h1, "Method");
   h1->AddFrame(label4, new TGLayoutHints(kLHintsNormal |
                                          kLHintsCenterY, 2, 2, 0, 0));
   TGHorizontal3DLine *hline1 = new TGHorizontal3DLine(h1);
   h1->AddFrame(hline1, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX));
   gf->AddFrame(h1, new TGLayoutHints(kLHintsExpandX));

   TGHorizontalFrame *h2 = new TGHorizontalFrame(gf);
   TGVerticalFrame *v1 = new TGVerticalFrame(h2);
   fMethodList = BuildMethodList(v1, kFP_MLIST);
   fMethodList->Select(1, kFALSE);
   fMethodList->Resize(130, 20);
   lb = fMethodList->GetListBox();
   Int_t lbe = lb->GetNumberOfEntries();
   lb->Resize(lb->GetWidth(), lbe*16);
   v1->AddFrame(fMethodList, new TGLayoutHints(kLHintsLeft, 0, 0, 2, 5));

   fLinearFit = new TGCheckButton(v1, "Linear fit", kFP_MLINF);
   fLinearFit->Associate(this);
   fLinearFit->SetToolTipText("Perform Linear fitter if selected");
   v1->AddFrame(fLinearFit, new TGLayoutHints(kLHintsNormal, 0, 0, 2, 2));

   TGHorizontalFrame *v1h = new TGHorizontalFrame(v1);
   TGLabel *label41 = new TGLabel(v1h, "Robust:");
   v1h->AddFrame(label41, new TGLayoutHints(kLHintsNormal |
                                            kLHintsCenterY, 25, 5, 5, 2));
   fRobustValue = new TGNumberEntry(v1h, 1., 5, kFP_RBUST,
                                    TGNumberFormat::kNESRealTwo,
                                    TGNumberFormat::kNEAPositive,
                                    TGNumberFormat::kNELLimitMinMax,0.,1.);
   v1h->AddFrame(fRobustValue, new TGLayoutHints(kLHintsLeft));
   v1->AddFrame(v1h, new TGLayoutHints(kLHintsNormal));
   fRobustValue->SetState(kFALSE);
   fRobustValue->GetNumberEntry()->SetToolTipText("Available only for graphs");

   h2->AddFrame(v1, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   TGVerticalFrame *v2 = new TGVerticalFrame(h2);
   TGCompositeFrame *v21 = new TGCompositeFrame(v2, 120, 20,
                                                kHorizontalFrame | kFixedWidth);
   fUserButton = new TGTextButton(v21, "User-Defined...", kFP_MUSR);
   v21->AddFrame(fUserButton, new TGLayoutHints(kLHintsRight   |
                                                kLHintsCenterY |
                                                kLHintsExpandX));
   fUserButton->SetToolTipText("Open a dialog for entering a user-defined method");
   fUserButton->SetState(kButtonDisabled);
   v2->AddFrame(v21, new TGLayoutHints(kLHintsRight | kLHintsTop));

   fNoChi2 = new TGCheckButton(v2, "No Chi-square", kFP_NOCHI);
   fNoChi2->Associate(this);
   fNoChi2->SetToolTipText("'C'- do not calculate Chi-square (for Linear fitter)");
   v2->AddFrame(fNoChi2, new TGLayoutHints(kLHintsNormal, 0, 0, 34, 2));

   h2->AddFrame(v2, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 20, 0, 0, 0));
   gf->AddFrame(h2, new TGLayoutHints(kLHintsExpandX, 20, 0, 0, 0));

   // 'fit option' sub-group
   TGHorizontalFrame *h3 = new TGHorizontalFrame(gf);
   TGLabel *label5 = new TGLabel(h3, "Fit Options");
   h3->AddFrame(label5, new TGLayoutHints(kLHintsNormal |
                                          kLHintsCenterY, 2, 2, 0, 0));
   TGHorizontal3DLine *hline2 = new TGHorizontal3DLine(h3);
   h3->AddFrame(hline2, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX));
   gf->AddFrame(h3, new TGLayoutHints(kLHintsExpandX));

   TGHorizontalFrame *h = new TGHorizontalFrame(gf);
   TGVerticalFrame *v3 = new TGVerticalFrame(h);
   fIntegral = new TGCheckButton(v3, "Integral", kFP_INTEG);
   fIntegral->Associate(this);
   fIntegral->SetToolTipText("'I'- use integral of function instead of value in bin center");
   v3->AddFrame(fIntegral, new TGLayoutHints(kLHintsNormal, 0, 0, 2, 2));

   fBestErrors = new TGCheckButton(v3, "Best errors", kFP_IMERR);
   fBestErrors->Associate(this);
   fBestErrors->SetToolTipText("'E'- better errors estimation using Minos technique");
   v3->AddFrame(fBestErrors, new TGLayoutHints(kLHintsNormal, 0, 0, 2, 2));

   fAllWeights1 = new TGCheckButton(v3, "All weights = 1", kFP_ALLW1);
   fAllWeights1->Associate(this);
   fAllWeights1->SetToolTipText("'W'- all weights=1 for non empty bins; error bars ignored");
   v3->AddFrame(fAllWeights1, new TGLayoutHints(kLHintsNormal, 0, 0, 2, 2));

   fEmptyBinsWghts1 = new TGCheckButton(v3, "Empty bins, weights=1", kFP_EMPW1);
   fEmptyBinsWghts1->Associate(this);
   fEmptyBinsWghts1->SetToolTipText("'WW'- all weights=1 including empty bins; error bars ignored");
   v3->AddFrame(fEmptyBinsWghts1, new TGLayoutHints(kLHintsNormal, 0, 0, 2, 2));

   h->AddFrame(v3, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   TGVerticalFrame *v4 = new TGVerticalFrame(h);
   fUseRange = new TGCheckButton(v4, "Use range", kFP_USERG);
   fUseRange->Associate(this);
   fUseRange->SetToolTipText("'R'- fit only data within the specified function range");
   v4->AddFrame(fUseRange, new TGLayoutHints(kLHintsNormal, 0, 0, 2, 2));
   if (fFitOption.Contains('R'))
      fUseRange->SetState(kButtonDown);

   fImproveResults = new TGCheckButton(v4, "Improve fit results", kFP_IFITR);
   fImproveResults->Associate(this);
   fImproveResults->SetToolTipText("'M'- after minimum is found, search for a new one");
   v4->AddFrame(fImproveResults, new TGLayoutHints(kLHintsNormal, 0, 0, 2, 2));

   fAdd2FuncList = new TGCheckButton(v4, "Add to list", kFP_ADDLS);
   fAdd2FuncList->Associate(this);
   fAdd2FuncList->SetToolTipText("'+'- add function to the list without deleting the previous");
   v4->AddFrame(fAdd2FuncList, new TGLayoutHints(kLHintsNormal, 0, 0, 2, 2));

   h->AddFrame(v4, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY, 20, 0, 0, 0));
   gf->AddFrame(h, new TGLayoutHints(kLHintsExpandX, 20, 0, 0, 0));

   // 'draw option' sub-group
   TGHorizontalFrame *h5 = new TGHorizontalFrame(gf);
   TGLabel *label6 = new TGLabel(h5, "Draw Options");
   h5->AddFrame(label6, new TGLayoutHints(kLHintsNormal |
                                          kLHintsCenterY, 2, 2, 2, 2));
   TGHorizontal3DLine *hline3 = new TGHorizontal3DLine(h5);
   h5->AddFrame(hline3, new TGLayoutHints(kLHintsCenterY | kLHintsExpandX));
   gf->AddFrame(h5, new TGLayoutHints(kLHintsExpandX));

   TGHorizontalFrame *h6 = new TGHorizontalFrame(gf);
   TGVerticalFrame *v5 = new TGVerticalFrame(h6);

   fDrawSame = new TGCheckButton(v5, "SAME", kFP_DSAME);
   fDrawSame->Associate(this);
   fDrawSame->SetToolTipText("Superimpose on previous picture in the same pad");
   v5->AddFrame(fDrawSame, new TGLayoutHints(kLHintsNormal, 0, 0, 2, 2));

   fNoDrawing = new TGCheckButton(v5, "No drawing", kFP_DNONE);
   fNoDrawing->Associate(this);
   fNoDrawing->SetToolTipText("'0'- do not draw function graphics");
   v5->AddFrame(fNoDrawing, new TGLayoutHints(kLHintsNormal, 0, 0, 2, 2));

   fNoStoreDrawing = new TGCheckButton(v5, "Do not store/draw", kFP_DNOST);
   fNoStoreDrawing->Associate(this);
   fNoStoreDrawing->SetToolTipText("'N'- do not store the function, do not draw it");
   v5->AddFrame(fNoStoreDrawing, new TGLayoutHints(kLHintsNormal, 0, 0, 2, 2));

   h6->AddFrame(v5, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   TGVerticalFrame *v6 = new TGVerticalFrame(h6);
   TGCompositeFrame *v61 = new TGCompositeFrame(v6, 120, 20,
                                                kHorizontalFrame | kFixedWidth);
   fDrawAdvanced = new TGTextButton(v61, "Advanced...", kFP_DADVB);
   v61->AddFrame(fDrawAdvanced, new TGLayoutHints(kLHintsRight   |
                                                  kLHintsCenterY |
                                                  kLHintsExpandX));
   fDrawAdvanced->SetToolTipText("Open a dialog for advanced draw options");
   fDrawAdvanced->SetState(kButtonDisabled);

   v6->AddFrame(v61, new TGLayoutHints(kLHintsRight | kLHintsTop,
                                       0, 0, (4+fDrawSame->GetHeight())*2, 0));

   h6->AddFrame(v6, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));
   gf->AddFrame(h6, new TGLayoutHints(kLHintsExpandX, 20, 0, 2, 0));

   fGeneral->AddFrame(gf, new TGLayoutHints(kLHintsExpandX |
                                            kLHintsExpandY, 5, 5, 0, 0));
   // sliderX
   fSliderXParent = new TGHorizontalFrame(fGeneral);
   TGLabel *label8 = new TGLabel(fSliderXParent, "X:");
   fSliderXParent->AddFrame(label8, new TGLayoutHints(kLHintsLeft |
                                                      kLHintsCenterY, 0, 5, 0, 0));
   fSliderX = new TGDoubleHSlider(fSliderXParent, 1, kDoubleScaleBoth);
   fSliderX->SetScale(5);
   fSliderXParent->AddFrame(fSliderX, new TGLayoutHints(kLHintsExpandX | 
                                                        kLHintsCenterY));
   fGeneral->AddFrame(fSliderXParent, new TGLayoutHints(kLHintsExpandX, 5, 5, 0, 0));

   // sliderY - no implemented functionality yet
   fSliderYParent = new TGHorizontalFrame(fGeneral);
   TGLabel *label9 = new TGLabel(fSliderYParent, "Y:");
   fSliderYParent->AddFrame(label9, new TGLayoutHints(kLHintsLeft |
                                                      kLHintsCenterY, 0, 5, 0, 0));
   fSliderY = new TGDoubleHSlider(fSliderYParent, 1, kDoubleScaleBoth);
   fSliderY->SetScale(5);
   fSliderYParent->AddFrame(fSliderY, new TGLayoutHints(kLHintsExpandX | 
                                                        kLHintsCenterY));
   fGeneral->AddFrame(fSliderYParent, new TGLayoutHints(kLHintsExpandX, 5, 5, 0, 0));

   // sliderZ
   fSliderZParent = new TGHorizontalFrame(fGeneral);
   TGLabel *label10 = new TGLabel(fSliderZParent, "Z:");
   fSliderZParent->AddFrame(label10, new TGLayoutHints(kLHintsLeft |
                                                       kLHintsCenterY, 0, 5, 0, 0));
   fSliderZ = new TGDoubleHSlider(fSliderZParent, 1, kDoubleScaleBoth);
   fSliderZ->SetScale(5);
   fSliderZParent->AddFrame(fSliderZ, new TGLayoutHints(kLHintsExpandX | 
                                                        kLHintsCenterY));
   fGeneral->AddFrame(fSliderZParent, new TGLayoutHints(kLHintsExpandX, 5, 5, 0, 0));
}


//______________________________________________________________________________
void TFitEditor::CreateMinimizationTab()
{
   // Create 'Minimization' tab.
   
   fTabContainer = fTab->AddTab("Minimization");
   fMinimization = new TGCompositeFrame(fTabContainer, 10, 10, kVerticalFrame);
   fTabContainer->AddFrame(fMinimization, new TGLayoutHints(kLHintsTop |
                                                            kLHintsExpandX,
                                                            5, 5, 2, 2));
   MakeTitle(fMinimization, "Library");

   TGHorizontalFrame *hl = new TGHorizontalFrame(fMinimization);
   fLibMinuit = new TGRadioButton(hl, "Minuit", kFP_LMIN);
   fLibMinuit->Associate(this);
   fLibMinuit->SetToolTipText("Use minimization from libMinuit (default)");
   hl->AddFrame(fLibMinuit, new TGLayoutHints(kLHintsNormal, 40, 0, 0, 1));
   fLibMinuit->SetState(kButtonDown);
   fStatusBar->SetText("LIB Minuit",0);

   fLibMinuit2 = new TGRadioButton(hl, "Minuit2", kFP_LMIN2);
   fLibMinuit2->Associate(this);
   fLibMinuit2->SetToolTipText("New C++ version of Minuit");
   hl->AddFrame(fLibMinuit2, new TGLayoutHints(kLHintsNormal, 35, 0, 0, 1));

   fLibFumili = new TGRadioButton(hl, "Fumili", kFP_LFUM);
   fLibFumili->Associate(this);
   fLibFumili->SetToolTipText("Use minimization from libFumili");
   hl->AddFrame(fLibFumili, new TGLayoutHints(kLHintsNormal, 30, 0, 0, 1));
   fMinimization->AddFrame(hl, new TGLayoutHints(kLHintsExpandX, 20, 0, 5, 1));

   MakeTitle(fMinimization, "Method");

   TGHorizontalFrame *hm = new TGHorizontalFrame(fMinimization);
   fMigrad = new TGRadioButton(hm, "MIGRAD", kFP_MIGRAD);
   fMigrad->Associate(this);
   fMigrad->SetToolTipText("Use MIGRAD as minimization method");
   hm->AddFrame(fMigrad, new TGLayoutHints(kLHintsNormal, 40, 0, 0, 1));
   fMigrad->SetState(kButtonDown);
   fStatusBar->SetText("MIGRAD",1);

   fSimplex = new TGRadioButton(hm, "SIMPLEX", kFP_SIMPLX);
   fSimplex->Associate(this);
   fSimplex->SetToolTipText("Use SIMPLEX as minimization method");
   // Simplex functionality will come with the new fitter design    
   fSimplex->SetState(kButtonDisabled);
   hm->AddFrame(fSimplex, new TGLayoutHints(kLHintsNormal, 20, 0, 0, 1));

   fFumili = new TGRadioButton(hm, "FUMILI", kFP_FUMILI);
   fFumili->Associate(this);
   fFumili->SetToolTipText("Use FUMILI as minimization method");
   fFumili->SetState(kButtonDisabled);
   hm->AddFrame(fFumili, new TGLayoutHints(kLHintsNormal, 18, 0, 0, 1));
   fMinimization->AddFrame(hm, new TGLayoutHints(kLHintsExpandX, 20, 0, 5, 1));

   MakeTitle(fMinimization, "Settings");
   TGLabel *hslabel1 = new TGLabel(fMinimization,"Use ENTER key to validate a new value or click");
   fMinimization->AddFrame(hslabel1, new TGLayoutHints(kLHintsNormal, 61, 0, 5, 1));
   TGLabel *hslabel2 = new TGLabel(fMinimization,"on Reset button to set the defaults.");
   fMinimization->AddFrame(hslabel2, new TGLayoutHints(kLHintsNormal, 61, 0, 1, 10));

   TGHorizontalFrame *hs = new TGHorizontalFrame(fMinimization);
   
   TGVerticalFrame *hsv1 = new TGVerticalFrame(hs, 180, 10, kFixedWidth);
   TGLabel *errlabel = new TGLabel(hsv1,"Error definition (default = 1): ");
   hsv1->AddFrame(errlabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 
                                              1, 1, 5, 7));
   TGLabel *tollabel = new TGLabel(hsv1,"Max tolerance (precision): ");
   hsv1->AddFrame(tollabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 
                                              1, 1, 5, 7));
   TGLabel *itrlabel = new TGLabel(hsv1,"Max number of iterations: ");
   hsv1->AddFrame(itrlabel, new TGLayoutHints(kLHintsLeft | kLHintsCenterY, 
                                              1, 1, 5, 5));
   hs->AddFrame(hsv1, new TGLayoutHints(kLHintsNormal, 60, 0, 0, 0));
   
   TGVerticalFrame *hsv2 = new TGVerticalFrame(hs, 90,10, kFixedWidth);
   fErrorScale = new TGNumberEntryField(hsv2, kFP_MERR, 1.0,
                                        TGNumberFormat::kNESRealTwo,
                                        TGNumberFormat::kNEAPositive,
                                        TGNumberFormat::kNELLimitMinMax,0.,100.);
   hsv2->AddFrame(fErrorScale, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 
                                                 1, 1, 0, 3));
   fTolerance = new TGNumberEntryField(hsv2, kFP_MTOL, 1.0E-9, 
                                       TGNumberFormat::kNESReal,
                                       TGNumberFormat::kNEAPositive,
                                       TGNumberFormat::kNELLimitMinMax, 0., 1.);
   fTolerance->SetNumber(TVirtualFitter::GetPrecision());
   hsv2->AddFrame(fTolerance, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 
                                                1, 1, 3, 3));
   fIterations = new TGNumberEntryField(hsv2, kFP_MITR, 5000, 
                                   TGNumberFormat::kNESInteger,
                                   TGNumberFormat::kNEAPositive,
                                   TGNumberFormat::kNELNoLimits);
   fIterations->SetNumber(TVirtualFitter::GetMaxIterations());
   hsv2->AddFrame(fIterations, new TGLayoutHints(kLHintsLeft | kLHintsExpandX, 
                                                 1, 1, 3, 3));
   hs->AddFrame(hsv2, new TGLayoutHints(kLHintsNormal, 0, 0, 0, 0));
   fMinimization->AddFrame(hs, new TGLayoutHints(kLHintsExpandX, 0, 0, 1, 1));
   fStatusBar->SetText("Itr: 5000",2);


   MakeTitle(fMinimization, "Print Options");

   TGHorizontalFrame *h8 = new TGHorizontalFrame(fMinimization);
   fOptDefault = new TGRadioButton(h8, "Default", kFP_PDEF);
   fOptDefault->Associate(this);
   fOptDefault->SetToolTipText("Default is between Verbose and Quiet");
   h8->AddFrame(fOptDefault, new TGLayoutHints(kLHintsNormal, 40, 0, 0, 1));
   fOptDefault->SetState(kButtonDown);
   fStatusBar->SetText("Prn: DEF",3);

   fOptVerbose = new TGRadioButton(h8, "Verbose", kFP_PVER);
   fOptVerbose->Associate(this);
   fOptVerbose->SetToolTipText("'V'- print results after each iteration");
   h8->AddFrame(fOptVerbose, new TGLayoutHints(kLHintsNormal, 30, 0, 0, 1));

   fOptQuiet = new TGRadioButton(h8, "Quiet", kFP_PQET);
   fOptQuiet->Associate(this);
   fOptQuiet->SetToolTipText("'Q'- no print");
   h8->AddFrame(fOptQuiet, new TGLayoutHints(kLHintsNormal, 25, 0, 0, 1));

   fMinimization->AddFrame(h8, new TGLayoutHints(kLHintsExpandX, 20, 0, 5, 1));

}

//______________________________________________________________________________
void TFitEditor::ConnectSlots()
{
   // Connect GUI signals to fit panel slots.

   // list of predefined functions
   fFuncList->Connect("Selected(Int_t)", "TFitEditor", this, "DoFunction(Int_t)");
   // entered formula or function name
   fEnteredFunc->Connect("ReturnPressed()", "TFitEditor", this, "DoEnteredFunction()");
   // set parameters dialog
   fSetParam->Connect("Clicked()", "TFitEditor", this, "DoSetParameters()");
   // allowed function operations
   fNone->Connect("Toggled(Bool_t)","TFitEditor", this, "DoNoOperation(Bool_t)");
   fAdd->Connect("Toggled(Bool_t)","TFitEditor", this, "DoAddition(Bool_t)");

   // fit options
   fIntegral->Connect("Toggled(Bool_t)","TFitEditor",this,"DoIntegral()");
   fBestErrors->Connect("Toggled(Bool_t)","TFitEditor",this,"DoBestErrors()");
   fUseRange->Connect("Toggled(Bool_t)","TFitEditor",this,"DoUseRange()");
   fAdd2FuncList->Connect("Toggled(Bool_t)","TFitEditor",this,"DoAddtoList()");
   fAllWeights1->Connect("Toggled(Bool_t)","TFitEditor",this,"DoAllWeights1()");
   fEmptyBinsWghts1->Connect("Toggled(Bool_t)","TFitEditor",this,"DoEmptyBinsAllWeights1()");
   fImproveResults->Connect("Toggled(Bool_t)","TFitEditor",this,"DoImproveResults()");

   // linear fit
   fLinearFit->Connect("Toggled(Bool_t)","TFitEditor",this,"DoLinearFit()");
   fNoChi2->Connect("Toggled(Bool_t)","TFitEditor",this,"DoNoChi2()");
   fRobustValue->Connect("ValueSet(Long_t)", "TFitEditor", this, "DoRobust()");
   (fRobustValue->GetNumberEntry())->Connect("ReturnPressed()", "TFitEditor",
                                              this, "DoRobust()");

   // draw options
   fNoStoreDrawing->Connect("Toggled(Bool_t)","TFitEditor",this,"DoNoStoreDrawing()");
   fNoDrawing->Connect("Toggled(Bool_t)","TFitEditor",this,"DoNoDrawing()");
   fDrawSame->Connect("Toggled(Bool_t)","TFitEditor",this,"DoDrawSame()");
   // fit method
   fMethodList->Connect("Selected(Int_t)", "TFitEditor", this, "DoMethod(Int_t)");

   // fit, reset, close buttons
   fFitButton->Connect("Clicked()", "TFitEditor", this, "DoFit()");
   fResetButton->Connect("Clicked()", "TFitEditor", this, "DoReset()");
   fCloseButton->Connect("Clicked()", "TFitEditor", this, "DoClose()");

   // user method button
   fUserButton->Connect("Clicked()", "TFitEditor", this, "DoUserDialog()");
   // advanced draw options
   fDrawAdvanced->Connect("Clicked()", "TFitEditor", this, "DoAdvancedOptions()");

   if (fDim > 0) {
      fSliderX->Connect("PositionChanged()","TFitEditor",this, "DoSliderXMoved()");
      fSliderX->Connect("Pressed()","TFitEditor",this, "DoSliderXPressed()");
      fSliderX->Connect("Released()","TFitEditor",this, "DoSliderXReleased()");
   }
   if (fDim > 1) {
      fSliderY->Connect("PositionChanged()","TFitEditor",this, "DoSliderYMoved()");
      fSliderY->Connect("Pressed()","TFitEditor",this, "DoSliderYPressed()");
      fSliderY->Connect("Released()","TFitEditor",this, "DoSliderYReleased()");
   }
   if (fDim > 2) {
      fSliderZ->Connect("PositionChanged()","TFitEditor",this, "DoSliderZMoved()");
      fSliderZ->Connect("Pressed()","TFitEditor",this, "DoSliderZPressed()");
      fSliderZ->Connect("Released()","TFitEditor",this, "DoSliderZReleased()");
   }
   fParentPad->Connect("RangeAxisChanged()", "TFitEditor", this, "UpdateGUI()");
   
   // 'Minimization' tab
   // library
   fLibMinuit->Connect("Toggled(Bool_t)","TFitEditor",this,"DoLibrary(Bool_t)");
   fLibMinuit2->Connect("Toggled(Bool_t)","TFitEditor",this,"DoLibrary(Bool_t)");
   fLibFumili->Connect("Toggled(Bool_t)","TFitEditor",this,"DoLibrary(Bool_t)");

   // minimization method
   fMigrad->Connect("Toggled(Bool_t)","TFitEditor",this,"DoMinMethod(Bool_t)");
   // Simplex functionality will come with the new fitter design
   //fSimplex->Connect("Toggled(Bool_t)","TFitEditor",this,"DoMinMethod(Bool_t)");
   fFumili->Connect("Toggled(Bool_t)","TFitEditor",this,"DoMinMethod(Bool_t)");

   // fitter settings
   fErrorScale->Connect("ReturnPressed()", "TFitEditor", this, "DoErrorsDef()");
   fTolerance->Connect("ReturnPressed()", "TFitEditor", this, "DoMaxTolerance()");
   fIterations->Connect("ReturnPressed()", "TFitEditor", this, "DoMaxIterations()");
   
   // print options
   fOptDefault->Connect("Toggled(Bool_t)","TFitEditor",this,"DoPrintOpt(Bool_t)");
   fOptVerbose->Connect("Toggled(Bool_t)","TFitEditor",this,"DoPrintOpt(Bool_t)");
   fOptQuiet->Connect("Toggled(Bool_t)","TFitEditor",this,"DoPrintOpt(Bool_t)");

}

//______________________________________________________________________________
void TFitEditor::DisconnectSlots()
{
   // Disconnect GUI signals from fit panel slots.

   Disconnect("CloseWindow()");

   fFuncList->Disconnect("Selected(Int_t)");
   fEnteredFunc->Disconnect("ReturnPressed()");
   fSetParam->Disconnect("Clicked()");
   fNone->Disconnect("Toggled(Bool_t)");
   fAdd->Disconnect("Toggled(Bool_t)");

   // fit options
   fIntegral->Disconnect("Toggled(Bool_t)");
   fBestErrors->Disconnect("Toggled(Bool_t)");
   fUseRange->Disconnect("Toggled(Bool_t)");
   fAdd2FuncList->Disconnect("Toggled(Bool_t)");
   fAllWeights1->Disconnect("Toggled(Bool_t)");
   fEmptyBinsWghts1->Disconnect("Toggled(Bool_t)");
   fImproveResults->Disconnect("Toggled(Bool_t)");

   // linear fit
   fLinearFit->Disconnect("Toggled(Bool_t)");
   fNoChi2->Disconnect("Toggled(Bool_t)");
   fRobustValue->Disconnect("ValueSet(Long_t)");
   (fRobustValue->GetNumberEntry())->Disconnect("ReturnPressed()");

   // draw options
   fNoStoreDrawing->Disconnect("Toggled(Bool_t)");
   fNoDrawing->Disconnect("Toggled(Bool_t)");
   fDrawSame->Disconnect("Toggled(Bool_t)");

   // fit method
   fMethodList->Disconnect("Selected(Int_t)");

   // fit, reset, close buttons
   fFitButton->Disconnect("Clicked()");
   fResetButton->Disconnect("Clicked()");
   
   // other methods
   fUserButton->Disconnect("Clicked()");
   fDrawAdvanced->Disconnect("Clicked()");

   if (fDim > 0) {
      fSliderX->Disconnect("PositionChanged()");
      fSliderX->Disconnect("Pressed()");
      fSliderX->Disconnect("Released()");
   }
   if (fDim > 1) {
      fSliderY->Disconnect("PositionChanged()");
      fSliderY->Disconnect("Pressed()");
      fSliderY->Disconnect("Released()");
   }
   if (fDim > 2) {
      fSliderZ->Disconnect("PositionChanged()");
      fSliderZ->Disconnect("Pressed()");
      fSliderZ->Disconnect("Released()");
   }
   
   // slots related to 'Minimization' tab
   fLibMinuit->Disconnect("Toggled(Bool_t)");
   fLibMinuit2->Disconnect("Toggled(Bool_t)");
   fLibFumili->Disconnect("Toggled(Bool_t)");

   // minimization method
   fMigrad->Disconnect("Toggled(Bool_t)");
   // Simplex functionality will come with the new fitter design
   //fSimplex->Disconnect("Toggled(Bool_t)");
   fFumili->Disconnect("Toggled(Bool_t)");

   // fitter settings
   fErrorScale->Disconnect("ReturnPressed()");
   fTolerance->Disconnect("ReturnPressed()");
   fIterations->Disconnect("ReturnPressed()");

   // print options
   fOptDefault->Disconnect("Toggled(Bool_t)");
   fOptVerbose->Disconnect("Toggled(Bool_t)");
   fOptQuiet->Disconnect("Toggled(Bool_t)");

}

//______________________________________________________________________________
void TFitEditor::SetCanvas(TCanvas *newcan)
{
   // Connect to another canvas.

   if (!newcan || (fCanvas == newcan)) return;

   fCanvas = newcan;
   ConnectToCanvas();
}

//______________________________________________________________________________
void TFitEditor::ConnectToCanvas()
{
   // Connect fit panel to the 'Selected' signal of canvas 'c'.

   TQObject::Connect("TCanvas", "Selected(TVirtualPad *, TObject *, Int_t)", 
                     "TFitEditor",this, 
                     "SetFitObject(TVirtualPad *, TObject *, Int_t)");
   TQObject::Connect("TCanvas", "Closed()", "TFitEditor", this, "DoNoSelection()");
}

//______________________________________________________________________________
void TFitEditor::Hide()
{
   // Hide the fit panel and set it to non-active state. 

   if (fgFitDialog) {
      fgFitDialog->UnmapWindow();
   }
   fParentPad->Disconnect("RangeAxisChanged()");
   DoReset();
   fCanvas = 0;
   fParentPad = 0;
   fFitObject = 0;
   gROOT->GetListOfCleanups()->Remove(this);
}

//______________________________________________________________________________
void TFitEditor::Show(TVirtualPad* pad, TObject *obj)
{
   // Show the fit panel (possible only via context menu).

   if (!gROOT->GetListOfCleanups()->FindObject(this))
      gROOT->GetListOfCleanups()->Add(this);   

   if (!fgFitDialog->IsMapped()) {
      fgFitDialog->MapWindow();
      gVirtualX->RaiseWindow(GetId());
   }
   SetCanvas(pad->GetCanvas());
   fCanvas->Selected(pad, obj, kButton1Down);
}
//______________________________________________________________________________
void TFitEditor::CloseWindow()
{
   // Close fit panel window.

   Hide();
}

//______________________________________________________________________________
void TFitEditor::Terminate()
{
   //  Called to delete the fit panel. 

   TQObject::Disconnect("TCanvas", "Closed()");
   delete fgFitDialog;
   fgFitDialog = 0;
}

//______________________________________________________________________________
void TFitEditor::UpdateGUI()
{
   //  Set the fit panel GUI according to the selected object. 

   // sliders
   if (fDim > 0) {
      fSliderX->Disconnect("PositionChanged()");
      fSliderX->Disconnect("Pressed()");
      fSliderX->Disconnect("Released()");

      switch (fType) {
         case kObjectHisto: {
            fXaxis = ((TH1*)fFitObject)->GetXaxis();
            fYaxis = ((TH1*)fFitObject)->GetYaxis();
            fZaxis = ((TH1*)fFitObject)->GetZaxis();
            fXrange = fXaxis->GetNbins();
            fXmin = fXaxis->GetFirst();
            fXmax = fXaxis->GetLast();
            break;
         }
         case kObjectGraph: {
            TGraph *gr = (TGraph*)fFitObject; //TBV
            TH1F *hist = gr->GetHistogram();
            if (hist) {
               fXaxis = hist->GetXaxis();
               fYaxis = hist->GetYaxis();
               fZaxis = hist->GetZaxis();
               fXrange = fXaxis->GetNbins();
               fXmin = fXaxis->GetFirst();
               fXmax = fXaxis->GetLast();
            }
            break;
         }
         case kObjectGraph2D: {
            //not implemented
            break;
         }
         case kObjectHStack: {
            TH1 *hist = (TH1 *)((THStack *)fFitObject)->GetHists()->First();
            fXaxis = hist->GetXaxis();
            fYaxis = hist->GetYaxis();
            fZaxis = hist->GetZaxis();
            fXrange = fXaxis->GetNbins();
            fXmin = fXaxis->GetFirst();
            fXmax = fXaxis->GetLast();
            break;
         }
         case kObjectTree:  {
            //not implemented
            break;
         }
      }
      if (fXmin > 1 || fXmax < fXrange) {
         fSliderX->SetRange(fXmin,fXmax);
         fSliderX->SetPosition(fXmin, fXmax);
      } else {
         fSliderX->SetRange(1,fXrange);
         fSliderX->SetPosition(fXmin,fXmax);
      }
      fSliderX->SetScale(5);
      fSliderX->Connect("PositionChanged()","TFitEditor",this, "DoSliderXMoved()");
      fSliderX->Connect("Pressed()","TFitEditor",this, "DoSliderXPressed()");
      fSliderX->Connect("Released()","TFitEditor",this, "DoSliderXReleased()");
   }

/*  no implemented functionality for y & z sliders yet 
   if (fDim > 1) {
      fSliderY->Disconnect("PositionChanged()");
      fSliderY->Disconnect("Pressed()");
      fSliderY->Disconnect("Released()");

      if (!fSliderYParent->IsMapped())
         fSliderYParent->MapWindow();
      if (fSliderZParent->IsMapped())
         fSliderZParent->UnmapWindow();

      switch (fType) {
         case kObjectHisto: {
            fYrange = fYaxis->GetNbins();
            fYmin = fYaxis->GetFirst();
            fYmax = fYaxis->GetLast();
            break;
         }
         case kObjectGraph: {
            //not implemented
            break;
         }
         case kObjectGraph2D: {
            //not implemented
            break;
         }
         case kObjectHStack: {
            fYrange = fYaxis->GetNbins();
            fYmin = fYaxis->GetFirst();
            fYmax = fYaxis->GetLast();
            break;
         }
         case kObjectTree:  {
            //not implemented
            break;
         }
      }
      fSliderY->SetRange(1,fYrange);
      fSliderY->SetPosition(fYmin,fYmax);
      fSliderY->SetScale(5);
   }

   if (fDim > 2) {
      fSliderZ->Disconnect("PositionChanged()");
      fSliderZ->Disconnect("Pressed()");
      fSliderZ->Disconnect("Released()");

      if (!fSliderZParent->IsMapped())
         fSliderZParent->MapWindow();

      switch (fType) {
         case kObjectHisto: {
            fZrange = fZaxis->GetNbins();
            fZmin = fZaxis->GetFirst();
            fZmax = fZaxis->GetLast();
            break;
         }
         case kObjectGraph: {
            //not implemented
            break;
         }
         case kObjectGraph2D: {
            //not implemented
            break;
         }
         case kObjectHStack: {
            //TH1 *hist = (TH1 *)((THStack *)fFitObject)->GetHists()->First();
            fZrange = fZaxis->GetNbins();
            fZmin = fZaxis->GetFirst();
            fZmax = fZaxis->GetLast();
            break;
         }
         case kObjectTree:  {
            //not implemented
            break;
         }
      }
      fSliderZ->SetRange(1,fZrange);
      fSliderZ->SetPosition(fZmin,fZmax);
      fSliderZ->SetScale(5);
   }

   switch (fDim) {
      case 1:
         fGeneral->HideFrame(fSliderYParent);
         fGeneral->HideFrame(fSliderZParent);
         break;
      case 2:
         fGeneral->HideFrame(fSliderZParent);
         break;
   }
   Layout();*/
}

//______________________________________________________________________________
void TFitEditor::SetFitObject(TVirtualPad *pad, TObject *obj, Int_t event)
{
   // Slot called when the user clicks on an object inside a canvas. 
   // Updates pointers to the parent pad and the selected object
   // for fitting (if suitable).

   if (event != kButton1Down) return;

   if (!pad || !obj) {
      DoNoSelection();
      return;
   }
   
   // is obj suitable for fitting?
   if (!SetObjectType(obj)) return;
   
   fParentPad = pad;
   fFitObject = obj;
   fDrawOption = GetDrawOption();
   ShowObjectName(obj);
   UpdateGUI();

   ConnectSlots();  //TBS do we need it every time?

   TGTextLBEntry *te = (TGTextLBEntry *)fFuncList->GetSelectedEntry();
   if (fNone->GetState() == kButtonDown)
      fFunction = te->GetTitle();
   else if (fAdd->GetState() == kButtonDown) {
      fFunction += '+';
      fFunction +=te->GetTitle();
   }
   fEnteredFunc->SetText(fFunction.Data());
   fEnteredFunc->SelectAll();
   if (!fFitFunc) {
      fFitFunc = new TF1("fitFunc",fFunction.Data(),fXmin,fXmax);
   }
   
   // Update the information about the selected object.
   if (fSetParam->GetState() == kButtonDisabled)
      fSetParam->SetEnabled(kTRUE);
   if (fFitButton->GetState() == kButtonDisabled)
      fFitButton->SetEnabled(kTRUE);
   if (fResetButton->GetState() == kButtonDisabled)
      fResetButton->SetEnabled(kTRUE);
   DoLinearFit();
}

//______________________________________________________________________________
void TFitEditor::DoNoSelection()
{
   // Slot called when users close a TCanvas. 

   if (gROOT->GetListOfCanvases()->IsEmpty()) {
      Terminate();
      return;
   }
   
   DisconnectSlots();
   fParentPad = 0;
   fFitObject = 0;
   fObjLabel->SetText("No object selected");
   fObjLabelParent->Resize(GetDefaultSize());
   Layout();

   fSetParam->SetEnabled(kFALSE);
   fFitButton->SetEnabled(kFALSE);
   fResetButton->SetEnabled(kFALSE);
}

//______________________________________________________________________________
void TFitEditor::RecursiveRemove(TObject* obj)
{
   // When obj is deleted, clear fFitObject if fFitObject = obj.

   if (obj == fFitObject) {

      fFitObject = 0;
      DisconnectSlots();
      fObjLabel->SetText("No object selected");
      fObjLabelParent->Resize(GetDefaultSize());
      Layout();

      fFitButton->SetEnabled(kFALSE);
      fResetButton->SetEnabled(kFALSE);
      fSetParam->SetEnabled(kFALSE);
      TQObject::Connect("TCanvas", "Selected(TVirtualPad *, TObject *, Int_t)", 
                        "TFitEditor",this, 
                        "SetFitObject(TVirtualPad *, TObject *, Int_t)");
      TQObject::Connect("TCanvas", "Closed()", "TFitEditor", this, 
                        "DoNoSelection()");
      return;
   }
   if (obj == fParentPad) {

      fFitObject = 0;
      fParentPad = 0;
      DisconnectSlots();
      fObjLabel->SetText("No object selected");
      fObjLabelParent->Resize(GetDefaultSize());
      Layout();

      fFitButton->SetEnabled(kFALSE);
      fResetButton->SetEnabled(kFALSE);
      fSetParam->SetEnabled(kFALSE);
   }
}

//______________________________________________________________________________
TGComboBox* TFitEditor::BuildFunctionList(TGFrame* parent, Int_t id)
{
   // Create function list combo box.

   TGComboBox *c = new TGComboBox(parent, id);

   c->AddEntry("gaus" ,  kFP_GAUS);
   c->AddEntry("gausn",  kFP_GAUSN);
   c->AddEntry("expo",   kFP_EXPO);
   c->AddEntry("landau", kFP_LAND);
   c->AddEntry("landaun",kFP_LANDN);
   c->AddEntry("pol0",   kFP_POL0);
   c->AddEntry("pol1",   kFP_POL1);
   c->AddEntry("pol2",   kFP_POL2);
   c->AddEntry("pol3",   kFP_POL3);
   c->AddEntry("pol4",   kFP_POL4);
   c->AddEntry("pol5",   kFP_POL5);
   c->AddEntry("pol6",   kFP_POL6);
   c->AddEntry("pol7",   kFP_POL7);
   c->AddEntry("pol8",   kFP_POL8);
   c->AddEntry("pol9",   kFP_POL9);
   c->AddEntry("user",   kFP_USER);

   if (!gROOT->GetFunction("gaus")) {
      Float_t xmin = 1.;
      Float_t xmax = 2.;
      new TF1("gaus","gaus",xmin,xmax);
      new TF1("landau","landau",xmin,xmax);
      new TF1("expo","expo",xmin,xmax);
      for (Int_t i=0; i<10; i++) {
         new TF1(Form("pol%d",i),Form("pol%d",i),xmin,xmax);
      }
   }
   return c;
}

//______________________________________________________________________________
TGComboBox* TFitEditor::BuildMethodList(TGFrame* parent, Int_t id)
{
   // Create method list in a combo box.

   TGComboBox *c = new TGComboBox(parent, id);
   c->AddEntry("Chi-square", kFP_MCHIS);
   c->AddEntry("Binned Likelihood", kFP_MBINL);
   //c->AddEntry("Unbinned Likelihood", kFP_MUBIN); //for later use
   //c->AddEntry("User", kFP_MUSER);                //for later use
   c->Select(kFP_MCHIS);
   return c;
}

//______________________________________________________________________________
void TFitEditor::DoAddtoList()
{
   // Slot connected to 'add to list of function' setting.

   if (fAdd2FuncList->GetState() == kButtonDown)
      fFitOption += '+';
   else {
      Int_t eq = fFitOption.First('+');
      fFitOption.Remove(eq, 1);
   }
}

//______________________________________________________________________________
void TFitEditor::DoAdvancedOptions()
{
   // Slot connected to advanced option button (opens a dialog).

   new TGMsgBox(fClient->GetRoot(), GetMainFrame(),
                "Info", "Advanced option dialog is not implemented yet",
                kMBIconAsterisk,kMBOk, 0);
}

//______________________________________________________________________________
void TFitEditor::DoEmptyBinsAllWeights1()
{
   // Slot connected to 'include emtry bins and forse all weights to 1' setting.

   if (fEmptyBinsWghts1->GetState() == kButtonDown) {
      if (fAllWeights1->GetState() == kButtonDown) {
         fAllWeights1->SetState(kButtonUp, kTRUE);
      }
      fFitOption += "WW";
   } else {
      Int_t eq = fFitOption.First("WW");
      fFitOption.Remove(eq, 2);
   }
}

//______________________________________________________________________________
void TFitEditor::DoAllWeights1()
{
   // Slot connected to 'set all weights to 1' setting.

   if (fAllWeights1->GetState() == kButtonDown) {
      if (fEmptyBinsWghts1->GetState() == kButtonDown) {
         fEmptyBinsWghts1->SetState(kButtonUp, kTRUE);
      }
      fFitOption += 'W';
   } else {
      Int_t eq = fFitOption.First('W');
      fFitOption.Remove(eq, 1);
   }
}

//______________________________________________________________________________
void TFitEditor::DoClose()
{
   // Close the fit panel.

   Hide();
}

//______________________________________________________________________________
void TFitEditor::DoDrawSame()
{
   // Slot connected to 'same' draw option.

   fFitOption.ToUpper();
   
   if (fDrawSame->GetState() == kButtonDown) {
      if (fDrawOption.Contains("SAME"))
         return;
      else
         fDrawOption += "SAME";
   } else {
      if (fDrawOption.Contains("SAME"))
         fDrawOption.ReplaceAll("SAME", "");
   }
}

//______________________________________________________________________________
void TFitEditor::DoFit()
{
   // Perform a fit with current parameters' settings.

   if (!fFitObject) return;
   if (!fParentPad) return;
   if (!fFitFunc) {
      new TGMsgBox(fClient->GetRoot(), GetMainFrame(),
                   "Error", Form("Function with name '%s' does not exist",
                   fFunction.Data()), kMBIconExclamation, kMBClose, 0);
      return;
   }

   fParentPad->Disconnect("RangeAxisChanged()");
   TVirtualPad *save = 0;
   save = gPad;
   gPad = fParentPad;
   fParentPad->cd();
    
   fParentPad->GetCanvas()->SetCursor(kWatch);
   Double_t xmin = 0;
   Double_t xmax = 0;
   switch (fType) {
      case kObjectHisto: {
         TH1 *h1 = (TH1*)fFitObject;
         xmin = fXaxis->GetBinLowEdge((Int_t)(fSliderX->GetMinPosition()));
         xmax = fXaxis->GetBinUpEdge((Int_t)(fSliderX->GetMaxPosition()));
         fFitFunc->SetRange(xmin,xmax);
         fDrawOption = GetDrawOption();
         h1->Fit(fFitFunc, fFitOption.Data(), fDrawOption.Data(), xmin, xmax);
         break;
      }
      case kObjectGraph: {
         TGraph *gr = (TGraph*)fFitObject;
         TH1F *hist = gr->GetHistogram();
         if (hist) { //!!! for many graphs in a pad, use the xmin/xmax of pad!!!
            xmin = fXaxis->GetBinLowEdge((Int_t)(fSliderX->GetMinPosition()));
            xmax = fXaxis->GetBinUpEdge((Int_t)(fSliderX->GetMaxPosition()));
            Int_t npoints = gr->GetN();
            Double_t *gx = gr->GetX();
            Double_t gxmin, gxmax;
            gxmin = gx[0];
            gxmax = gx[npoints-1];
            Double_t err0 = gr->GetErrorX(0);
            Double_t errn = gr->GetErrorX(npoints-1);
            if (err0 > 0)
               gxmin -= 2*err0;
            if (errn > 0)
               gxmax += 2*errn;
            for (Int_t i=0; i<npoints; i++) {
               if (gx[i] < xmin)
                  gxmin = gx[i];
               if (gx[i] > xmax)
                  gxmax = gx[i];
            }
            if (xmin < gxmin) xmin = gxmin;
            if (xmax > gxmax) xmax = gxmax;
         }
         fFitFunc->SetRange(xmin,xmax);
         fDrawOption = GetDrawOption();
         gr->Fit(fFitFunc, fFitOption.Data(), fDrawOption.Data(), xmin, xmax);
         break;
      }
      case kObjectGraph2D: {
         // N/A
         break;
      }
      case kObjectHStack: {
         // N/A
         break;
      }
      case kObjectTree:  {
         // N/A
         break;
      }
   }

   fParentPad->Modified();
   fParentPad->Update();
   fParentPad->GetCanvas()->SetCursor(kPointer);
   fParentPad->Connect("RangeAxisChanged()", "TFitEditor", this, "UpdateGUI()");
   
   if (save) gPad = save;
   if (fSetParam->GetState() == kButtonDisabled && 
       fLinearFit->GetState() == kButtonUp)
      fSetParam->SetState(kButtonUp);
}

//______________________________________________________________________________
Int_t TFitEditor::CheckFunctionString(const char *fname)
{
   // Check entered function string.

   TFormula *form = 0;
   form = new TFormula(fname, fname);
   if (form) {
      return form->Compile();
   }
   return -1;
}

//______________________________________________________________________________
void TFitEditor::DoAddition(Bool_t on)
{
   // Slot connected to addition of predefined functions.

   static Bool_t first = kFALSE;
   TString s = fEnteredFunc->GetText();
   if (on) {
      if (!first) {
         s += "(0)";
         fEnteredFunc->SetText(s.Data());
         first = kTRUE;
         fSelLabel->SetText(fFunction.Data());
         ((TGCompositeFrame *)fSelLabel->GetParent())->Layout();
      }
   } else {
      first = kFALSE;
   }
}

//______________________________________________________________________________
void TFitEditor::DoNoOperation(Bool_t on)
{
   // Slot connected to NOP of predefined functions.

   TGTextLBEntry *te = (TGTextLBEntry *)fFuncList->GetSelectedEntry();
   if (on) {
      fEnteredFunc->SetText(te->GetTitle());
   }
   fFunction = fEnteredFunc->GetText();
   fSelLabel->SetText(fFunction.Data());
   ((TGCompositeFrame *)fSelLabel->GetParent())->Layout();
   if (fFitFunc) delete fFitFunc;
   fFitFunc = new TF1("fitFunc",fFunction.Data(),fXmin,fXmax);
}

//______________________________________________________________________________
void TFitEditor::DoFunction(Int_t /*sel*/)
{
   // Slot connected to predefined fit function settings.

   TGTextLBEntry *te = (TGTextLBEntry *)fFuncList->GetSelectedEntry();
   if (fNone->GetState() == kButtonDown) {
      fEnteredFunc->SetText(te->GetTitle());
   } else if (fAdd->GetState() == kButtonDown) {
      TString s = fEnteredFunc->GetText();
      TFormula tmp("tmp", fFunction.Data());
      Int_t np = tmp.GetNpar();
      s += Form("+%s(%d)", te->GetTitle(), np);
      fEnteredFunc->SetText(s.Data());
   }
   fFunction = fEnteredFunc->GetText();

   // create TF1 with the passed string. Delete previous one if existing
   if (fFunction.Contains("gaus") || fFunction.Contains("expo") ||
       fFunction.Contains("landau") || fFunction.Contains("user")) {
      fLinearFit->SetState(kButtonUp, kTRUE);
   } else {
      fLinearFit->SetState(kButtonDown, kTRUE);
   }
/*   if (fFunction.Contains("user")) {
      fFitOption += 'U';
   } else if (fFitOption.Contains('U')) {
      Int_t eq = fFitOption.First('U');
      fFitOption.Remove(eq, 1);
   }*/

   fEnteredFunc->SelectAll();
   fSelLabel->SetText(fFunction.Data());
   ((TGCompositeFrame *)fSelLabel->GetParent())->Layout();

   if (fFitFunc) delete fFitFunc;
   fFitFunc = new TF1("fitFunc",fFunction.Data(),fXmin,fXmax);
}

//______________________________________________________________________________
void TFitEditor::DoEnteredFunction()
{
   // Slot connected to entered function in text entry.

   Int_t ok = CheckFunctionString(fEnteredFunc->GetText());

   if (ok != 0) {
      new TGMsgBox(fClient->GetRoot(), GetMainFrame(),
                   "Error...", "Verify the entered function string!",
                   kMBIconStop,kMBOk, 0);
   }

   fFunction = fEnteredFunc->GetText();
   if (fFitFunc) delete fFitFunc;
   fFitFunc = new TF1("fitFunc",fFunction.Data(),fXmin,fXmax);

   fSelLabel->SetText(fFunction.Data());
   ((TGCompositeFrame *)fSelLabel->GetParent())->Layout();
   if (fFunction.Contains("++")) {
      fLinearFit->SetState(kButtonDown, kTRUE);
      fAdd->SetState(kButtonDown, kTRUE);
   } else if (fFunction.Contains('+')) {
      fAdd->SetState(kButtonDown, kTRUE);
   } else {
      fNone->SetState(kButtonDown, kTRUE);
   }
}

//______________________________________________________________________________
void TFitEditor::DoImproveResults()
{
   // Slot connected to 'improve fit results' option settings.

   if (fImproveResults->GetState() == kButtonDown)
      fFitOption += 'M';
   else if (fFitOption.Contains('M'))
      fFitOption.ReplaceAll('M', "");
}

//______________________________________________________________________________
void TFitEditor::DoBestErrors()
{
   // Slot connected to 'best errors' option settings.

   if (fBestErrors->GetState() == kButtonDown)
      fFitOption += 'E';
   else if (fFitOption.Contains('E'))
      fFitOption.ReplaceAll('E', "");
}

//______________________________________________________________________________
void TFitEditor::DoIntegral()
{
   // Slot connected to 'integral' option settings.

   if (fIntegral->GetState() == kButtonDown)
      fFitOption += 'I';
   else if (fFitOption.Contains('I'))
      fFitOption.ReplaceAll('I', "");
}

//______________________________________________________________________________
void TFitEditor::DoLinearFit()
{
   // Slot connected to linear fit settings.

   if (fLinearFit->GetState() == kButtonDown) {
      fPlus = "++";
      if (fFitOption.Contains('F'))
         fFitOption.ReplaceAll('F', "");
      fSetParam->SetState(kButtonDisabled);
      fBestErrors->SetState(kButtonDisabled);
      fImproveResults->SetState(kButtonDisabled);
   } else {
      fPlus = '+';
      if (fFunction.Contains("pol") || fFunction.Contains("++"))
         fFitOption += 'F';
      fSetParam->SetState(kButtonUp);
      fBestErrors->SetState(kButtonUp);
      fImproveResults->SetState(kButtonUp);
   }
}

//______________________________________________________________________________
void TFitEditor::DoMethod(Int_t id)
{
   // Slot connected to fit method settings.

   if (id == kFP_MCHIS) {
      if (fFitOption.Contains('L'))
         fFitOption.ReplaceAll('L', "");
   } else {
      fFitOption += 'L';
   }
}

//______________________________________________________________________________
void TFitEditor::DoNoChi2()
{
   // Slot connected to 'no chi2' option settings.

   if (fNoChi2->GetState() == kButtonDown)
      fFitOption += 'C';
   else if (fFitOption.Contains('C'))
      fFitOption.ReplaceAll('C', "");

   if (fLinearFit->GetState() == kButtonUp)
      fLinearFit->SetState(kButtonDown, kTRUE);
}

//______________________________________________________________________________
void TFitEditor::DoNoDrawing()
{
   // Slot connected to 'no drawing' settings.

   if (fNoDrawing->GetState() == kButtonDown)
      fFitOption += '0';
   else if (fFitOption.Contains('0'))
      fFitOption.ReplaceAll('0', "");
}

//______________________________________________________________________________
void TFitEditor::DoNoStoreDrawing()
{
   // Slot connected to 'no storing, no drawing' settings.

   if (fNoStoreDrawing->GetState() == kButtonDown)
      fFitOption += 'N';
   else if (fFitOption.Contains('N'))
      fFitOption.ReplaceAll('N', "");

   if (fNoDrawing->GetState() == kButtonUp)
      fNoDrawing->SetState(kButtonDown);
}

//______________________________________________________________________________
void TFitEditor::DoPrintOpt(Bool_t on)
{
   // Slot connected to print option settings.

   TGButton *btn = (TGButton *) gTQSender;
   Int_t id = btn->WidgetId();
   switch (id) {
      case kFP_PDEF:
         if (on) {
            fOptDefault->SetState(kButtonDown);
            fOptVerbose->SetState(kButtonUp);
            fOptQuiet->SetState(kButtonUp);
            if (fFitOption.Contains('Q')) {
               fFitOption.ReplaceAll('Q', "");
            }
            if (fFitOption.Contains('V')) {
               fFitOption.ReplaceAll('V', "");
            }
         }
         fStatusBar->SetText("Prn: DEF",3);
         break;
      case kFP_PVER:
         if (on) {
            fOptVerbose->SetState(kButtonDown);
            fOptDefault->SetState(kButtonUp);
            fOptQuiet->SetState(kButtonUp);
            if (fFitOption.Contains('Q')) {
               fFitOption.ReplaceAll('Q', "");
            }
            fFitOption += 'V';
         }
         fStatusBar->SetText("Prn: VER",3);
         break;
      case kFP_PQET:
         if (on) {
            fOptQuiet->SetState(kButtonDown);
            fOptDefault->SetState(kButtonUp);
            fOptVerbose->SetState(kButtonUp);
            if (fFitOption.Contains('V')) {
               fFitOption.ReplaceAll('V', "");
            }
            fFitOption += 'Q';
         }
         fStatusBar->SetText("Prn: QT",3);
      default:
         break;
   }
}

//______________________________________________________________________________
void TFitEditor::DoReset()
{
   // Reset all fit parameters.

   fParentPad->Modified();
   fParentPad->Update();
   fFitOption = 'R';
   fDrawOption = GetDrawOption();
   fFunction = "gaus";
   if (fFitFunc) {
      delete fFitFunc;
      fFitFunc = new TF1("fitFunc", fFunction.Data(), fXmin, fXmax);
   }
   if (fXmin > 1 || fXmax < fXrange) {
      fSliderX->SetRange(fXmin,fXmax);
      fSliderX->SetPosition(fXmin, fXmax);
   } else {
      fSliderX->SetRange(1,fXrange);
      fSliderX->SetPosition(fXmin,fXmax);
   }
   fPlus = '+';
   if (fLinearFit->GetState() == kButtonDown)
      fLinearFit->SetState(kButtonUp, kTRUE);
   if (fBestErrors->GetState() == kButtonDown)
      fBestErrors->SetState(kButtonUp, kFALSE);
   if (fUseRange->GetState() == kButtonUp)
      fUseRange->SetState(kButtonDown, kFALSE);
   if (fAllWeights1->GetState() == kButtonDown)
      fAllWeights1->SetState(kButtonUp, kFALSE);
   if (fEmptyBinsWghts1->GetState() == kButtonDown)
      fEmptyBinsWghts1->SetState(kButtonUp, kFALSE);
   if (fImproveResults->GetState() == kButtonDown)
      fImproveResults->SetState(kButtonUp, kFALSE);
   if (fAdd2FuncList->GetState() == kButtonDown)
      fAdd2FuncList->SetState(kButtonUp, kFALSE);
   if (fNoChi2->GetState() == kButtonDown)
      fNoChi2->SetState(kButtonUp, kFALSE);
   if (fDrawSame->GetState() == kButtonDown)
      fDrawSame->SetState(kButtonUp, kFALSE);
   if (fNoDrawing->GetState() == kButtonDown)
      fNoDrawing->SetState(kButtonUp, kFALSE);
   if (fNoStoreDrawing->GetState() == kButtonDown)
      fNoStoreDrawing->SetState(kButtonUp, kFALSE);
   fNone->SetState(kButtonDown);
   fFuncList->Select(1, kTRUE);

   // minimization tab
   if (fLibMinuit->GetState() != kButtonDown)
      fLibMinuit->SetState(kButtonDown, kTRUE);
   if (fMigrad->GetState() != kButtonDown)
      fMigrad->SetState(kButtonDown, kTRUE);
   if (fOptDefault->GetState() != kButtonDown)
      fOptDefault->SetState(kButtonDown, kTRUE);
   fErrorScale->SetNumber(1.0);
   fErrorScale->ReturnPressed();
   fTolerance->SetNumber(1e-6);
   fTolerance->ReturnPressed();
   fIterations->SetIntNumber(5000);
   fIterations->ReturnPressed();
}

//______________________________________________________________________________
void TFitEditor::DoRobust()
{
   // Slot connected to robust setting of linear fit.

   if (fType != kObjectGraph) return;

   fRobustValue->SetState(kTRUE);
   if (fFitOption.Contains("ROB")) {
      Int_t pos = fFitOption.Index("=");
      fFitOption.Replace(pos+1, 4, Form("%g",fRobustValue->GetNumber()));
   } else {
      fFitOption += Form("ROB=%g", fRobustValue->GetNumber());
   }
}

//______________________________________________________________________________
void TFitEditor::DoBound(Bool_t on)
{
   // Slot connected to 'B' option setting.

   TString s = fFitOption;
   if (s.Contains("ROB")) {
      s.ReplaceAll("ROB", "H");
   }
   if (on) {
      if (s.Contains('B'))
         return;
      else 
         fFitOption += 'B';
   } else {
      if (s.Contains('B')) {
      Int_t pos = fFitOption.First('B');
      Int_t rob = fFitOption.Index("ROB");
      if (pos != rob+2)
         fFitOption.Remove(pos, 1);
      }
   }
}

//______________________________________________________________________________
void TFitEditor::DoSetParameters()
{
   // Open set parameters dialog.

   if (!fFitFunc) {
      printf("SetParamters - create fit function %s\n",fFunction.Data());
      fFitFunc = new TF1("fitFunc",Form("%s",fFunction.Data()), fXmin, fXmax);
   }
   fParentPad->Disconnect("RangeAxisChanged()");
   Double_t xmin, xmax;
   fFitFunc->GetRange(xmin, xmax);
   Int_t ret = 0;
   new TFitParametersDialog(gClient->GetDefaultRoot(), GetMainFrame(), 
                            fFitFunc, fParentPad, xmin, xmax, &ret);

   TGTextLBEntry *te = (TGTextLBEntry *)fFuncList->GetSelectedEntry();
   if ((fNone->GetState() == kButtonDown) && 
       strcmp(te->GetTitle(), "user")) {
      if (ret == kFPDBounded) {
         DoBound(kTRUE);
      } else {
         DoBound(kFALSE);
      }
   }
   fParentPad->Connect("RangeAxisChanged()", "TFitEditor", this, "UpdateGUI()");
}

//______________________________________________________________________________
void TFitEditor::DoSliderXPressed()
{
   // Slot connected to range settings on x-axis.

   if (!fParentPad) return;

   TVirtualPad *save = 0;
   save = gPad;
   gPad = fParentPad;
   fParentPad->cd();

   fParentPad->GetCanvas()->FeedbackMode(kFALSE);
   fParentPad->SetLineWidth(1);
   fParentPad->SetLineColor(2);
   Float_t xleft = 0;
   Double_t xright = 0;
   switch (fType) {
      case kObjectHisto: {
         //hist 1dim
         xleft  = fXaxis->GetBinLowEdge((Int_t)((fSliderX->GetMinPosition())+0.5));
         xright = fXaxis->GetBinUpEdge((Int_t)((fSliderX->GetMaxPosition())+0.5));
         break;
      }
      case kObjectGraph: {
         // graph
         xleft  = fXaxis->GetBinLowEdge((Int_t)((fSliderX->GetMinPosition())+0.5));
         xright = fXaxis->GetBinUpEdge((Int_t)((fSliderX->GetMaxPosition())+0.5));
         break;
      }
      case kObjectGraph2D: {
         // N/A
         break;
      }
      case kObjectHStack: {
         // N/A
         break;
      }
      case kObjectTree:  {
         // N/A
         break;
      }
   }
   Float_t ymin = fParentPad->GetUymin();
   Float_t ymax = fParentPad->GetUymax();
   fPx1old = fParentPad->XtoAbsPixel(xleft);
   fPy1old = fParentPad->YtoAbsPixel(ymin);
   fPx2old = fParentPad->XtoAbsPixel(xright);
   fPy2old = fParentPad->YtoAbsPixel(ymax);
   gVirtualX->DrawBox(fPx1old, fPy1old, fPx2old, fPy2old, TVirtualX::kHollow);

   if(save) gPad = save;
}

//______________________________________________________________________________
void TFitEditor::DoSliderXMoved()
{
   // Slot connected to range settings on x-axis.

   Int_t px1,py1,px2,py2;
   Float_t xleft = 0;
   Double_t xright = 0;

   TVirtualPad *save = 0;
   save = gPad;
   gPad = fParentPad;
   gPad->cd();

   switch (fType) {
      case kObjectHisto: {
         xleft  = fXaxis->GetBinLowEdge((Int_t)((fSliderX->GetMinPosition())+0.5));
         xright = fXaxis->GetBinUpEdge((Int_t)((fSliderX->GetMaxPosition())+0.5));
         break;
      }
      case kObjectGraph: {
         // graph
         xleft  = fXaxis->GetBinLowEdge((Int_t)((fSliderX->GetMinPosition())+0.5));
         xright = fXaxis->GetBinUpEdge((Int_t)((fSliderX->GetMaxPosition())+0.5));
/*        TGraph *gr = (TGraph *)fFitObject;
         Int_t np = gr->GetN();
         Double_t *x = gr->GetX();
         xleft  = x[0];
         xright = x[0];
         for(Int_t i=0; i<np; i++) {
            if (xleft > x[i])
               xleft = x[i];
            if (xright < x[i])
               xright = x[i];
         }*/
         break;
      }
      case kObjectGraph2D: {
         // N/A
         break;
      }
      case kObjectHStack: {
         // N/A
         break;
      }
      case kObjectTree:  {
         // N/A
         break;
      }
   }
   Float_t ymin = gPad->GetUymin();
   Float_t ymax = gPad->GetUymax();
   px1 = gPad->XtoAbsPixel(xleft);
   py1 = gPad->YtoAbsPixel(ymin);
   px2 = gPad->XtoAbsPixel(xright);
   py2 = gPad->YtoAbsPixel(ymax);
   gPad->GetCanvas()->FeedbackMode(kTRUE);
   gPad->SetLineWidth(1);
   gPad->SetLineColor(2);
   gVirtualX->DrawBox(fPx1old, fPy1old, fPx2old, fPy2old, TVirtualX::kHollow);
   gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
   fPx1old = px1;
   fPy1old = py1;
   fPx2old = px2 ;
   fPy2old = py2;

   if(save) gPad = save;
}

//______________________________________________________________________________
void TFitEditor::DoSliderXReleased()
{
   // Slot connected to range settings on x-axis.

   gVirtualX->Update(0);
}

//______________________________________________________________________________
void TFitEditor::DoSliderYPressed()
{
   // Slot connected to range settings on y-axis.

   fParentPad->cd();
   switch (fType) {
      case kObjectHisto: {
         if (!fParentPad) return;
         fParentPad->cd();
         fParentPad->GetCanvas()->FeedbackMode(kFALSE);
         fParentPad->SetLineWidth(1);
         fParentPad->SetLineColor(2);
         //hist 1dim
         Float_t ybottom = fYaxis->GetBinLowEdge((Int_t)((fSliderY->GetMinPosition())+0.5));
         Float_t ytop = fYaxis->GetBinUpEdge((Int_t)((fSliderY->GetMaxPosition())+0.5));
         Float_t xmin = fParentPad->GetUxmin();
         Float_t xmax = fParentPad->GetUxmax();
         fPx1old = fParentPad->XtoAbsPixel(xmin);
         fPy1old = fParentPad->YtoAbsPixel(ybottom);
         fPx2old = fParentPad->XtoAbsPixel(xmax);
         fPy2old = fParentPad->YtoAbsPixel(ytop);
         gVirtualX->DrawBox(fPx1old, fPy1old, fPx2old, fPy2old, TVirtualX::kHollow);
      }
      case kObjectGraph: {
         // N/A
         break;
      }
      case kObjectGraph2D: {
         // N/A
         break;
      }
      case kObjectHStack: {
         // N/A
         break;
      }
      case kObjectTree:  {
         // N/A
         break;
      }
   }
}

//______________________________________________________________________________
void TFitEditor::DoSliderYMoved()
{
   // Slot connected to range settings on y-axis.

   switch (fType) {
      case kObjectHisto: {
         Int_t px1,py1,px2,py2;
         Float_t ybottom = fYaxis->GetBinLowEdge((Int_t)((fSliderY->GetMinPosition())+0.5));
         Float_t ytop = fYaxis->GetBinUpEdge((Int_t)((fSliderY->GetMaxPosition())+0.5));
         Float_t xmin = fParentPad->GetUxmin();
         Float_t xmax = fParentPad->GetUxmax();
         px1 = fParentPad->XtoAbsPixel(xmin);
         py1 = fParentPad->YtoAbsPixel(ybottom);
         px2 = fParentPad->XtoAbsPixel(xmax);
         py2 = fParentPad->YtoAbsPixel(ytop);
         fParentPad->GetCanvas()->FeedbackMode(kTRUE);
         fParentPad->cd();
         fParentPad->SetLineWidth(1);
         fParentPad->SetLineColor(2);
         gVirtualX->DrawBox(fPx1old, fPy1old, fPx2old, fPy2old, TVirtualX::kHollow);
         gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         fPx1old = px1;
         fPy1old = py1;
         fPx2old = px2 ;
         fPy2old = py2;
         gVirtualX->Update(0);
      }
      case kObjectGraph: {
         // N/A
         break;
      }
      case kObjectGraph2D: {
         // N/A
         break;
      }
      case kObjectHStack: {
         // N/A
         break;
      }
      case kObjectTree:  {
         // N/A
         break;
      }
   }
}

//______________________________________________________________________________
void TFitEditor::DoSliderYReleased()
{
   // Slot connected to range settings on y-axis.

   fParentPad->Modified();
   fParentPad->Update();
}

//______________________________________________________________________________
void TFitEditor::DoSliderZPressed()
{
   // Slot connected to range settings on z-axis.

}

//______________________________________________________________________________
void TFitEditor::DoSliderZMoved()
{
   // Slot connected to range settings on z-axis.

}

//______________________________________________________________________________
void TFitEditor::DoSliderZReleased()
{
   // Slot connected to range settings on z-axis.

}

//______________________________________________________________________________
void TFitEditor::DoUserDialog()
{
   // Open a dialog for getting a user defined method.

   new TGMsgBox(fClient->GetRoot(), GetMainFrame(),
                "Info", "Dialog of user method is not implemented yet",
                kMBIconAsterisk,kMBOk, 0);
}

//______________________________________________________________________________
void TFitEditor::DoUseRange()
{
   // Slot connected to fit range settings.

   if (fUseRange->GetState() == kButtonDown)
      fFitOption.Insert(0,'R');
   else  {
      Int_t pos = fFitOption.First('R');
      Int_t rob = fFitOption.Index("ROB");
      if (pos != rob)
         fFitOption.Remove(pos, 1);
   }
}

//______________________________________________________________________________
void TFitEditor::SetFunction(const char *function)
{
   // Set the function to be used in performed fit.

   fFunction = function;
}

//______________________________________________________________________________
Bool_t TFitEditor::SetObjectType(TObject* obj)
{
   // Check whether the object suitable for fitting and set 
   // its type, dimension and method combo box accordingly.
   
   Bool_t set = kFALSE;

   if (obj->InheritsFrom("TGraph")) {
      fType = kObjectGraph;
      TF1 *f1 =((TGraph *)obj)->GetFunction("fitFunc");
      if (f1) {
         fFitFunc = new TF1();
         f1->Copy(*fFitFunc);
      }
      set = kTRUE;
      fDim = 1;
      if (fMethodList->FindEntry("Binned Likelihood"))
         fMethodList->RemoveEntry(kFP_MBINL);
      if (!fMethodList->FindEntry("Chi-square"))
         fMethodList->AddEntry("Chi-square", kFP_MCHIS);
      fMethodList->Select(kFP_MCHIS, kFALSE);
      fRobustValue->SetState(kTRUE);
      fRobustValue->GetNumberEntry()->SetToolTipText("Set robust value");
   } else if (obj->InheritsFrom("TGraph2D")) {
      fType = kObjectGraph2D;
      set = kTRUE;
      fDim = 2;
      if (fMethodList->FindEntry("Unbinned Likelihood"))
         fMethodList->RemoveEntry(kFP_MUBIN);
      if (!fMethodList->FindEntry("Chi-square"))
         fMethodList->AddEntry("Chi-square", kFP_MCHIS);
      fMethodList->Select(kFP_MCHIS, kFALSE);
   } else if (obj->InheritsFrom("THStack")) {
      fType = kObjectHStack;
      set = kTRUE;
      TH1 *hist = (TH1 *)((THStack *)obj)->GetHists()->First();
      fDim = hist->GetDimension();
      if (fMethodList->FindEntry("Unbinned Likelihood"))
         fMethodList->RemoveEntry(kFP_MUBIN);
      if (!fMethodList->FindEntry("Chi-square"))
         fMethodList->AddEntry("Chi-square", kFP_MCHIS);
      fMethodList->Select(kFP_MCHIS, kFALSE);
   } else if (obj->InheritsFrom("TTree")) {
      fType = kObjectTree;
      set = kTRUE;
      fDim = -1; //not implemented
      fMethodList->SetEnabled(kFALSE);
   } else if (obj->InheritsFrom("TH1")){
      fType = kObjectHisto;
      TF1 *f1 =((TH1 *)obj)->GetFunction("fitFunc");
      if (f1) {
         fFitFunc = new TF1();
         f1->Copy(*fFitFunc);
      }
      set = kTRUE;
      fDim = ((TH1*)obj)->GetDimension();
      if (!fMethodList->FindEntry("Binned Likelihood"))
         fMethodList->AddEntry("Binned Likelihood", kFP_MBINL);
      if (!fMethodList->FindEntry("Chi-square"))
         fMethodList->AddEntry("Chi-square", kFP_MCHIS);
      fMethodList->Select(kFP_MCHIS, kFALSE);
   }
   return set;
}

//______________________________________________________________________________
void TFitEditor::ShowObjectName(TObject* obj)
{
   // Show object name on the top.
   
   TString name;
   
   if (obj) {
      name = obj->GetName();
      name.Append("::");
      name.Append(obj->ClassName());
   } else {
      name = "No object selected";
   }
   fObjLabel->SetText(name.Data());
   fObjLabelParent->Resize(GetDefaultSize());
   Layout();
}

//______________________________________________________________________________
Option_t *TFitEditor::GetDrawOption() const
{
   // Get draw options of the selected object.

   if (!fParentPad) return "";

   TListIter next(fParentPad->GetListOfPrimitives());
   TObject *obj;
   while ((obj = next())) {
      if (obj == fFitObject) return next.GetOption();
   }
   return "";
}

//______________________________________________________________________________
void TFitEditor::DoLibrary(Bool_t on)
{
   // Set selected minimization library in use.

   TGButton *bt = (TGButton *)gTQSender;
   Int_t id = bt->WidgetId(); 

   switch (id) {

      case kFP_LMIN:
         {
            if (on) {
               fLibMinuit->SetState(kButtonDown);
               fLibMinuit2->SetState(kButtonUp);
               fLibFumili->SetState(kButtonUp);
               if (fFumili->GetState() != kButtonDisabled) {
                  fFumili->SetState(kButtonDisabled);
               }
               fMigrad->SetState(kButtonDown);
               fStatusBar->SetText("MIGRAD", 1);
               // Simplex functionality will come with the new fitter design    
               //if (fSimplex->GetState() == kButtonDisabled)
               //   fSimplex->SetState(kButtonUp);
               TVirtualFitter::SetDefaultFitter("Minuit");
               fStatusBar->SetText("LIB Minuit", 0);
            }
            
         }
         break;
      
      case kFP_LMIN2:
         {
            if (on) {
               fLibMinuit->SetState(kButtonUp);
               fLibMinuit2->SetState(kButtonDown);
               fLibFumili->SetState(kButtonUp);
               // Simplex functionality will come with the new fitter design    
               //if (fSimplex->GetState() == kButtonDisabled)
               //   fSimplex->SetState(kButtonUp);
               if (fMigrad->GetState() == kButtonDisabled)
                  fMigrad->SetState(kButtonUp);
               if (fFumili->GetState() == kButtonDisabled)
                  fFumili->SetState(kButtonUp);
               if (fMigrad->GetState() == kButtonDown)
                  TVirtualFitter::SetDefaultFitter("Minuit2");
               else if (fFumili->GetState() == kButtonDown)
                  TVirtualFitter::SetDefaultFitter("Fumili2");
               fStatusBar->SetText("LIB Minuit2", 0);
            }
         }
         break;
      
      case kFP_LFUM:
         {
            if (on) {
               if (fFumili->GetState() != kButtonDown) {
                  fFumili->SetState(kButtonDown);
                  fStatusBar->SetText("FUMILI", 1);
               }
               fLibMinuit->SetState(kButtonUp);
               fLibMinuit2->SetState(kButtonUp);
               fLibFumili->SetState(kButtonDown);
               TVirtualFitter::SetDefaultFitter("Fumili");
               fMigrad->SetState(kButtonDisabled);
               // Simplex functionality will come with the new fitter design    
               //fSimplex->SetState(kButtonDisabled);
               fStatusBar->SetText("LIB Fumili", 0);
            }
         }
      default:
         break;
   }
   
}

//______________________________________________________________________________
void TFitEditor::DoMinMethod(Bool_t on)
{
   // Set selected minimization method in use.

   TGButton *bt = (TGButton *)gTQSender;
   Int_t id = bt->WidgetId(); 

   switch (id) {

      case kFP_MIGRAD:
         {
            if (on) {
               // Simplex functionality will come with the new fitter design    
               //fSimplex->SetState(kButtonUp);
               if (fLibMinuit->GetState() == kButtonDown)
                  fFumili->SetState(kButtonDisabled);
               else
                  fFumili->SetState(kButtonUp);
               fMigrad->SetState(kButtonDown);
               fStatusBar->SetText("MIGRAD",1);
               if (fLibMinuit2->GetState() == kButtonDown)
                  if (strncmp(TVirtualFitter::GetDefaultFitter(),"Minuit2",7) != 0) 
                     TVirtualFitter::SetDefaultFitter("Minuit2");
            }
         }
         break;
      
      case kFP_SIMPLX:
         {
            if (on) {
               // Simplex functionality will come with the new fitter design    
               //fMigrad->SetState(kButtonUp);
               //if (fLibMinuit->GetState() == kButtonDown)
               //   fFumili->SetState(kButtonDisabled);
               //else
               //   fFumili->SetState(kButtonUp);
               //fSimplex->SetState(kButtonDown);
               //fStatusBar->SetText("SIMPLEX",1);
            }
         }
         break;
      
      case kFP_FUMILI:
         {
            if (on) {
               fMigrad->SetState(kButtonUp);
               // Simplex functionality will come with the new fitter design    
               //fSimplex->SetState(kButtonUp);
               fFumili->SetState(kButtonDown);
               fStatusBar->SetText("FUMILI",1);
               if (fLibMinuit2->GetState() == kButtonDown)
                  TVirtualFitter::SetDefaultFitter("Fumili2");
               else
                  TVirtualFitter::SetDefaultFitter("Fumili");
            }
         }
         break;
      
   }
}

//______________________________________________________________________________
void TFitEditor::DoErrorsDef()
{
   // Set the error definition for default fitter.
   
   Double_t err = fErrorScale->GetNumber();
   TVirtualFitter::SetErrorDef(err);
}

//______________________________________________________________________________
void TFitEditor::DoMaxTolerance()
{
   // Set the fit relative precision.
   
   Double_t tol = fTolerance->GetNumber();
   TVirtualFitter::SetPrecision(tol);
}

//______________________________________________________________________________
void TFitEditor::DoMaxIterations()
{
   // Set the maximum number of iterations.

   Long_t itr = fIterations->GetIntNumber();
   TVirtualFitter::SetMaxIterations(itr);
   fStatusBar->SetText(Form("Itr: %ld",itr),2);
}

//______________________________________________________________________________
void TFitEditor::MakeTitle(TGCompositeFrame *parent, const char *title)
{
   // Create section title in the GUI.

   TGCompositeFrame *ht = new TGCompositeFrame(parent, 350, 10, 
                                               kFixedWidth | kHorizontalFrame);
   ht->AddFrame(new TGLabel(ht, title),
                new TGLayoutHints(kLHintsLeft, 1, 1, 0, 0));
   ht->AddFrame(new TGHorizontal3DLine(ht),
                new TGLayoutHints(kLHintsExpandX | kLHintsCenterY, 5, 5, 2, 2));
   parent->AddFrame(ht, new TGLayoutHints(kLHintsTop, 5, 0, 5, 0));
}


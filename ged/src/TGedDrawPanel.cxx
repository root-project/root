// @(#)root/ged:$Name:  $:$Id: TGedDrawPanel.cxx,v 1.2 2004/02/18 20:31:36 brun Exp $
// Author: Marek Biskup, Ilka Antcheva 15/08/2003

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGedDrawPanel                                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TGedDrawPanel.h"
#include "TCanvas.h"
#include "TGButton.h"
#include "TH1.h"
#include "TF1.h"
#include "TGDoubleSlider.h"
#include <ctype.h>
#include <snprintf.h>


ClassImp(TGedDrawPanel)

enum {
  kType = 1,
     kTypeHist        = 0,
     kTypeLego1       = 1,
     kTypeLego2       = 2,
     kTypeLego3       = 3,
     kTypeSurf        = 4,
     kTypeSurfColors  = 5,
     kTypeSurfContour = 6,
     kTypeGouraud     = 7,
     kTypeCont0       = 8,
     kTypeCont1       = 9,
     kTypeCont2       = 10, 
     kTypeCont3       = 11,
  kCoords = 2,
     kCoordsCartesian = 0,
     kCoordsPolar     = 1,
     kCoordsCylindric = 2,
     kCoordsSpheric   = 3,
  kErrors = 3,
     kErrorsNone = 0,
     kErrorsEdges      = 1,
     kErrorsRectangles = 2,
     kErrorsFill       = 3,
     kErrorsContour    = 4,

  kDraw     = 1000,
  kDefaults = 1001,
  kClose    = 1002
};


static const char * const TypeData[] = {
   "hist",  "lego1", "lego2", "lego3", 
   "surf1", "surf2", "surf3", "surf4",
   "cont1", "cont2", "cont3", "cont4"
};

static const char * const CoordsData[] = {
   "", "pol", "cyl", "sph"
};

static const char * const ErrorsData[] = {
   "", "e1", "e2", "e3", "e4"
};


TGedDrawPanel* TGedDrawPanel::fDrawPanel = 0;

//______________________________________________________________________________
void TGedDrawPanel::ShowPanel(TObject* histogram, TPad* pad)
{
   if (!fDrawPanel)
      fDrawPanel = new TGedDrawPanel();
   fDrawPanel->SetHistogram(histogram, pad);
   fDrawPanel->MapRaised();
}

//______________________________________________________________________________
TGedDrawPanel::TGedDrawPanel() :
   TGMainFrame(gClient->GetRoot(), 300, 500, kVerticalFrame)
{
   fHistogram =0;
   fRefPad = 0;
   Build();
   MapWindow();
}

//______________________________________________________________________________
TGedDrawPanel::~TGedDrawPanel()
{
   fDrawPanel = 0;
   Cleanup();
}

//______________________________________________________________________________
void TGedDrawPanel::Build()
{
   TGGroupFrame *type = new TGGroupFrame(this, "type", kHorizontalFrame);
   AddFrame(type, new TGLayoutHints(kLHintsExpandX));

   TGCompositeFrame *t1 = new TGCompositeFrame(type, 10, 10);
   type->AddFrame(t1, new TGLayoutHints(kLHintsExpandY));
   TGCompositeFrame *t2 = new TGCompositeFrame(type, 10, 10);
   type->AddFrame(t2, new TGLayoutHints(kLHintsExpandY));
   TGCompositeFrame *t3 = new TGCompositeFrame(type, 10, 10);
   type->AddFrame(t3, new TGLayoutHints(kLHintsExpandY));
   TGCompositeFrame *t4 = new TGCompositeFrame(type, 10, 10);
   type->AddFrame(t4, new TGLayoutHints(kLHintsExpandY));
   
   fType[0] = new TGRadioButton(t1, "hist", kTypeHist);
   t1->AddFrame(fType[0], new TGLayoutHints(kLHintsTop));

   fType[1] = new TGRadioButton(t2, "lego1", kTypeLego1);
   t2->AddFrame(fType[1], new TGLayoutHints(kLHintsTop));

   fType[2] = new TGRadioButton(t2, "lego2", kTypeLego2);
   t2->AddFrame(fType[2], new TGLayoutHints(kLHintsTop));
   
   fType[3] = new TGRadioButton(t2, "lego3", kTypeLego3);
   t2->AddFrame(fType[3], new TGLayoutHints(kLHintsTop));

   fType[4] = new TGRadioButton(t3, "surf", kTypeSurf);
   t3->AddFrame(fType[4], new TGLayoutHints(kLHintsTop));

   fType[5] = new TGRadioButton(t3, "surf/colors", kTypeSurfColors);
   t3->AddFrame(fType[5], new TGLayoutHints(kLHintsTop));

   fType[6] = new TGRadioButton(t3, "surf/contour", kTypeSurfContour);
   t3->AddFrame(fType[6], new TGLayoutHints(kLHintsTop));

   fType[7] = new TGRadioButton(t3, "gouraud", kTypeGouraud);
   t3->AddFrame(fType[7], new TGLayoutHints(kLHintsTop));

   fType[8] = new TGRadioButton(t4, "cont0", kTypeCont0);
   t4->AddFrame(fType[8], new TGLayoutHints(kLHintsTop));

   fType[9] = new TGRadioButton(t4, "cont1", kTypeCont1);
   t4->AddFrame(fType[9], new TGLayoutHints(kLHintsTop));

   fType[10] = new TGRadioButton(t4, "cont2", kTypeCont2);
   t4->AddFrame(fType[10], new TGLayoutHints(kLHintsTop));

   fType[11] = new TGRadioButton(t4, "cont3", kTypeCont3);
   t4->AddFrame(fType[11], new TGLayoutHints(kLHintsTop));

   for (int i = 0; i < 12; i++) {
      fType[i]->SetUserData( (void*) kType);
      fType[i]->Associate(this);
   }

   TGCompositeFrame *row2 = new TGCompositeFrame(this, 10, 10, kHorizontalFrame);
   AddFrame(row2, new TGLayoutHints(kLHintsExpandX));
   
   TGGroupFrame *coords = new TGGroupFrame(row2, "coords", kVerticalFrame);
   row2->AddFrame(coords, new TGLayoutHints(kLHintsExpandX | kLHintsExpandY));

   fCoords[0] = new TGRadioButton(coords, "cartesian", kCoordsCartesian);
   coords->AddFrame(fCoords[0], new TGLayoutHints(kLHintsTop));

   fCoords[1] = new TGRadioButton(coords, "polar", kCoordsPolar);
   coords->AddFrame(fCoords[1], new TGLayoutHints(kLHintsTop));

   fCoords[2] = new TGRadioButton(coords, "cylindric", kCoordsCylindric);
   coords->AddFrame(fCoords[2], new TGLayoutHints(kLHintsTop));
   
   fCoords[3] = new TGRadioButton(coords, "spheric", kCoordsSpheric);
   coords->AddFrame(fCoords[3], new TGLayoutHints(kLHintsTop));

   for (int i = 0; i < 4; i++) {
      fCoords[i]->SetUserData( (void*) kCoords);
      fCoords[i]->Associate(this);
   }

   TGGroupFrame *errors = new TGGroupFrame(row2, "errors", kVerticalFrame);
   row2->AddFrame(errors, new TGLayoutHints(kLHintsExpandX));

   fErrors[0] = new TGRadioButton(errors, "No errors", kErrorsNone);
   errors->AddFrame(fErrors[0], new TGLayoutHints(kLHintsTop));

   fErrors[1] = new TGRadioButton(errors, "E1: edges", kErrorsEdges);
   errors->AddFrame(fErrors[1], new TGLayoutHints(kLHintsTop));
   
   fErrors[2] = new TGRadioButton(errors, "E2: rectangles", kErrorsRectangles);
   errors->AddFrame(fErrors[2], new TGLayoutHints(kLHintsTop));
   
   fErrors[3] = new TGRadioButton(errors, "E3: fill", kErrorsFill);
   errors->AddFrame(fErrors[3], new TGLayoutHints(kLHintsTop));

   fErrors[4] = new TGRadioButton(errors, "E4: contour", kErrorsContour);
   errors->AddFrame(fErrors[4], new TGLayoutHints(kLHintsTop));
      
   for (int i = 0; i < 5; i++) {
      fErrors[i]->SetUserData( (void*) kErrors);
      fErrors[i]->Associate(this);
   }

   fSlider = new TGDoubleHSlider(this, 1, 1);
   AddFrame(fSlider, new TGLayoutHints(kLHintsExpandX));
   
   TGCompositeFrame *row3 = new TGCompositeFrame(this, 10, 10, kHorizontalFrame);
   AddFrame(row3, new TGLayoutHints(kLHintsExpandX));
   TGTextButton *draw = new TGTextButton(row3, "Draw", kDraw);
   row3->AddFrame(draw, new TGLayoutHints(kLHintsLeft));
   draw->Associate(this);
   
   TGTextButton *defaults = new TGTextButton(row3, "Defaults", kDefaults);
   row3->AddFrame(defaults, new TGLayoutHints(kLHintsLeft));
   defaults->Associate(this);
   
   TGTextButton *close = new TGTextButton(row3, "Close", kClose);
   row3->AddFrame(close, new TGLayoutHints(kLHintsLeft));
   close->Associate(this);

   MapSubwindows();
   Resize(GetDefaultSize());
   Reset();
}

//______________________________________________________________________________
void TGedDrawPanel::SetRadio(TGRadioButton **group, Int_t count, Int_t index)
{
   for (int i = 0; i < count; i++) {
      if (i == index)
         group[i]->SetState(kButtonDown);
      else
         group[i]->SetState(kButtonUp);
   }
}

//______________________________________________________________________________
Bool_t TGedDrawPanel::ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2)
{
   switch (GET_MSG(msg)) {
      case kC_COMMAND:
         switch (GET_SUBMSG(msg)) {
            case kCM_RADIOBUTTON:
               switch (parm2) {
                  case kType:
                     SetRadio(fType, 12, parm1);
                     break;
                  case kCoords:
                     SetRadio(fCoords, 4, parm1);
                     break;
                  case kErrors:
                     SetRadio(fErrors, 5, parm1);
                     break;
                  default:
                     break;
               }
               break;
            case kCM_BUTTON:
               switch (parm1) {
                  case kDraw:
                     DrawHistogram();
                     break;
                  case kDefaults:
                     Reset();
                     break;
                  case kClose:
                     CloseWindow();
                     break;
                  default:
                     break;
               }
         }
         break;
      case kC_HSLIDER:
         ProcessSlider(GET_SUBMSG(msg));
         break;
   }
   return kTRUE;
}

//______________________________________________________________________________
void TGedDrawPanel::CloseWindow()
{
   DeleteWindow();
}

//______________________________________________________________________________
void TGedDrawPanel::ProcessSlider(Long_t submsg)
{
   // Control mousse events when slider is used in a drawpanel

   Float_t xpmin = fSlider->GetMinPosition();
   Float_t xpmax = fSlider->GetMaxPosition();

   static TH1 *h1;
   static Bool_t done = kFALSE;
   static Int_t px1,py1,px2,py2,nbins;
   static Int_t pxmin,pxmax;
   static Float_t xmin,xmax,ymin,ymax,xleft,xright;
   Int_t first,last;

   if (!fRefPad) return;
   fRefPad->cd();

   fRefPad->GetCanvas()->FeedbackMode(kTRUE);
   gVirtualX->SetLineWidth(2);
   gVirtualX->SetLineColor(-1);

   switch (submsg) {
      case kSL_PRESS:      // button 1 down
         if (done) gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         done = kTRUE;
         if (fHistogram->InheritsFrom("TF1")) {
            h1 = (TH1*)((TF1*)fHistogram)->GetHistogram();
         } else if (fHistogram->InheritsFrom("TH1")) {
            h1 = (TH1*)fHistogram;
         } else {
            h1 = 0;
            break;
         }
         nbins = h1->GetXaxis()->GetNbins();
         xmin  = fRefPad->GetUxmin();
         xmax  = fRefPad->GetUxmax();
         xleft = xmin+(xmax-xmin)*xpmin;
         xright= xmin+(xmax-xmin)*xpmax;
         ymin  = fRefPad->GetUymin();
         ymax  = fRefPad->GetUymax();
         px1   = fRefPad->XtoAbsPixel(xleft);
         py1   = fRefPad->YtoAbsPixel(ymin);
         px2   = fRefPad->XtoAbsPixel(xright);
         py2   = fRefPad->YtoAbsPixel(ymax);
         pxmin = fRefPad->XtoAbsPixel(xmin);
         pxmax = fRefPad->XtoAbsPixel(xmax);
         gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         break;
      case kSL_POS:        //kButton1Motion:
         if (h1 == 0) break;
         gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         first  = 1 + Int_t(nbins*xpmin);
         last   =     Int_t(nbins*xpmax);
         xleft  = fRefPad->XtoPad(h1->GetXaxis()->GetBinLowEdge(first));
         xright = fRefPad->XtoPad(h1->GetXaxis()->GetBinLowEdge(last)+h1->GetXaxis()->GetBinWidth(last));
         px1 = fRefPad->XtoAbsPixel(xleft);
         px2 = fRefPad->XtoAbsPixel(xright);
         if (px1 < pxmin) px1 = pxmin;
         if (px2 > pxmax) px2 = pxmax;
         gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         break;
      case kSL_RELEASE:
         gVirtualX->DrawBox(px1, py1, px2, py2, TVirtualX::kHollow);
         done = kFALSE;
         fRefPad->GetCanvas()->FeedbackMode(kFALSE);
         gVirtualX->SetLineWidth(-1);
         break;
   }
}

//______________________________________________________________________________
void TGedDrawPanel::AddRadioOption(TGRadioButton **group, 
                                   const char * const * const data, Int_t count)
{
   for (int i = 0; i < count; i++)
      if (group[i]->GetState() == kButtonDown)
         fOption += data[i];
}

//______________________________________________________________________________
void TGedDrawPanel::ReadOption()
{
   fOption.Resize(0);
   AddRadioOption(fType, TypeData, 12);
   AddRadioOption(fCoords, CoordsData, 4);
   AddRadioOption(fErrors, ErrorsData, 5);
}


//______________________________________________________________________________
void TGedDrawPanel::ParseRadio(const char * const option, TGRadioButton **group, 
                               const char * const * const data, Int_t count)
{
   for (int i = 0; i < count; i++) {
      if (data[i][0] != '\0')
         if (strstr(option, data[i]))
            group[i]->SetState(kButtonDown);
   }
}

//______________________________________________________________________________
void TGedDrawPanel::ParseOption(const char * const option)
{
   char o[128];
   int i;

   for (i = 0; i < 128 - 1 && option[i] != '\0'; i++) 
      o[i] = tolower(option[i]);

   o[i] = 0;
   ParseRadio(o, fType, TypeData, 12);
   ParseRadio(o, fCoords, CoordsData, 4);
   ParseRadio(o, fErrors, ErrorsData, 5);
}

//______________________________________________________________________________
void TGedDrawPanel::DrawHistogram()
{
   if (!fHistogram || !fRefPad) return;

   ReadOption();
   fRefPad->cd();
   TH1 *h1;

   if (fHistogram->InheritsFrom("TF1")) {
      h1 = (TH1*)((TF1*)fHistogram)->GetHistogram();
   } else if (fHistogram->InheritsFrom("TH1")) {
      h1 = (TH1*)fHistogram;
   } else {
      h1 = 0;
   }

   if (h1) {
      Int_t nbins   = h1->GetXaxis()->GetNbins();
      Int_t first   = 1 + Int_t(nbins*fSlider->GetMinPosition());
      Int_t last    =     Int_t(nbins*fSlider->GetMaxPosition());
      h1->GetXaxis()->SetRange(first,last);
   }

   Int_t keep = fHistogram->TestBit(kCanDelete);
   fHistogram->SetBit(kCanDelete,0);
   fHistogram->Draw((char*)fOption.Data());
   fHistogram->SetBit(kCanDelete,keep);
   fRefPad->Update();
}

//______________________________________________________________________________
void TGedDrawPanel::Reset()
{
   SetRadio(fType, 12, 0);
   SetRadio(fCoords, 4, 0);
   SetRadio(fErrors, 5, 0);
   fSlider->SetPosition(0.0, 1.0);
}

//______________________________________________________________________________
void TGedDrawPanel::Clean()
{
   SetRadio(fType, 12, -1);
   SetRadio(fCoords, 4, -1);
   SetRadio(fErrors, 5, -1);
   fSlider->SetPosition(0.0, 1.0);
}

//______________________________________________________________________________
void TGedDrawPanel::SetHistogram(TObject* histogram, TPad* pad)
{
  // Set default draw panel options.

   ParseOption(histogram->GetDrawOption());
   fRefPad    = pad;
   fHistogram = histogram;

   Clean();
   
   char cmd[64];
   if (fHistogram) {
      snprintf(cmd, 64, "drawpanel: %s",fHistogram->GetName());
      SetWindowName(cmd);
      ParseOption(histogram->GetDrawOption());
   }
}

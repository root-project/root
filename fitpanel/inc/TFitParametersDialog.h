// @(#)root/fitpanel:$Name:$:$Id:$
// Author:Ilka Antcheva, Lorenzo Moneta 03/10/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFitParametersDialog
#define ROOT_TFitParametersDialog

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TFitParametersDialog                                                //
//                                                                      //
//  This class is used for fit function parameter settings.             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGButton
#include "TGWidget.h"
#endif
#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif
#ifndef ROOT_TF1
#include "TF1.h"
#endif


class TGNumberEntry;
class TGTextEntry;
class TGCheckButton;
class TGTextButton;
class TGTripleHSlider;
class TGNumberEntryField;
class TVirtualPad;


class TFitParametersDialog : public TGTransientFrame {

protected:
   TF1                 *fFunc;            // function passed to this dialog
   TVirtualPad         *fFpad;            // pad where the function is drawn
   Int_t                fNP;              // number of function parameters
   Double_t            *fPmin;            // min limits of patameters range
   Double_t            *fPmax;            // max limits of patameters range
   Double_t            *fPval;            // original patameters' values
   Double_t            *fPerr;            // original patameters' errors
   Double_t            *fPstp;            // original patameters' step
   Double_t             fRangexmin;       // min limits of patameters range
   Double_t             fRangexmax;       // max limits of patameters range
   Double_t             fRXmin;           // original min range
   Double_t             fRXmax;           // original max range
   TGCompositeFrame    *fContNam;         // container of parameter names
   TGCompositeFrame    *fContVal;         // container of parameter values
   TGCompositeFrame    *fContFix;         // container of fix settings
   TGCompositeFrame    *fContBnd;         // container of bound settings
   TGCompositeFrame    *fContSld;         // container of sliders
   TGCompositeFrame    *fContMin;         // container of min range values
   TGCompositeFrame    *fContMax;         // container of max range values
   TGCompositeFrame    *fContStp;         // container of step values
   TGCompositeFrame    *fContErr;         // container of error values
   TGTextEntry         **fParNam;         // parameter names
   TGCheckButton       **fParBnd;         // bound setting switch
   TGCheckButton       **fParFix;         // fix setting switch
   TGNumberEntry       **fParVal;         // parameter values
   TGNumberEntryField  **fParMin;         // min range values
   TGNumberEntryField  **fParMax;         // max range values
   TGNumberEntry       **fParStp;         // step values
   TGTripleHSlider     **fParSld;         // triple sliders
   TGNumberEntryField  **fParErr;         // error values
   TGCheckButton       *fUpdate;          // immediate update switch
   TGTextButton        *fApply;           // Apply button
   TGTextButton        *fReset;           // Reset button
   TGTextButton        *fOK;              // OK button
   TGTextButton        *fCancel;          // Cancel button
   Bool_t              fHasChanges;       // kTRUE if function was redrawn;
   Bool_t              fImmediateDraw;    // kTRUE if function is updated on run-time

public:
   TFitParametersDialog(const TGWindow *p, const TGWindow *main,
                             TF1 *func, TVirtualPad *pad,
                             Double_t rmin=1., Double_t rmax=2.);
   virtual ~TFitParametersDialog();

   virtual void  CloseWindow();
   virtual void  DoApply();
   virtual void  DoCancel();
   virtual void  DoFix(Bool_t on);
   virtual void  DoBound(Bool_t on);
   virtual void  DoOK();
   virtual void  DoParMaxLimit();
   virtual void  DoParMinLimit();
   virtual void  DoParStep();
   virtual void  DoParValue();
   virtual void  DoReset();
   virtual void  DoSlider();
   virtual void  HandleButtons(Bool_t update);
   virtual void  RedrawFunction();

   ClassDef(TFitParametersDialog, 0)  // Fit function parameters dialog
};

#endif

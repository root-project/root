// @(#)root/ged:$Id$
// Author: Ilka Antcheva 08/05/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFunctionParametersDialog
#define ROOT_TFunctionParametersDialog

//////////////////////////////////////////////////////////////////////////
//                                                                      //
//  TFunctionParametersDialog                                           //
//                                                                      //
//  This class is used for function parameter settings.                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif


class TF1;
class TGNumberEntry;
class TGTextEntry;
class TGCheckButton;
class TGTextButton;
class TGTripleHSlider;
class TGNumberEntryField;
class TVirtualPad;


class TFunctionParametersDialog : public TGTransientFrame {

protected:
   TF1                 *fFunc;            // function passed to this dialog
   TVirtualPad         *fFpad;            // pad where the function is drawn
   Int_t                fNP;              // number of function parameters
   Double_t            *fPmin;            // min limits of patameters range
   Double_t            *fPmax;            // max limits of patameters range
   Double_t            *fPval;            // original patameters' values
   Double_t            *fPerr;            // original patameters' errors
   Double_t             fRangexmin;       // min limits of patameters range
   Double_t             fRangexmax;       // max limits of patameters range
   Double_t             fRXmin;           // original min range
   Double_t             fRXmax;           // original max range
   TGCompositeFrame    *fContNam;         // container of parameter names
   TGCompositeFrame    *fContVal;         // container of parameter values
   TGCompositeFrame    *fContFix;         // container of fix settings
   TGCompositeFrame    *fContSld;         // container of sliders
   TGCompositeFrame    *fContMin;         // container of min range values
   TGCompositeFrame    *fContMax;         // container of max range values
   TGTextEntry         **fParNam;         // parameter names
   TGCheckButton       **fParFix;         // fix setting check buttons
   TGNumberEntry       **fParVal;         // parameter values
   TGNumberEntryField  **fParMin;         // min range values
   TGNumberEntryField  **fParMax;         // max range values
   TGTripleHSlider     **fParSld;         // triple sliders
   TGCheckButton       *fUpdate;          // Immediate update check button
   TGTextButton        *fApply;           // Apply button
   TGTextButton        *fReset;           // Reset button
   TGTextButton        *fOK;              // Cancel button
   TGTextButton        *fCancel;          // Cancel button
   Bool_t              fHasChanges;       // kTRUE if function was redrawn;
   Bool_t              fImmediateDraw;    // kTRUE if function is updated on run-time

public:
   TFunctionParametersDialog(const TGWindow *p, const TGWindow *main,
                             TF1 *func, TVirtualPad *pad,
                             Double_t rmin, Double_t rmax);
   virtual ~TFunctionParametersDialog();

   virtual void  CloseWindow();
   virtual void  DoApply();
   virtual void  DoCancel();
   virtual void  DoFix(Bool_t on);
   virtual void  DoOK();
   virtual void  DoParMaxLimit();
   virtual void  DoParMinLimit();
   virtual void  DoParValue();
   virtual void  DoReset();
   virtual void  DoSlider();
   virtual void  HandleButtons(Bool_t update);
   virtual void  RedrawFunction();

   ClassDef(TFunctionParametersDialog, 0)  // Function parameters dialog
};

#endif

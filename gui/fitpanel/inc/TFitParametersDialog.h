// @(#)root/fitpanel:$Id$
// Author: Ilka Antcheva, Lorenzo Moneta, David Gonzalez Maline 03/10/06

/*************************************************************************
 * Copyright (C) 1995-2006, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TFitParametersDialog
#define ROOT_TFitParametersDialog


#include "TGWidget.h"
#include "TGFrame.h"

enum EFPDialogBound {
   kFPDBounded,
   kFPDNoneBounded
};

/// Enumeration specifying if parameters
/// have been modified by the user
enum EFPDialogChange {
   kFPDNoChange,
   kFPDChange
};

class TF1;
class TGNumberEntry;
class TGTextEntry;
class TGCheckButton;
class TGTextButton;
class TGTripleHSlider;
class TGNumberEntryField;
class TVirtualPad;


class TFitParametersDialog : public TGTransientFrame {

protected:
   TF1                 *fFunc;            ///< function passed to this dialog
   TVirtualPad         *fFpad;            ///< pad where the function is drawn
   Bool_t               fHasChanges;      ///< kTRUE if function was redrawn;
   Bool_t               fImmediateDraw;   ///< kTRUE if function is updated on run-time
   Int_t               *fRetCode;         ///< address to store return code
   Int_t                fNP;              ///< number of function parameters
   Double_t             fRangexmin;       ///< min function range
   Double_t             fRangexmax;       ///< max function range
   Double_t            *fPmin;            ///< min limits of patameters range
   Double_t            *fPmax;            ///< max limits of patameters range
   Double_t            *fPval;            ///< original patameters' values
   Double_t            *fPerr;            ///< original patameters' errors
   Double_t            *fPstp;            ///< original patameters' step
   TGCompositeFrame    *fContNam;         ///< container of parameter names
   TGCompositeFrame    *fContVal;         ///< container of parameter values
   TGCompositeFrame    *fContFix;         ///< container of fix settings
   TGCompositeFrame    *fContBnd;         ///< container of bound settings
   TGCompositeFrame    *fContSld;         ///< container of sliders
   TGCompositeFrame    *fContMin;         ///< container of min range values
   TGCompositeFrame    *fContMax;         ///< container of max range values
   TGCompositeFrame    *fContStp;         ///< container of step values
   TGCompositeFrame    *fContErr;         ///< container of error values
   TGTextEntry         **fParNam;         ///< parameter names
   TGCheckButton       **fParBnd;         ///< bound setting switch
   TGCheckButton       **fParFix;         ///< fix setting switch
   TGNumberEntry       **fParVal;         ///< parameter values
   TGNumberEntryField  **fParMin;         ///< min range values
   TGNumberEntryField  **fParMax;         ///< max range values
   TGNumberEntry       **fParStp;         ///< step values
   TGTripleHSlider     **fParSld;         ///< triple sliders
   TGNumberEntryField  **fParErr;         ///< error values
   TGCheckButton       *fUpdate;          ///< immediate update switch
   TGTextButton        *fApply;           ///< Apply button
   TGTextButton        *fReset;           ///< Reset button
   TGTextButton        *fOK;              ///< OK button
   TGTextButton        *fCancel;          ///< Cancel button
   TList                fTextEntries;     ///< list of text entries used for keyboard navigation

   void  DisconnectSlots();
public:
   TFitParametersDialog(const TGWindow *p, const TGWindow *main, TF1 *func,
                        TVirtualPad *pad, Int_t *ret_code = 0);
   virtual ~TFitParametersDialog();

   virtual void  CloseWindow();
   virtual void  DoApply();
   virtual void  DoCancel();
   virtual void  DoOK();
   virtual void  DoParFix(Bool_t on);
   virtual void  DoParBound(Bool_t on);
   virtual void  DoParMaxLimit();
   virtual void  DoParMinLimit();
   virtual void  DoParStep();
   virtual void  DoParValue();
   virtual void  DoReset();
   virtual void  DoSlider();
   virtual void  DrawFunction();
   virtual void  HandleButtons(Bool_t update);
   virtual void  HandleShiftTab();
   virtual void  HandleTab();
   virtual Bool_t HasChanges() { return fHasChanges; }

protected:
   void SetParameters();

   ClassDef(TFitParametersDialog, 0)  // Fit function parameters dialog
};

#endif

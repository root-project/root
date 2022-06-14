// @(#)root/gui:$Id$
// Author: Fons Rademakers   10/7/2000

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTextEditDialogs
#define ROOT_TGTextEditDialogs


#include "TGFrame.h"


class TGSearchType {
public:
   Bool_t  fDirection;
   Bool_t  fCaseSensitive;
   char   *fBuffer;
   Bool_t  fClose;
   TGSearchType() { fDirection = kTRUE; fCaseSensitive = kFALSE; fBuffer = nullptr; fClose = kTRUE; }
   ~TGSearchType() { if (fBuffer) delete [] fBuffer;}
};

class TGButton;
class TGRadioButton;
class TGCheckButton;
class TGTextEntry;
class TGTextBuffer;
class TGLabel;
class TGIcon;
class TGComboBox;

class TGSearchDialog : public TGTransientFrame {

protected:
   TGCompositeFrame   *fF1, *fF2, *fF3, *fF4;  ///< sub frames
   TGLayoutHints      *fL1, *fL2, *fL3, *fL4;  ///< layout hints
   TGLayoutHints      *fL5, *fL6, *fL21, *fL9; ///< layout hints
   TGLayoutHints      *fL10;                   ///< layout hints
   TGButton           *fSearchButton;          ///< search button
   TGButton           *fCancelButton;          ///< cancel button
   TGRadioButton      *fDirectionRadio[2];     ///< search direction radio buttons
   TGCheckButton      *fCaseCheck;             ///< case check box
   TGGroupFrame       *fG2;                    ///< group frame
   TGTextEntry        *fSearch;                ///< search text entry widget
   TGTextBuffer       *fBSearch;               ///< search text buffer
   TGLabel            *fLSearch;               ///< label
   TGSearchType       *fType;                  ///< search type structure
   Int_t              *fRetCode;               ///< return code
   TGComboBox         *fCombo;                 ///< text entry combobox

   static TGSearchDialog *fgSearchDialog;      ///< global singleton

public:
   TGSearchDialog(const TGWindow *p = nullptr, const TGWindow *main = nullptr, UInt_t w = 1, UInt_t h = 1,
                  TGSearchType *sstruct = 0, Int_t *ret_code = 0,
                  UInt_t options = kVerticalFrame);
   virtual ~TGSearchDialog();

   void   CloseWindow() override;
   Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2) override;
   virtual void   SetClose(Bool_t on) { fType->fClose = on; }
   virtual Bool_t IsClose() const {  return fType->fClose; }
   virtual void   TextEntered(const char *text); //*SIGNAL*
   virtual TGSearchType *GetType() const { return fType; }

   static TGSearchDialog *&SearchDialog();

   ClassDefOverride(TGSearchDialog,0)  // Text search dialog used by TGTextEdit widget
};


class TGPrintDialog : public TGTransientFrame {

protected:
   char             **fPrinter;                    ///< printer to be used
   char             **fPrintCommand;               ///< printer command to be used
   TGCompositeFrame  *fF1, *fF2, *fF3, *fF4, *fF5; ///< sub frames
   TGLayoutHints     *fL1, *fL2, *fL3, *fL5, *fL6; ///< layout hints
   TGLayoutHints     *fL7, *fL21;                  ///< layout hints
   TGIcon            *fPrinterIcon;                ///< printer icon
   TGButton          *fPrintButton;                ///< print button
   TGButton          *fCancelButton;               ///< cancel button
   TGComboBox        *fPrinterEntry;               ///< printer list combo widget
   TGTextEntry       *fPrintCommandEntry;          ///< command text entry widget
   TGTextBuffer      *fBPrinter, *fBPrintCommand;  ///< printer and command text buffers
   TGLabel           *fLPrinter, *fLPrintCommand;  ///< printer and command labels
   Int_t             *fRetCode;                    ///< return code

public:
   TGPrintDialog(const TGWindow *p = nullptr, const TGWindow *main = nullptr, UInt_t w = 1, UInt_t h = 1,
                 char **printerName = nullptr, char **printProg = nullptr, Int_t *ret_code = nullptr,
                 UInt_t options = kVerticalFrame);
   virtual ~TGPrintDialog();

   void   CloseWindow() override;
   virtual void   GetPrinters();
   Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2) override;

   ClassDefOverride(TGPrintDialog,0)  // Print dialog used by TGTextEdit widget
};


class TGGotoDialog : public TGTransientFrame {

protected:
   TGCompositeFrame *fF1, *fF2;                ///< sub frames
   TGButton         *fGotoButton;              ///< goto button
   TGButton         *fCancelButton;            ///< cancel button
   TGLayoutHints    *fL1, *fL5, *fL6, *fL21;   ///< layout hints
   TGTextEntry      *fGoTo;                    ///< goto line number entry widget
   TGTextBuffer     *fBGoTo;                   ///< goto line number text buffer
   TGLabel          *fLGoTo;                   ///< goto label
   Long_t           *fRetCode;                 ///< return code

public:
   TGGotoDialog(const TGWindow *p = nullptr, const TGWindow *main = nullptr, UInt_t w = 1, UInt_t h = 1,
                Long_t *ret_code = nullptr, UInt_t options = kVerticalFrame);
   virtual ~TGGotoDialog();

   void   CloseWindow() override;
   Bool_t ProcessMessage(Longptr_t msg, Longptr_t parm1, Longptr_t parm2) override;

   ClassDefOverride(TGGotoDialog,0)  // Goto line dialog used by TGTextEdit widget
};

#endif

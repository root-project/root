// @(#)root/gui:$Name:  $:$Id: TGTextEditDialogs.h,v 1.1 2000/07/11 09:26:37 rdm Exp $
// Author: Fons Rademakers   10/7/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TGTextEditDialogs
#define ROOT_TGTextEditDialogs


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TGTextEditDialogs                                                    //
//                                                                      //
// This file defines several dialogs that are used by the TGTextEdit    //
// widget via its associated context popup menu.                        //
// The following dialogs are available: TGSearchDialog, TGGotoDialog    //
// and TGPrintDialog.                                                   //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif


class TGSearchType {
public:
   Bool_t  fDirection;
   Bool_t  fCaseSensitive;
   char   *fBuffer;
   TGSearchType() { fDirection = kTRUE; fCaseSensitive = kFALSE; fBuffer = 0; }
};

class TGButton;
class TGRadioButton;
class TGCheckButton;
class TGTextEntry;
class TGTextBuffer;
class TGLabel;
class TGIcon;


class TGSearchDialog : public TGTransientFrame {

protected:
   TGCompositeFrame   *fF1, *fF2, *fF3, *fF4;  // sub frames
   TGLayoutHints      *fL1, *fL2, *fL3, *fL4;  // layout hints
   TGLayoutHints      *fL5, *fL6, *fL21, *fL9; // layout hints
   TGLayoutHints      *fL10;                   // layout hints
   TGButton           *fSearchButton;          // search button
   TGButton           *fCancelButton;          // cancel button
   TGRadioButton      *fDirectionRadio[2];     // search direction radio buttons
   TGCheckButton      *fCaseCheck;             // case check box
   TGGroupFrame       *fG2;                    // group frame
   TGTextEntry        *fSearch;                // search text entry widget
   TGTextBuffer       *fBSearch;               // search text buffer
   TGLabel            *fLSearch;               // label
   TGSearchType       *fType;                  // search type structure
   Int_t              *fRetCode;               // return code

public:
   TGSearchDialog(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h,
                  TGSearchType *sstruct, Int_t *ret_code,
                  UInt_t options = kMainFrame | kVerticalFrame);
   virtual ~TGSearchDialog();

   virtual void   CloseWindow();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(TGSearchDialog,0)  // Text search dialog used by TGTextEdit widget
};


class TGPrintDialog : public TGTransientFrame {

protected:
   char             **fPrinter;                    // printer to be used
   char             **fPrintCommand;               // printer command to be used
   TGCompositeFrame  *fF1, *fF2, *fF3, *fF4, *fF5; // sub frames
   TGLayoutHints     *fL1, *fL2, *fL3, *fL5, *fL6; // layout hints
   TGLayoutHints     *fL7, *fL21;                  // layout hints
   TGIcon            *fPrinterIcon;                // printer icon
   TGButton          *fPrintButton;                // print button
   TGButton          *fCancelButton;               // cancel button
   TGTextEntry       *fPrinterEntry;               // printer text entry widget
   TGTextEntry       *fPrintCommandEntry;          // command text entry widget
   TGTextBuffer      *fBPrinter, *fBPrintCommand;  // printer and command text buffers
   TGLabel           *fLPrinter, *fLPrintCommand;  // printer and command labels
   Int_t             *fRetCode;                    // return code

public:
   TGPrintDialog(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h,
                 char **printerName, char **printProg, Int_t *ret_code,
                 UInt_t options = kMainFrame | kVerticalFrame);
   virtual ~TGPrintDialog();

   virtual void   CloseWindow();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(TGPrintDialog,0)  // Print dialog used by TGTextEdit widget
};


class TGGotoDialog : public TGTransientFrame {

protected:
   TGCompositeFrame *fF1, *fF2;                // sub frames
   TGButton         *fGotoButton;              // goto button
   TGButton         *fCancelButton;            // cancel button
   TGLayoutHints    *fL1, *fL5, *fL6, *fL21;   // layout hints
   TGTextEntry      *fGoTo;                    // goto line number entry widget
   TGTextBuffer     *fBGoTo;                   // goto line number text buffer
   TGLabel          *fLGoTo;                   // goto label
   Long_t           *fRetCode;                 // return code

public:
   TGGotoDialog(const TGWindow *p, const TGWindow *main, UInt_t w, UInt_t h,
                Long_t *ret_code, UInt_t options = kMainFrame | kVerticalFrame);
   virtual ~TGGotoDialog();

   virtual void   CloseWindow();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t parm2);

   ClassDef(TGGotoDialog,0)  // Goto line dialig used by TGTextEdit widget
};

#endif

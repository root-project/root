// @(#)root/gui:$Id$
// Author: David Gonzalez Maline  21/10/2008

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TTreeInput
#define ROOT_TTreeInput

///////////////////////////////////////////////////////////////////////////
//                                                                       //
// Tree Input Dialog Widget                                              //
//                                                                       //
// An input dialog box to select the variables and the cuts from         //
// a TTree                                                               //
//                                                                       //
///////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TGFrame
#include "TGFrame.h"
#endif

class TGLabel;
class TGTextEntry;
class TGTextButton;

class TTreeInput : public TGTransientFrame {

private:
   TGTextEntry      *fTEVars;  // text entry widget for variables
   TGTextEntry      *fTECuts;  // text entry widget for cuts
   TGTextButton     *fOk;      // ok button
   TGTextButton     *fCancel;  // cancel button
   char             *fStrvars;  // address to store variables string
   char             *fStrcuts;  // address to store cuts string

   TTreeInput(const TTreeInput&);  // Not implemented
   TTreeInput &operator= (const TTreeInput&); // Not implemented

public:
   TTreeInput(const TGWindow *p, const TGWindow *main,
              char *strvars, char* strcuts);
   ~TTreeInput();
   virtual Bool_t ProcessMessage(Long_t msg, Long_t parm1, Long_t);

   ClassDef(TTreeInput, 0)  // Simple input dialog

};

#endif

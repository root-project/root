// @(#)root/rint:$Name:  $:$Id: TRint.h,v 1.2 2001/02/22 13:32:00 rdm Exp $
// Author: Rene Brun   17/02/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef ROOT_TRint
#define ROOT_TRint

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// Rint                                                                 //
//                                                                      //
// Rint is the ROOT Interactive Interface. It allows interactive access //
// to the ROOT system via a C++ interpreter.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TApplication
#include "TApplication.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif


class TRint : public TApplication {

private:
   Int_t       fNcmd;               //Command history number
   TString     fDefaultPrompt;      //Default prompt: "root [%d] "
   char        fPrompt[64];         //Interpreter prompt
   Bool_t      fInterrupt;          //If true macro execution will be stopped

public:
   TRint(const char *appClassName, int *argc, char **argv,
         void *options = 0, int numOptions = 0, Bool_t noLogo = kFALSE);
   virtual             ~TRint();
   virtual char       *GetPrompt();
   virtual const char *SetPrompt(const char *newPrompt);
   virtual void        HandleTermInput();
   virtual void        PrintLogo();
   virtual void        Run(Bool_t retrn = kFALSE);
   virtual void        Terminate(int status);
           void        Interrupt() { fInterrupt = kTRUE; }

   ClassDef(TRint,0)  //ROOT Interactive Application Interface
};

#endif

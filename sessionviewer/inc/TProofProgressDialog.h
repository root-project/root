// @(#)root/proof:$Name:  $:$Id: TProofProgressDialog.h,v 1.1 2003/04/04 00:40:27 rdm Exp $
// Author: Fons Rademakers   21/03/03

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TProofProgressDialog
#define ROOT_TProofProgressDialog


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TProofProgressDialog                                                 //
//                                                                      //
// This class provides a query progress bar.                            //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TTime
#include "TTime.h"
#endif

class TGTransientFrame;
class TGProgressBar;
class TGTextButton;
class TGCheckButton;
class TGLabel;
class TVirtualProof;


class TProofProgressDialog {

private:
   TGTransientFrame   *fDialog;  // transient frame, main dialog window
   TGProgressBar      *fBar;     // progress bar
   TGTextButton       *fClose;
   TGTextButton       *fStop;
   TGCheckButton      *fKeep;
   TGLabel            *fFilesEvents;
   TGLabel            *fProcessed;
   TGLabel            *fTotal;
   TGLabel            *fRate;
   TVirtualProof      *fProof;
   TTime               fStartTime;
   TTime               fEndTime;
   Long64_t            fPrevProcessed;
   Long64_t            fFirst;
   Long64_t            fEntries;
   Int_t               fFiles;

   static Bool_t       fgKeep;

public:
   TProofProgressDialog(TVirtualProof *proof, const char *selector,
                        Int_t files, Long64_t first, Long64_t entries);
   ~TProofProgressDialog();
   void Progress(Long64_t total, Long64_t processed);

   void CloseWindow();
   void DoClose();
   void DoKeep(Bool_t on);
   void DoStop();
};

#endif

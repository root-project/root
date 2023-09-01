// Author: Sangsu Ryu 28/06/2011

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSelVerifyDataSet
#define ROOT_TSelVerifyDataSet

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSelVerifyDataSet                                                    //
//                                                                      //
// PROOF selector to parallel-process dataset on workers                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <TSelector.h>
#include <TString.h>

class TTree;
class TFileCollection;

class TSelVerifyDataSet : public TSelector {

private:

   Int_t fFopt;
   Int_t fSopt;
   Int_t fRopt;

   // File selection, Reopen and Touch options
   Bool_t fAllf;
   Bool_t fCheckstg;
   Bool_t fNonStgf;
   Bool_t fReopen;
   Bool_t fTouch;
   Bool_t fStgf;

   // File processing options
   Bool_t fNoaction;
   Bool_t fFullproc;
   Bool_t fLocateonly;
   Bool_t fStageonly;

   // Run options
   Bool_t fDoall;
   Bool_t fGetlistonly;
   Bool_t fScanlist;

   Bool_t fDbg;

   TString fMss;
   TString fStageopts;

   Bool_t fChangedDs;
   Int_t fTouched;
   Int_t fOpened;
   Int_t fDisappeared;

   TFileCollection *fSubDataSet; // Sub-dataset being verified

   void InitMembers();

public :

   TSelVerifyDataSet(TTree *);
   TSelVerifyDataSet();
   ~TSelVerifyDataSet() override {}
   Int_t   Version() const override {return 1;}
   void    Begin(TTree *) override { }
   void    SlaveBegin(TTree *tree) override;
   void    Init(TTree *) override { }
   Bool_t  Notify() override { return kTRUE; }
   Bool_t  Process(Long64_t entry) override;
   void    SetOption(const char *option) override { fOption = option; }
   void    SetObject(TObject *obj) override { fObject = obj; }
   void    SetInputList(TList *input) override {fInput = input;}
   TList  *GetOutputList() const override { return fOutput; }
   void    SlaveTerminate() override;
   void    Terminate() override { }

   ClassDefOverride(TSelVerifyDataSet,0) //PROOF selector for parallel dataset verification
};

#endif

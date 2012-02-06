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

#ifndef ROOT_TSelector
#include <TSelector.h>
#endif
#ifndef ROOT_TString
#include <TString.h>
#endif

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

public :

   TSelVerifyDataSet(TTree *);
   TSelVerifyDataSet();
   virtual ~TSelVerifyDataSet() {}
   virtual Int_t   Version() const {return 1;}
   virtual void    Begin(TTree *);
   virtual void    SlaveBegin(TTree *tree);
   virtual void    Init(TTree *tree);
   virtual Bool_t  Notify();
   virtual Bool_t  Process(Long64_t entry);
   virtual void    SetOption(const char *option) { fOption = option; }
   virtual void    SetObject(TObject *obj) { fObject = obj; }
   virtual void    SetInputList(TList *input) {fInput = input;}
   virtual TList  *GetOutputList() const { return fOutput; }
   virtual void    SlaveTerminate();
   virtual void    Terminate();

   ClassDef(TSelVerifyDataSet,0) //PROOF selector for parallel dataset verification
};

#endif

#ifdef TSelVerifyDataSet_cxx
void TSelVerifyDataSet::Init(TTree *)
{
   fFopt = -1;
   fSopt = 0;
   fRopt = 0;

   fAllf = 0;
   fCheckstg = 0;
   fNonStgf = 0;
   fReopen = 0;
   fTouch = 0;
   fStgf = 0;
   fNoaction = 0;
   fFullproc = 0;
   fLocateonly = 0;
   fStageonly = 0;
   fDoall       = 0;
   fGetlistonly = 0;
   fScanlist    = 0;
   fDbg = 0;

   fChangedDs = kFALSE;
   fTouched = 0;
   fOpened = 0;
   fDisappeared = 0;
   fSubDataSet = 0;
}

Bool_t TSelVerifyDataSet::Notify()
{
   return kTRUE;
}
#endif // #ifdef TSelVerifyDataSet_cxx

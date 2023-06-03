// @(#)root/proof:$Id$
// Author: Sangsu Ryu 22/06/2010

/*************************************************************************
 * Copyright (C) 1995-2005, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSelHandleDataSet
#define ROOT_TSelHandleDataSet

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSelHandleDataSet                                                    //
//                                                                      //
// PROOF selector for file cache release.                               //
// List of files to be cleaned for each node is provided by client.     //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <TSelector.h>
#include <TString.h>

class TList;
class TPBHandleDSType;

class TSelHandleDataSet : public TSelector {

private:
   TPBHandleDSType *fType;
   TString          fDestDir;
   void ReleaseCache(const char *fn);
   void CheckCache(const char *fn);
   void RemoveFile(const char *fn);
   void CopyFile(const char *fn);

public :

   TSelHandleDataSet() : fType(0) { }
   ~TSelHandleDataSet() override { }
   Int_t   Version() const override {return 2;}
   void    Begin(TTree *) override { }
   void    SlaveBegin(TTree *) override;
   void    Init(TTree *) override { }
   Bool_t  Notify() override { return kTRUE; }
   Bool_t  Process(Long64_t entry) override;
   void    SetOption(const char *option) override { fOption = option; }
   void    SetObject(TObject *obj) override { fObject = obj; }
   void    SetInputList(TList *input) override {fInput = input;}
   TList  *GetOutputList() const override { return fOutput; }
   void    SlaveTerminate() override { }
   void    Terminate() override { }

   ClassDefOverride(TSelHandleDataSet,0)     //PROOF selector for event file generation
};

#endif


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
   virtual ~TSelHandleDataSet() { }
   virtual Int_t   Version() const {return 2;}
   virtual void    Begin(TTree *) { }
   virtual void    SlaveBegin(TTree *);
   virtual void    Init(TTree *) { }
   virtual Bool_t  Notify() { return kTRUE; }
   virtual Bool_t  Process(Long64_t entry);
   virtual void    SetOption(const char *option) { fOption = option; }
   virtual void    SetObject(TObject *obj) { fObject = obj; }
   virtual void    SetInputList(TList *input) {fInput = input;}
   virtual TList  *GetOutputList() const { return fOutput; }
   virtual void    SlaveTerminate() { }
   virtual void    Terminate() { }

   ClassDef(TSelHandleDataSet,0)     //PROOF selector for event file generation
};

#endif


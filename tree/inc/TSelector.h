// @(#)root/treeplayer:$Name:  $:$Id: TSelector.h,v 1.9 2002/02/15 22:12:30 brun Exp $
// Author: Rene Brun   05/02/97

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSelector
#define ROOT_TSelector


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TSelector                                                            //
//                                                                      //
// A utility class for Trees selections.                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////


#ifndef ROOT_TObject
#include "TObject.h"
#endif
#ifndef ROOT_TString
#include "TString.h"
#endif

typedef long Long64_t;

class TTree;

class TSelector : public TObject {
protected:
   TString   fOption;  //option given to TTree::Process
   TObject  *fObject;  //current object if processing object (vs. TTree)
   TList    *fInput;   //list of objects available during processing (on PROOF)
   TList    *fOutput;  //list of objects created during processing (on PROOF)

public:
   TSelector();
   virtual            ~TSelector();
   virtual void        Init(TTree *) { }
   virtual void        Begin(TTree *) { }
   virtual Bool_t      Notify() { return kTRUE; }
           const char *GetOption() const { return fOption.Data();}
   virtual Bool_t      ProcessCut(int entry) { return kTRUE; }
   virtual void        ProcessFill(int entry) { }
   virtual Bool_t      Process(int entry) { return kFALSE; }
           void        SetOption(const char *option) { fOption = option; }
           void        SetObject(TObject *obj) { fObject = obj; }
           void        SetInputList(TList *input) { fInput = input; }
           TList      *GetOutputList() const { return fOutput; }
   virtual void        Terminate() { }

   static  TSelector  *GetSelector(const char *filename);

   ClassDef(TSelector,0)  //A utility class for tree and object processing
};

#endif

